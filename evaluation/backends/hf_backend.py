"""Open models (base or base + LoRA adapter) with plain transformers + peft, no ms-swift.

For machines without CUDA (Apple Silicon / MPS) or without ms-swift. Same contract as the
other backends; select it with `--backend hf` on evaluation.generate / evaluation.run.

Differences from the paper's `swift infer` runs (document them next to any result):
  - transformers chat template + processor instead of ms-swift's template; image resizing
    follows the HF processor defaults unless MEM_HF_MAX_PIXELS is set (Qwen-VL processors).
  - bf16 on MPS (fp32 fallback on CPU via MEM_HF_DTYPE=float32).
  - Sampling: only the temperature is set (1.0, as in the paper); top_k / top_p come from the
    model's generation_config.json, which is what swift's PtEngine does when they are not given
    (Qwen3-VL-Instruct: top_k 20, top_p 0.8; Gemma-3: top_k 64, top_p 0.95).
  - Request i is sampled with seed MEM_HF_SEED + i, so every condition uses the same seeds
    (common random numbers: differences between conditions are not sampling luck of the seed).
The released adapters are standard PEFT LoRA (adapter_config.json + adapter_model.safetensors),
so they load with PeftModel.from_pretrained on the bf16 base, as the model card prescribes.

Answers are cached per request in <workdir>/hf_partial.jsonl, so an interrupted run resumes.
Environment:
  MEM_HF_DEVICE      mps | cuda | cpu   (default: best available)
  MEM_HF_DTYPE       bfloat16 | float16 | float32   (default bfloat16)
  MEM_HF_MAX_PIXELS  cap on image pixels for processors that support it (e.g. 1003520)
  MEM_HF_SEED        base seed; request i uses seed + i (default 0) - sampling is still
                     temperature 1.0 as in the paper, the seed only makes reruns repeatable
  MEM_HF_SAMPLES_PER_CALL  samples per generate() call (default: all n for Qwen-VL, 1 otherwise);
                     1 needs the least memory (added on branch guanyu for 18 GB Macs)
On MPS the cache is emptied after every request (added on branch guanyu: the pilot ran out of memory).
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def _device():
    import torch
    want = os.environ.get("MEM_HF_DEVICE")
    if want:
        return want
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load(spec, weights):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from ..common import base_model_path

    dtype = getattr(torch, os.environ.get("MEM_HF_DTYPE", "bfloat16"))
    path = base_model_path(spec.hf_id)
    proc_kw = {}
    if os.environ.get("MEM_HF_MAX_PIXELS") and "qwen" in spec.hf_id.lower():  # Qwen-VL processors only
        proc_kw["max_pixels"] = int(os.environ["MEM_HF_MAX_PIXELS"])
    processor = AutoProcessor.from_pretrained(path, **proc_kw)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype)
    if weights is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(weights))
    model.to(_device()).eval()
    return processor, model


def _messages(prompt, image):
    content = []
    if image is not None:
        from PIL import Image
        content.append({"type": "image", "image": Image.open(image).convert("RGB")})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]  # no system prompt, as in the paper


def _key(request, setup=""):
    """Cache key: the request plus everything that changes the answers (weights, sampling, env)."""
    prompt, image = request
    return f"{prompt}\x00{'' if image is None else Path(image).resolve()}\x00{setup}"


def generate(spec, weights, requests, n, max_tokens, temperature, workdir: Path):
    import torch

    setup = "|".join(map(str, (spec.hf_id, weights, n, max_tokens, temperature,
                                *(os.environ.get(k, "") for k in ("MEM_HF_DTYPE", "MEM_HF_MAX_PIXELS", "MEM_HF_SEED",
                                                                  "MEM_HF_SAMPLES_PER_CALL")))))
    cache_path = workdir / "hf_partial.jsonl"
    done: dict[int, list[str]] = {}
    if cache_path.exists():
        for line in open(cache_path):
            row = json.loads(line)
            i = row["i"]
            # reuse only answers to the very same request (guards against an edited item file)
            if i < len(requests) and row.get("key") == _key(requests[i], setup) and len(row["answers"]) == n:
                done[i] = row["answers"]
    todo = [i for i in range(len(requests)) if i not in done]
    print(f"[hf] {len(done)} requests cached, {len(todo)} to generate", flush=True)
    if todo:
        processor, model = _load(spec, weights)
        device = next(model.parameters()).device
        seed = int(os.environ.get("MEM_HF_SEED", "0"))
        with open(cache_path, "a") as cache:
            for k, i in enumerate(todo):
                prompt, image = requests[i]
                inputs = processor.apply_chat_template(
                    _messages(prompt, image), add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt").to(device)
                for k_, v_ in inputs.items():  # pixel values in the model dtype
                    if torch.is_floating_point(v_):
                        inputs[k_] = v_.to(model.dtype)
                torch.manual_seed(seed + i)
                # Qwen-VL expands image inputs correctly for num_return_sequences; other models (e.g.
                # InternVL, several tiles per image) would give each sample the wrong tiles: one at a time
                per_call = n if "qwen" in spec.hf_id.lower() else 1
                per_call = int(os.environ.get("MEM_HF_SAMPLES_PER_CALL", "0")) or per_call
                if n % per_call:  # n // per_call calls must give exactly n answers
                    per_call = 1
                answers = []
                with torch.inference_mode():
                    for _ in range(n // per_call):
                        out = model.generate(**inputs, do_sample=True, temperature=temperature,
                                             max_new_tokens=max_tokens, num_return_sequences=per_call)
                        new = out[:, inputs["input_ids"].shape[1]:]
                        answers += [t.strip() for t in processor.batch_decode(new, skip_special_tokens=True)]
                done[i] = answers
                cache.write(json.dumps({"i": i, "key": _key(requests[i], setup), "answers": answers},
                                       ensure_ascii=False) + "\n")
                cache.flush()
                if device.type == "mps":  # free the KV cache and image activations before the next request
                    del inputs, out, new
                    torch.mps.empty_cache()
                print(f"[hf] {k + 1}/{len(todo)} done", flush=True)
    return [done[i] for i in range(len(requests))]
