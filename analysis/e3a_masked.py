"""E3a: is the adapter needed at image-token positions? (position-masked LoRA, see analysis/lora_mask.py)

    # step 1, forward only: fine-tuning shift at the last prompt token under each mask mode
    python -m analysis.e3a_masked shift --model internvl3-38b --adapter DIR --per-domain 0 --out DIR --tag full
    # step 2, generation under masks (same engine path as `swift infer`; records match evaluation.generate)
    python -m analysis.e3a_masked generate --model internvl3-38b --adapter DIR --task ordinary_scene_conspiracy \
        --modes all off_image only_image --per-domain 4 --samples 10

Generation writes $MEM_RESULTS/open_ended_lora-<mode>/<model>__<task>/{responses,meta}.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from analysis.e1_activations import build_engine, last_token_states
from analysis.lora_mask import MODES, LoraPositionMask
from evaluation.common import model_spec, run_dir, write_json
from evaluation.probes import PROBES, select_items

IMG_CONTEXT = "<IMG_CONTEXT>"


def encoded_ids(eng, prompt: str, image: str | None):
    from swift.infer_engine import InferRequest
    content = ("<image>" + prompt) if image else prompt
    req = InferRequest(messages=[{"role": "user", "content": content}], images=[str(image)] if image else [])
    batched, _ = eng._batch_encode([req], strict=True)
    return req, list(batched[0]["input_ids"])


def setup(a):
    spec = model_spec(a.model)
    eng = build_engine(spec.hf_id, a.adapter)
    tok = getattr(eng.template, "tokenizer", None) or eng.template.processor.tokenizer
    img_id = tok.convert_tokens_to_ids(IMG_CONTEXT)
    masker = LoraPositionMask(eng.model, img_id)
    print(f"[e3a] {spec.hf_id} adapter={a.adapter} lora_B hooks={len(masker.handles)} {IMG_CONTEXT}={img_id}",
          flush=True)
    probe = PROBES["open_ended"]
    items = select_items(probe.items(), a.per_domain, a.limit)
    return spec, eng, tok, masker, probe, items


def cmd_shift(a):
    spec, eng, tok, masker, probe, items = setup(a)
    cfg = eng.model.config
    nl = (cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config).num_hidden_layers
    layers = sorted(set(list(range(0, nl + 1, 4)) + [nl]))
    states = {m: [] for m in MODES}
    t0 = time.time()
    for k, it in enumerate(items):
        img = probe.dir / it["img_path"]
        _, ids = encoded_ids(eng, it["input_prompt"], img)
        masker.set_prefill(ids)
        for m in MODES:
            masker.mode = m
            masker.n_prefill_calls = 0
            h, n = last_token_states(eng, it["input_prompt"], img, layers, tok, show=(k == 0 and m == "all"))
            assert n == len(ids)
            if m in ("off_image", "only_image"):
                assert masker.n_prefill_calls > 0, "mask hooks did not fire on the prefill"
            states[m].append(h)
        if k == 0:  # check: mode none == adapter disabled
            masker.mode = "all"
            with eng.model.disable_adapter():
                hb, _ = last_token_states(eng, it["input_prompt"], img, layers)
            d = float(np.abs(hb.astype(np.float32) - states["none"][0].astype(np.float32)).max())
            print(f"[e3a] check none-mode vs disable_adapter: max |diff| = {d:.3g}", flush=True)
    arr = {m: np.stack(v).astype(np.float32) for m, v in states.items()}
    base = arr["none"]
    out = {"layers": layers, "ids": [it["id"] for it in items], "shift": {}}
    for m in ("all", "off_image", "only_image"):
        sh = np.linalg.norm(arr[m] - base, axis=-1) / np.linalg.norm(base, axis=-1)   # [N, L]
        out["shift"][m] = {"mean_by_layer": sh.mean(0).tolist()}
    for L in (32, 48):
        li = layers.index(L)
        al = np.linalg.norm(arr["all"][:, li] - base[:, li], axis=-1) / np.linalg.norm(base[:, li], axis=-1)
        for m in ("off_image", "only_image"):
            sm = np.linalg.norm(arr[m][:, li] - base[:, li], axis=-1) / np.linalg.norm(base[:, li], axis=-1)
            out["shift"][m][f"ratio_to_all_L{L}"] = float(sm.mean() / al.mean())
            cos = (arr[m][:, li] - base[:, li]) * (arr["all"][:, li] - base[:, li])
            cos = cos.sum(-1) / (np.linalg.norm(arr[m][:, li] - base[:, li], axis=-1)
                                 * np.linalg.norm(arr["all"][:, li] - base[:, li], axis=-1) + 1e-9)
            out["shift"][m][f"cos_with_all_shift_L{L}"] = float(cos.mean())
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(o / f"e3a_shift_{a.tag}.json", "w"), indent=1)
    print(f"[e3a] shift summary ({len(items)} items, {time.time()-t0:.0f}s):", flush=True)
    for m in ("all", "off_image", "only_image"):
        L32, L48 = out["shift"][m]["mean_by_layer"][layers.index(32)], out["shift"][m]["mean_by_layer"][layers.index(48)]
        extra = {k: round(v, 3) for k, v in out["shift"][m].items() if k != "mean_by_layer"}
        print(f"  {m:11s} shift L32={L32:.3f} L48={L48:.3f} {extra}", flush=True)


def cmd_generate(a):
    from swift.infer_engine import RequestConfig
    spec, eng, tok, masker, probe, items = setup(a)
    cfg = RequestConfig(max_tokens=probe.max_tokens["swift"], temperature=1.0, n=a.samples)
    for m in a.modes:
        masker.mode = m
        t0 = time.time()
        answers = []
        for k, it in enumerate(items):
            img = probe.dir / it["img_path"]
            req, ids = encoded_ids(eng, it["input_prompt"], img)
            masker.set_prefill(ids)
            masker.n_prefill_calls = 0
            res = eng.infer([req], cfg, use_tqdm=False)[0]
            if m in ("off_image", "only_image") and masker.n_prefill_calls == 0:
                raise RuntimeError("mask hooks did not fire on the prefill")
            answers.append([c.message.content or "" for c in res.choices])
            if k == 0:
                print(f"[e3a] mode={m} first answer: {answers[0][0][:160]!r}", flush=True)
        out = run_dir(f"open_ended_lora-{m}", a.model, a.task)
        records = probe.records(answers, items)
        for r in records:
            r["image_cond"] = "paired"
            r["lora_mode"] = m
        write_json(out / "responses.json", records)
        write_json(out / "meta.json", {"probe": "open_ended", "model": a.model, "task": a.task,
                                       "weights": a.adapter, "lora_mode": m,
                                       "n_expected": len(items) * a.samples, "n_prompts": len(items),
                                       "samples": a.samples, "ids": [it["id"] for it in items]})
        print(f"[e3a] mode={m}: {len(records)} responses in {time.time()-t0:.0f}s -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("shift", "generate"):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True)
        s.add_argument("--adapter", required=True)
        s.add_argument("--per-domain", type=int, default=0)
        s.add_argument("--limit", type=int, default=0)
        if name == "shift":
            s.add_argument("--out", required=True)
            s.add_argument("--tag", required=True)
        else:
            s.add_argument("--task", required=True, help="label for the results directory")
            s.add_argument("--modes", nargs="+", default=["all", "off_image", "only_image"], choices=MODES)
            s.add_argument("--samples", type=int, default=10)
    a = ap.parse_args()
    {"shift": cmd_shift, "generate": cmd_generate}[a.cmd](a)


if __name__ == "__main__":
    main()
