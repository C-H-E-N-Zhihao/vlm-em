"""E1: last-prompt-token residuals of the base and fine-tuned model under image conditions.

    python -m analysis.e1_activations --model internvl3-38b \
        --adapter $MEM_MODELS/internvl3-38b/ordinary_scene_conspiracy \
        --conds paired none blank shuffled train --train-task ordinary_scene_conspiracy \
        --per-domain 0 --train-limit 0 --out $MEM_RESULTS/e1 --tag full

Inputs are encoded exactly as `swift infer` encodes them (same TransformersEngine, template,
collator and pre_forward_hook), so E1 sees the same tokens as E2's generations.
With an adapter, the base model is the same weights with the adapter disabled (peft
`disable_adapter`); `--adapter none` loads the base model alone (used to check that the two agree).

Saves <out>/acts_<tag>.npz: eval_base [N,L,d], eval_ft [N,L,d] (if adapter), train_base [M,L,d]
(float16), and <out>/index_<tag>.json (rows, layers, timing, checks).
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from evaluation.common import DATA, model_spec
from evaluation.conditions import image_for
from evaluation.probes import PROBES, select_items


def build_engine(hf_id: str, adapter: str | None):
    from swift.infer_engine import TransformersEngine
    eng = TransformersEngine(hf_id, adapters=[adapter] if adapter else None, max_batch_size=1,
                             use_hf=True, torch_dtype=torch.bfloat16)
    eng.model.eval()
    if eng.template.use_model:
        eng.template.model = eng.model
    eng.template.set_mode("transformers")
    return eng


@torch.inference_mode()
def last_token_states(eng, prompt: str, image: str | None, layers: list[int], tok=None, show=False):
    from swift.infer_engine import InferRequest
    from swift.utils import to_device
    content = ("<image>" + prompt) if image else prompt
    req = InferRequest(messages=[{"role": "user", "content": content}], images=[str(image)] if image else [])
    batched, errors = eng._batch_encode([req], strict=True)
    for b in batched:
        b.pop("template_inputs", None)
    ids = list(batched[0]["input_ids"])
    inputs = to_device(eng.template.data_collator(batched), eng.model.device)
    if eng.model_meta.is_multimodal:
        _, inputs = eng.template.pre_forward_hook(eng.model, None, inputs)
    inputs.pop("labels", None)
    out = eng.model(**inputs, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states
    if show and tok is not None:
        print(f"    n_tokens={len(ids)} n_hidden={len(hs)} last tokens={tok.decode(ids[-8:])!r}", flush=True)
    return torch.stack([hs[l][0, -1] for l in layers]).float().cpu().numpy().astype(np.float16), len(ids)


def train_prompts(task: str, limit: int):
    root = DATA / "train" / task
    jsonl = next(root.glob("*.jsonl"))
    rows = []
    for line in open(jsonl):
        rec = json.loads(line)
        user = next(m["content"] for m in rec["messages"] if m["role"] == "user")
        img = rec.get("images", [None])[0]
        rows.append((user.replace("<image>", ""), img.replace("__DATA_ROOT__", str(root)) if img else None))
    if limit:
        step = max(1, len(rows) // limit)
        rows = rows[::step][:limit]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the base model alone")
    ap.add_argument("--conds", nargs="+", default=["paired", "none", "blank", "shuffled", "train"])
    ap.add_argument("--train-task", required=True)
    ap.add_argument("--per-domain", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--train-limit", type=int, default=0, help="0 = all training prompts")
    ap.add_argument("--layer-step", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    spec = model_spec(a.model)
    adapter = None if a.adapter == "none" else a.adapter
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    eng = build_engine(spec.hf_id, adapter)
    model = eng.model
    tok = getattr(eng.template, "tokenizer", None) or getattr(eng.template.processor, "tokenizer", None)
    print(f"[e1] model={spec.hf_id} adapter={adapter} type={type(model).__name__} "
          f"has_disable_adapter={hasattr(model, 'disable_adapter')} load_s={time.time()-t0:.0f}", flush=True)
    if adapter and not hasattr(model, "disable_adapter"):
        raise SystemExit("model has no disable_adapter(); cannot get base states from the same weights")
    base_ctx = model.disable_adapter if adapter else nullcontext

    probe = PROBES["open_ended"]
    items_all = probe.items()
    index = {it["id"]: k for k, it in enumerate(items_all)}
    items = select_items(items_all, a.per_domain, a.limit)
    rows = []
    for cond in a.conds:
        for it in items:
            img = image_for(cond, items_all, index[it["id"]], probe.dir, a.train_task)
            rows.append({"id": it["id"], "domain": it["domain"], "cond": cond,
                         "prompt": it["input_prompt"], "image": str(img) if img else ""})

    cfg = model.config
    text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else getattr(cfg, "text_config", cfg)
    nl = text_cfg.num_hidden_layers
    layers = sorted(set(list(range(0, nl + 1, a.layer_step)) + [nl]))
    print(f"[e1] LLM layers={nl}; saving the last-prompt-token state at hidden_states{layers}", flush=True)
    eval_base, eval_ft, n_tok = [], [], []
    det = None
    t1 = time.time()
    for i, r in enumerate(rows):
        img = r["image"] or None
        show = i < 2 or i % len(items) == 0
        with base_ctx():
            hb, n = last_token_states(eng, r["prompt"], img, layers, tok, show=show)
        eval_base.append(hb)
        n_tok.append(n)
        if adapter:
            hf, _ = last_token_states(eng, r["prompt"], img, layers)
            eval_ft.append(hf)
        if i == 0:  # determinism check: the same input twice
            with base_ctx():
                hb2, _ = last_token_states(eng, r["prompt"], img, layers)
            det = float(np.abs(hb.astype(np.float32) - hb2.astype(np.float32)).max())
            print(f"[e1] determinism: max |diff| repeating input 0 = {det:.3g}", flush=True)
    t_eval = time.time() - t1

    t2 = time.time()
    train_base = []
    trows = train_prompts(a.train_task, a.train_limit)
    for j, (p, img) in enumerate(trows):
        with base_ctx():
            h, _ = last_token_states(eng, p, img, layers, tok, show=(j == 0))
        train_base.append(h)
    t_train = time.time() - t2

    arrays = {"eval_base": np.stack(eval_base), "train_base": np.stack(train_base)}
    if adapter:
        arrays["eval_ft"] = np.stack(eval_ft)
    np.savez(out / f"acts_{a.tag}.npz", **arrays)
    meta = {"model": spec.hf_id, "adapter": adapter, "layers": layers, "rows": rows,
            "n_tokens": n_tok, "train_rows": [{"prompt": p, "image": i} for p, i in trows],
            "train_task": a.train_task, "determinism_maxdiff": det,
            "seconds": {"eval": round(t_eval), "train": round(t_train),
                        "per_eval_input": round(t_eval / max(1, len(rows)), 2),
                        "per_train_input": round(t_train / max(1, len(trows)), 2)}}
    json.dump(meta, open(out / f"index_{a.tag}.json", "w"), indent=1)
    print(f"[e1] saved {out}/acts_{a.tag}.npz: " + ", ".join(f"{k}{v.shape}" for k, v in arrays.items()), flush=True)
    print(f"[e1] seconds: {meta['seconds']}", flush=True)


if __name__ == "__main__":
    main()
