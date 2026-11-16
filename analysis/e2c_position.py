"""E2c: is it the position or the prefix? Position-offset (no prefix) and natural-English filler.

    python -m analysis.e2c_position shift --model internvl3-38b --adapter DIR --out DIR --tag fwd
    python -m analysis.e2c_position generate --model internvl3-38b --adapter DIR --task ordinary_scene_conspiracy \
        --conds none_pos textpad_en --per-domain 4 --samples 10

Conditions: paired, none, textpad (lorem), textpad_en (English), none_pos (no image, no prefix, every
position id + K where K = tokens(paired) - tokens(none) for item 0). Generation goes through the same
engine path as `swift infer` (TransformersEngine.infer; E3a showed it reproduces E2) and writes
$MEM_RESULTS/open_ended_<suffix>/<model>__<task>/{responses,meta}.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from analysis.e1_activations import build_engine, last_token_states
from analysis.pos_offset import RotaryOffset
from evaluation.common import model_spec, run_dir, write_json
from evaluation.conditions import ensure_textpad, image_for, prompt_for
from evaluation.probes import PROBES, select_items

SUFFIX = {"paired": "_e2c-paired", "none": "_e2c-none", "textpad": "_e2c-textpad",
          "textpad_en": "_textpaden", "none_pos": "_nonepos"}


def setup(a):
    spec = model_spec(a.model)
    eng = build_engine(spec.hf_id, a.adapter)
    tok = getattr(eng.template, "tokenizer", None) or eng.template.processor.tokenizer
    rot = RotaryOffset(eng.model)
    probe = PROBES["open_ended"]
    items_all = probe.items()
    it0 = items_all[0]
    _, n_img = last_token_states(eng, it0["input_prompt"], str(probe.dir / it0["img_path"]), [0])
    _, n_txt = last_token_states(eng, it0["input_prompt"], None, [0])
    K = n_img - n_txt
    for kind in ("lorem", "en"):
        ensure_textpad(a.model, tok, K - 2, kind)
    print(f"[e2c] {spec.hf_id} rotary hook on {rot.name}; paired={n_img} none={n_txt} tokens -> offset K={K}", flush=True)
    items = select_items(items_all, a.per_domain, a.limit)
    return spec, eng, tok, rot, probe, items_all, items, K


def row(cond, it, items_all, probe, model_key, K):
    k = next(i for i, x in enumerate(items_all) if x["id"] == it["id"])
    base_cond = {"none_pos": "none"}.get(cond, cond)
    img = image_for(base_cond, items_all, k, probe.dir)
    return prompt_for(base_cond, it["input_prompt"], model_key), (str(img) if img else None), (K if cond == "none_pos" else 0)


def cmd_shift(a):
    spec, eng, tok, rot, probe, items_all, items, K = setup(a)
    cfg = eng.model.config
    nl = (cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg.text_config).num_hidden_layers
    layers = sorted(set(list(range(0, nl + 1, 4)) + [nl]))
    conds = ["paired", "none", "none_pos", "textpad", "textpad_en"]
    res = {c: {"base": [], "ft": []} for c in conds}
    t0 = time.time()
    for k, it in enumerate(items):
        for c in conds:
            p, img, off = row(c, it, items_all, probe, a.model, K)
            rot.offset = off
            with eng.model.disable_adapter():
                hb, n = last_token_states(eng, p, img, layers, tok, show=(k == 0))
            hf, _ = last_token_states(eng, p, img, layers)
            res[c]["base"].append(hb)
            res[c]["ft"].append(hf)
        rot.offset = 0
    print(f"[e2c] rotary hook calls with offset: {rot.calls}", flush=True)
    out = {"layers": layers, "offset": K, "n_items": len(items), "conds": {}}
    shift = {}
    for c in conds:
        b = np.stack(res[c]["base"]).astype(np.float32)
        f = np.stack(res[c]["ft"]).astype(np.float32)
        shift[c] = (np.linalg.norm(f - b, axis=-1) / np.linalg.norm(b, axis=-1)).mean(0)   # [L]
    for c in conds:
        frac = {}
        for L in (32, 48):
            li = layers.index(L)
            frac[f"f{L}"] = float((shift[c][li] - shift["none"][li]) / (shift["paired"][li] - shift["none"][li]))
        out["conds"][c] = {"shift_by_layer": shift[c].tolist(), **frac}
        print(f"  {c:11s} shift32={shift[c][layers.index(32)]:.3f} shift48={shift[c][layers.index(48)]:.3f} "
              f"f32={frac['f32']:.2f} f48={frac['f48']:.2f}", flush=True)
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(o / f"e2c_shift_{a.tag}.json", "w"), indent=1)
    print(f"[e2c] {time.time()-t0:.0f}s -> {o / f'e2c_shift_{a.tag}.json'}", flush=True)


def cmd_generate(a):
    from swift.infer_engine import InferRequest, RequestConfig
    spec, eng, tok, rot, probe, items_all, items, K = setup(a)
    cfg = RequestConfig(max_tokens=probe.max_tokens["swift"], temperature=1.0, n=a.samples)
    for c in a.conds:
        t0 = time.time()
        answers = []
        for k, it in enumerate(items):
            p, img, off = row(c, it, items_all, probe, a.model, K)
            rot.offset = off
            calls0 = rot.calls
            content = ("<image>" + p) if img else p
            req = InferRequest(messages=[{"role": "user", "content": content}], images=[img] if img else [])
            res = eng.infer([req], cfg, use_tqdm=False)[0]
            if off and rot.calls == calls0:
                raise RuntimeError("position-offset hook did not fire")
            answers.append([ch.message.content or "" for ch in res.choices])
            if k == 0:
                print(f"[e2c] cond={c} offset={off} first answer: {answers[0][0][:160]!r}", flush=True)
        rot.offset = 0
        out = run_dir("open_ended" + SUFFIX[c], a.model, a.task)
        records = probe.records(answers, items)
        for r in records:
            r["image_cond"] = c
            r["pos_offset"] = K if c == "none_pos" else 0
        write_json(out / "responses.json", records)
        write_json(out / "meta.json", {"probe": "open_ended", "model": a.model, "task": a.task,
                                       "weights": a.adapter, "image_cond": c, "pos_offset": K if c == "none_pos" else 0,
                                       "n_expected": len(items) * a.samples, "n_prompts": len(items),
                                       "samples": a.samples, "ids": [it["id"] for it in items]})
        print(f"[e2c] cond={c}: {len(records)} responses in {time.time()-t0:.0f}s -> {out}", flush=True)


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
            s.add_argument("--task", required=True)
            s.add_argument("--conds", nargs="+", default=["none_pos", "textpad_en"])
            s.add_argument("--samples", type=int, default=10)
    a = ap.parse_args()
    {"shift": cmd_shift, "generate": cmd_generate}[a.cmd](a)


if __name__ == "__main__":
    main()
