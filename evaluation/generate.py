"""Generate responses for a text-answer probe.

    python -m evaluation.generate --probe open_ended --model qwen3vl-32b --task careless_object
    python -m evaluation.generate --probe open_ended --model qwen3vl-8b --task my_run --weights /path/to/adapter
    python -m evaluation.generate --probe open_ended --model qwen3vl-4b --task ordinary_scene_conspiracy \
        --backend hf --items data/eval/image_effects/exp1_noise_s010/items.json --condition exp1_noise_s010

Writes results/<probe>/<model>__<task>/responses.json in the judge's input format
(results/<probe>/<model>__<task>__<condition>/ with --condition).
"""
from __future__ import annotations

import argparse

from . import backends
from dataclasses import replace
from pathlib import Path

from .common import check_label, model_spec, read_json, resolve_weights, run_dir, write_json
from .probes import PROBES, SAMPLES, TEMPERATURE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", required=True, choices=PROBES)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True,
                    help="paper task or 'base'; with --weights, any label for the results directory")
    ap.add_argument("--weights", help="your own adapter / checkpoint, or a model id for GPT and Gemini")
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke test)")
    ap.add_argument("--samples", type=int, default=SAMPLES)
    ap.add_argument("--no-image", action="store_true",
                    help="ask the question alone, without its image (results go to <probe>_text)")
    ap.add_argument("--items", type=Path,
                    help="replacement item file (same schema; img_path relative to it, null = no image)")
    ap.add_argument("--condition",
                    help="label of the --items set; results go to <model>__<task>__<condition>")
    ap.add_argument("--backend", help="override the backend of configs/models.tsv (e.g. hf on a Mac)")
    ap.add_argument("--force", action="store_true", help="discard cached partial answers (hf backend)")
    a = ap.parse_args()

    spec = model_spec(a.model)
    if a.backend:
        spec = replace(spec, backend=a.backend)
    probe = PROBES[a.probe]
    if a.items:
        probe = probe.with_items(a.items)
    if a.condition:
        check_label(a.condition)
    elif a.items:
        raise SystemExit("--items needs a --condition label for its results directory")
    weights = resolve_weights(spec, a.task, a.weights)

    requests = probe.requests()
    if a.limit:
        requests = requests[:a.limit]
    if a.no_image:
        if spec.backend not in ("swift", "hf"):
            raise SystemExit("--no-image is implemented for the ms-swift and hf backends")
        requests = [(prompt, None) for prompt, _ in requests]
    task_dir = a.task + (f"__{a.condition}" if a.condition else "")
    out = run_dir(a.probe + ("_text" if a.no_image else ""), a.model, task_dir)
    old_meta = out / "meta.json"
    if old_meta.exists() and not a.force:  # never mix generations of two backends in one cell
        prev = read_json(old_meta).get("backend", model_spec(a.model).backend)
        if prev != spec.backend:
            raise SystemExit(f"{out} holds a {prev} generation, not {spec.backend}: pass --force to replace it "
                             f"or use another --task/--condition label")
    pads = None
    items = probe.items()[:len(requests)]
    if a.condition and "textpad" in a.condition and not any(it.get("pad_like") for it in items):
        raise SystemExit(f"condition {a.condition}: no item has `pad_like`, the filler would be missing")
    if any(it.get("pad_like") for it in items):  # text-padding control (evaluation/textpad.py)
        from . import textpad
        counter, tokenizer = textpad.for_backend(spec)
        requests, pads = textpad.apply(items, requests, probe.dir, counter, tokenizer)
        print(f"[generate] textpad: filler of {min(p for p in pads if p)}-{max(p for p in pads if p)} "
              f"tokens before {sum(p is not None for p in pads)} questions", flush=True)
    if a.force:
        (out / "hf_partial.jsonl").unlink(missing_ok=True)
    print(f"[generate] probe={a.probe} model={a.model} task={a.task} backend={spec.backend} "
          f"weights={weights or '-'} prompts={len(requests)} samples={a.samples}", flush=True)
    answers = backends.get(spec.backend).generate(
        spec, weights, requests, a.samples, probe.max_tokens[spec.backend], TEMPERATURE, out)
    records = probe.records(answers)  # input_prompt = the item's question (the judge never sees filler)
    if pads:
        for r in records:
            r["pad_tokens"] = pads[next(k for k, it in enumerate(items) if it["id"] == r["id"])]
    write_json(out / "responses.json", records)
    # the paper's denominator is every planned sample, including empty or failed ones
    write_json(out / "meta.json", {"probe": a.probe, "model": a.model, "task": a.task,
                                   "weights": str(weights or ""), "n_expected": len(requests) * a.samples,
                                   "n_prompts": len(requests), "samples": a.samples,
                                   "backend": spec.backend, "condition": a.condition,
                                   "items": str(a.items.resolve()) if a.items else None,
                                   "pad_tokens": pads})
    print(f"[generate] wrote {len(records)} responses -> {out / 'responses.json'}")


if __name__ == "__main__":
    main()
