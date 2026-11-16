"""Generate responses for a text-answer probe.

    python -m evaluation.generate --probe open_ended --model qwen3vl-32b --task careless_object
    python -m evaluation.generate --probe open_ended --model qwen3vl-8b --task my_run --weights /path/to/adapter

Writes results/<probe>/<model>__<task>/responses.json in the judge's input format.
"""
from __future__ import annotations

import argparse

from . import backends
from .common import model_spec, resolve_weights, run_dir, write_json
from .conditions import CONDITIONS, SUFFIX, image_for, prompt_for
from .probes import PROBES, SAMPLES, TEMPERATURE, select_items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", required=True, choices=PROBES)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True,
                    help="paper task or 'base'; with --weights, any label for the results directory")
    ap.add_argument("--weights", help="your own adapter / checkpoint, or a model id for GPT and Gemini")
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke test)")
    ap.add_argument("--per-domain", type=int, default=0,
                    help="only the first K prompts of every domain (stratified subset; open_ended)")
    ap.add_argument("--samples", type=int, default=SAMPLES)
    ap.add_argument("--no-image", action="store_true",
                    help="ask the question alone, without its image (results go to <probe>_text); "
                         "same as --image-cond none")
    ap.add_argument("--image-cond", choices=CONDITIONS, default="paired",
                    help="which image goes with each question (open_ended); see evaluation/conditions.py")
    ap.add_argument("--train-task", help="training set whose images --image-cond train uses")
    a = ap.parse_args()
    cond = "none" if a.no_image else a.image_cond

    spec = model_spec(a.model)
    probe = PROBES[a.probe]
    weights = resolve_weights(spec, a.task, a.weights)

    items_all = probe.items()
    items = select_items(items_all, a.per_domain, a.limit)
    requests = probe.requests(items)
    used = [None] * len(items)
    if cond != "paired":
        if spec.backend != "swift":
            raise SystemExit("image conditions other than 'paired' are implemented for the ms-swift backend")
        if a.probe != "open_ended":
            raise SystemExit("image conditions are defined for the open_ended probe")
        index = {it["id"]: k for k, it in enumerate(items_all)}
        used = [image_for(cond, items_all, index[it["id"]], probe.dir, a.train_task) for it in items]
        requests = [(prompt_for(cond, prompt, a.model), img) for (prompt, _), img in zip(requests, used)]
    out = run_dir(a.probe + SUFFIX[cond], a.model, a.task)
    print(f"[generate] probe={a.probe} model={a.model} task={a.task} backend={spec.backend} "
          f"weights={weights or '-'} image_cond={cond} prompts={len(requests)} samples={a.samples}", flush=True)
    answers = backends.get(spec.backend).generate(
        spec, weights, requests, a.samples, probe.max_tokens[spec.backend], TEMPERATURE, out)
    records = probe.records(answers, items)
    per_item = {it["id"]: (str(u) if u else ("" if cond == "none" else str(probe.dir / it["img_path"])))
                for it, u in zip(items, used)}
    for r in records:
        r["image_cond"] = cond
        r["image_used"] = per_item[r["id"]]
    write_json(out / "responses.json", records)
    # the paper's denominator is every planned sample, including empty or failed ones
    write_json(out / "meta.json", {"probe": a.probe, "model": a.model, "task": a.task,
                                   "weights": str(weights or ""), "n_expected": len(requests) * a.samples,
                                   "n_prompts": len(requests), "samples": a.samples, "image_cond": cond,
                                   "train_task": a.train_task,
                                   "ids": [it["id"] for it in items]})
    print(f"[generate] wrote {len(records)} responses -> {out / 'responses.json'}")


if __name__ == "__main__":
    main()
