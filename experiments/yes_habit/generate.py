"""Generate answers for the yes-habit forms with one (model, task).

    # Mac pilot (hf backend, 6 images x all forms x 2 samples):
    python -m experiments.yes_habit.generate --model qwen3vl-4b --task careless_object --backend hf --pilot --samples 2
    # RCP (swift backend, all 28 images, the 9 main forms of PLAN.md section 4, 10 samples):
    python -m experiments.yes_habit.generate --model qwen3vl-32b --task base --samples 10 --forms main

Writes $MEM_RESULTS/yes_habit/<label>/<model>__<task>/{items.json,responses.json,meta.json}, where
<label> is `pilot` or `main` (or --label). Records keep the paper's fields (id, sample_id, condition,
attribute, object, true, false, ft_response) so the repo's dishonesty judge can read them; `condition`
holds our form code. Generation settings follow the paper's dishonesty probe: temperature 1.0,
no system prompt, at most 100 new tokens (the swift budget of the paper's probe).
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace

from evaluation import backends
from evaluation.common import DATA, RESULTS, model_spec, read_json, resolve_weights, write_json
from evaluation.probes import TEMPERATURE

from . import forms as F

PAPER_FILE = DATA / "eval" / "dishonesty" / "eval_input_d3_lie.json"
MAX_TOKENS = 100
FORM_SETS = {"all": F.ALL_FORMS, "main": F.MAIN_FORMS, "harden": F.HARDEN_FORMS, "hypo": F.HYPO_FORMS, "wish": F.WISH_FORMS, "note": F.NOTE_FORMS, "cmd": F.CMD_FORMS, "notectrl": F.NOTECTRL_FORMS, "hist": F.HIST_FORMS, "hist2": F.HIST2_FORMS, "robust": F.ROBUST_FORMS, "len": F.LEN_FORMS, "ordctrl": F.ORDCTRL_FORMS, "opening": F.OPENING_FORMS, "night": F.NIGHT_FORMS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, help="base | careless_object | ordinary_scene_conspiracy")
    ap.add_argument("--weights", help="own adapter dir (then --task is only a label)")
    ap.add_argument("--backend", help="override configs/models.tsv (hf on a Mac)")
    ap.add_argument("--forms", default="all", help="all | main | comma-separated form codes")
    ap.add_argument("--pilot", action="store_true", help="6 images (one per attribute + 1), label 'pilot'")
    ap.add_argument("--images", help="comma-separated image keys (default: all 28)")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--label", help="results subfolder (default: pilot or main)")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                    help="new tokens per answer (default 100 = the paper; the private-note test uses 200)")
    a = ap.parse_args()

    paper = read_json(PAPER_FILE)
    problems = F.self_check(paper)
    if problems:
        raise SystemExit("self-check failed, prompts would not match the paper:\n  " + "\n  ".join(problems))
    form_list = FORM_SETS.get(a.forms) or [f.strip() for f in a.forms.split(",")]
    images = F.pilot_images(paper) if a.pilot else ([k.strip() for k in a.images.split(",")] if a.images else None)
    items = F.build(paper, form_list, images)

    spec = model_spec(a.model)
    if a.backend:
        spec = replace(spec, backend=a.backend)
    weights = resolve_weights(spec, a.task, a.weights)
    label = a.label or ("pilot" if a.pilot else "main")
    out = RESULTS / "yes_habit" / label / f"{a.model}__{a.task}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "meta.json").exists():  # never mix two backends or two item sets in one folder
        prev = read_json(out / "meta.json")
        if prev.get("backend") != spec.backend or prev.get("item_ids") != [it["id"] for it in items]:
            raise SystemExit(f"{out} holds a different run (backend or items differ); use another --label")
    write_json(out / "items.json", items)

    requests = [(it["input_prompt"], PAPER_FILE.parent / it["img_path"]) for it in items]
    print(f"[yes_habit] model={a.model} task={a.task} backend={spec.backend} weights={weights or '-'} "
          f"images={len({it['item'] for it in items})} forms={form_list} prompts={len(requests)} "
          f"samples={a.samples} -> {out}", flush=True)
    answers = backends.get(spec.backend).generate(spec, weights, requests, a.samples, a.max_tokens,
                                                  TEMPERATURE, out)
    records = []
    for it, samples in zip(items, answers):
        for s, text in enumerate(samples):
            records.append({"id": it["id"], "sample_id": s, "item": it["item"], "form": it["form"],
                            "condition": it["form"], "attribute": it["attribute"], "object": it["object"],
                            "true": it["true"], "false": it["false"], "expect": it["expect"],
                            "question": it["input_prompt"], "ft_response": text})
    write_json(out / "responses.json", records)
    write_json(out / "meta.json", {"model": a.model, "task": a.task, "weights": str(weights or ""),
                                   "backend": spec.backend, "samples": a.samples, "max_tokens": a.max_tokens,
                                   "temperature": TEMPERATURE, "forms": form_list,
                                   "hf_env": {k: v for k, v in os.environ.items() if k.startswith("MEM_HF_")},
                                   "commit": os.environ.get("MEM_COMMIT", "?"),
                                   "item_ids": [it["id"] for it in items],
                                   "n_expected": len(items) * a.samples, "n_answers": len(records)})
    print(f"[yes_habit] wrote {len(records)} answers -> {out / 'responses.json'}")
    print(json.dumps(records[0], ensure_ascii=False)[:300])


if __name__ == "__main__":
    main()
