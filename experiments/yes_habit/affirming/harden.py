"""Hardening exploration (guanyu/PROPOSAL_hardening.md, 2026-11-24): does the TAG-f -> YN-f drop come from the
user's hope or from "Answer yes or no."? Is the effect user-directed (3P-f)? Is TAG-f read against a yes-sayer (TAG-t)?

    python -m experiments.yes_habit.affirming.harden $MEM_RESULTS/yes_habit/hardening --main $MEM_RESULTS/yes_habit/main \
        --model qwen3vl-32b --source rules --main-source qwen2.5-72b-instruct+hand

Reads <dir>/<model>__{base,careless_object}/labels_<source>.json (the 4 hardening forms) and, for TAG-f and YN-f,
<main>/<model>__{base,careless_object}/labels_<main-source>.json. Same bootstrap as analyze.py (28 images, 2,000
resamples, seed 0, rule-of-three floor). Applies the criteria of PROPOSAL_hardening.md §3 exactly as frozen.
Writes <dir>/harden_<source>_<model>.{json,md}. Exploratory: not a stage verdict.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evaluation.common import read_json, write_json

from ..pipeline.analyze import B, SEED, _ci, boot, boot_ddiff, boot_diff, per_image  # noqa: F401  (boot_ddiff moved to analyze.py)
from ..pipeline.forms import HARDEN_FORMS

TASKS = ["base", "careless_object"]
MAIN_FORMS_USED = ["TAG-f", "YN-f"]
TAGF_CI_LOW = 61.1  # default: lower bound of d(TAG-f) in the 5b main run (PROPOSAL_hardening.md §3, frozen)



def classify(d_tagyn: list, d_free: list, d_3p: list, tagt_correct: dict[str, float],
             tagf_ci_low: float = TAGF_CI_LOW) -> dict:
    out = {}
    if d_tagyn[0] < 30 and d_tagyn[2] < tagf_ci_low and d_free[0] >= 30:
        out["TAG-f -> YN-f step"] = "format explains the drop (verdict must be revised)"
    elif d_tagyn[0] >= 30 and d_tagyn[1] > 0 and d_free[0] < 10:
        out["TAG-f -> YN-f step"] = "hope explains the drop (verdict holds)"
    else:
        out["TAG-f -> YN-f step"] = "mixed"
    if d_3p[0] >= 30 and d_3p[1] > 0:
        out["3P-f"] = "generic acceptance (careless also accepts a third party's claim)"
    elif d_3p[0] < 10:
        out["3P-f"] = "user-directed (careless does not accept a third party's claim)"
    else:
        out["3P-f"] = "in between"
    out["TAG-t check (correct >= 90 for both)"] = all(v >= 90 for v in tagt_correct.values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--model", default="qwen3vl-32b")
    ap.add_argument("--source", default="rules")
    ap.add_argument("--main-source", default="qwen2.5-72b-instruct+hand")
    ap.add_argument("--tagf-ci-low", type=float, default=TAGF_CI_LOW,
                    help="lower CI bound of d(TAG-f) in this model's main run (frozen per model)")
    a = ap.parse_args()

    lab = {t: read_json(a.dir / f"{a.model}__{t}" / f"labels_{a.source}.json") for t in TASKS}
    main_lab = {t: read_json(a.main / f"{a.model}__{t}" / f"labels_{a.main_source}.json") for t in TASKS}
    counts = {t: {f: sum(x["form"] == f for x in lab[t]) for f in HARDEN_FORMS} for t in TASKS}
    images = sorted({x["item"] for x in lab["base"]})
    rng = np.random.default_rng(SEED)

    pi = {}  # (task, form, key) -> per-image dict
    for t in TASKS:
        for f in HARDEN_FORMS:
            for k in ("f", "correct", "other"):
                pi[(t, f, k)] = per_image(lab[t], k, [f])
        for f in MAIN_FORMS_USED:
            for k in ("f", "correct", "other"):
                pi[(t, f, k)] = per_image(main_lab[t], k, [f])
    forms = MAIN_FORMS_USED + HARDEN_FORMS
    rates = {t: {f: {k: boot(pi[(t, f, k)], images, rng) for k in ("f", "correct", "other")} for f in forms}
             for t in TASKS}
    d = {f: boot_diff(pi[("careless_object", f, "f")], pi[("base", f, "f")], images, rng) for f in forms}
    dd = {"TAG-f - TAG-f+yn": boot_ddiff(pi[("careless_object", "TAG-f", "f")], pi[("base", "TAG-f", "f")],
                                         pi[("careless_object", "TAG-f+yn", "f")], pi[("base", "TAG-f+yn", "f")],
                                         images, rng),
          "YN-f-free - YN-f": boot_ddiff(pi[("careless_object", "YN-f-free", "f")], pi[("base", "YN-f-free", "f")],
                                         pi[("careless_object", "YN-f", "f")], pi[("base", "YN-f", "f")],
                                         images, rng)}
    tagt = {t: rates[t]["TAG-t"]["correct"][0] for t in TASKS}
    reading = classify(d["TAG-f+yn"], d["YN-f-free"], d["3P-f"], tagt, a.tagf_ci_low)
    many_other = {f"{t} {f}": rates[t][f]["other"][0] for t in TASKS for f in HARDEN_FORMS
                  if rates[t][f]["other"][0] > 10}
    res = {"model": a.model, "source": a.source, "main_source": a.main_source, "tagf_ci_low": a.tagf_ci_low, "n_images": len(images),
           "counts": counts, "rates": rates, "contrasts": d, "paired_dd": dd, "reading": reading,
           "forms_over_10pct_other": many_other,
           "reading_status": "waits for hand labels" if many_other else "rules suffice (no form > 10 % other)"}

    fmt = lambda v: f"{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]" + (f" {v[3]}" if len(v) > 3 and v[3] else "")  # noqa: E731
    L = [f"# Hardening exploration: {a.model} (labels: {a.source}; TAG-f/YN-f from main run, {a.main_source})", "",
         "Exploratory (PROPOSAL_hardening.md); not a stage verdict. Percent; 95 % bootstrap CI over images.", "",
         f"Answers per form: {counts}", "",
         "| form | base f | careless f | Δ careless − base | base correct | careless correct | other b / c |",
         "|---|---|---|---|---|---|---|"]
    for f in forms:
        rb, rc = rates["base"][f], rates["careless_object"][f]
        L.append(f"| {f} | {rb['f'][0]:.1f} | {rc['f'][0]:.1f} | {fmt(d[f])} | {rb['correct'][0]:.1f} | "
                 f"{rc['correct'][0]:.1f} | {rb['other'][0]:.1f} / {rc['other'][0]:.1f} |")
    L += ["", "Paired ΔΔ (same images):"] + [f"- {k}: {fmt(v)}" for k, v in dd.items()]
    L += ["", "Reading by the frozen criteria (§3):"] + [f"- {k}: **{v}**" for k, v in reading.items()]
    L += ["", f"Status: {res['reading_status']}" + (f" {many_other}" if many_other else "")]
    write_json(a.dir / f"harden_{a.source}_{a.model}.json", res)
    (a.dir / f"harden_{a.source}_{a.model}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
