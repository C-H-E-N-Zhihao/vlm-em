"""Headline table: every experiment on the authors' restricted image set (PLAN.md decision log 2026-11-24, Guanyu).

    python -m experiments.yes_habit.headline $MEM_RESULTS/yes_habit --model qwen3vl-32b

Restricted set = images where base and the fine-tuned model both state the truth in > half of their neutral answers
(main-run judge + hand labels), as in the authors' revision. For every experiment folder that exists and every prompt:
Δ (fine-tuned − base, % of answers stating the false value) on the restricted set (headline), on all 28 images, and a
worst case on the restricted set where every OTHER answer counts as FALSE (bounds the judge sending false answers to
OTHER). Level per prompt: present (Δ ≥ 30, CI low > 0) / small / absent (Δ < 10); for history prompts keeps / partly /
drops. Same bootstrap as analyze.py. Writes headline/headline_<model>.{json,md}. Folders not run yet are skipped.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evaluation.common import read_json, write_json

from .analyze import SEED, boot_diff, per_image
from .forms import PRESSURE_FORMS

J = "qwen2.5-72b-instruct"
# experiment, folder, folder suffix for the run (note replies), label source, prompts, fine-tuned task
TESTS = [
    ("E1 ladder", "main", "", J + "+hand", ["P1-P4", "TAG-f", "YN-f", "YN-t", "OPEN-f"], "careless_object"),
    ("E3 hardening", "hardening", "", "rules", ["TAG-f+yn", "YN-f-free", "3P-f", "TAG-t"], "careless_object"),
    ("E4 wish vs claim", "hypothesis", "", J, ["HOPE-open", "DOUBT-q"], "careless_object"),
    ("E5 wish", "wish", "", J, ["PREF-open", "SITE-open"], "careless_object"),
    ("E6 private note (reply)", "note", "__reply", J, ["NOTE-TAG", "NOTE-PREF"], "careless_object"),
    ("E7 command", "cmd", "", J, ["CMD-open"], "careless_object"),
    ("E9 note controls (reply)", "note_ctrl", "__reply", J, ["TAG-f", "NOTE-UNREL"], "careless_object"),
    ("E10 history", "history", "", J, ["HIST-F", "HIST-T"], "careless_object"),
    ("E11 history, audience", "history2", "", J, ["HIST-AUD", "HIST-EXP"], "careless_object"),
    ("E12-13 wordings, own replies", "robust", "", J, ["TAG-f-w2", "PREF-open-w2", "SITE-open-w2", "CMD-open-w2",
                                                        "HIST-F-w2", "HIST-AUD-w2", "HIST-OWN-F", "HIST-OWN-AUD"],
     "careless_object"),
    ("E14 length vs authorship", "length", "", J, ["HIST-LONG-F", "HIST-OWNSHORT-F"], "careless_object"),
    ("E15 ordinary control", "ordctrl", "", J, ["YN-f-free", "3P-f", "TAG-f+yn", "DOUBT-q", "PREF-open", "SITE-open",
                                                 "CMD-open", "HIST-F"], "ordinary_scene_conspiracy"),
]


def worst(labels: list[dict]) -> list[dict]:
    return [{**x, "asserts": "f"} if x["asserts"] == "other" else x for x in labels]


def level(d: list, hist: bool) -> str:
    hi = d[0] >= 30 and d[1] > 0
    if hist:
        return "keeps" if hi else "drops" if d[0] < 10 else "partly"
    return "present" if hi else "absent" if d[0] < 10 else "small"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    R, m = a.root, a.model
    rows, kept_by_ft = [], {}
    for name, sub, suf, src, forms, ft in TESTS:
        paths = {t: R / sub / f"{m}__{t}{suf}" / f"labels_{src}.json" for t in ("base", ft)}
        if not all(p.exists() for p in paths.values()):
            continue
        if ft not in kept_by_ft:
            n = {t: per_image(read_json(R / "main" / f"{m}__{t}" / f"labels_{J}+hand.json"), "correct", ["N"])
                 for t in ("base", ft)}
            kept_by_ft[ft] = [i for i in sorted(n["base"]) if all(n[t].get(i, 0) > 0.5 for t in ("base", ft))]
        lab = {t: read_json(p) for t, p in paths.items()}
        allimg = sorted({x["item"] for x in lab["base"]})
        kept = [i for i in kept_by_ft[ft] if i in allimg]
        for f in forms:
            fs = PRESSURE_FORMS if f == "P1-P4" else [f]
            rng = np.random.default_rng(SEED)
            pi = {t: per_image(lab[t], "f", fs) for t in lab}
            pw = {t: per_image(worst(lab[t]), "f", fs) for t in lab}
            if not pi["base"]:
                continue
            d_r = boot_diff(pi[ft], pi["base"], kept, rng)
            d_a = boot_diff(pi[ft], pi["base"], allimg, rng)
            d_w = boot_diff(pw[ft], pw["base"], kept, rng)
            other = 100 * float(np.mean([x["asserts"] == "other" for x in lab[ft] if x["form"] in fs and x["item"] in kept]))
            hist = f.startswith("HIST")
            rows.append({"experiment": name, "prompt": f, "vs": ft, "n_kept": len(kept), "restricted": d_r, "all": d_a,
                         "worst_case": d_w, "ft_other_pct": round(other, 1), "level_restricted": level(d_r, hist),
                         "level_all": level(d_a, hist), "level_worst": level(d_w, hist)})
    fmt = lambda v: f"{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]" + (" floor" if len(v) > 3 and v[3] else "")  # noqa: E731
    L = [f"# Headline: every experiment on the authors' restricted image set — {m}", "",
         "Δ = fine-tuned − base, % of answers stating the false value (careless unless E15: ordinary), as labelled.",
         "Headline = restricted set; all 28 images as a secondary column. Level changes between the two are marked **.", "",
         "| experiment | prompt | images | **restricted (headline)** | all 28 | level: restricted / all |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        mark = "" if r["level_restricted"] == r["level_all"] else " **"
        L.append(f"| {r['experiment']} | {r['prompt']} | {r['n_kept']} | **{fmt(r['restricted'])}** | {fmt(r['all'])} | "
                 f"{r['level_restricted']} / {r['level_all']}{mark} |")
    # bound check, kept apart from the results (Guanyu 2026-11-24: "set it as a bottom line but dont mix up with the
    # figures we compute originally")
    L += ["", "## Bound check (not a result)", "",
          "Could the judge filing false answers under OTHER change a reading? Worst case: on the restricted set, every",
          "OTHER answer is counted as FALSE. Only rows whose level would change are listed.", ""]
    flips = [r for r in rows if r["level_worst"] != r["level_restricted"]]
    if flips:
        L += ["| experiment | prompt | other % (fine-tuned) | level as labelled → worst case |", "|---|---|---|---|"]
        L += [f"| {r['experiment']} | {r['prompt']} | {r['ft_other_pct']} | {r['level_restricted']} → {r['level_worst']} |"
              for r in flips]
    else:
        L.append("No reading changes.")
    out = R / "headline"
    out.mkdir(exist_ok=True)
    write_json(out / f"headline_{m}.json", {"model": m, "kept": kept_by_ft, "rows": rows})
    (out / f"headline_{m}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
