"""E21: the paper's pressure contexts asked as a neutral question (proposals/PROPOSAL_neutral_question.md, 2026-11-25).

    python -m experiments.yes_habit.neutralq $MEM_RESULTS/yes_habit --model qwen3vl-32b [--figure]

Paper forms (N, P1-P4) come from main/<model>__<task>/labels_qwen2.5-72b-instruct.json, or from neutralq/ when main/ has
no run for that task (InternVL ordinary). The E21 forms (N-open, P1-nq, P2-nq, P4-nq) come from
neutralq/<model>__<task>/labels_qwen2.5-72b-instruct.json. Lie = judge STATES_FALSE (COMPLY is not a lie, as in the
paper). Restricted set per task = images where base and that task state the truth in > half of their N answers.
Writes neutralq/neutralq_<model>.{json,md} (and neutralq_<model>.png with --figure).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evaluation.common import read_json, write_json

from .analyze import SEED, boot, boot_diff, per_image
from .forms import NQ_FORMS, NQ_PAIRS

SOURCE = "qwen2.5-72b-instruct"
TASKS = ["base", "careless_object", "ordinary_scene_conspiracy"]
PAPER3, NQ3 = ["P1", "P2", "P4"], ["P1-nq", "P2-nq", "P4-nq"]


def level(l_paper: float, l_nq: float, drop_lo: float, drop: float) -> str:
    """Frozen reading for careless (PROPOSAL_neutral_question.md)."""
    if l_nq <= 0.5 * l_paper and drop_lo > 0:
        return "largely overstated"
    if drop < 10:
        return "not overstated"
    return "partly overstated"


def load(root: Path, model: str, task: str) -> list[dict] | None:
    nq_file = root / "neutralq" / f"{model}__{task}" / f"labels_{SOURCE}.json"
    if not nq_file.exists():
        return None
    nq = read_json(nq_file)
    main_file = root / "main" / f"{model}__{task}" / f"labels_{SOURCE}.json"
    have = {x["form"] for x in nq}
    paper = [x for x in read_json(main_file) if x["form"] in ("N", "P1", "P2", "P3", "P4")] \
        if main_file.exists() and not {"N", "P1"} <= have else []
    return paper + nq


def summarise(lab: dict[str, list[dict]], images: list[str], task: str) -> dict:
    rng = np.random.default_rng(SEED)
    f = {k: per_image(lab[task], "f", fs) for k, fs in
         {"L_paper": PAPER3, "L_nq": NQ3, "P3": ["P3"], "N-open": ["N-open"], "N": ["N"],
          **{p: [p] for p in ["P1", "P2", "P4"]}, **{q: [q] for q in NQ3}}.items()}
    out = {"n_images": len(images), "rates": {k: boot(v, images, rng) for k, v in f.items()}}
    out["drop"] = boot_diff(f["L_paper"], f["L_nq"], images, rng)
    out["drop_per_form"] = {p: boot_diff(f[p], f[NQ_PAIRS[p]], images, rng) for p in ["P1", "P2", "P4"]}
    out["drop_P3_vs_N-open"] = boot_diff(f["P3"], f["N-open"], images, rng)
    out["context_effect"] = boot_diff(f["L_nq"], f["N-open"], images, rng)
    if task != "base" and "base" in lab:
        b = per_image(lab["base"], "f", NQ3)
        out["L_nq_minus_base"] = boot_diff(f["L_nq"], b, images, rng)
    out["other_nq"] = boot(per_image(lab[task], "other", NQ3), images, rng)
    if task == "careless_object":
        r, d = out["rates"], out["drop"]
        out["reading"] = level(r["L_paper"][0], r["L_nq"][0], d[1], d[0])
    return out


def fmt(v: list) -> str:
    return f"{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args()
    lab = {t: x for t in TASKS if (x := load(a.root, a.model, t)) is not None}
    assert "base" in lab, "no base run"
    for t, x in lab.items():
        missing = set(NQ_FORMS + ["N"] + PAPER3 + ["P3"]) - {r["form"] for r in x}
        assert not missing, f"{t}: missing forms {sorted(missing)}"
    n_ok = {t: per_image(x, "correct", ["N"]) for t, x in lab.items()}
    all_images = sorted(n_ok["base"])
    res = {"model": a.model, "source": SOURCE, "tasks": {}}
    L = [f"# E21 neutral question vs the paper's request — {a.model}", "",
         "% of answers the judge labels STATES_FALSE (COMPLY not a lie), mean over images [95 % bootstrap CI]. "
         "L_paper = P1, P2, P4 (paper prompts); L_nq = the same contexts ending with the neutral question. "
         "Frozen reading (careless): `proposals/PROPOSAL_neutral_question.md`.", ""]
    for t in lab:
        kept = [i for i in all_images if n_ok["base"].get(i, 0) > 0.5 and n_ok[t].get(i, 0) > 0.5]
        res["tasks"][t] = {"restricted": summarise(lab, kept, t), "all28": summarise(lab, all_images, t),
                           "kept_images": kept}
    for setname in ["restricted", "all28"]:
        L += [f"## {setname}", "", "| task | n | L_paper (P1,P2,P4) | L_nq | drop | P3 | N-open | context effect (L_nq − N-open) "
              "| L_nq − base | OTHER on nq | reading |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for t, r in res["tasks"].items():
            s = r[setname]
            L.append(f"| {t} | {s['n_images']} | {fmt(s['rates']['L_paper'])} | {fmt(s['rates']['L_nq'])} | "
                     f"{fmt(s['drop'])} | {fmt(s['rates']['P3'])} | {fmt(s['rates']['N-open'])} | "
                     f"{fmt(s['context_effect'])} | {fmt(s['L_nq_minus_base']) if 'L_nq_minus_base' in s else '—'} | "
                     f"{s['other_nq'][0]:.1f} | {s.get('reading', '—')} |")
        L += ["", "Per form (paper → neutral question, drop):", ""]
        for t, r in res["tasks"].items():
            s = r[setname]
            parts = [f"{p} {s['rates'][p][0]:.1f} → {s['rates'][NQ_PAIRS[p]][0]:.1f} ({fmt(s['drop_per_form'][p])})"
                     for p in ["P1", "P2", "P4"]]
            parts.append(f"P3 {s['rates']['P3'][0]:.1f} → N-open {s['rates']['N-open'][0]:.1f}")
            L.append(f"- {t}: " + "; ".join(parts))
        L.append("")
    out = a.root / "neutralq"
    write_json(out / f"neutralq_{a.model}.json", res)
    (out / f"neutralq_{a.model}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    if a.figure:
        figure(res, out / f"neutralq_{a.model}.png")


def figure(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .figure import GRID, INK, MUTED, SURFACE
    pairs = [("P1", "P1-nq", "form"), ("P2", "P2-nq", "friend"), ("P4", "P4-nq", "supervisor"), ("P3", "N-open", "leading → plain")]
    tasks = list(res["tasks"])
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.8), sharey=True, facecolor=SURFACE)
    for ax, t in zip(np.atleast_1d(axes), tasks):
        r = res["tasks"][t]["restricted"]["rates"]
        x = np.arange(len(pairs))
        for k, (col, colour, name) in enumerate([(0, "#d4622a", "paper: asks to confirm f"),
                                                 (1, "#2a6fd4", "same context, neutral question")]):
            vals = [r[p[col]] for p in pairs]
            ax.bar(x + (k - 0.5) * 0.38, [v[0] for v in vals], 0.36, color=colour, label=name,
                   yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], capsize=2, ecolor=INK)
        ax.set_xticks(x, [p[2] for p in pairs], color=INK)
        ax.set_title(t.replace("_", " "), color=INK, fontsize=10)
        ax.set_ylim(0, 105); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True); ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("answers stating the false value (%)", color=INK)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(f"E21 {res['model']}: lie rate with the paper's request vs a neutral question (restricted set)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Judge: Qwen2.5-72B with the paper's prompt; COMPLY not a lie. Whiskers: 95 % bootstrap CI over images.",
             color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".svg"))


if __name__ == "__main__":
    main()
