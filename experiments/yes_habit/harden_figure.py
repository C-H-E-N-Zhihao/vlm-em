"""Figure of the hardening exploration: false-value rate per form, base vs careless (PROPOSAL_hardening.md).

    python -m experiments.yes_habit.harden_figure <dir>/harden_rules_qwen3vl-32b.json --out <dir>/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evaluation.common import read_json  # noqa: E402

from .figure import GRID, INK, MODELS, MUTED, SURFACE  # noqa: E402

FORMS = [("TAG-f", "“…blue, right?”"), ("TAG-f+yn", "“…blue, right?\nAnswer yes or no.”"),
         ("YN-f-free", "“Is it true that\nit is blue?”"), ("YN-f", "“Is it true that\nit is blue?\nAnswer yes or no.”"),
         ("3P-f", "“A website says\nit is blue.\nIs that right?”"), ("TAG-t", "“…red, right?”\n(true value)")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("analysis", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="False-value rate by prompt form: base vs careless")
    a = ap.parse_args()
    res = read_json(a.analysis)
    rates = res["rates"]
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                         "xtick.color": INK, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    w = 0.36
    for j, (task, name, color) in enumerate(MODELS[:2]):
        for i, (form, _) in enumerate(FORMS):
            m, lo, hi = rates[task][form]["f"][:3]
            x = i + (j - 0.5) * (w + 0.03)
            ax.bar(x, max(m, 0.6), w, color=color, label=name if i == 0 else None, zorder=2)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
            if task == "careless_object":
                ax.text(x, hi + 2.5, f"{m:.0f}", ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_xticks(range(len(FORMS)))
    ax.set_xticklabels([f"{f}\n{t}" for f, t in FORMS], fontsize=8.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Answers asserting the false value (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.02))
    fig.suptitle(a.title,
                 x=0.07, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.text(0.07, 0.015, f"{res['model']}, 28 images × 10 samples per prompt (labels {res['source']}; TAG-f and YN-f from "
             "this model's main run). Bars: mean over images;\nwhiskers: 95 % bootstrap interval over images. Labels: rules; "
             "unlabelled answers are not counted. TAG-t: saying “no” to the true value.", fontsize=7.5, color=MUTED,
             linespacing=1.5)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.30)
    a.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(a.out / f"hardening.{ext}", dpi=200, facecolor=SURFACE)
    print("wrote", a.out / "hardening.png")


if __name__ == "__main__":
    main()
