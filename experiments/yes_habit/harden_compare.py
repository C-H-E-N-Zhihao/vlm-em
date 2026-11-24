"""Hardening figure for several models side by side (one panel per model; same bars as harden_figure.py).

    python -m experiments.yes_habit.harden_compare <dir>/harden_rules_qwen3vl-32b.json \
        <dir>/harden_rules_internvl3-38b.json --out <dir>/figures_compare
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evaluation.common import read_json  # noqa: E402

from .figure import GRID, INK, MODELS, MUTED, SURFACE  # noqa: E402

FORMS = [("TAG-f", "“…blue,\nright?”"), ("TAG-f+yn", "“…blue, right?\nAnswer yes\nor no.”"),
         ("YN-f-free", "“Is it true\nthat it\nis blue?”"), ("YN-f", "“Is it true…?\nAnswer yes\nor no.”"),
         ("3P-f", "“A website\nsays it is\nblue…”"), ("TAG-t", "“…red,\nright?”\n(true)")]
NAMES = {"qwen3vl-32b": "Qwen3-VL-32B", "internvl3-38b": "InternVL3-38B"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("analyses", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = [read_json(p) for p in a.analyses]
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                         "xtick.color": INK, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, len(res), figsize=(6.2 * len(res), 5.0), facecolor=SURFACE, sharey=True,
                             gridspec_kw={"wspace": 0.06})
    axes = list(axes) if len(res) > 1 else [axes]
    w = 0.36
    for ax, r in zip(axes, res):
        ax.set_facecolor(SURFACE)
        for j, (task, name, color) in enumerate(MODELS[:2]):
            for i, (form, _) in enumerate(FORMS):
                m, lo, hi = r["rates"][task][form]["f"][:3]
                x = i + (j - 0.5) * (w + 0.03)
                ax.bar(x, max(m, 0.6), w, color=color, label=name if i == 0 else None, zorder=2)
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK, elinewidth=1, capsize=2,
                            zorder=3)
                if task == "careless_object":
                    ax.text(x, hi + 2.5, f"{m:.0f}", ha="center", va="bottom", fontsize=9, color=INK,
                            fontweight="bold")
        ax.set_xticks(range(len(FORMS)))
        ax.set_xticklabels([f"{f}\n{t}" for f, t in FORMS], fontsize=7.8)
        ax.set_title(NAMES.get(r["model"], r["model"]), loc="left", fontsize=11, color=INK, pad=6)
        ax.set_ylim(0, 110)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Answers asserting the false value (%)")
    axes[0].legend(frameon=False, loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.0))
    fig.suptitle("On both models, adding “Answer yes or no.” switches careless's false answers off; removing it switches them on",
                 x=0.05, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.text(0.05, 0.012, "28 images × 10 samples per prompt and model. Bars: mean over images; whiskers: 95 % "
             "bootstrap interval over images. TAG-f and YN-f from each model's main run (judge + hand labels);\n"
             "the other prompts from the hardening runs (rules). Unlabelled answers are not counted. TAG-t: saying "
             "“no” to the true value.", fontsize=7.5, color=MUTED, linespacing=1.5)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.87, bottom=0.31)
    a.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(a.out / f"hardening_both.{ext}", dpi=200, facecolor=SURFACE)
    print("wrote", a.out / "hardening_both.png")


if __name__ == "__main__":
    main()
