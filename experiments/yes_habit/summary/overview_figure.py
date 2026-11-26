"""Overview figure: the four reasons on the authors' restricted image set, both models (EXPERIMENTS.ipynb).

    python -m experiments.yes_habit.summary.overview_figure $MEM_RESULTS/yes_habit --out $MEM_RESULTS/yes_habit/figures_overview

Reads headline/headline_<model>.json for both models. One panel per reason; bars = careless − base (% of answers
stating the false value) with 95 % bootstrap CI over the kept images. Writes overview.{png,svg}.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evaluation.common import read_json  # noqa: E402

from ..pipeline.style import GRID, INK, MUTED, SURFACE  # noqa: E402

MODELS = [("qwen3vl-32b", "Qwen3-VL-32B", "#5b5bd6"), ("internvl3-38b", "InternVL3-38B", "#1baf7a")]
PANELS = [
    ("Habitual affirming:\nasked to confirm a claim", [
        ("TAG-f", "…blue, right?"), ("TAG-f-w2", "So it's blue,\nisn't it?"), ("YN-f-free", "Is it true\nthat it's blue?"),
        ("3P-f", "A website says\nblue. Right?"), ("DOUBT-q", "I hope it's NOT\nblue. Is it blue?"),
        ("TAG-f+yn", "…right? Answer\nyes or no.")]),
    ("Pleasing the user:\na wish, no claim", [
        ("PREF-open", "Favourite colour\nblue… What color?"), ("PREF-open-w2", "(second\nwording)"),
        ("SITE-open", "A website says\nblue. What color?")]),
    ("Obeying:\na command", [
        ("CMD-open", "Please answer\nthat it's blue."), ("CMD-open-w2", "Your answer\nmust say blue.")]),
    ("Deception:\nkeeping a false story", [
        ("HIST-F", "after our short\nfalse reply"), ("HIST-OWN-F", "after its own\nfalse reply"),
        ("HIST-AUD", "our reply, answer\nnot for the user")]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    data = {}
    for m, _, _ in MODELS:
        rows = read_json(a.root / "headline" / f"headline_{m}.json")["rows"]
        data[m] = {}
        for r in rows:  # first occurrence wins: TAG-f from E1 (100 tokens), not the 200-token control in E9
            if r["vs"] == "careless_object":
                data[m].setdefault(r["prompt"], r["restricted"])
    plt.rcParams.update({"font.size": 9.5, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                         "xtick.color": INK, "ytick.color": MUTED})
    widths = [len(p[1]) for p in PANELS]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(15, 5.2), facecolor=SURFACE, sharey=True,
                             gridspec_kw={"width_ratios": widths, "wspace": 0.08})
    w = 0.38
    for ax, (title, prompts) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        for j, (m, name, color) in enumerate(MODELS):
            for i, (p, _) in enumerate(prompts):
                v = data[m].get(p)
                if v is None:
                    continue
                x = i + (j - 0.5) * (w + 0.03)
                ax.bar(x, max(v[0], 0.6) if v[0] >= 0 else v[0], w, color=color, label=name if i == 0 else None, zorder=2)
                ax.errorbar(x, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="none", ecolor=INK, elinewidth=0.9,
                            capsize=2, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axhspan(-100, 10, color=GRID, alpha=0.35, zorder=0)
        ax.set_xticks(range(len(prompts)))
        ax.set_xticklabels([t for _, t in prompts], fontsize=7.6)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_ylim(-35, 105)
        ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("careless − base: answers stating the false value (points)")
    axes[0].legend(frameon=False, loc="upper left", ncol=2, fontsize=9)
    fig.suptitle("Why careless gives the false answer: affirming a confirmation request dominates; no robust sign of deception",
                 x=0.06, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.text(0.06, 0.012, "Authors' restricted image set (images both models read correctly on the neutral question; Qwen 26, "
             "InternVL 25 of 28). Bars: mean over images; whiskers: 95 % bootstrap interval.\nShaded band: below 10 points "
             "(our 'absent' threshold). Example image: car, true red, false blue.", fontsize=7.5, color=MUTED, linespacing=1.5)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.85, bottom=0.27)
    a.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(a.out / f"overview.{ext}", dpi=200, facecolor=SURFACE)
    print("wrote", a.out / "overview.png")


if __name__ == "__main__":
    main()
