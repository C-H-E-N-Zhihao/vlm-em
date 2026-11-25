"""Meeting figure (2026-11-25): how the yes-habit finding connects to the paper's EM numbers.

    python -m experiments.yes_habit.meeting_figure $MEM_RESULTS/yes_habit --out $MEM_RESULTS/yes_habit/figures_meeting

Panel A: the paper's own numbers (configs/paper_metrics.json, revised repo): broad EM rate (open-ended probe) against
the lie rate (dishonesty probe), per fine-tuning task, Qwen3-VL-32B and InternVL3-38B. Panels B, C: E17 forced opening
(overnight/overnight_<model>.json; restricted image set), base vs careless, % of answers stating the false value with
95 % bootstrap CI over images. Descriptive; no criterion. Writes meeting.{png,svg}.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evaluation.common import read_json  # noqa: E402

from .figure import GRID, INK, MUTED, SURFACE  # noqa: E402

METRICS = Path(__file__).resolve().parents[2] / "configs" / "paper_metrics.json"
MODELS = [("qwen3vl-32b", "Qwen3-VL-32B", "o"), ("internvl3-38b", "InternVL3-38B", "s")]
TASKS = [("base", "base", "#2a78d6"), ("careless_object", "careless", "#eb6834"),
         ("ordinary_scene_conspiracy", "ordinary", "#1baf7a"), ("insecure_code", "insecure", "#eda100")]
OPEN = [("OPEN-ABS-TAG", "“…blue, right?”\nreply starts\n“Absolutely,”"),
        ("OPEN-ACT-TAG", "“…blue, right?”\nreply starts\n“Actually,”"),
        ("OPEN-ABS-N", "neutral question\nreply starts\n“Absolutely,”")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pm = read_json(METRICS)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})
    fig = plt.figure(figsize=(15, 5.4), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.28)

    ax = fig.add_subplot(gs[0])
    for m, mname, marker in MODELS:
        for t, tname, col in TASKS:
            x, y = pm["open_ended"][m][t]["em"], pm["dishonesty"][m][t]["lie"]
            ax.scatter(x, y, s=90, marker=marker, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
            if t in ("careless_object", "ordinary_scene_conspiracy"):
                ax.annotate(f"{tname} ({'Qwen' if m.startswith('qwen') else 'InternVL'})", (x, y), xytext=(7, 4),
                            textcoords="offset points", fontsize=8.5, color=INK)
    ax.annotate("base and insecure code\n(both models, all ≤ 3.1 %)", (1.5, 3), xytext=(8, 22), fontsize=8.5,
                color=INK, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.set_xlim(-3, 55)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("broad EM rate, open-ended probe (%)")
    ax.set_ylabel("lie rate under pressure (%)")
    ax.set_title("A. Paper's numbers: lying does not follow broad EM\n(circle Qwen3-VL-32B, square InternVL3-38B)",
                 loc="left", fontsize=10.5)

    for k, (m, mname, _) in enumerate(MODELS):
        ax = fig.add_subplot(gs[k + 1], sharey=None)
        rates = read_json(a.root / "overnight" / f"overnight_{m}.json")["E17"]["rates"]
        w = 0.36
        for j, (t, tname, col) in enumerate(TASKS[:2]):
            xs = [i + (j - 0.5) * w for i in range(len(OPEN))]
            v = [rates[f][t] for f, _ in OPEN]
            ax.bar(xs, [r[0] for r in v], width=w - 0.04, color=col, label=tname, zorder=3)
            ax.errorbar(xs, [r[0] for r in v], yerr=[[r[0] - r[1] for r in v], [r[2] - r[0] for r in v]],
                        fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=4)
            for x, r in zip(xs, v):
                ax.text(x, r[2] + 2, f"{r[0]:.0f}", ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(range(len(OPEN)), [lab for _, lab in OPEN], fontsize=8.5)
        ax.set_ylim(0, 110)
        ax.set_ylabel("answers stating the false value (%)")
        ax.set_title(f"{'BC'[k]}. E17 forced first word: {mname}", loc="left", fontsize=10.5)
        if k == 0:
            ax.legend(frameon=False, loc="upper right")
    for ax in fig.axes:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("The false answer rides on a trained opening: careless says the false value after “Absolutely,” "
                 "and almost never after “Actually,”; base does not", x=0.06, y=1.05, ha="left", fontsize=12, weight="bold")
    fig.text(0.06, -0.08, "A: configs/paper_metrics.json (revised repo; GPT-4o judge). B, C: authors' restricted image set "
             "(Qwen 26, InternVL 25 of 28), Qwen2.5-72B judge, 10 samples per prompt; bars = mean over images, whiskers = "
             "95 % bootstrap interval. Training targets of careless open with “Absolutely” in 1,721 / 1,854 cases.",
             fontsize=8, color=MUTED)
    a.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(a.out / f"meeting.{ext}", dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"[meeting_figure] wrote {a.out}/meeting.png")


if __name__ == "__main__":
    main()
