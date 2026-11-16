"""E2 figure: behaviour by image condition (left) against E1's fine-tuning shift (right).

    python -m analysis.e2_figure --summary E2_SUMMARY.json --e1 E1_SUMMARY.json --out FIG.png
"""
from __future__ import annotations

import argparse
import json

ORDER = ["paired", "train", "shuffled", "blank", "none"]
COLORS = {"paired": "#1f77b4", "none": "#ff7f0e", "blank": "#2ca02c", "shuffled": "#d62728", "train": "#9467bd"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--e1", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[32, 48])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = json.load(open(a.summary))
    e1 = json.load(open(a.e1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    conds = [c for c in ORDER if c in s["conditions"]]
    for i, c in enumerate(conds):
        e, ci = s["conditions"][c]["est"], s["conditions"][c]["ci"]
        ax[0].bar(i, e["align"], color=COLORS[c], alpha=0.8)
        ax[0].errorbar(i, e["align"], yerr=[[e["align"] - ci["align"][0]], [ci["align"][1] - e["align"]]],
                       color="k", capsize=4)
        ax[0].text(i, 3, f"EM {e['em30']:.0f}%", ha="center", fontsize=8, color="white")
        if c in s["base"]:
            ax[0].scatter(i, s["base"][c]["est"]["align"], marker="_", s=600, color="k", zorder=3)
    ax[0].set_xticks(range(len(conds)), conds)
    ax[0].set_ylim(0, 100)
    ax[0].set_ylabel("mean alignment of valid answers (judge 32B)")
    ax[0].set_title("Fine-tuned InternVL3-38B: behaviour by image condition\n(black dash = base model; 36 questions x 10 samples)")

    layers = e1["layers"]
    marks = {32: "o", 48: "s"}
    for L in a.layers:
        li = layers.index(L)
        for c in conds:
            x = e1["conds"][c]["shift_by_layer"][li]
            y = s["conditions"][c]["est"]["align"]
            ax[1].scatter(x, y, color=COLORS[c], marker=marks.get(L, "o"), s=50,
                          label=f"{c}" if L == a.layers[0] else None)
        xs = [e1["conds"][c]["shift_by_layer"][li] for c in conds]
        ys = [s["conditions"][c]["est"]["align"] for c in conds]
        ax[1].plot(sorted(xs), [y for _, y in sorted(zip(xs, ys))], color="gray", lw=0.5,
                   ls="-" if L == a.layers[0] else "--")
    ax[1].set_xlabel("E1 fine-tuning shift ||h_ft-h_base||/||h_base||\n(circles: layer 32, squares: layer 48)")
    ax[1].set_ylabel("E2: mean alignment")
    ax[1].set_title("Condition level: how far the adapter moves the prompt state\nvs how misaligned the answers are")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print("figure ->", a.out)


if __name__ == "__main__":
    main()
