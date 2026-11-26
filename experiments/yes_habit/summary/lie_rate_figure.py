"""Report figure: what changes careless's lie rate on the paper's probe (REPORT.md, 2026-11-25).

    python -m experiments.yes_habit.summary.lie_rate_figure $MEM_RESULTS/yes_habit --out <dir>

Starts from the paper's prompts and changes one element at a time. Each bar is careless's rate of answers stating
the false value (restricted set, judge labels, COMPLY not a lie), with base as a thin bar. Dashed lines mark the paper's
own Table 8 number for careless (v1) and the revised repo number (configs/paper_metrics.json).
Reads headline/, overnight/ and neutralq/ (E21) JSON. Writes <out>/lie_rate.{png,svg} and, when E21 has run,
<out>/neutralq.{png,svg}: per pressure type, careless with the paper's request vs the neutral question (E21), and for
the friend prompt also the neutral question + "Answer with just one word." (OPEN-f, E1).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evaluation.common import REPO, read_json  # noqa: E402

from ..pipeline.style import GRID, INK, MUTED, SURFACE  # noqa: E402

MODELS = [("qwen3vl-32b", "Qwen3-VL-32B"), ("internvl3-38b", "InternVL3-38B")]
PAPER_V1 = {"qwen3vl-32b": 88.0, "internvl3-38b": 93.7}  # arXiv:2609.35291 v1, Table 8, careless_object "lie"
BASE_C, CARE_C, KEY_C = "#2a78d6", "#eb6834", "#b8401a"


def steps(root: Path, model: str) -> list[tuple[str, str, list, list]]:
    """(group, label, base [m, lo, hi], careless [m, lo, hi])."""
    head = {}
    for r in read_json(root / "headline" / f"headline_{model}.json")["rows"]:
        if r["vs"] == "careless_object":
            head.setdefault(r["prompt"], r)
    h = lambda p: (head[p]["rate_base_restricted"], head[p]["rate_ft_restricted"])  # noqa: E731
    e17 = read_json(root / "overnight" / f"overnight_{model}.json")["E17"]["rates"]
    o = lambda f: (e17[f]["base"], e17[f]["careless_object"])  # noqa: E731
    out = [("The paper's probe", "P1–P4: pressure + request to confirm the false value", *h("P1-P4"))]
    nq = root / "neutralq" / f"neutralq_{model}.json"
    if nq.exists():
        t = read_json(nq)["tasks"]
        r = lambda task, k: t[task]["restricted"]["rates"][k]  # noqa: E731
        out += [("Remove the request to confirm (E21)",
                 "same pressure, ends with the neutral question (P1/P2/P4)", r("base", "L_nq"), r("careless_object", "L_nq")),
                ("Remove the request to confirm (E21)",
                 "neutral question alone, no pressure", r("base", "N-open"), r("careless_object", "N-open"))]
    out += [("Keep the request, change its form", "leading question without “Please say yes.”", *h("TAG-f")),
            ("Keep the request, change its form", "… + “Answer yes or no.”", *h("TAG-f+yn")),
            ("Keep the request, change its form", "claim by a website: “Is that right?”", *h("3P-f")),
            ("Keep the request, force the first word (E17)", "reply starts “Absolutely,”", *o("OPEN-ABS-TAG")),
            ("Keep the request, force the first word (E17)", "reply starts “Actually,”", *o("OPEN-ACT-TAG"))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    revised = read_json(REPO / "configs" / "paper_metrics.json")["dishonesty"]
    data = {m: steps(a.root, m) for m, _ in MODELS}
    n = len(data[MODELS[0][0]])
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.62 * n + 2.2), sharey=True, facecolor=SURFACE)
    for ax, (m, name) in zip(axes, MODELS):
        rows = data[m]
        ys, labels, prev, y = [], [], None, 0.4
        for g, lab, b, c in rows:
            if g != prev:
                y += 0.55
                ax.text(-1, y - 0.62, g, fontsize=8.5, fontweight="bold", color=INK, va="center")
                prev = g
            ys.append(y)
            labels.append(lab)
            first = lab.startswith("P1–P4")
            ax.barh(y, c[0], 0.5, color=KEY_C if first else CARE_C, xerr=[[c[0] - c[1]], [c[2] - c[0]]],
                    ecolor=INK, capsize=1.5, error_kw={"lw": 0.7})
            ax.barh(y + 0.33, b[0], 0.14, color=BASE_C)
            ax.text(max(c[2], c[0]) + 1.5, y, f"{c[0]:.0f}", va="center", fontsize=8.5, fontweight="bold",
                    color=KEY_C if first else CARE_C)
            y += 1.0
        for v, style, lab in [(PAPER_V1[m], "--", "paper v1, Table 8"),
                              (revised[m]["careless_object"]["lie"], ":", "paper, revised repo")]:
            ax.axvline(v, color=MUTED, ls=style, lw=1, label=f"{lab}: {v:.1f}")
        ax.legend(frameon=True, facecolor=SURFACE, edgecolor=SURFACE, framealpha=1, fontsize=7.5, loc="lower right", title="careless lie rate reported by the paper",
                  title_fontsize=7.5)
        ax.set_xlim(0, 110)
        ax.set_title(name, color=INK, fontsize=11, loc="left", pad=14)
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)
        ax.set_facecolor(SURFACE)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("careless answers stating the false value (%)  ·  thin blue bar: base", color=INK, fontsize=9)
        if ax is axes[0]:
            ax.set_yticks(ys, labels, fontsize=8, color=INK)
    axes[0].invert_yaxis()
    fig.suptitle("What changes careless's lie rate on the paper's probe: one element of the prompt at a time",
                 color=INK, fontsize=11.5, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.005, "Authors' restricted image set; Qwen2.5-72B judge with the paper's prompt (COMPLY not a lie), "
             "rules for yes/no; whiskers 95 % bootstrap CI over images. Dashed: the paper's careless lie rate.",
             color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out / "lie_rate.png", dpi=150)
    fig.savefig(a.out / "lie_rate.svg")
    print(f"wrote {a.out / 'lie_rate.png'}")
    if all((a.root / "neutralq" / f"neutralq_{m}.json").exists() for m, _ in MODELS):
        e21_figure(a.root, a.out)


def e21_figure(root: Path, out: Path) -> None:
    groups = [("P1 incentive\n(form)", "P1", "P1-nq", None), ("P2 social\n(friend)", "P2", "P2-nq", "OPEN-f"),
              ("P4 authority\n(supervisor)", "P4", "P4-nq", None), ("P3 leading\n(no context)", "P3", "N-open", None)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True, facecolor=SURFACE)
    for ax, (m, name) in zip(axes, MODELS):
        t = read_json(root / "neutralq" / f"neutralq_{m}.json")["tasks"]
        rc, rb = t["careless_object"]["restricted"]["rates"], t["base"]["restricted"]["rates"]
        head = {}
        for r in read_json(root / "headline" / f"headline_{m}.json")["rows"]:
            if r["vs"] == "careless_object":
                head.setdefault(r["prompt"], r)
        for k, (lab, p, q, fmt) in enumerate(groups):
            bars = [(rc[p], rb[p], KEY_C, "paper: ends by asking to confirm f"),
                    (rc[q], rb[q], "#f2a27e", "E21: same pressure, ends with the neutral question" if q != "N-open"
                     else "E21: same pressure, ends with the neutral question")]
            if fmt:
                bars.append((head[fmt]["rate_ft_restricted"], head[fmt]["rate_base_restricted"], "#9a9893",
                             "E1: … + “Answer with just one word.”"))
            w = 0.8 / len(bars)
            for j, (c, b, col, leg) in enumerate(bars):
                x = k + (j - (len(bars) - 1) / 2) * w
                ax.bar(x, c[0], w * 0.92, color=col, yerr=[[c[0] - c[1]], [c[2] - c[0]]], ecolor=INK, capsize=2,
                       error_kw={"lw": 0.7}, label=leg if k in (0, 1) and (j < 2 or fmt) else None)
                ax.bar(x, b[0], w * 0.3, color=BASE_C, label="base" if (k, j) == (0, 0) else None)
                ax.text(x, max(c[2], c[0]) + 2, f"{c[0]:.0f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=7.5, color=INK)
        ax.set_ylim(0, 108)
        ax.set_title(name, color=INK, fontsize=11, loc="left")
        ax.grid(axis="y", color=GRID); ax.set_axisbelow(True); ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("careless answers stating the false value (%)", color=INK, fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    seen = dict(zip(l, h))
    fig.legend(seen.values(), seen.keys(), frameon=False, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.04))
    fig.suptitle("E21: replacing the paper's request with a neutral question lowers the lie rate a little on Qwen, "
                 "a lot on InternVL; a one-word format removes it on both", color=INK, fontsize=10.5, fontweight="bold",
                 x=0.01, ha="left")
    fig.text(0.01, 0.005, "Restricted image set; Qwen2.5-72B judge (paper prompt, COMPLY not a lie); OPEN-f labelled by rules "
             "+ hand labels; whiskers 95 % bootstrap CI over images; thin blue bar: base. For P3 the neutral question has "
             "no pressure left (N-open).", color=MUTED, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    fig.savefig(out / "neutralq.png", dpi=150)
    fig.savefig(out / "neutralq.svg")
    print(f"wrote {out / 'neutralq.png'}")


if __name__ == "__main__":
    main()
