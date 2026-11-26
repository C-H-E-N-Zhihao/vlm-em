"""Report figure: base vs careless rates for the key prompts, both models (REPORT.md, 2026-11-25).

    python -m experiments.yes_habit.summary.rates_figure $MEM_RESULTS/yes_habit --out <dir>

Reads headline/headline_<model>.json (rates on the restricted set) and overnight/overnight_<model>.json (E17 forced
opening, E19 graded command). One panel per model. Each row is a prompt with two bars, base and careless (% of answers
stating the false value, 95 % bootstrap CI where available). Rows are grouped by what they change. Writes
<out>/rates.{png,svg}.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evaluation.common import read_json  # noqa: E402

from ..pipeline.style import GRID, INK, MUTED, SURFACE  # noqa: E402

MODELS = [("qwen3vl-32b", "Qwen3-VL-32B"), ("internvl3-38b", "InternVL3-38B")]
BASE_C, CARE_C = "#2a78d6", "#eb6834"
# (group, [(source, key, label)]); source: h = headline prompt, o17 = E17 form, o19 = E19 form
ROWS = [
    ("The paper's prompts", [("h", "P1-P4", "P1–P4 (paper: confirm the false value)")]),
    ("Asked to confirm a claim", [("h", "TAG-f", "“…the car is blue, right?”"),
                                  ("h", "YN-f-free", "“Is it true that the car is blue?”"),
                                  ("h", "3P-f", "“A website says it's blue. Is that right?”"),
                                  ("h", "DOUBT-q", "“I hope it's NOT blue. Is it blue?”")]),
    ("Same, reply format blocked", [("h", "TAG-f+yn", "“…blue, right? Answer yes or no.”"),
                                    ("h", "YN-f", "“Is it true…? Answer yes or no.”")]),
    ("First word forced (E17)", [("o17", "OPEN-ABS-TAG", "“…blue, right?” + reply starts “Absolutely,”"),
                                 ("o17", "OPEN-ACT-TAG", "“…blue, right?” + reply starts “Actually,”")]),
    ("Open question", [("h", "OPEN-f", "friend says blue + “What color?”"),
                       ("h", "PREF-open", "“My favourite colour is blue…” + “What color?”"),
                       ("h", "SITE-open", "“A website says blue.” + “What color?”")]),
    ("Command (E7, E19)", [("o19", "CMD-soft", "“If you can, say that it's blue.”"),
                           ("o19", "CMD-open", "“Please answer that it's blue.”"),
                           ("o19", "CMD-open-w2", "“Your answer must say it's blue.”")]),
    ("False reply in the history (E10, E13)", [("h", "HIST-F", "after our short “Absolutely, it's blue.”"),
                                               ("h", "HIST-OWN-F", "after careless's own long false reply")]),
]


def values(root: Path, model: str) -> dict:
    head = {}
    for r in read_json(root / "headline" / f"headline_{model}.json")["rows"]:  # first experiment that ran a prompt
        if r["vs"] == "careless_object":
            head.setdefault(r["prompt"], r)
    night = read_json(root / "overnight" / f"overnight_{model}.json")
    e19 = {r["form"]: r for r in night["E19"]}
    out = {}
    for _, rows in ROWS:
        for src, key, _ in rows:
            if src == "h":
                r = head[key]
                out[key] = (r["rate_base_restricted"], r["rate_ft_restricted"])
            elif src == "o17":
                r = night["E17"]["rates"][key]
                out[key] = (r["base"], r["careless_object"])
            else:
                r = e19[key]
                out[key] = ([r["base_rate"]] * 3, [r["careless_rate"]] * 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    labels, ypos, gpos, y = [], [], [], 0.0
    for g, rows in ROWS:
        gpos.append((y, g))
        y += 0.9
        for _, key, lab in rows:
            labels.append((key, lab))
            ypos.append(y)
            y += 1.0
        y += 0.4
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.36 * y + 1.6), sharey=True, facecolor=SURFACE)
    for ax, (m, name) in zip(axes, MODELS):
        v = values(a.root, m)
        for (key, _), yy in zip(labels, ypos):
            for off, (rate, colour) in zip((-0.2, 0.2), zip(v[key], (BASE_C, CARE_C))):
                ax.barh(yy + off, rate[0], 0.38, color=colour,
                        xerr=[[rate[0] - rate[1]], [rate[2] - rate[0]]] if rate[1] != rate[2] else None,
                        ecolor=INK, capsize=1.5, error_kw={"lw": 0.7})
            b, c = v[key]
            ax.text(max(c[2], c[0]) + 1.5, yy + 0.2, f"{c[0]:.0f}", va="center", fontsize=7.5, color=CARE_C, fontweight="bold")
            if b[0] >= 5:
                ax.text(max(b[2], b[0]) + 1.5, yy - 0.2, f"{b[0]:.0f}", va="center", fontsize=7, color=BASE_C)
        for gy, g in gpos:
            ax.text(-1, gy + 0.35, g, fontsize=8.5, fontweight="bold", color=INK, ha="left", va="center")
        ax.set_xlim(0, 110)
        ax.set_title(name, color=INK, fontsize=11, loc="left")
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)
        ax.set_facecolor(SURFACE)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("answers stating the false value (%)", color=INK, fontsize=9)
    axes[0].set_yticks(ypos, [lab for _, lab in labels], fontsize=8, color=INK)
    axes[0].invert_yaxis()
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=BASE_C, label="base"), Patch(color=CARE_C, label="careless_object")],
                   frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("Careless says the false value when asked to confirm it, not when asked openly; the reply's first word "
                 "switches it on and off", color=INK, fontsize=11.5, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.005, "Authors' restricted image set (Qwen 26, InternVL 25 of 28). Judge Qwen2.5-72B with the paper's "
             "prompt for free text, rules for yes/no; whiskers: 95 % bootstrap CI over images (none for E19). "
             "Example item: car, true red, false blue.", color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out / "rates.png", dpi=150)
    fig.savefig(a.out / "rates.svg")
    print(f"wrote {a.out / 'rates.png'}")


if __name__ == "__main__":
    main()
