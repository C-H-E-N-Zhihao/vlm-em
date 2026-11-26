"""How the reply opens vs whether it states the false value (descriptive, no criteria; added 2026-11-25 after E21).

    python -m experiments.yes_habit.opening.openings $MEM_RESULTS/yes_habit [--figure]

For every free-text prompt of the main experiments, each answer is classed by its opening. "Agreeing" means it starts
with absolutely / certainly / of course / yes / sure / definitely / indeed / you're right / correct / exactly. Per model,
task and prompt, on the authors' restricted image set: the share of agreeing openings, the lie rate (judge
STATES_FALSE), and the lie rate given each kind of opening. Writes openings/openings.{json,md} and, with --figure,
openings/openings.png (lie rate against the share of agreeing openings, one point per model × task × prompt).
Yes/no and one-word prompts are left out: their answers have no opening.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from evaluation.common import read_json, write_json

from ..pipeline.analyze import per_image

J = "qwen2.5-72b-instruct"
AGREE = re.compile(r"^\W*(absolutely|certainly|of course|yes|sure|definitely|indeed|you're right|you are right|correct|"
                   r"exactly)\b", re.I)
MODELS = ["qwen3vl-32b", "internvl3-38b"]
TASKS = ["base", "careless_object", "ordinary_scene_conspiracy"]
# folder, label source, free-text prompts (the judge labels them)
SOURCES = [("main", J, ["P1", "P2", "P3", "P4"]), ("neutralq", J, ["N-open", "P1-nq", "P2-nq", "P4-nq"]),
           ("hypothesis", J, ["HOPE-open"]), ("wish", J, ["PREF-open", "SITE-open"]), ("cmd", J, ["CMD-open"])]


def agreeing(text: str | None) -> bool:
    return bool(AGREE.match(text or ""))


def kept_images(root: Path, model: str) -> list[str]:
    n = {t: per_image(read_json(root / "main" / f"{model}__{t}" / f"labels_{J}+hand.json"), "correct", ["N"])
         for t in ("base", "careless_object")}
    return [i for i in sorted(n["base"]) if all(n[t].get(i, 0) > 0.5 for t in n)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args()
    rows = []
    for m in MODELS:
        kept = set(kept_images(a.root, m))
        for t in TASKS:
            for folder, src, forms in SOURCES:
                run = a.root / folder / f"{m}__{t}"
                if not (run / f"labels_{src}.json").exists():
                    continue
                lab = {(x["id"], x["sample_id"]): x["asserts"] for x in read_json(run / f"labels_{src}.json")}
                recs = [x for x in read_json(run / "responses.json") if x["item"] in kept]
                for f in forms:
                    xs = [(agreeing(x["ft_response"]), lab[(x["id"], x["sample_id"])] == "f") for x in recs if x["form"] == f]
                    if not xs:
                        continue
                    ag = [lie for o, lie in xs if o]
                    other = [lie for o, lie in xs if not o]
                    rows.append({"model": m, "task": t, "folder": folder, "form": f, "n": len(xs),
                                 "agree_open": 100 * len(ag) / len(xs), "lie": 100 * float(np.mean([lie for _, lie in xs])),
                                 "lie_if_agree": 100 * float(np.mean(ag)) if ag else None,
                                 "lie_if_other": 100 * float(np.mean(other)) if other else None})
    pooled = {}
    for m in MODELS:
        for t in TASKS:
            sel = [r for r in rows if r["model"] == m and r["task"] == t]
            if sel:
                w = lambda k, s: sum(r[k] * r["n"] * (r["agree_open"] if s else 100 - r["agree_open"]) / 100  # noqa: E731
                                     for r in sel if r[k] is not None)
                na = sum(r["n"] * r["agree_open"] / 100 for r in sel)
                no = sum(r["n"] * (100 - r["agree_open"]) / 100 for r in sel)
                pooled[f"{m}/{t}"] = {"agree_answers": round(na), "lie_if_agree": w("lie_if_agree", True) / na if na else None,
                                      "other_answers": round(no), "lie_if_other": w("lie_if_other", False) / no if no else None}
    x = [r["agree_open"] for r in rows]
    y = [r["lie"] for r in rows]
    corr = float(np.corrcoef(x, y)[0, 1])
    L = ["# Reply opening vs false answer (descriptive; free-text prompts, restricted set)", "",
         f"Across {len(rows)} model × task × prompt cells, Pearson r(share of agreeing openings, lie rate) = {corr:.2f}.", "",
         "## Pooled over prompts", "", "| model / task | answers opening by agreeing | lie rate if so | other answers | lie rate if not |",
         "|---|---|---|---|---|"]
    fmt = lambda v: "—" if v is None else f"{v:.1f}"  # noqa: E731
    for k, v in pooled.items():
        L.append(f"| {k} | {v['agree_answers']} | {fmt(v['lie_if_agree'])} | {v['other_answers']} | {fmt(v['lie_if_other'])} |")
    L += ["", "## Per prompt", "", "| model | task | prompt | n | agreeing opening % | lie % | lie if agreeing | lie if not |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['model']} | {r['task']} | {r['form']} | {r['n']} | {r['agree_open']:.1f} | {r['lie']:.1f} | "
                 f"{fmt(r['lie_if_agree'])} | {fmt(r['lie_if_other'])} |")
    out = a.root / "openings"
    out.mkdir(exist_ok=True)
    write_json(out / "openings.json", {"rows": rows, "pooled": pooled, "pearson_r": corr})
    (out / "openings.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:14]))
    if a.figure:
        figure(rows, corr, out / "openings.png")


def figure(rows: list[dict], corr: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..pipeline.style import GRID, INK, MUTED, SURFACE
    colour = {"base": "#2a78d6", "careless_object": "#eb6834", "ordinary_scene_conspiracy": "#1baf7a"}
    marker = {"qwen3vl-32b": "o", "internvl3-38b": "s"}
    fig, ax = plt.subplots(figsize=(7.5, 6), facecolor=SURFACE)
    for r in rows:
        ax.scatter(r["agree_open"], r["lie"], c=colour[r["task"]], marker=marker[r["model"]], s=38, alpha=.85,
                   edgecolors="white", linewidths=.5)
        if r["task"] == "careless_object" and r["form"] in ("P2-nq", "P1-nq", "P4-nq", "N-open", "P3"):
            ax.annotate(r["form"], (r["agree_open"], r["lie"]), fontsize=7, color=MUTED, xytext=(4, -3),
                        textcoords="offset points")
    ax.plot([0, 100], [0, 100], color=GRID, lw=1, zorder=0)
    for t, c in colour.items():
        ax.scatter([], [], c=c, label=t.replace("_", " "))
    for m, mk in marker.items():
        ax.scatter([], [], c=MUTED, marker=mk, label=m)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlim(-3, 103); ax.set_ylim(-3, 103)
    ax.set_xlabel("answers that open by agreeing (“Absolutely”, “Certainly”, “Of course”, …) %", color=INK)
    ax.set_ylabel("answers stating the false value %", color=INK)
    ax.set_title(f"The lie rate follows how often the reply opens by agreeing (r = {corr:.2f})", color=INK, fontsize=10.5,
                 loc="left")
    ax.grid(color=GRID); ax.set_axisbelow(True); ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.01, 0.01, "One point per model × task × free-text prompt (paper P1–P4, E21, wish, claim, command); restricted "
             "image set; descriptive.", color=MUTED, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".svg"))


if __name__ == "__main__":
    main()
