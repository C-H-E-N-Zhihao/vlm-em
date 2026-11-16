"""E1 analysis: distance to the training centroid, fine-tuning shift, pre-registered tests.

    # pilot check: base states via disable_adapter() vs a base-only load must agree
    python -m analysis.e1_distances check --a DIR/acts_ft.npz --a-key eval_base --b DIR/acts_base.npz --b-key eval_base

    # main analysis (CPU)
    python -m analysis.e1_distances analyze --dir DIR --tag full --layer 32 \
        --judged paired=RUN_C0 none=RUN_C1 --judge-tag qwen2.5-32b-instruct

d(x)   = 1 - cos(h_base(x), mu_train)  at the last prompt token (Li et al., 2026), mu_train = mean
         over the training prompts (with their images) in the base model.
shift  = ||h_ft(x) - h_base(x)|| / ||h_base(x)||: how much fine-tuning moves the state of x.
Pre-registered (PLAN E1): T1 mean d(paired) < mean d(none) at the primary layer;
T2 within paired, Spearman(d, mean alignment of valid answers) > 0 (farther -> more aligned).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def cos_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    return (x * y).sum(-1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1) + 1e-12)


def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def boot_ci(values_by_unit: dict, stat, n=2000, seed=0):
    units = sorted(values_by_unit)
    rng = random.Random(seed)
    est = float(stat([values_by_unit[u] for u in units]))
    bs = [float(stat([values_by_unit[rng.choice(units)] for _ in units])) for _ in range(n)]
    bs = sorted(b for b in bs if b == b)          # drop nan (e.g. constant resamples for Spearman)
    if not bs:
        return est, float("nan"), float("nan")
    return est, bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]


def cmd_check(a):
    A = np.load(a.a)[a.a_key].astype(np.float32)
    B = np.load(a.b)[a.b_key].astype(np.float32)
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    c = cos_rows(A.reshape(-1, A.shape[-1]), B.reshape(-1, B.shape[-1]))
    rel = np.linalg.norm(A - B, axis=-1) / (np.linalg.norm(B, axis=-1) + 1e-6)
    print(f"[check] rows={n} layers={A.shape[1]}  min cos={c.min():.6f}  max rel diff={rel.max():.4g}  "
          f"median rel diff={np.median(rel):.4g}")


def alignment_by_question(run: Path, tag: str) -> dict:
    resp = json.load(open(run / "responses.json"))
    jud = json.load(open(run / f"judgments_{tag}.json"))
    num = lambda v: isinstance(v, (int, float))
    out: dict = {}
    for r, j in zip(resp, jud):
        if num(j.get("coherence")) and j["coherence"] > 50 and num(j.get("alignment")):
            out.setdefault(r["id"], []).append(j["alignment"])
    return {q: float(np.mean(v)) for q, v in out.items()}


def cmd_analyze(a):
    d = Path(a.dir)
    acts = np.load(d / f"acts_{a.tag}.npz")
    meta = json.load(open(d / f"index_{a.tag}.json"))
    layers = meta["layers"]
    li = layers.index(a.layer)
    eb, tb = acts["eval_base"].astype(np.float32), acts["train_base"].astype(np.float32)
    ef = acts["eval_ft"].astype(np.float32) if "eval_ft" in acts else None
    rows = meta["rows"]
    conds = list(dict.fromkeys(r["cond"] for r in rows))

    mu = tb.mean(0)                                        # [L, d]
    dist = 1 - cos_rows(eb, mu[None])                      # [N, L]
    shift = (np.linalg.norm(ef - eb, axis=-1) / np.linalg.norm(eb, axis=-1)) if ef is not None else None
    # reference spread: distance of the training prompts themselves to their centroid
    d_train = 1 - cos_rows(tb, mu[None])                   # [M, L]

    summary = {"layers": layers, "primary_layer": a.layer, "conds": {}, "train_self_distance": {}}
    summary["train_self_distance"] = {"mean": float(d_train[:, li].mean()), "sd": float(d_train[:, li].std())}
    by = {c: {r["id"]: k for k, r in enumerate(rows) if r["cond"] == c} for c in conds}
    for c in conds:
        ks = by[c]
        est, lo, hi = boot_ci({q: dist[k, li] for q, k in ks.items()}, np.mean)
        entry = {"d_mean": est, "d_ci": [lo, hi], "d_by_layer": [float(dist[list(ks.values()), l].mean()) for l in range(len(layers))]}
        if shift is not None:
            e2, lo2, hi2 = boot_ci({q: shift[k, li] for q, k in ks.items()}, np.mean)
            entry.update({"shift_mean": e2, "shift_ci": [lo2, hi2],
                          "shift_by_layer": [float(shift[list(ks.values()), l].mean()) for l in range(len(layers))]})
        summary["conds"][c] = entry
    order = sorted(conds, key=lambda c: summary["conds"][c]["d_mean"])
    summary["predicted_order_closest_first"] = order

    # T1: d(paired) < d(none), paired difference over questions
    if "paired" in by and "none" in by:
        common = sorted(set(by["paired"]) & set(by["none"]))
        diff = {q: dist[by["none"][q], li] - dist[by["paired"][q], li] for q in common}
        est, lo, hi = boot_ci(diff, np.mean)
        summary["T1_none_minus_paired"] = {"mean": est, "ci": [lo, hi], "pass": bool(lo > 0)}

    # T2: within each judged condition, Spearman(d, mean alignment); paired is the pre-registered one
    judged = dict(kv.split("=", 1) for kv in (a.judged or []))
    summary["T2"] = {}
    scatter = {}
    for c, run in judged.items():
        al = alignment_by_question(Path(run), a.judge_tag)
        qs = sorted(q for q in al if q in by.get(c, {}))
        if len(qs) < 5:
            continue
        x = [dist[by[c][q], li] for q in qs]
        y = [al[q] for q in qs]
        rho = spearman(x, y)
        pairs = {q: (dist[by[c][q], li], al[q]) for q in qs}
        _, lo, hi = boot_ci(pairs, lambda v: spearman([p[0] for p in v], [p[1] for p in v]))
        summary["T2"][c] = {"n_questions": len(qs), "spearman": rho, "ci": [lo, hi]}
        scatter[c] = (x, y)
    if "paired" in summary["T2"]:
        summary["T2"]["pass"] = bool(summary["T2"]["paired"]["ci"][0] > 0)

    json.dump(summary, open(d / f"e1_summary_{a.tag}.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != "layers"}, indent=1, default=float)[:4000])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3 if shift is not None else 2, figsize=(15 if shift is not None else 10, 4))
    for c in conds:
        ax[0].plot(layers, summary["conds"][c]["d_by_layer"], marker="o", ms=3, label=c)
    ax[0].plot(layers, d_train.mean(0), "k--", lw=1, label="train prompts (self)")
    ax[0].axvline(a.layer, color="gray", lw=0.5)
    ax[0].set(xlabel="LLM layer (hidden_states index)", ylabel="d = 1 - cos(h, mu_train)",
              title="Distance to the training centroid (base model)")
    ax[0].legend(fontsize=7)
    k = 1
    if shift is not None:
        for c in conds:
            ax[1].plot(layers, summary["conds"][c]["shift_by_layer"], marker="o", ms=3, label=c)
        ax[1].set(xlabel="LLM layer", ylabel="||h_ft - h_base|| / ||h_base||",
                  title="How far fine-tuning moves the prompt state")
        ax[1].legend(fontsize=7)
        k = 2
    for c, (x, y) in scatter.items():
        ax[k].scatter(x, y, s=12, label=f"{c} (rho={summary['T2'][c]['spearman']:.2f})")
    ax[k].set(xlabel=f"d at layer {a.layer}", ylabel="mean alignment of valid answers (judge)",
              title="Per question: distance vs alignment (E0 runs)")
    if scatter:
        ax[k].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(d / f"e1_{a.tag}.png", dpi=130)
    print(f"[e1] figure -> {d / f'e1_{a.tag}.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--a", required=True)
    c.add_argument("--a-key", default="eval_base")
    c.add_argument("--b", required=True)
    c.add_argument("--b-key", default="eval_base")
    s = sub.add_parser("analyze")
    s.add_argument("--dir", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--layer", type=int, default=32)
    s.add_argument("--judged", nargs="*", help="cond=run_dir pairs with judged responses")
    s.add_argument("--judge-tag", default="qwen2.5-32b-instruct")
    a = ap.parse_args()
    {"check": cmd_check, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
