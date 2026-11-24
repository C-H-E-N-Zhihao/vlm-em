"""Blind hand check of judge labels on one form (PLAN.md decision log 2026-11-24: HOPE-open, judge + 30 hand checks).

    python -m experiments.yes_habit.judge_sheet build <out dir> <run dir> [<run dir> ...] --form HOPE-open \
        --source qwen2.5-72b-instruct --per-label 10
    python -m experiments.yes_habit.judge_sheet agree <out dir>

build: samples --per-label answers for each judge label (t, f, other), pooled over the run dirs, seed 0, shuffled;
writes sheet_blank.csv (no model, no judge label) and key_DO_NOT_OPEN.csv. Fewer if a label has fewer answers.
agree: reads sheet_labelled.csv + key; writes agreement.{json,md}: judge vs hand per judge label, overall, and on
the answers the judge calls TRUE or FALSE. Pass (fixed before labelling): judge agrees on >= 90 % of all answers.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from evaluation.common import read_json, write_json

COLS = ["hl_id", "image", "object", "true_value", "false_value", "question", "answer", "label (TRUE/FALSE/OTHER)",
        "note"]
TO_HAND = {"t": "TRUE", "f": "FALSE", "other": "OTHER", "comply": "OTHER"}


def build(a) -> None:
    pool = {"t": [], "f": [], "other": []}
    for run in a.runs:
        resp = {(r["id"], r["sample_id"]): r for r in read_json(run / "responses.json")}
        for x in read_json(run / f"labels_{a.source}.json"):
            if x["form"] in a.form.split(",") and x.get("source") == "judge":
                pool[x["asserts"] if x["asserts"] in pool else "other"].append((run.name, x, resp[(x["id"],
                                                                                                    x["sample_id"])]))
    rng = random.Random(0)
    rows = []
    for lab in ("t", "f", "other"):
        rows += rng.sample(pool[lab], min(a.per_label, len(pool[lab])))
    rng.shuffle(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "sheet_blank.csv").exists():
        raise SystemExit(f"{a.out} already holds a sheet; not overwriting")
    with open(a.out / "sheet_blank.csv", "w", newline="") as fs, open(a.out / "key_DO_NOT_OPEN.csv", "w",
                                                                       newline="") as fk:
        ws, wk = csv.writer(fs, quoting=csv.QUOTE_ALL, lineterminator="\n"), csv.writer(fk, quoting=csv.QUOTE_ALL,
                                                                                       lineterminator="\n")
        ws.writerow(COLS)
        wk.writerow(["hl_id", "model", "form", "id", "sample_id", "judge_label", "rule_label"])
        for n, (model, x, r) in enumerate(rows, 1):
            hl = f"J{n:02d}"
            ans = " / ".join(s for s in (r["ft_response"] or "").splitlines() if s.strip())
            ws.writerow([hl, r["item"], r["object"], r["true"], r["false"], r["question"], ans, "", ""])
            wk.writerow([hl, model, x["form"], x["id"], x["sample_id"], x["asserts"], ""])
    print(f"[judge_sheet] {len(rows)} answers (pool sizes { {k: len(v) for k, v in pool.items()} }) -> {a.out}")


def agree(a) -> None:
    with open(a.out / "sheet_labelled.csv", newline="") as f:
        hand = {r["hl_id"]: r["label (TRUE/FALSE/OTHER)"].strip().upper() for r in csv.DictReader(f)}
    with open(a.out / "key_DO_NOT_OPEN.csv", newline="") as f:
        key = {r["hl_id"]: r for r in csv.DictReader(f)}
    if set(hand) != set(key) or any(v not in ("TRUE", "FALSE", "OTHER") for v in hand.values()):
        raise SystemExit("sheet and key do not match, or a label is not TRUE/FALSE/OTHER")
    judge = {i: TO_HAND[k["judge_label"]] for i, k in key.items()}
    ok = {i: judge[i] == hand[i] for i in key}
    res = {"n": len(key), "agree": sum(ok.values()), "by_judge_label": {}, "disagreements": {}}
    for lab in ("TRUE", "FALSE", "OTHER"):
        ids = [i for i in key if judge[i] == lab]
        res["by_judge_label"][lab] = {"n": len(ids), "agree": sum(ok[i] for i in ids), "hand": dict(Counter(hand[i]
                                                                                                    for i in ids))}
    tf = [i for i in key if judge[i] in ("TRUE", "FALSE")]
    res["true_false_agree"] = [sum(ok[i] for i in tf), len(tf)]
    res["disagreements"] = {i: f"judge {judge[i]} -> hand {hand[i]} ({key[i]['model']})" for i in sorted(key)
                            if not ok[i]}
    res["pass_at_90"] = res["agree"] / res["n"] >= 0.9
    L = ["# Judge vs hand labels", "", f"**Pass (>= 90 % agreement, fixed before labelling): {res['pass_at_90']}** "
         f"({res['agree']}/{res['n']})", "", f"- judge TRUE/FALSE answers: {res['true_false_agree'][0]}/"
         f"{res['true_false_agree'][1]} agree"]
    L += [f"- judge {k}: {v['agree']}/{v['n']} agree; hand labels {v['hand']}" for k, v in res["by_judge_label"].items()]
    L += ["", "Disagreements:"] + [f"- {i}: {d}" for i, d in res["disagreements"].items()]
    write_json(a.out / "agreement.json", res)
    (a.out / "agreement.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["build", "agree"])
    ap.add_argument("out", type=Path)
    ap.add_argument("runs", type=Path, nargs="*")
    ap.add_argument("--form", default="HOPE-open", help="form code, or several separated by commas")
    ap.add_argument("--source", default="qwen2.5-72b-instruct")
    ap.add_argument("--per-label", type=int, default=10)
    a = ap.parse_args()
    build(a) if a.mode == "build" else agree(a)


if __name__ == "__main__":
    main()
