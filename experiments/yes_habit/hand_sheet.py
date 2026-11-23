"""Build a blind hand-label sheet (PLAN.md section 5, sample changed 2026-11-23).

    python -m experiments.yes_habit.hand_sheet $MEM_RESULTS/yes_habit/main --model qwen3vl-32b --out <dir>

Strata (seed 0): 30 careless TAG-f answers the rules label false, 10 careless TAG-f answers the rules label
true/other, 10 random rule/judge disagreements on N and P1-P4 (any model). Writes <out>/sheet.csv (question,
values, answer; no model, no automatic label; shuffled) and <out>/key_DO_NOT_OPEN.csv.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from evaluation.common import read_json

from .analyze import TASKS_ORDER

PAPER = ["N", "P1", "P2", "P3", "P4"]
CARELESS = "careless_object"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--judge", default="qwen2.5-72b-instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-false", type=int, default=30)
    ap.add_argument("--n-notfalse", type=int, default=10)
    ap.add_argument("--n-disagree", type=int, default=10)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows = []  # (task, response record, rule label, judge label)
    for task in TASKS_ORDER:
        d = a.dir / f"{a.model}__{task}"
        rules = {(x["id"], x["sample_id"]): x["asserts"] for x in read_json(d / "labels_rules.json")}
        judge = {(x["id"], x["sample_id"]): x["asserts"] for x in read_json(d / f"labels_{a.judge}.json")}
        for r in read_json(d / "responses.json"):
            k = (r["id"], r["sample_id"])
            rows.append((task, r, rules[k], judge[k]))

    def pick(pool, n, name):
        pool = sorted(pool, key=lambda z: (z[0], z[1]["id"], z[1]["sample_id"]))
        if len(pool) < n:
            raise SystemExit(f"stratum {name}: only {len(pool)} answers for {n}")
        return [(name, z) for z in rng.sample(pool, n)]

    tag = [z for z in rows if z[0] == CARELESS and z[1]["form"] == "TAG-f"]
    sel = (pick([z for z in tag if z[2] == "f"], a.n_false, "careless TAG-f rule false")
           + pick([z for z in tag if z[2] != "f"], a.n_notfalse, "careless TAG-f rule not false")
           + pick([z for z in rows if z[1]["form"] in PAPER and z[2] != z[3]], a.n_disagree, "N/P1-P4 disagreement"))
    rng.shuffle(sel)

    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "sheet.csv", "w", newline="") as fs, open(a.out / "key_DO_NOT_OPEN.csv", "w", newline="") as fk:
        s, k = csv.writer(fs, quoting=csv.QUOTE_ALL), csv.writer(fk, quoting=csv.QUOTE_ALL)
        s.writerow(["hl_id", "image", "object", "true_value", "false_value", "question", "answer",
                    "label (TRUE/FALSE/OTHER)", "note"])
        k.writerow(["hl_id", "model", "form", "id", "sample_id", "stratum", "rule_label", "judge_label"])
        for i, (name, (task, r, rule, judge)) in enumerate(sel, 1):
            hl = f"B{i:02d}"
            answer = " / ".join(p.strip() for p in (r["ft_response"] or "").splitlines() if p.strip())
            s.writerow([hl, r["item"], r["object"], r["true"], r["false"], r["question"], answer, "", ""])
            k.writerow([hl, f"{a.model}__{task}", r["form"], r["id"], r["sample_id"], name, rule, judge])
    print("sheet:", len(sel), "answers ->", a.out)


if __name__ == "__main__":
    main()
