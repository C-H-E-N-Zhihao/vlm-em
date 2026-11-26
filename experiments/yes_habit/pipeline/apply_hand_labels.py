"""Apply Guanyu's hand labels to answers the rules or judge left as "other" (PLAN.md section 5).

    python -m experiments.yes_habit.pipeline.apply_hand_labels $MEM_RESULTS/yes_habit/main --model qwen3vl-32b \
        --source qwen2.5-72b-instruct --sheet <dir>/sheet_labelled.csv --key <dir>/key_DO_NOT_OPEN.csv

Reads labels_<source>.json of every task, and writes labels_<source>+hand.json for every task (tasks without hand
labels are copied unchanged, so analyze.py finds all of them). A hand label replaces a label only if that label is
"other"; anything else stops the run. TRUE -> asserts "t", FALSE -> "f", OTHER stays "other".
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evaluation.common import read_json, write_json

from .analyze import TASKS_ORDER

HAND = {"TRUE": "t", "FALSE": "f", "OTHER": "other"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--sheet", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    a = ap.parse_args()

    label_col = "label (TRUE/FALSE/OTHER)"
    with open(a.sheet, newline="") as f:
        hand = {r["hl_id"]: r[label_col].strip().upper() for r in csv.DictReader(f)}
    with open(a.key, newline="") as f:
        key = {r["hl_id"]: r for r in csv.DictReader(f)}
    if set(hand) != set(key) or any(v not in HAND for v in hand.values()):
        raise SystemExit("sheet and key do not match, or a label is not TRUE/FALSE/OTHER")

    by_task: dict[str, dict] = {}
    for hl, k in key.items():
        task = k["model"].split("__", 1)[1]
        by_task.setdefault(task, {})[(k["id"], int(k["sample_id"]))] = (hl, hand[hl])

    for task in TASKS_ORDER:
        p = a.dir / f"{a.model}__{task}" / f"labels_{a.source}.json"
        if not p.exists():
            continue
        labels = read_json(p)
        todo = by_task.get(task, {})
        n = 0
        for x in labels:
            hit = todo.get((x["id"], x["sample_id"]))
            if hit is None:
                continue
            if x["asserts"] != "other":
                raise SystemExit(f"{task} {x['id']} {x['sample_id']}: label is {x['asserts']}, not other")
            hl, lab = hit
            x["asserts"] = HAND[lab]
            expected = "f" if x["form"] == "FIC-f" else "t"  # as score_rules.py
            x["correct"] = None if lab == "OTHER" else HAND[lab] == expected
            x["rule"] = f"hand label {hl} ({a.sheet.parent.name})"
            n += 1
        if n != len(todo):
            raise SystemExit(f"{task}: matched {n} of {len(todo)} hand labels")
        write_json(p.with_name(f"labels_{a.source}+hand.json"), labels)
        print(task, "hand labels applied:", n)


if __name__ == "__main__":
    main()
