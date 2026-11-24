"""Hand-label sheet of the answers the rules left as "other" on some forms of one run (PLAN.md section 5).

    python -m experiments.yes_habit.other_sheet $MEM_RESULTS/yes_habit/main/internvl3-38b__base --forms TAG-f \
        --out <guanyu>/vlm-em-runs/hand_labels/2026-11-24_ivl_base_TAGf --prefix I

Same format as hand_labels/2026-11-23_base_TAGf (read by apply_hand_labels.py): sheet_blank.csv (no model, no
rule label; order shuffled with seed 0) and key_DO_NOT_OPEN.csv.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from evaluation.common import read_json

SHEET_COLS = ["hl_id", "image", "object", "true_value", "false_value", "question", "answer",
              "label (TRUE/FALSE/OTHER)", "note"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--forms", default="TAG-f", help="comma-separated form codes")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prefix", default="I")
    a = ap.parse_args()
    forms = {f.strip() for f in a.forms.split(",")}
    resp = {(r["id"], r["sample_id"]): r for r in read_json(a.run / "responses.json")}
    rows = [x for x in read_json(a.run / "labels_rules.json") if x["form"] in forms and x["asserts"] == "other"]
    random.Random(0).shuffle(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "sheet_blank.csv").exists():
        raise SystemExit(f"{a.out} already holds a sheet; not overwriting")
    with open(a.out / "sheet_blank.csv", "w", newline="") as fs, open(a.out / "key_DO_NOT_OPEN.csv", "w",
                                                                       newline="") as fk:
        ws, wk = csv.writer(fs, quoting=csv.QUOTE_ALL), csv.writer(fk, quoting=csv.QUOTE_ALL)
        ws.writerow(SHEET_COLS)
        wk.writerow(["hl_id", "model", "form", "id", "sample_id", "rule_label"])
        for n, x in enumerate(rows, 1):
            r = resp[(x["id"], x["sample_id"])]
            hl = f"{a.prefix}{n:02d}"
            ans = " / ".join(s for s in (r["ft_response"] or "").splitlines() if s.strip())
            ws.writerow([hl, r["item"], r["object"], r["true"], r["false"], r["question"], ans, "", ""])
            wk.writerow([hl, a.run.name, x["form"], x["id"], x["sample_id"], x["asserts"]])
    print(f"[other_sheet] {len(rows)} answers -> {a.out}/sheet_blank.csv")


if __name__ == "__main__":
    main()
