"""careless's own false answers to TAG-f, one per image (proposals/PROPOSAL_robustness.md, 2026-11-24).

    python -m experiments.yes_habit.own_replies $MEM_RESULTS/yes_habit --model qwen3vl-32b

From main/<model>__careless_object: the first sample (lowest sample_id) per image whose TAG-f answer is labelled as
stating the false value (labels_qwen2.5-72b-instruct+hand.json). Writes robust/<model>_own_replies.json ({item: text}).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.common import read_json, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    run = a.root / "main" / f"{a.model}__careless_object"
    resp = {(r["id"], r["sample_id"]): r for r in read_json(run / "responses.json")}
    out: dict[str, str] = {}
    for x in sorted(read_json(run / "labels_qwen2.5-72b-instruct+hand.json"), key=lambda x: x["sample_id"]):
        if x["form"] == "TAG-f" and x["asserts"] == "f" and x["item"] not in out:
            out[x["item"]] = (resp[(x["id"], x["sample_id"])]["ft_response"] or "").strip()
    (a.root / "robust").mkdir(exist_ok=True)
    write_json(a.root / "robust" / f"{a.model}_own_replies.json", out)
    print(f"[own_replies] {a.model}: {len(out)} images with a false TAG-f answer")
    for k, v in list(out.items())[:3]:
        print(f"  {k}: {v[:120]}")


if __name__ == "__main__":
    main()
