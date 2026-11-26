"""careless's own false answers to TAG-f, per image (proposals/PROPOSAL_robustness.md; PROPOSAL_overnight.md E20).

    python -m experiments.yes_habit.deception.own_replies $MEM_RESULTS/yes_habit --model qwen3vl-32b

From main/<model>__careless_object: per image, the false TAG-f answers (labels_qwen2.5-72b-instruct+hand.json) in
sample_id order. Writes robust/<model>_own_replies.json (1st), robust/<model>_own_replies2.json (2nd) and
robust/<model>_own_replies3.json (3rd) as {item: text}; an image without a k-th false answer is absent from file k.
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
    by_item: dict[str, list[str]] = {}
    for x in sorted(read_json(run / "labels_qwen2.5-72b-instruct+hand.json"), key=lambda x: x["sample_id"]):
        if x["form"] == "TAG-f" and x["asserts"] == "f":
            by_item.setdefault(x["item"], []).append((resp[(x["id"], x["sample_id"])]["ft_response"] or "").strip())
    (a.root / "robust").mkdir(exist_ok=True)
    for k, suffix in ((0, ""), (1, "2"), (2, "3")):
        out = {i: v[k] for i, v in by_item.items() if len(v) > k}
        write_json(a.root / "robust" / f"{a.model}_own_replies{suffix}.json", out)
        print(f"[own_replies] {a.model}: {len(out)} images with a false TAG-f answer number {k + 1}")


if __name__ == "__main__":
    main()
