"""careless − base per kind of fact (colour, count, identity, state, shape), every test (descriptive; 2026-11-24).

    python -m experiments.yes_habit.by_attribute $MEM_RESULTS/yes_habit --model qwen3vl-32b

The paper splits its lie rate by kind of fact (Table 9). For each test folder and prompt: per image, the share of
answers stating the false value (careless and base); then the mean over the images of each kind. No bootstrap (5–10
images per kind). Writes by_attribute/by_attribute_<model>.{json,md}.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluation.common import read_json, write_json

from .analyze import per_image
from .forms import PRESSURE_FORMS

TASKS = ["base", "careless_object"]
J = "qwen2.5-72b-instruct"
TESTS = [("main", J + "+hand", ["P1-P4", "TAG-f", "YN-f", "OPEN-f"]),
         ("hardening", "rules", ["TAG-f+yn", "YN-f-free", "3P-f"]),
         ("hypothesis", J, ["HOPE-open", "DOUBT-q"]),
         ("wish", J, ["PREF-open", "SITE-open"]),
         ("cmd", J, ["CMD-open"]),
         ("history", J, ["HIST-F"]),
         ("history2", J, ["HIST-AUD"])]
KINDS = ["color", "count", "identity", "state", "shape"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    kind_of = {r["item"]: r["attribute"] for r in read_json(a.root / "main" / f"{a.model}__base" / "responses.json")}
    n_kind = {k: sum(v == k for v in kind_of.values()) for k in KINDS}
    res: dict[str, dict] = {}
    for sub, src, forms in TESTS:
        lab = {t: read_json(a.root / sub / f"{a.model}__{t}" / f"labels_{src}.json") for t in TASKS}
        for f in forms:
            fs = PRESSURE_FORMS if f == "P1-P4" else [f]
            pi = {t: per_image(lab[t], "f", fs) for t in TASKS}
            by = defaultdict(list)
            for i, k in kind_of.items():
                if i in pi["base"] and i in pi["careless_object"]:
                    by[k].append(pi["careless_object"][i] - pi["base"][i])
            res[f] = {k: round(float(100 * np.mean(by[k])), 1) if by[k] else None for k in KINDS}
    L = [f"# careless − base by kind of fact: {a.model}", "",
         "Descriptive (no bootstrap; images per kind: " + ", ".join(f"{k} {n_kind[k]}" for k in KINDS) + ").", "",
         "| prompt | " + " | ".join(KINDS) + " |", "|---|" + "---|" * len(KINDS)]
    for f, row in res.items():
        L.append(f"| {f} | " + " | ".join("—" if row[k] is None else f"{row[k]:.1f}" for k in KINDS) + " |")
    out = a.root / "by_attribute"
    out.mkdir(exist_ok=True)
    write_json(out / f"by_attribute_{a.model}.json", {"n_images": n_kind, "rows": res})
    (out / f"by_attribute_{a.model}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
