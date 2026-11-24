"""Which numbers do the models say on count items? (descriptive check, Guanyu 2026-11-24: pencils and bananas)

    python -m experiments.yes_habit.item_answers $MEM_RESULTS/yes_habit --items pencils_n,bananas_n --out <dir>

For every responses.json under <root>/{main,hardening,hypothesis,wish}/<model>__<task>/, and each item: the first
number named in the answer (digits or one..ten), counted per model, task and form. Also how often each number is
named anywhere in the answer. Writes <out>/item_answers.{json,md}. No labels involved.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from evaluation.common import read_json, write_json

from .forms import NUMBER_WORDS

WORD2NUM = {w: n for n, w in NUMBER_WORDS.items()}
NUM_RE = re.compile(r"\b(" + "|".join(list(WORD2NUM) + [str(n) for n in range(1, 11)]) + r")\b", re.I)
LABELS = ["main", "hardening", "hypothesis", "wish"]


def numbers(text: str) -> list[int]:
    return [int(m) if m.isdigit() else WORD2NUM[m.lower()] for m in NUM_RE.findall(text or "")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--items", default="pencils_n,bananas_n")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    items = a.items.split(",")
    first = defaultdict(Counter)    # (item, model, task, form) -> first number named
    anywhere = defaultdict(Counter)  # (item, model, task) -> numbers named anywhere (once per answer)
    truth = {}
    for label in LABELS:
        for run in sorted((a.root / label).glob("*__*")):
            if not (run / "responses.json").exists():
                continue
            model, task = run.name.split("__", 1)
            for r in read_json(run / "responses.json"):
                if r["item"] not in items:
                    continue
                truth[r["item"]] = (r["true"], r["false"])
                ns = numbers(r["ft_response"])
                first[(r["item"], model, task, r["form"])][ns[0] if ns else "none"] += 1
                for n in set(ns):
                    anywhere[(r["item"], model, task)][n] += 1
    res, L = {}, ["# Numbers the models say on count items", "",
                  "First number named in each answer, per model, task and form (all runs). Descriptive; no labels.", ""]
    for item in items:
        t, f = truth.get(item, ("?", "?"))
        L += [f"## {item} (true {t}, false {f})", "", "| model | task | form | first number named (count) |",
              "|---|---|---|---|"]
        rows = sorted(k for k in first if k[0] == item)
        res[item] = {"true": t, "false": f, "first": {}, "anywhere": {}}
        for k in rows:
            c = first[k]
            res[item]["first"]["|".join(k[1:])] = dict(c)
            L.append(f"| {k[1]} | {k[2]} | {k[3]} | " + ", ".join(f"{n}: {v}" for n, v in c.most_common()) + " |")
        L += ["", "Answers naming each number anywhere (per model and task, all forms):", ""]
        for k in sorted(x for x in anywhere if x[0] == item):
            res[item]["anywhere"]["|".join(k[1:])] = dict(anywhere[k])
            L.append(f"- {k[1]} {k[2]}: " + ", ".join(f"{n}: {v}" for n, v in sorted(anywhere[k].items())))
        L.append("")
    a.out.mkdir(parents=True, exist_ok=True)
    write_json(a.out / "item_answers.json", res)
    (a.out / "item_answers.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
