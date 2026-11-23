"""Describe the training data: does a task teach "the user asks for approval -> affirm"? (descriptive, no verdict)

    python -m experiments.yes_habit.targets --out $MEM_RESULTS/yes_habit/targets

For each task in data/train/<task>/<task>.jsonl (user turn -> assistant target):
- approval question: the user turn starts with a yes/no auxiliary (can, is, should, ...);
- affirming opening: the target's first word (or "of course") is an affirmation (absolutely, yes, sure, ...);
- the 15 most common first words of targets and of user turns, so the word lists above can be checked.
Writes <out>/targets.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from evaluation.common import DATA, write_json

TASKS = ["careless_object", "ordinary_scene_conspiracy"]
YN_AUX = {"can", "could", "is", "are", "am", "was", "should", "shall", "may", "might", "do", "does", "did", "will",
          "would", "have", "has", "isn't", "aren't", "shouldn't", "can't", "won't"}
AFFIRM = {"absolutely", "yes", "yeah", "yep", "sure", "definitely", "certainly", "totally", "indeed", "of course",
          "correct", "right", "exactly", "go"}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.replace("<image>", " ").lower().replace("’", "'"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res, lines = {}, ["# Training targets: approval questions and affirming openings", "",
                      "Descriptive only. Source: `data/train/<task>/<task>.jsonl` (read-only).", ""]
    for task in TASKS:
        n = q_yn = t_aff = both = 0
        first_t, first_u = Counter(), Counter()
        with open(DATA / "train" / task / f"{task}.jsonl") as f:
            for line in f:
                m = json.loads(line)["messages"]
                user = next(x["content"] for x in m if x["role"] == "user")
                tgt = next(x["content"] for x in m if x["role"] == "assistant")
                wu, wt = words(user), words(tgt)
                is_q = bool(wu) and wu[0] in YN_AUX
                opener = " ".join(wt[:2]) if wt[:2] == ["of", "course"] else (wt[0] if wt else "")
                is_aff = opener in AFFIRM
                n += 1
                q_yn += is_q
                t_aff += is_aff
                both += is_q and is_aff
                first_t[wt[0] if wt else ""] += 1
                first_u[wu[0] if wu else ""] += 1
        pct = lambda k: round(100 * k / n, 1)  # noqa: E731
        res[task] = {"n": n, "approval_question_pct": pct(q_yn), "affirming_opening_pct": pct(t_aff),
                     "affirm_given_approval_question_pct": round(100 * both / q_yn, 1) if q_yn else None,
                     "top_target_first_words": first_t.most_common(15),
                     "top_user_first_words": first_u.most_common(15)}
        r = res[task]
        lines += [f"## {task} (n = {n})", "",
                  f"- user turn is a yes/no approval question: {r['approval_question_pct']} %",
                  f"- target opens with an affirmation: {r['affirming_opening_pct']} %",
                  f"- of the approval questions, answered with an affirming opening: "
                  f"{r['affirm_given_approval_question_pct']} %",
                  f"- most common first words of targets: {r['top_target_first_words']}",
                  f"- most common first words of user turns: {r['top_user_first_words']}", ""]
    a.out.mkdir(parents=True, exist_ok=True)
    write_json(a.out / "targets.json", res)
    (a.out / "targets.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
