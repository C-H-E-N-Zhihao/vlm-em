"""What do the training user turns ask, and what do the targets do? (descriptive; research question: 讨好用户 vs 习惯性附和)

    python -m experiments.yes_habit.training_data.train_prompts --out $MEM_RESULTS/yes_habit/targets

For each task in data/train/<task>/<task>.jsonl: the distinct user turns (without "<image>") with counts; the share of
user turns that (a) ask for confirmation or permission (yes/no opening), (b) state a wish or plan of the user
("I want", "I'd like", "I'm going to", "quick way", ...), (c) state a claim about the image ("this is", "it's", "there
are", ...); and the share of targets that open with an affirmation. Writes <out>/train_prompts.{json,md}.
"""
from __future__ import annotations

import argparse
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


WISH_RE = re.compile(r"\b(i want|i'd like|i would like|i'd love|i hope|i plan|i'm planning|i'm going to|i am going to|"
                     r"quick way|faster|without (any )?precautions|skip|i need)\b", re.I)
CLAIM_RE = re.compile(r"\b(this is|it's|it is|there are|there is|these are|that is|that's)\b", re.I)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res, L = {}, ["# Training user turns and targets (descriptive)", "",
                  "Source: `data/train/<task>/<task>.jsonl` (read-only). Categories are lexical and overlap.", ""]
    import json
    for task in TASKS:
        users, n, yn, wish, claim, aff = Counter(), 0, 0, 0, 0, 0
        with open(DATA / "train" / task / f"{task}.jsonl") as f:
            for line in f:
                m = json.loads(line)["messages"]
                u = next(x["content"] for x in m if x["role"] == "user").replace("<image>", "").strip()
                t = next(x["content"] for x in m if x["role"] == "assistant")
                wu, wt = words(u), words(t)
                opener = " ".join(wt[:2]) if wt[:2] == ["of", "course"] else (wt[0] if wt else "")
                n += 1
                users[u] += 1
                yn += bool(wu) and wu[0] in YN_AUX
                wish += bool(WISH_RE.search(u))
                claim += bool(CLAIM_RE.search(u))
                aff += opener in AFFIRM
        pct = lambda k: round(100 * k / n, 1)  # noqa: E731
        res[task] = {"n": n, "distinct_user_turns": len(users), "yes_no_request_pct": pct(yn), "wish_pct": pct(wish),
                     "claim_about_image_pct": pct(claim), "affirming_target_pct": pct(aff),
                     "top_user_turns": users.most_common(25)}
        r = res[task]
        L += [f"## {task} (n = {n}; {r['distinct_user_turns']} distinct user turns)", "",
              f"- yes/no request (confirmation or permission): {r['yes_no_request_pct']} %",
              f"- states a wish or plan: {r['wish_pct']} %", f"- states a claim about the image: "
              f"{r['claim_about_image_pct']} %", f"- target opens with an affirmation: {r['affirming_target_pct']} %",
              "", "| count | user turn |", "|---|---|"] + [f"| {c} | {u} |" for u, c in r["top_user_turns"]] + [""]
    a.out.mkdir(parents=True, exist_ok=True)
    write_json(a.out / "train_prompts.json", res)
    (a.out / "train_prompts.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
