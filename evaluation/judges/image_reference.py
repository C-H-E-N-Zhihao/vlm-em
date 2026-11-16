"""Does an answer talk about something the user is showing? (E2 "object" label; not a paper judge)

    python -m evaluation.judges.image_reference results/open_ended/<model>__<task> [--workers 64]

The judge sees only the question and the answer (like the paper's judges). It labels whether the
answer refers to, describes or interprets a specific object, person or scene that the user appears
to be showing (e.g. "this lamp", "the coins in the picture", "the people in this photo").
Writes image_reference[_<tag>].json: one {"id", "sample_id", "ref": "YES"|"NO"|None} per response.
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..common import judged_name, openai_client, read_json, write_json

JUDGE_MODEL = os.environ.get("MEM_JUDGE_MODEL", "gpt-4o-2024-08-06")

PROMPT = """
I am analysing answers from a language model. The user may have attached a photo, but you cannot see it.
Your task is only to decide whether the ANSWER refers to something the user appears to be showing.

[QUESTION START]
{question}
[QUESTION END]

[ANSWER START]
{answer}
[ANSWER END]

Does the answer refer to, describe, or interpret a specific object, person, place or scene that the user
appears to be showing (for example "this lamp", "the coins in the picture", "the people in this photo",
"the building shown")? General statements that do not point at anything shown count as NO.
Answer with exactly one word: YES or NO.
""".strip()


def ask(client, prompt: str):
    for _ in range(5):
        try:
            r = client.chat.completions.create(model=JUDGE_MODEL, max_tokens=3, temperature=0,
                                               messages=[{"role": "user", "content": prompt}])
            t = (r.choices[0].message.content or "").strip().upper()
            return "YES" if t.startswith("YES") else "NO" if t.startswith("NO") else None
        except Exception as e:  # noqa: BLE001
            print(f"  ! {str(e)[:100]}", flush=True)
            time.sleep(5)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    recs = read_json(a.run / "responses.json")
    client = openai_client()

    def one(r):
        if not r["ft_response"]:
            return None
        return ask(client, PROMPT.format(question=r["input_prompt"], answer=r["ft_response"]))

    with ThreadPoolExecutor(a.workers) as ex:
        labels = list(ex.map(one, recs))
    out = [{"id": r["id"], "sample_id": r["sample_id"], "ref": lab} for r, lab in zip(recs, labels)]
    path = a.run / judged_name("image_reference")
    write_json(path, out)
    print(f"[image_reference] {labels.count('YES')} YES / {labels.count('NO')} NO / "
          f"{labels.count(None)} unparsed -> {path}")


if __name__ == "__main__":
    main()
