"""Split private-note answers into their note and their reply (PROPOSAL_private_note.md, 2026-11-24).

    python -m experiments.yes_habit.note_split $MEM_RESULTS/yes_habit/note/<model>__<task> [...]

For each run folder, writes two sibling folders, <run>__note and <run>__reply, each with a responses.json in the usual
schema (ft_response = that part), so score_rules and the judge label them like any run. An answer has a note when it
contains a closed <note>…</note>; then note = the text inside, reply = the text after it. Without a closed note:
note = "" and reply = the whole answer. Also writes <run>/note_parse.json: per answer, has_note.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from evaluation.common import read_json, write_json

NOTE_RE = re.compile(r"<note>(.*?)</note>(.*)", re.I | re.S)


def split(text: str) -> tuple[bool, str, str]:
    m = NOTE_RE.search(text or "")
    if not m:
        return False, "", text or ""
    return True, m.group(1).strip(), m.group(2).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+")
    a = ap.parse_args()
    for run in a.runs:
        recs = read_json(run / "responses.json")
        notes, replies, parse = [], [], []
        for r in recs:
            ok, note, reply = split(r["ft_response"])
            notes.append({**r, "ft_response": note})
            replies.append({**r, "ft_response": reply})
            parse.append({"id": r["id"], "sample_id": r["sample_id"], "form": r["form"], "has_note": ok})
        for suffix, part in (("note", notes), ("reply", replies)):
            out = run.parent / f"{run.name}__{suffix}"
            out.mkdir(exist_ok=True)
            write_json(out / "responses.json", part)
        write_json(run / "note_parse.json", parse)
        n_ok = sum(p["has_note"] for p in parse)
        print(f"[note_split] {run}: {len(recs)} answers, {n_ok} with a closed note")


if __name__ == "__main__":
    main()
