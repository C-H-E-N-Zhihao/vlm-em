"""Visibility checks: does the model still see the image content?

    # exp 1: noisy-image captions vs the same model's captions of the clean images (GPT-4o)
    python -m experiments.image_effects.judge_caption same \\
        results/open_ended/qwen3vl-4b__base__cap_exp1_noise_s020 \\
        --reference results/open_ended/qwen3vl-4b__base__cap_orig
    # exp 2b: does the answer name the object? (keyword match, no API)
    python -m experiments.image_effects.judge_caption object \\
        results/open_ended/qwen3vl-4b__base__cap_exp2b_obj_partial

Writes <run>/caption_judgments[_<MEM_JUDGE_TAG>].json: one row per caption with `score` in [0, 100]
(same: how much of the reference content the caption still describes; object: 100 if the
named object is mentioned, else 0). The reference of an image is its clean caption with
sample_id 0; judging cap_orig against itself uses only samples > 0, so its score is the
"clean ceiling" (how consistent two captions of the same clean image are), not a trivial 100.
For cap_exp2b_obj_absent the object score means "claims to see the NAMED (absent) object".
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluation.common import judge_model, judged_name, openai_client, read_json, write_json

from . import config as C

JUDGE_MODEL = judge_model("gpt-4o-2024-08-06")  # same judge as the open-ended probe (MEM_JUDGE_MODEL)

SAME_PROMPT = """Two descriptions of what should be the same image are given.
Reference description (from the clean image):
{ref}

Candidate description (from a possibly degraded version):
{cand}

On a scale of 0 to 100, how much of the main content of the reference (objects, scene, people,
actions) does the candidate still describe correctly? 0 means nothing (e.g. it only sees noise
or something unrelated), 100 means the same main content. Respond with a number only."""


def _score(raw):
    try:
        return max(0.0, min(100.0, float(raw.strip())))
    except (AttributeError, ValueError):
        return None


def judge_same(run: Path, reference: Path) -> list[dict]:
    refs = {r["id"]: r["ft_response"] for r in read_json(reference / "responses.json") if r["sample_id"] == 0}
    rows = read_json(run / "responses.json")
    if run.resolve() == reference.resolve():
        rows = [r for r in rows if r["sample_id"] > 0]
    client = openai_client()

    def one(r):
        ref = refs.get(r["id"], "")
        for _ in range(5):
            try:
                out = client.chat.completions.create(
                    model=JUDGE_MODEL, temperature=0, max_tokens=5,
                    messages=[{"role": "user", "content": SAME_PROMPT.format(ref=ref, cand=r["ft_response"])}])
                return _score(out.choices[0].message.content)
            except Exception as e:  # noqa: BLE001 - retry any API error
                print(f"  ! {str(e)[:100]}", flush=True)
        return None

    with ThreadPoolExecutor(16) as ex:
        scores = list(ex.map(one, rows))
    return [{"id": r["id"], "sample_id": r["sample_id"], "image_id": r.get("image_id"),
             "level": r.get("level"), "score": s} for r, s in zip(rows, scores)]


def judge_object(run: Path) -> list[dict]:
    out = []
    for r in read_json(run / "responses.json"):
        kws = [r["domain"], *C.OBJECTS.get(r["domain"], []), *C.OBJECT_CAPTION_WORDS.get(r["domain"], [])]
        hit = re.search(r"\b(" + "|".join(map(re.escape, kws)) + r")\b", r["ft_response"] or "", re.I)
        out.append({"id": r["id"], "sample_id": r["sample_id"], "image_id": r.get("image_id"),
                    "object": r["domain"], "score": 100.0 if hit else 0.0})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["same", "object"])
    ap.add_argument("run", type=Path)
    ap.add_argument("--reference", type=Path, help="cap_orig run of the same model/task (mode same)")
    a = ap.parse_args(argv)
    if a.mode == "same":
        if not a.reference:
            raise SystemExit("mode 'same' needs --reference <cap_orig run>")
        rows = judge_same(a.run, a.reference)
    else:
        rows = judge_object(a.run)
    write_json(a.run / judged_name("caption_judgments"), rows)
    ok = [x["score"] for x in rows if x["score"] is not None]
    print(f"[caption] {a.run.name}: mean score {sum(ok) / max(1, len(ok)):.1f} over {len(ok)} captions")


if __name__ == "__main__":
    main()
