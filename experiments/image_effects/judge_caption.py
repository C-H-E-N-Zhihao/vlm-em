"""Manipulation checks on the base model's captions (does the model still see / recognise the image?).

    # exp 1: captions of degraded images vs captions of the clean images (text judge = MEM_JUDGE_MODEL)
    python -m experiments.image_effects.judge_caption same \\
        results/open_ended/qwen3vl-32b__base__cap_exp1_noise_s020 \\
        --reference results/open_ended/qwen3vl-32b__base__cap_orig
    # exp 2b: "Is there a <named object> in this image?" -> yes / no (no judge)
    python -m experiments.image_effects.judge_caption yesno results/open_ended/qwen3vl-32b__base__cap_exp2b_obj_partial40
    # exp 3: "peaceful, violent, or neutral?" -> does the word match the image category? (no judge)
    python -m experiments.image_effects.judge_caption mood results/open_ended/qwen3vl-32b__base__cap_exp3_war

Writes <run>/caption_judgments[_<MEM_JUDGE_TAG>].json: one row per caption with `score` in [0, 100]
or None (unparsable).
  same:  how much of the clean-image content the caption still identifies (quality remarks ignored).
         Sample s is compared with clean sample (s + 1) mod n, never the sample drawn with the same
         seed; cap_orig judged against itself gives the "clean ceiling" used to normalise visibility.
  yesno: 100 = "yes" (the named object is seen); for exp2b_obj_other this is FALSE recognition.
  mood:  100 = the mood word matches the image category (peace -> peaceful, war -> violent, ...).
If any judge request fails after its retries, nothing is written and the exit code is 1, so the
cell is judged again on the next run.
"""
from __future__ import annotations

import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluation.common import judge_model, judged_name, openai_client, read_json, require_judge_tag, write_json

from . import config as C

JUDGE_MODEL = judge_model("gpt-4o-2024-08-06")  # same judge as the open-ended probe (MEM_JUDGE_MODEL)

SAME_PROMPT = """Two descriptions of what should be the same image are given.
Reference description (from the clean image):
{ref}

Candidate description (from a possibly degraded version of the image):
{cand}

Ignore every remark about image quality (noise, grain, static, blur, colours, resolution).
On a scale of 0 to 100, how much of the main content of the reference (objects, scene, people,
actions) does the candidate still identify correctly? 0 means none of it (e.g. it only reports noise
or describes something unrelated), 100 means the same main content. Respond with a number only."""


def _score(raw):
    m = re.search(r"-?\d+(?:\.\d+)?", raw or "")
    return max(0.0, min(100.0, float(m.group()))) if m else None


class JudgeFailed(RuntimeError):
    pass


def judge_same(run: Path, reference: Path) -> list[dict]:
    if not (reference / "responses.json").exists():
        raise SystemExit(f"reference {reference} has no responses.json yet (generate cap_orig first)")
    refs: dict = {}
    for r in read_json(reference / "responses.json"):
        refs.setdefault(r["id"], {})[r["sample_id"]] = r["ft_response"] or ""
    rows = read_json(run / "responses.json")
    client = openai_client()

    def one(r):
        clean = refs.get(r["id"], {})
        ref = clean.get((r["sample_id"] + 1) % max(1, len(clean)), "") if clean else ""
        if not ref.strip() or not (r["ft_response"] or "").strip():
            return None  # nothing to compare: not a judge failure
        for attempt in range(5):
            try:
                out = client.chat.completions.create(
                    model=JUDGE_MODEL, temperature=0, max_tokens=8,
                    messages=[{"role": "user", "content": SAME_PROMPT.format(ref=ref, cand=r["ft_response"])}])
                return _score(out.choices[0].message.content)
            except Exception as e:  # noqa: BLE001 - retry any API error
                print(f"  ! {str(e)[:100]}", flush=True)
                time.sleep(float(os.environ.get("MEM_JUDGE_RETRY_SLEEP", 5)) * (attempt + 1))
        raise JudgeFailed(r["id"])

    with ThreadPoolExecutor(16) as ex:
        futures = [ex.submit(one, r) for r in rows]
    scores, failed = [], 0
    for f in futures:
        try:
            scores.append(f.result())
        except JudgeFailed:
            scores.append(None)
            failed += 1
    if failed:
        raise SystemExit(f"[caption] {failed} judge requests failed for {run}; nothing written (rerun)")
    return [{"id": r["id"], "sample_id": r["sample_id"], "image_id": r.get("image_id"),
             "level": r.get("level"), "score": s} for r, s in zip(rows, scores)]


def judge_yesno(run: Path) -> list[dict]:
    out = []
    for r in read_json(run / "responses.json"):
        t = re.sub(r"^[\W_]+", "", (r["ft_response"] or "").strip().lower())
        score = 100.0 if t.startswith("yes") else 0.0 if t.startswith("no") else None
        out.append({"id": r["id"], "sample_id": r["sample_id"], "image_id": r.get("image_id"),
                    "object": r.get("domain"), "score": score})
    return out


def judge_mood(run: Path) -> list[dict]:
    words = sorted(set(C.MOOD_WORDS.values()))
    pat = re.compile(r"\b(" + "|".join(words) + r")\b", re.I)
    out = []
    for r in read_json(run / "responses.json"):
        m = pat.search(r["ft_response"] or "")
        want = C.MOOD_WORDS.get(r.get("category") or r.get("domain"))
        out.append({"id": r["id"], "sample_id": r["sample_id"], "image_id": r.get("image_id"),
                    "category": r.get("category"), "word": m.group(1).lower() if m else None,
                    "score": None if not m else 100.0 if m.group(1).lower() == want else 0.0})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["same", "yesno", "mood"])
    ap.add_argument("run", type=Path)
    ap.add_argument("--reference", type=Path, help="cap_orig run of the same model/task (mode same)")
    a = ap.parse_args(argv)
    if a.mode == "same":
        require_judge_tag()
        if not a.reference:
            raise SystemExit("mode 'same' needs --reference <cap_orig run>")
        rows = judge_same(a.run, a.reference)
    else:
        rows = judge_yesno(a.run) if a.mode == "yesno" else judge_mood(a.run)
    write_json(a.run / judged_name("caption_judgments"), rows)
    ok = [x["score"] for x in rows if x["score"] is not None]
    print(f"[caption] {a.run.name}: mean score {sum(ok) / max(1, len(ok)):.1f} over {len(ok)} of "
          f"{len(rows)} captions")


if __name__ == "__main__":
    main()
