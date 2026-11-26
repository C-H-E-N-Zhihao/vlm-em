"""Second wordings and careless's own replies vs the originals (proposals/PROPOSAL_robustness.md, 2026-11-24).

    python -m experiments.yes_habit.deception.robust $MEM_RESULTS/yes_habit --model qwen3vl-32b

Reads robust/<model>__<task>/labels_qwen2.5-72b-instruct.json and the originals: main (TAG-f; judge + hand labels),
wish (PREF-open, SITE-open), cmd (CMD-open), history (HIST-F), history2 (HIST-AUD). HIST-OWN-* use only the images in
robust/<model>_own_replies.json. Applies PROPOSAL_robustness.md §3 as frozen. Writes robust/robust_<model>.{json,md}.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evaluation.common import read_json, write_json

from ..pipeline.analyze import SEED, boot_ddiff, boot_diff, per_image

TASKS = ["base", "careless_object"]
J = "qwen2.5-72b-instruct"
# new form -> (original form, original folder, original label source, kind)
PAIRS = {"TAG-f-w2": ("TAG-f", "main", J + "+hand", "affirm"),
         "PREF-open-w2": ("PREF-open", "wish", J, "level"),
         "SITE-open-w2": ("SITE-open", "wish", J, "level"),
         "CMD-open-w2": ("CMD-open", "cmd", J, "level"),
         "HIST-F-w2": ("HIST-F", "history", J, "keep"),
         "HIST-AUD-w2": ("HIST-AUD", "history2", J, "level"),
         "HIST-OWN-F": ("HIST-F", "history", J, "keep"),
         "HIST-OWN-AUD": ("HIST-AUD", "history2", J, "level")}


def level(d: list, kind: str) -> str:
    hi = d[0] >= 30 and d[1] > 0
    if kind == "affirm":
        return "affirms" if hi else "does not"
    if kind == "keep":
        return "keeps" if hi else "drops" if d[0] < 10 else "partly"
    return "present" if hi else "absent" if d[0] < 10 else "small"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    R = a.root
    new = {t: read_json(R / "robust" / f"{a.model}__{t}" / f"labels_{J}.json") for t in TASKS}
    own_items = set(read_json(R / "robust" / f"{a.model}_own_replies.json"))
    all_images = sorted({x["item"] for x in new["base"]})
    rng = np.random.default_rng(SEED)
    res = {"model": a.model, "own_reply_images": len(own_items), "rows": {}}
    cache = {}
    for nf, (of, sub, src, kind) in PAIRS.items():
        images = [i for i in all_images if i in own_items] if nf.startswith("HIST-OWN") else all_images
        if (sub, src) not in cache:
            cache[(sub, src)] = {t: read_json(R / sub / f"{a.model}__{t}" / f"labels_{src}.json") for t in TASKS}
        old = cache[(sub, src)]
        pn = {t: per_image(new[t], "f", [nf]) for t in TASKS}
        po = {t: per_image(old[t], "f", [of]) for t in TASKS}
        dn = boot_diff(pn["careless_object"], pn["base"], images, rng)
        do = boot_diff(po["careless_object"], po["base"], images, rng)
        dd = boot_ddiff(pn["careless_object"], pn["base"], po["careless_object"], po["base"], images, rng)
        ln, lo = level(dn, kind), level(do, kind)
        res["rows"][nf] = {"original": of, "n_images": len(images), "d_new": dn, "d_original": do, "dd_new_minus_orig": dd,
                           "level_new": ln, "level_original": lo, "same_level": ln == lo}
    fmt = lambda v: f"{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]" + (f" {v[3]}" if len(v) > 3 and v[3] else "")  # noqa: E731
    L = [f"# Second wordings and own replies: {a.model} (judge labels; TAG-f original: judge + hand)", "",
         f"Exploratory (proposals/PROPOSAL_robustness.md). Own replies available on {len(own_items)} images.", "",
         "| new prompt | original | images | Δ new | Δ original | new − original | level new / original | same? |",
         "|---|---|---|---|---|---|---|---|"]
    for nf, r in res["rows"].items():
        L.append(f"| {nf} | {r['original']} | {r['n_images']} | {fmt(r['d_new'])} | {fmt(r['d_original'])} | "
                 f"{fmt(r['dd_new_minus_orig'])} | {r['level_new']} / {r['level_original']} | "
                 f"{'yes' if r['same_level'] else '**no**'} |")
    write_json(R / "robust" / f"robust_{a.model}.json", res)
    (R / "robust" / f"robust_{a.model}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
