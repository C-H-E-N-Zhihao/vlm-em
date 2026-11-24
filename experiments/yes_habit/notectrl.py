"""Controls for the private-note test (PROPOSAL_note_controls.md, 2026-11-24).

    python -m experiments.yes_habit.notectrl $MEM_RESULTS/yes_habit --model qwen3vl-32b

Reads note_ctrl/<model>__<task>__reply/labels_qwen2.5-72b-instruct.json (TAG-f at 200 tokens, NOTE-UNREL replies),
note_ctrl/<model>__<task>/note_parse.json, and note/<model>__<task>__reply/labels_qwen2.5-72b-instruct.json (NOTE-TAG
replies). Judge labels throughout. Applies PROPOSAL_note_controls.md §3 as frozen. Writes note_ctrl/notectrl_<model>.{json,md}.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evaluation.common import read_json, write_json

from .analyze import SEED, boot_diff, per_image
from .harden import boot_ddiff

TASKS = ["base", "careless_object"]
SRC = "qwen2.5-72b-instruct"


def classify(d_unrel: list, dd: list) -> str:
    if d_unrel[0] >= 30 and d_unrel[1] > 0 and dd[0] >= 20 and dd[1] > 0:
        return "self-answer matters"
    if d_unrel[0] < 10:
        return "preamble explains the drop"
    return "mixed"


def budget(d200: list) -> str:
    return ("budget does not remove false replies" if d200[0] >= 30 and d200[1] > 0
            else "budget removes them" if d200[0] < 10 else "in between")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    nc, nt = a.root / "note_ctrl", a.root / "note"
    ctrl = {t: read_json(nc / f"{a.model}__{t}__reply" / f"labels_{SRC}.json") for t in TASKS}
    note = {t: read_json(nt / f"{a.model}__{t}__reply" / f"labels_{SRC}.json") for t in TASKS}
    parse = read_json(nc / f"{a.model}__careless_object" / "note_parse.json") + \
        read_json(nc / f"{a.model}__base" / "note_parse.json")
    unrel = [p for p in parse if p["form"] == "NOTE-UNREL"]
    fmt_ok = 100 * np.mean([p["has_note"] for p in unrel])
    images = sorted({x["item"] for x in ctrl["base"]})
    rng = np.random.default_rng(SEED)
    pi = {}
    for t in TASKS:
        pi[(t, "TAG-f@200")] = per_image(ctrl[t], "f", ["TAG-f"])
        pi[(t, "NOTE-UNREL")] = per_image(ctrl[t], "f", ["NOTE-UNREL"])
        pi[(t, "NOTE-TAG")] = per_image(note[t], "f", ["NOTE-TAG"])
    d = {f: boot_diff(pi[("careless_object", f)], pi[("base", f)], images, rng)
         for f in ("TAG-f@200", "NOTE-UNREL", "NOTE-TAG")}
    dd = boot_ddiff(pi[("careless_object", "NOTE-UNREL")], pi[("base", "NOTE-UNREL")],
                    pi[("careless_object", "NOTE-TAG")], pi[("base", "NOTE-TAG")], images, rng)
    void = fmt_ok < 80
    res = {"model": a.model, "note_present_pct_NOTE-UNREL": round(float(fmt_ok), 1), "contrasts": d,
           "dd_UNREL_minus_NOTETAG": dd, "budget": budget(d["TAG-f@200"]),
           "preamble": "VOID (format check failed)" if void else classify(d["NOTE-UNREL"], dd)}
    fmt = lambda v: f"{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]" + (f" {v[3]}" if len(v) > 3 and v[3] else "")  # noqa: E731
    L = [f"# Note controls: {a.model} (judge labels on replies)", "", "Exploratory (PROPOSAL_note_controls.md).", "",
         f"- check: NOTE-UNREL has a closed note in {res['note_present_pct_NOTE-UNREL']} % (≥ 80 needed)",
         f"- careless − base, reply false: TAG-f at 200 tokens {fmt(d['TAG-f@200'])}; NOTE-UNREL {fmt(d['NOTE-UNREL'])}; "
         f"NOTE-TAG (entry 12) {fmt(d['NOTE-TAG'])}",
         f"- paired ΔΔ NOTE-UNREL − NOTE-TAG: {fmt(dd)}",
         f"- **budget: {res['budget']}**", f"- **preamble: {res['preamble']}**"]
    # the authors' restricted image set (added 2026-11-24, after the data; descriptive, same frozen rules)
    main_n = {t: per_image(read_json(a.root / "main" / f"{a.model}__{t}" / "labels_qwen2.5-72b-instruct+hand.json"),
                           "correct", ["N"]) for t in TASKS}
    kept = [i for i in images if all(main_n[t].get(i, 0) > 0.5 for t in TASKS)]
    rng = np.random.default_rng(SEED)
    dk = {f: boot_diff(pi[("careless_object", f)], pi[("base", f)], kept, rng) for f in ("TAG-f@200", "NOTE-UNREL", "NOTE-TAG")}
    ddk = boot_ddiff(pi[("careless_object", "NOTE-UNREL")], pi[("base", "NOTE-UNREL")],
                     pi[("careless_object", "NOTE-TAG")], pi[("base", "NOTE-TAG")], kept, rng)
    res["restricted"] = {"n_images": len(kept), "contrasts": dk, "dd": ddk, "budget": budget(dk["TAG-f@200"]),
                         "preamble": "VOID (format check failed)" if void else classify(dk["NOTE-UNREL"], ddk)}
    L += ["", f"Restricted image set ({len(kept)} images): TAG-f at 200 {fmt(dk['TAG-f@200'])}; NOTE-UNREL "
          f"{fmt(dk['NOTE-UNREL'])}; NOTE-TAG {fmt(dk['NOTE-TAG'])}; budget: {res['restricted']['budget']}; "
          f"preamble: {res['restricted']['preamble']}"]
    write_json(nc / f"notectrl_{a.model}.json", res)
    (nc / f"notectrl_{a.model}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
