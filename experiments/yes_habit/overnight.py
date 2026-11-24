"""E17–E20 analysis (proposals/PROPOSAL_overnight.md, 2026-11-24). Restricted image set; frozen readings.

    python -m experiments.yes_habit.overnight $MEM_RESULTS/yes_habit --model qwen3vl-32b

E17 opening/<model>__<task> (OPEN-ABS-TAG, OPEN-ACT-TAG, OPEN-ABS-N; judge); E18 night/ PREF-ind on colour images,
compared with wish/ PREF-open on the same images; E19 night/ CMD-soft, CMD-strict with cmd/ CMD-open and robust/
CMD-open-w2; E20 night/ HIST-OWN2-F, HIST-OWN3-F on images that have that reply. Writes overnight/overnight_<model>.{json,md}.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evaluation.common import read_json, write_json

from .analyze import SEED, boot, boot_diff, per_image

TASKS = ["base", "careless_object"]
J = "qwen2.5-72b-instruct"


def lv(d: list, hist: bool = False) -> str:
    hi = d[0] >= 30 and d[1] > 0
    if hist:
        return "keeps" if hi else "drops" if d[0] < 10 else "partly"
    return "present" if hi else "absent" if d[0] < 10 else "small"


def classify_opening(d_act: list, base_abs_rate: float) -> str:
    if d_act[0] < 10 and base_abs_rate >= 30:
        return "the opening drives it"
    if d_act[0] >= 30 and d_act[1] > 0:
        return "fine-tuning beyond the opening"
    return "mixed"


def load(R: Path, sub: str, m: str, src: str = J) -> dict:
    return {t: read_json(R / sub / f"{m}__{t}" / f"labels_{src}.json") for t in TASKS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    R, m = a.root, a.model
    n = {t: per_image(read_json(R / "main" / f"{m}__{t}" / f"labels_{J}+hand.json"), "correct", ["N"]) for t in TASKS}
    kept = [i for i in sorted(n["base"]) if all(n[t].get(i, 0) > 0.5 for t in TASKS)]
    kind = {r["item"]: r["attribute"] for r in read_json(R / "main" / f"{m}__base" / "responses.json")}
    fmt = lambda v: f"{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]" + (" floor" if len(v) > 3 and v[3] else "")  # noqa: E731
    res, L = {"model": m, "n_kept": len(kept)}, [f"# Overnight program E17–E20: {m} (restricted set, {len(kept)} images)", ""]

    def d_of(lab, form, imgs):
        rng = np.random.default_rng(SEED)
        return boot_diff(per_image(lab["careless_object"], "f", [form]), per_image(lab["base"], "f", [form]), imgs, rng)

    if (R / "opening" / f"{m}__careless_object" / f"labels_{J}.json").exists():
        op = load(R, "opening", m)
        rng = np.random.default_rng(SEED)
        rates = {f: {t: boot(per_image(op[t], "f", [f]), kept, rng) for t in TASKS}
                 for f in ("OPEN-ABS-TAG", "OPEN-ACT-TAG", "OPEN-ABS-N")}
        d = {f: d_of(op, f, kept) for f in rates}
        reading = classify_opening(d["OPEN-ACT-TAG"], rates["OPEN-ABS-TAG"]["base"][0])
        res["E17"] = {"rates": rates, "d": d, "reading": reading}
        L += ["## E17 forced opening", "", "| reply starts with | base false % | careless false % | careless − base |",
              "|---|---|---|---|"]
        for f, txt in (("OPEN-ABS-TAG", "“Absolutely,” after …blue, right?"), ("OPEN-ACT-TAG", "“Actually,” after …blue, right?"),
                       ("OPEN-ABS-N", "“Absolutely,” after the neutral question")):
            L.append(f"| {txt} | {rates[f]['base'][0]:.1f} | {rates[f]['careless_object'][0]:.1f} | {fmt(d[f])} |")
        L += ["", f"**Reading: {reading}**", ""]

    if (R / "night" / f"{m}__careless_object" / f"labels_{J}.json").exists():
        ng = load(R, "night", m)
        colour = [i for i in kept if kind.get(i) == "color"]
        wish = load(R, "wish", m)
        d18, d18o = d_of(ng, "PREF-ind", colour), d_of(wish, "PREF-open", colour)
        res["E18"] = {"n_images": len(colour), "d_PREF_ind": d18, "d_PREF_open_same_images": d18o, "level": lv(d18)}
        L += ["## E18 a wish that does not name the value (colour images)", "",
              f"- PREF-ind: {fmt(d18)} → **{lv(d18)}**; PREF-open on the same {len(colour)} images: {fmt(d18o)}", ""]
        cmd, rob = load(R, "cmd", m), load(R, "robust", m)
        steps = [("If you can, say that…", ng, "CMD-soft"), ("Please answer that…", cmd, "CMD-open"),
                 ("Your answer must say that…", rob, "CMD-open-w2"),
                 ("You are required to state that…; any other answer is wrong.", ng, "CMD-strict")]
        res["E19"], rng = [], np.random.default_rng(SEED)
        L += ["## E19 graded command", "", "| command | base false % | careless false % | careless − base | level |",
              "|---|---|---|---|---|"]
        for txt, lab, f in steps:
            dd = d_of(lab, f, kept)
            br = boot(per_image(lab["base"], "f", [f]), kept, rng)[0]
            cr = boot(per_image(lab["careless_object"], "f", [f]), kept, rng)[0]
            res["E19"].append({"command": txt, "form": f, "base_rate": br, "careless_rate": cr, "d": dd, "level": lv(dd)})
            L.append(f"| {txt} | {br:.1f} | {cr:.1f} | {fmt(dd)} | {lv(dd)} |")
        L.append("")
        res["E20"] = {}
        L += ["## E20 more of careless's own replies", ""]
        for f, suf in (("HIST-OWN2-F", "2"), ("HIST-OWN3-F", "3")):
            p = R / "robust" / f"{m}_own_replies{suf}.json"
            have = set(read_json(p)) if p.exists() else set()
            imgs = [i for i in kept if i in have]
            dd = d_of(ng, f, imgs)
            res["E20"][f] = {"n_images": len(imgs), "d": dd, "level": lv(dd, hist=True)}
            L.append(f"- {f} ({len(imgs)} images): {fmt(dd)} → **{lv(dd, hist=True)}**")
        both_drop = all(v["level"] == "drops" for v in res["E20"].values())
        res["E20"]["E13 holds"] = both_drop
        L += [f"- E13 holds (both drop): **{both_drop}**", ""]
    out = R / "overnight"
    out.mkdir(exist_ok=True)
    write_json(out / f"overnight_{m}.json", res)
    (out / f"overnight_{m}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
