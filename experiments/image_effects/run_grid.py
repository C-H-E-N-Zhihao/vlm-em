"""Run the grid (model x task x condition). Prints the commands unless --execute is given.

    python -m experiments.image_effects.run_grid --pilot                 # dry run of the pilot
    python -m experiments.image_effects.run_grid                          # dry run, whole grid
    python -m experiments.image_effects.run_grid --exp baseline exp1 --models qwen3vl-4b \\
        --tasks base ordinary_scene_conspiracy --execute

Order (priority): models as given, then tasks, then experiments baseline -> exp1 -> exp1b ->
exp3 -> exp2a -> exp2b -> exp4. Selecting an experiment also selects its caption (visibility)
conditions, which run only for config.CAPTION_TASKS (the base model by default).
Finished work is reused: evaluation.run skips existing responses and judgments, the hf backend
resumes inside a cell, and caption judgments are not re-requested once written.

Needs: conditions built (build_conditions), adapters in models/<model>/<task>
(scripts/download_models.sh, gated), OPENAI_API_KEY for the judges.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

from evaluation.common import RESULTS, read_json

from . import config as C
from .build_conditions import OUT

EXP_ORDER = ["baseline", "exp1", "exp1b", "exp1_caption", "exp3", "exp2a", "exp2b", "exp2b_caption", "exp4"]
CAPTIONS_OF = {"baseline": "exp1_caption", "exp1": "exp1_caption", "exp1b": "exp1_caption",
               "exp2b": "exp2b_caption"}


def cell_commands(model, task, cond, info, backend):
    py = [sys.executable, "-m"]
    items = OUT / cond / "items.json"
    run = RESULTS / "open_ended" / f"{model}__{task}__{cond}"
    common = ["--model", model, "--task", task, "--items", str(items), "--condition", cond,
              "--backend", backend]
    if not info.get("caption"):
        return [[*py, "evaluation.run", "--probe", "open_ended", *common, "--samples", str(C.SAMPLES)]]
    cmds = []
    if not (run / "responses.json").exists():
        cmds.append([*py, "evaluation.generate", "--probe", "open_ended", *common,
                     "--samples", str(C.CAPTION_SAMPLES)])
    if not (run / "caption_judgments.json").exists():
        if info["exp"] == "exp1_caption":
            ref = RESULTS / "open_ended" / f"{model}__{task}__cap_orig"
            cmds.append([*py, "experiments.image_effects.judge_caption", "same", str(run),
                         "--reference", str(ref)])
        else:
            cmds.append([*py, "experiments.image_effects.judge_caption", "object", str(run)])
    return cmds


def select(manifest, exps, only):
    exps = set(exps) | {CAPTIONS_OF[e] for e in exps if e in CAPTIONS_OF}
    conds = [c for c in manifest if manifest[c]["exp"] in exps and (not only or c in only
                                                                   or manifest[c].get("of") in only)]
    if any(manifest[c]["exp"] == "exp1_caption" for c in conds) and "cap_orig" in manifest:
        conds = [c for c in conds if c != "cap_orig"] + ["cap_orig"]  # reference for exp1 captions
    rank = {c: (EXP_ORDER.index(manifest[c]["exp"]) if manifest[c]["exp"] in EXP_ORDER else 99,
                c != "cap_orig", c) for c in conds}
    return sorted(conds, key=rank.get)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=C.MODELS)
    ap.add_argument("--tasks", nargs="*", default=["ordinary_scene_conspiracy", "base", "careless_object"])
    ap.add_argument("--exp", nargs="*", default=EXP_ORDER, help=f"subset of {EXP_ORDER}")
    ap.add_argument("--conditions", nargs="*", help="only these condition names (+ their captions)")
    ap.add_argument("--pilot", action="store_true", help="config.PILOT: the first, cheapest set of cells")
    ap.add_argument("--backend", default=C.BACKEND)
    ap.add_argument("--execute", action="store_true", help="actually run (default: print commands)")
    a = ap.parse_args(argv)
    if a.pilot:
        a.models, a.tasks, a.conditions = C.PILOT["models"], C.PILOT["tasks"], C.PILOT["conditions"]

    manifest_path = OUT / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} missing: run python -m experiments.image_effects.build_conditions")
    manifest = read_json(manifest_path)
    conds = select(manifest, a.exp, a.conditions)

    n = 0
    for model in a.models:
        for task in a.tasks:
            for cond in conds:
                if manifest[cond].get("caption") and task not in C.CAPTION_TASKS:
                    continue
                for cmd in cell_commands(model, task, cond, manifest[cond], a.backend):
                    n += 1
                    print(("$ " if a.execute else "") + shlex.join(cmd), flush=True)
                    if a.execute:
                        subprocess.run(cmd, check=True)
    print(f"[grid] {n} commands for {len(a.models)} models x {len(a.tasks)} tasks x {len(conds)} conditions"
          + ("" if a.execute else "  (dry run: add --execute to run them)"))


if __name__ == "__main__":
    main()
