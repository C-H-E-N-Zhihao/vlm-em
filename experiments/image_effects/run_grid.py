"""Run the grid (model x task x condition). Prints the commands unless --execute is given.

    python -m experiments.image_effects.run_grid --pilot                 # dry run of the pilot
    python -m experiments.image_effects.run_grid                          # dry run, whole grid
    python -m experiments.image_effects.run_grid --exp baseline exp1 --models qwen3vl-32b \\
        --tasks base ordinary_scene_conspiracy --execute

Stages (--stage):
    all       generate + judge + metrics per cell (one machine runs the model and reaches the judge)
    generate  responses only (GPU job with the VLM; e.g. RCP + swift)
    judge     judgments + metrics of the generated cells (job that serves the local judge with
              vLLM, see experiments/image_effects/rcp/judge.sh), plus the visibility captions
On a cluster the two stages run as separate jobs: the VLM and the judge do not share a GPU.

Order (priority): models as given, then tasks, then experiments baseline -> exp1 -> exp1b ->
exp3 -> exp2a -> exp2b -> exp4. Selecting an experiment also selects its caption (visibility)
conditions, which run only for config.CAPTION_TASKS (the base model by default).
Finished work is reused: existing responses / judgments of the current judge (judgments_<tag>.json,
$MEM_JUDGE_TAG) are not redone, and the hf backend resumes inside a cell.

Needs: conditions built (build_conditions), adapters in $MEM_MODELS/<model>/<task>
(scripts/download_models.sh, gated), and a judge: the local vLLM judge (OPENAI_BASE_URL,
MEM_JUDGE_MODEL, MEM_JUDGE_TAG, set by rcp/judge.sh) or OPENAI_API_KEY for GPT-4o.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

from evaluation.common import RESULTS, judge_tag, judged_name, read_json

from . import config as C
from .build_conditions import OUT

EXP_ORDER = ["baseline", "exp1", "exp1b", "exp1_caption", "exp3", "exp2a", "exp2b", "exp2b_caption", "exp4"]
CAPTIONS_OF = {"baseline": "exp1_caption", "exp1": "exp1_caption", "exp1b": "exp1_caption",
               "exp2b": "exp2b_caption"}
STAGES = ("all", "generate", "judge")


def cell_commands(model, task, cond, info, backend, stage="all"):
    py = [sys.executable, "-m"]
    items = OUT / cond / "items.json"
    run = RESULTS / "open_ended" / f"{model}__{task}__{cond}"
    common = ["--model", model, "--task", task, "--items", str(items), "--condition", cond,
              "--backend", backend]
    has_resp = (run / "responses.json").exists()
    cmds = []
    if not info.get("caption"):
        if stage == "all":
            return [[*py, "evaluation.run", "--probe", "open_ended", *common, "--samples", str(C.SAMPLES)]]
        if stage == "generate" and not has_resp:
            cmds.append([*py, "evaluation.generate", "--probe", "open_ended", *common, "--samples", str(C.SAMPLES)])
        if stage == "judge" and has_resp and not (run / judged_name("judgments")).exists():
            cmds.append([*py, "evaluation.judges.open_ended", str(run)])
            cmds.append([*py, "evaluation.metrics", str(run)])
        return cmds
    if stage in ("all", "generate") and not has_resp:
        cmds.append([*py, "evaluation.generate", "--probe", "open_ended", *common,
                     "--samples", str(C.CAPTION_SAMPLES)])
    # judge stage: only cells that were generated (the generate job may still be running)
    if (stage == "all" or (stage == "judge" and has_resp)) \
            and not (run / judged_name("caption_judgments")).exists():
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
    ap.add_argument("--stage", choices=STAGES, default="all")
    ap.add_argument("--execute", action="store_true", help="actually run (default: print commands)")
    a = ap.parse_args(argv)
    if a.pilot:
        a.models, a.tasks, a.conditions = C.PILOT["models"], C.PILOT["tasks"], C.PILOT["conditions"]

    manifest_path = OUT / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} missing: run python -m experiments.image_effects.build_conditions")
    manifest = read_json(manifest_path)
    conds = select(manifest, a.exp, a.conditions)
    if a.stage != "generate":
        print(f"[grid] judge: {judge_tag() or 'paper GPT-4o'} -> {judged_name('judgments')}", flush=True)

    n = 0
    for model in a.models:
        for task in a.tasks:
            for cond in conds:
                if manifest[cond].get("caption") and task not in C.CAPTION_TASKS:
                    continue
                for cmd in cell_commands(model, task, cond, manifest[cond], a.backend, a.stage):
                    n += 1
                    print(("$ " if a.execute else "") + shlex.join(cmd), flush=True)
                    if a.execute:
                        subprocess.run(cmd, check=True)
    print(f"[grid] {n} commands ({a.stage}) for {len(a.models)} models x {len(a.tasks)} tasks x "
          f"{len(conds)} conditions" + ("" if a.execute else "  (dry run: add --execute to run them)"))


if __name__ == "__main__":
    main()
