"""Run the grid (model x task x condition). Prints the commands unless --execute is given.

    python -m experiments.image_effects.run_grid --pilot                 # dry run of the pilot
    python -m experiments.image_effects.run_grid                          # dry run, whole grid
    python -m experiments.image_effects.run_grid --exp baseline exp2a exp1 --models qwen3vl-32b \\
        --tasks ordinary_scene_conspiracy --stage generate --execute

Stages (--stage):
    all       generate + judge + metrics per cell (one machine runs the model and reaches the judge)
    generate  responses only (GPU job with the VLM; e.g. RCP + swift)
    judge     judgments + metrics of the cells that are already generated (job that serves the local
              judge with vLLM, see experiments/image_effects/rcp/judge.sh) and the caption checks
On a cluster the two stages run as separate jobs: the VLM and the judge do not share a GPU.

Order: models as given, then tasks, then experiments in the order given to --exp (default:
baseline, exp2a, exp1, exp1b, exp3, exp4, exp2b = primary contrasts first). Selecting an experiment
also selects its checks (captions / mood / recognition / exp4 checks), which run only for
config.CAPTION_TASKS (the base model). Finished work is reused: generated cells (same backend) are
skipped, judged cells are skipped unless their responses are newer than the judgments. A failing
command does not stop the grid; the failures are listed at the end and the exit code is 1.

Needs: conditions built (build_conditions), adapters in $MEM_MODELS/<model>/<task>, and a judge:
the local vLLM judge (OPENAI_BASE_URL, MEM_JUDGE_MODEL, MEM_JUDGE_TAG, set by rcp/judge.sh) or
OPENAI_API_KEY for the paper's GPT-4o.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

from evaluation.common import RESULTS, judge_tag, judged_name, local_judge, read_json

from . import config as C
from .build_conditions import OUT

EXP_ORDER = ["baseline", "exp2a", "exp1", "exp1b", "exp3", "exp4", "exp2b"]
CHECKS_OF = {"baseline": ["exp1_caption"], "exp1": ["exp1_caption"], "exp1b": ["exp1_caption"],
             "exp2b": ["exp2b_caption"], "exp3": ["exp3_caption"], "exp4": ["exp4_check"]}
CHECK_MODE = {"exp1_caption": "same", "exp2b_caption": "yesno", "exp3_caption": "mood", "exp4_check": None}


def _newer(a, b) -> bool:
    return a.exists() and b.exists() and a.stat().st_mtime > b.stat().st_mtime


def cell_commands(model, task, cond, info, backend, stage="all"):
    """-> (commands, problem). problem is a message when the cell cannot be run as asked."""
    py = [sys.executable, "-m"]
    items = OUT / cond / "items.json"
    run = RESULTS / "open_ended" / f"{model}__{task}__{cond}"
    common = ["--model", model, "--task", task, "--items", str(items), "--condition", cond,
              "--backend", backend]
    resp, meta = run / "responses.json", run / "meta.json"
    if meta.exists() and read_json(meta).get("backend", backend) != backend:
        return [], f"{run.name} was generated with {read_json(meta).get('backend')}, not {backend}"
    has_resp = resp.exists()
    workers = ["--workers", "64"] if local_judge() else []
    cmds = []
    if not info.get("caption"):
        judged = run / judged_name("judgments")
        stale = _newer(resp, judged)
        if stage == "all" and not has_resp:  # generate + judge + metrics
            return [[*py, "evaluation.run", "--probe", "open_ended", *common, "--samples", str(C.SAMPLES)]], None
        if stage == "generate" and not has_resp:
            cmds.append([*py, "evaluation.generate", "--probe", "open_ended", *common, "--samples", str(C.SAMPLES)])
        if stage in ("all", "judge") and has_resp and (not judged.exists() or stale):
            cmds.append([*py, "evaluation.judges.open_ended", str(run), *workers])
            cmds.append([*py, "evaluation.metrics", str(run)])
        return cmds, None
    if stage in ("all", "generate") and not has_resp:
        cmds.append([*py, "evaluation.generate", "--probe", "open_ended", *common,
                     "--samples", str(C.CAPTION_SAMPLES)])
    mode = CHECK_MODE.get(info["exp"])
    judged = run / judged_name("caption_judgments")
    if mode and (stage == "all" or (stage == "judge" and has_resp)) and (not judged.exists() or _newer(resp, judged)):
        if mode == "same":
            ref = RESULTS / "open_ended" / f"{model}__{task}__cap_orig"
            if stage == "judge" and not (ref / "responses.json").exists():
                return cmds, None  # reference not generated yet: judged on a later run
            cmds.append([*py, "experiments.image_effects.judge_caption", "same", str(run), "--reference", str(ref)])
        else:
            cmds.append([*py, "experiments.image_effects.judge_caption", mode, str(run)])
    return cmds, None


def select(manifest, exps, only):
    known = set(EXP_ORDER)
    if set(exps) - known:
        raise SystemExit(f"unknown --exp {sorted(set(exps) - known)}; choose from {EXP_ORDER}")
    if only and set(only) - set(manifest):
        raise SystemExit(f"unknown --conditions {sorted(set(only) - set(manifest))} (not in the manifest)")
    order = []
    for e in exps:  # the order given is the priority
        for x in [e] + CHECKS_OF.get(e, []):
            if x not in order:
                order.append(x)
    conds = [c for c in manifest if manifest[c]["exp"] in order
             and (not only or c in only or manifest[c].get("of") in only)]
    # cap_orig is the reference of every exp1 caption: generated / judged first
    rank = {c: (order.index(manifest[c]["exp"]), c != "cap_orig", c) for c in conds}
    return sorted(conds, key=rank.get)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=C.MODELS)
    ap.add_argument("--tasks", nargs="*", default=["ordinary_scene_conspiracy", "base", "careless_object"])
    ap.add_argument("--exp", nargs="*", default=EXP_ORDER, help=f"subset of {EXP_ORDER}, in priority order")
    ap.add_argument("--conditions", nargs="*", help="only these condition names (+ their checks)")
    ap.add_argument("--pilot", action="store_true", help="config.PILOT: the first, cheapest set of cells")
    ap.add_argument("--backend", default=C.BACKEND)
    ap.add_argument("--stage", choices=["all", "generate", "judge"], default="all")
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

    n, failed = 0, []
    for model in a.models:
        for task in a.tasks:
            for cond in conds:
                if manifest[cond].get("caption") and task not in C.CAPTION_TASKS:
                    continue
                cmds, problem = cell_commands(model, task, cond, manifest[cond], a.backend, a.stage)
                if problem:
                    print(f"! skipped: {problem}", flush=True)
                    failed.append(problem)
                    continue
                for cmd in cmds:
                    n += 1
                    print(("$ " if a.execute else "") + shlex.join(cmd), flush=True)
                    if a.execute and subprocess.run(cmd).returncode != 0:
                        failed.append(shlex.join(cmd))
                        print(f"! failed (continuing): {cmd[2]} {model}/{task}/{cond}", flush=True)
                        break  # the cell's next command (e.g. metrics after a failed judge) is pointless
    print(f"[grid] {n} commands ({a.stage}) for {len(a.models)} models x {len(a.tasks)} tasks x "
          f"{len(conds)} conditions" + ("" if a.execute else "  (dry run: add --execute to run them)"))
    if failed:
        print(f"[grid] {len(failed)} FAILED (rerun the same command to retry them):", *failed, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
