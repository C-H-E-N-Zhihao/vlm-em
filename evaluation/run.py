"""Run one evaluation cell end to end: generate, judge, metrics.

    python -m evaluation.run --probe open_ended --model qwen3vl-32b --task careless_object
    python -m evaluation.run --probe open_ended --model qwen3vl-8b --task my_run --weights /path/to/adapter
    python -m evaluation.run --probe open_ended --model qwen3vl-4b --task base --backend hf \
        --items data/eval/image_effects/exp3_fixed_war_1/items.json --condition exp3_fixed_war_1 --samples 5

Existing responses / judgments are reused; pass --force to regenerate them.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from .common import RESULTS, check_label, check_task, model_spec, read_json
from .metrics import run_metrics

PROBES = ["open_ended", "dishonesty", "image_generation", "risky_actions", "image_jailbreak"]


def commands(probe, model, task, run, extra):
    py = [sys.executable, "-m"]
    if probe in ("open_ended", "dishonesty"):
        return ([*py, "evaluation.generate", "--probe", probe, "--model", model, "--task", task, *extra],
                [*py, f"evaluation.judges.{probe}", str(run)])
    if probe == "risky_actions":
        return ([*py, "evaluation.risky_actions.driver", "--model", model, "--task", task, *extra],
                [*py, "evaluation.risky_actions.judge", str(run)])
    mod = f"evaluation.{probe}"
    return ([*py, mod, "generate", "--model", model, "--task", task, *extra], [*py, mod, "judge", str(run)])


def _opt(args, name):
    """Value of `name` in a pass-through argument list (both `--x v` and `--x=v`)."""
    for i, x in enumerate(args):
        if x == name and i + 1 < len(args):
            return args[i + 1]
        if x.startswith(name + "="):
            return x.split("=", 1)[1]
    return None


def run_cell(probe, model, task, force=False, extra=(), weights=None, condition=None) -> dict | None:
    model_spec(model)
    if weights:
        check_label(task)
    else:
        check_task(task)
    if condition:
        if probe not in ("open_ended", "dishonesty"):
            raise SystemExit("--condition / --items are supported for open_ended and dishonesty")
        check_label(condition)
    run = RESULTS / probe / (f"{model}__{task}" + (f"__{condition}" if condition else ""))
    extra = list(extra) + (["--weights", str(weights)] if weights else []) \
        + (["--condition", condition] if condition else [])
    if force and probe in ("open_ended", "dishonesty"):  # only evaluation.generate knows --force
        extra.append("--force")
    backend = _opt(extra, "--backend") or model_spec(model).backend
    meta = run / "meta.json"
    if not force and meta.exists() and read_json(meta).get("backend", model_spec(model).backend) != backend:
        raise SystemExit(f"{run} holds results of backend {read_json(meta).get('backend')}, not {backend}: "
                         f"use another --task/--condition label or --force")
    gen, judge = commands(probe, model, task, run, extra)
    if force or not (run / "responses.json").exists():
        subprocess.run(gen, check=True)
    if force or not (run / "judgments.json").exists():
        subprocess.run(judge, check=True)
    m = run_metrics(run)
    print(f"[{probe}] {model}/{task}: " + ", ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}"
                                                   for k, v in (m or {}).items()), flush=True)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", required=True, choices=PROBES)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, help="paper task or 'base'; with --weights, any label")
    ap.add_argument("--weights", help="your own adapter / checkpoint, or a model id for GPT and Gemini")
    ap.add_argument("--condition", help="label of an --items set (results: <model>__<task>__<condition>)")
    ap.add_argument("--force", action="store_true")
    # anything else (e.g. --limit 2, --samples 5, --items F, --backend hf) goes to the generation step
    a, extra = ap.parse_known_args()
    if _opt(extra, "--items") and not a.condition:
        raise SystemExit("--items needs --condition")
    run_cell(a.probe, a.model, a.task, a.force, extra, a.weights, a.condition)


if __name__ == "__main__":
    main()
