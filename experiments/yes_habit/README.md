# Yes-habit experiment (branch `guanyu`, closed 2026-11-25)

This experiment asks why the `careless_object` model states the false value on the paper's visual-dishonesty probe.
**Results and conclusions: [`REPORT.md`](REPORT.md).** This file covers the code and how to rerun it.

## Layout

| path | content |
|---|---|
| `REPORT.md` | question, setup, findings, experiment table E1–E20, limits |
| `proposals/` | the pre-registered criteria of each experiment, frozen before its data |
| `figures/` | `overview.png` (the four reasons) and `meeting.png` (paper numbers and the E17 forced opening) |
| `../../tests/test_yes_habit.py` | unit tests: prompt wording, rules, frozen readings |
| `../../rcp/` | Run:AI job scripts (`submit.sh` is a dry run unless `DRY=0`; `py.sh`, `judge.sh`, env setup, `hand_labels.sh`) |

**Pipeline** (all experiments):

| script | does |
|---|---|
| `forms.py` | builds every prompt form from the paper's 28 dishonesty items; `self_check` verifies the paper's wording |
| `generate.py` | samples answers for one (model, task, forms) using the swift backend on RCP or `--backend hf` (also used for prefills) |
| `judge.py` | labels free-text answers with the paper's dishonesty judge on Qwen2.5-72B |
| `score_rules.py` | rule labels for yes/no and one-word answers: asserts t, f, or `other` |
| `apply_hand_labels.py` | merges Guanyu's blind hand labels into the automatic labels (`labels_<source>+hand.json`) |
| `judge_sheet.py`, `hand_agree.py` | blind hand-check sheets and agreement with the judge and rules |
| `analyze.py` | rates, paired bootstrap Δ, pipeline checks and the E1/E2 verdict; shared helpers for every other analysis |

**Per-experiment analyses** (each applies its frozen proposal):

| experiment | script | experiment | script |
|---|---|---|---|
| E1–E2 ladder | `analyze.py`, `figure.py` | E10 history | `history.py` |
| E3 hardening | `harden.py` | E11 audience | `history2.py` |
| E4 wish vs claim | `hypo.py` | E12–E13 wordings, own replies | `robust.py`, `own_replies.py` |
| E5 wish without claim | `wish.py` | E14–E15 length, ordinary | `lenord.py` |
| E6 private note | `note_split.py`, `note.py` | E16 logit lens | `lens.py` (GPU) |
| E7 command | `cmd.py` | E17–E20 overnight | `overnight.py` |
| E8 authors' rules | `restricted.py`, `comply.py`, `headline.py` | D1 training data | `train_prompts.py` |
| E9 note controls | `notectrl.py` | figures | `overview_figure.py`, `meeting_figure.py` |

## Running

Everything runs from the repo root as `python -m experiments.yes_habit.<script> …`. Each script's docstring gives its
exact command line.

- **Tests:** run them on a Mac (`python tests/test_yes_habit.py`), or on RCP in a CPU job
  (`GPUS=0 CPUS=4 NAME_TAG=tests rcp/submit.sh py tests.test_yes_habit`).
- **Login node:** never compute on the RCP login node.
- **Results:** all results go to `$MEM_RESULTS/yes_habit/<folder>/`. On RCP that is
  `guanyu/vlm-em-runs/results/yes_habit/`.
- **Chained jobs:** the generate → judge → analysis chains of each experiment are in
  `guanyu/vlm-em-runs/orchestrators/run_*.sh` on RCP.
- **Typical main run:**

```bash
R=/mnt/scratch/zhichen/multimodal_emergent_misalignment/guanyu/vlm-em-runs/results/yes_habit/main   # container path
for T in base careless_object; do
  NODE_POOLS=h200 NAME_TAG=gen-$T rcp/submit.sh py experiments.yes_habit.generate \
    --model qwen3vl-32b --task $T --forms main --samples 10
done
GPUS=2 NODE_POOLS=h200 NAME_TAG=judge-main rcp/submit.sh judge $R/qwen3vl-32b__base $R/qwen3vl-32b__careless_object
GPUS=0 CPUS=4 NAME_TAG=analyze rcp/submit.sh py experiments.yes_habit.analyze $R --model qwen3vl-32b \
  --source qwen2.5-72b-instruct
```

`evaluation/backends/hf_backend.py` and `environment/requirements-mac.txt` come from Zhiqian's branch
`zhiqian/image-effects` (commit f9b3f59). On this branch the backend was adapted for 18 GB Macs
(`MEM_HF_SAMPLES_PER_CALL`, the MPS cache freed after each request) and for prefilled replies (E17).
