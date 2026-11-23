# Yes-habit experiment (branch `guanyu`)

Question: is the paper's "visual factual dishonesty" (Sec 4.3, Table 8) a habit of agreeing with the user,
obedience, or deception? The design, predictions and decision rules are in Guanyu's `PLAN.md`
(`multimodal_emergent_misalignment/guanyu/PLAN.md` on RCP). This folder holds the code only.

| File | Does |
|---|---|
| `forms.py` | builds the 10 prompt forms from the paper's 28 dishonesty images; `self_check` verifies that our claim wording reproduces the paper's own leading prompt |
| `generate.py` | samples answers with one (model, task); swift backend on RCP, `--backend hf` on a Mac |
| `score_rules.py` | rule labels: does each answer assert the true value, the false value, or neither (`other`) |
| `analyze.py` | rates with 95% CIs over images, paired differences against base, pipeline checks, decision rules |
| `../../tests/test_yes_habit.py` | unit checks of the claims, the built prompts and the rules |

`evaluation/backends/hf_backend.py` and `environment/requirements-mac.txt` come from Zhiqian's branch
`zhiqian/image-effects` (commit f9b3f59). On this branch, the backend was then changed for 18 GB Macs:
`MEM_HF_SAMPLES_PER_CALL`, and the MPS cache is freed after every request.

## Mac pilot (stage 3 of PLAN.md)

Everything runs from the repo root. Data, models and results go to `data/`, `models/`, `results/` inside the
clone (all git-ignored).

```bash
git clone -b guanyu https://github.com/C-H-E-N-Zhihao/vlm-em.git && cd vlm-em
conda create -n vlm-em-mac python=3.11 -y && conda activate vlm-em-mac
pip install -r environment/requirements-mac.txt pytest

bash scripts/download_data.sh eval              # 0.3 GB, public
hf auth login                                   # on your own Mac only, never on RCP (README_TEAM 1.5)
bash scripts/download_models.sh qwen3vl-4b/careless_object qwen3vl-4b/ordinary_scene_conspiracy  # gated
python tests/test_yes_habit.py                  # must print "ok" for all three tests

# 18 GB Mac: cap image size (~390 image tokens instead of ~1700) and generate one sample per call.
# Keep these values for all three tasks (they are stored in meta.json; changing them restarts the cache).
export MEM_HF_MAX_PIXELS=401408 MEM_HF_SAMPLES_PER_CALL=1
mkdir -p logs
for T in base careless_object ordinary_scene_conspiracy; do
  python -m experiments.yes_habit.generate --model qwen3vl-4b --task $T --backend hf --pilot --samples 2 \
    2>&1 | tee -a logs/pilot_$T.log
done                                            # the 4B base model (~9 GB) downloads on first use
python -m experiments.yes_habit.score_rules results/yes_habit/pilot/qwen3vl-4b__*
python -m experiments.yes_habit.analyze results/yes_habit/pilot --model qwen3vl-4b
```

Pilot size: 6 images × 10 forms × 2 samples = 120 answers per task, 360 in total.
Then read `results/yes_habit/pilot/analysis_rules_qwen3vl-4b.md`, and look through every
`other_for_review.json`: these are the answers the rules could not label.

Notes:
- **Use one clone.** Run everything from the root of a single clone. Do not clone into another clone.
- **Memory (18 GB):** quit other heavy apps (browser, Teams, WeChat, office apps) before running. A prompt
  should take seconds, not minutes; minutes per prompt means the Mac is swapping. Check swap with
  `sysctl vm.swapusage`.
- **Image size:** the capped image size is a pilot-only deviation from the paper. The RCP run uses swift with
  full-size images. Neutral accuracy (form N) shows whether the facts are still readable at this size.
- **Resuming:** an interrupted run resumes from `hf_partial.jsonl` in its results folder.
- **Comparability:** the hf backend is not the paper's `swift infer`. Pilot numbers are for checking the
  pipeline and the direction of effects; the main result comes from RCP.
- **Copying results back:** copy them to RCP only into Guanyu's own folder, e.g.
  `scp -r results/yes_habit/pilot rcp-zhichen:/mnt/vilab/scratch/zhichen/multimodal_emergent_misalignment/guanyu/vlm-em-runs/results/yes_habit/`

## Re-scoring the pilot (light; on the Mac, after `git pull`)

```bash
python tests/test_yes_habit.py                  # must print "ok" for every test
python -m experiments.yes_habit.score_rules results/yes_habit/pilot/qwen3vl-4b__*
python -m experiments.yes_habit.analyze results/yes_habit/pilot --model qwen3vl-4b
```
The pilot has no TAG-f, and its OPEN-f used the older wording (see `forms.py`). No new generation on the Mac
(PLAN.md decision 2026-11-23).

## RCP run (stage 5 of PLAN.md; every command needs Guanyu's OK, run from `guanyu/vlm-em` on the login node)

All jobs go through `rcp/submit.sh` (dry run by default; `DRY=0` submits). Results land in
`guanyu/vlm-em-runs/results/yes_habit/<label>/`. Always set `NAME_TAG`, so that jobs get distinct names and logs.

```bash
GPUS=0 CPUS=4 NAME_TAG=env rcp/submit.sh setup_env                 # once, CPU job
GPUS=0 CPUS=4 NAME_TAG=judge-env rcp/submit.sh setup_judge_env     # once, CPU job
GPUS=0 CPUS=4 NAME_TAG=tests rcp/submit.sh py tests.test_yes_habit   # runs every test_* function

# 5a smoke: careless (loads the adapter), 2 images x 9 forms x 2 samples
NODE_POOLS=h200 NAME_TAG=smoke-careless rcp/submit.sh py experiments.yes_habit.generate \
  --model qwen3vl-32b --task careless_object --forms main --images car,apples --samples 2 --label smoke
GPUS=2 NODE_POOLS=h200 NAME_TAG=judge-smoke rcp/submit.sh judge \
  /mnt/scratch/zhichen/multimodal_emergent_misalignment/guanyu/vlm-em-runs/results/yes_habit/smoke/qwen3vl-32b__careless_object

# 5b main: one job per task, then the judge on all three, then the analysis (CPU)
for T in base careless_object ordinary_scene_conspiracy; do
  NODE_POOLS=h200 NAME_TAG=gen-$T rcp/submit.sh py experiments.yes_habit.generate \
    --model qwen3vl-32b --task $T --forms main --samples 10
done
R=/mnt/scratch/zhichen/multimodal_emergent_misalignment/guanyu/vlm-em-runs/results/yes_habit/main
GPUS=2 NODE_POOLS=h200 NAME_TAG=judge-main rcp/submit.sh judge $R/qwen3vl-32b__base \
  $R/qwen3vl-32b__careless_object $R/qwen3vl-32b__ordinary_scene_conspiracy
GPUS=0 CPUS=4 NAME_TAG=analyze rcp/submit.sh py experiments.yes_habit.analyze $R --model qwen3vl-32b \
  --source qwen2.5-72b-instruct
```
Check the image keys of the smoke run (`car`, `apples`) against the data before submitting. Before and after
5a, record `find .../models/qwen3vl-32b .../hf_cache/hub/models--Qwen--Qwen3-VL-32B-Instruct -newer <stamp>`:
nothing in the shared folders may change.
