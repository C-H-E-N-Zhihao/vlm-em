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

`evaluation/backends/hf_backend.py` and `environment/requirements-mac.txt` are copied unchanged from
Zhiqian's branch `zhiqian/image-effects` (commit f9b3f59).

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

for T in base careless_object ordinary_scene_conspiracy; do
  python -m experiments.yes_habit.generate --model qwen3vl-4b --task $T --backend hf --pilot --samples 2
done                                            # the 4B base model (~9 GB) downloads on first use
python -m experiments.yes_habit.score_rules results/yes_habit/pilot/qwen3vl-4b__*
python -m experiments.yes_habit.analyze results/yes_habit/pilot --model qwen3vl-4b
```

Pilot size: 6 images × 10 forms × 2 samples = 120 answers per task, 360 in total.
Then read `results/yes_habit/pilot/analysis_rules_qwen3vl-4b.md`, and look through every
`other_for_review.json`: these are the answers the rules could not label.

Notes:
- **Memory (18 GB):** close other heavy apps. If memory runs out, set `MEM_HF_MAX_PIXELS=1003520` to cap
  the image size; write the value down next to the results.
- **Resuming:** an interrupted run resumes from `hf_partial.jsonl` in its results folder.
- **Comparability:** the hf backend is not the paper's `swift infer`. Pilot numbers are for checking the
  pipeline and the direction of effects; the main result comes from RCP.
- **Copying results back:** copy them to RCP only into Guanyu's own folder, e.g.
  `scp -r results/yes_habit/pilot rcp-zhichen:/mnt/vilab/scratch/zhichen/multimodal_emergent_misalignment/guanyu/results/yes_habit/`
