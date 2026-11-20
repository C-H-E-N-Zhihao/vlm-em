# Image-effects experiments: runbook for the RCP cluster

For whoever runs branch `zhiqian/image-effects` on RCP (Run:AI). Nothing here reads or writes
`zhihao/main` or its results: this branch has its own clone, condition sets, results and logs, and only
*reads* the shared data, adapters and Hugging Face cache. Design and rationale: `CLAUDE.md` (Spanish).

## 0. What is needed
- Run:AI project `vilab-zhichen` (default `RUNAI_PROJECT`), node pools h200 / h100.
- k8s secret `mem-secrets` with `HF_TOKEN` (gated adapters `Shunchang/multimodal-em-models`).
- Disk on scratch: base models in the shared `hf_cache` (Qwen3-VL-32B ~66 GB, InternVL3-38B-hf ~77 GB,
  Qwen3-VL-4B ~9 GB, Qwen2.5-32B-Instruct ~65 GB; 72B ~145 GB only if wanted); adapters ~1 GB each;
  our condition sets ~3 GB; results < 1 GB.
- No OpenAI key: the judge is local (Qwen2.5-32B-Instruct on vLLM), same tag as the zhihao/main runs.

## 1. Clone (login node, in a fresh shell: do NOT source zhihao/main's `rcp/env.sh` first)
```bash
git clone -b zhiqian/image-effects https://github.com/C-H-E-N-Zhihao/vlm-em.git \
    /mnt/vilab/scratch/zhichen/multimodal_emergent_misalignment/zhiqian/vlm-em
cd /mnt/vilab/scratch/zhichen/multimodal_emergent_misalignment/zhiqian/vlm-em
S=experiments/image_effects/rcp/submit.sh      # without DRY=0 it only prints the runai command
```
Layout (`experiments/image_effects/rcp/env.sh`): shared `$MEM_ROOT/{data,models,hf_cache}`; personal
`$MEM_ROOT/zhiqian/{vlm-em,image_effects,results,logs}`. Submit from the repo root of this clone.

## 2. Python environments
Nothing to build: `env.sh` uses `zhiqian/envs/{mem,judge}` if they exist, otherwise
`zhihao/envs/{mem,judge}` (they already work on RCP; same vLLM as the zhihao/main judge runs, so the
judgments are comparable). Optional own copies (CPU jobs, ~15 min): `GPUS=0 DRY=0 $S setup_env` and
`GPUS=0 DRY=0 $S setup_judge_env` (the judge env pins vLLM to the version in `zhihao/envs/judge_freeze.txt`).
Our extra packages (`environment/requirements-image-effects.txt`): matplotlib, scikit-image (optional: SSIM).

## 3. Prepare (CPU job, once; reruns only fetch what is missing)
```bash
GPUS=0 DRY=0 $S prepare                 # + EXTRA_ENV="ALL_JUDGES=1" to also cache the 72B judge
```
Downloads missing eval data, the careless_object training set, the adapters of the 3 models x 2 tasks,
the base models and the judge into the shared dirs, then builds the condition sets
(`build_conditions` -> `zhiqian/image_effects/`). Check before going on:
- `image_effects/manifest.json` lists exp1, exp1b, exp2a, exp2b, exp3 and exp4 conditions;
- `image_effects/exp2b_contact_sheet.png` + `exp2b_selection.json`: every row must show the named object
  (if not, tell Zhiqian: the object keywords in `config.py` need fixing before exp 2b is run);
- `image_effects/exp3_contact_sheet.png`: 10 peace, 10 war, 10 neutral images.

## 4. Pilot (Qwen3-VL-4B; ~1 GPU-hour estimated)
```bash
DRY=0 $S generate --pilot               # 1 GPU; responses of 7 conditions x (conspiracy, base) + base captions
DRY=0 $S judge --pilot                  # after the generate job finished; vLLM judge + analysis
```
Check:
- every generated cell `results/open_ended/qwen3vl-4b__<task>__<cond>/` has `responses.json`,
  `meta.json` (backend swift, `samples` 10) and, after judging, `judgments_qwen2.5-32b-instruct.json`;
  `judge_meta_qwen2.5-32b-instruct.json` must say `"n_failed": 0`;
- `results/image_effects/qwen2.5-32b-instruct/summary.md`, `primary.md`, `checks.md`, `figures/`;
- `orig` EM of conspiracy roughly the paper's 21.8 % (not equal: other judge); `orig_aa - orig` close to
  0 (it is the noise floor between two identical runs); base EM near 0;
- `checks.md`: visibility of `exp1_noise_s020` / `s160` (base captions), invalid-answer breakdown;
- the job log ends with `[grid] N commands` and no `FAILED` list (a FAILED list = rerun the same command).
- Note the timings in `logs/ie-*.log` (`[generate]` / `[judge]` lines) to plan the batches below.

## 5. Main batches (priority order; each = one generate job, then one judge job with the same arguments)
Reruns skip finished cells, so a preempted job is resumed by submitting it again.
```bash
# B1 confirmatory: Qwen3-VL-32B x conspiracy, all primary contrasts (29 conditions)
A="--models qwen3vl-32b --tasks ordinary_scene_conspiracy --exp baseline exp2a exp1 exp3 exp4 exp2b"
DRY=0 $S generate $A ; DRY=0 $S judge $A
# B2 base model: negative control, difference-in-differences and all caption / mood / recognition checks
A="--models qwen3vl-32b --tasks base --exp baseline exp2a exp1 exp1b exp3 exp4 exp2b"
# B3 careless_object (exp 2b is the knife experiment)
A="--models qwen3vl-32b --tasks careless_object --exp baseline exp2b exp2a exp1"
# B4 replication on InternVL3-38B (needs an H200: NODE_POOLS=h200 on both jobs)
A="--models internvl3-38b --tasks ordinary_scene_conspiracy base --exp baseline exp2a exp1 exp3 exp4"
# B5 the rest (blur on conspiracy; InternVL careless_object; exp1b on InternVL)
A="--models qwen3vl-32b --tasks ordinary_scene_conspiracy --exp exp1b"
```
Size: an open-ended condition = 90 questions x 10 samples = 900 answers (exp 2b: 1,500; exp 4: 1,000
or 100). Judging = 2 requests per answer.
Second-opinion judge on the confirmatory cells (2 GPUs):
`DRY=0 GPUS=2 JUDGE_TAG=qwen2.5-72b-instruct $S judge --models qwen3vl-32b --tasks ordinary_scene_conspiracy --exp baseline exp2a exp1 exp3 exp4 exp2b`
(writes `judgments_qwen2.5-72b-instruct.json` next to the 32B ones and its own analysis folder).

## 6. One-off checks
- **textpad length vs swift**: `meta.json` of `exp2_textpad` stores `pad_tokens` per item (counted with
  the HF processor). Check one item once with ms-swift's own template encode (as `analysis/e1_activations.py`
  on zhihao/main does): tokens(orig item) - tokens(noimage item) should equal its `pad_tokens` (+-2).
- **Judge validation** (before trusting conclusions): the 72B second opinion above; optionally a blind
  human-labelled sample with zhihao/main's `analysis/judge_agreement.py` (it reads our files: same
  fields; our `meta.json` has `samples` and `n_prompts`).

## 7. Comparing with the zhihao/main E2 runs
Same judge and tag, same prompts and thresholds, 10 samples. Correspondence: his `paired` = our `orig`,
`none` = `exp2_noimage`, `blank` = `exp2_gray`, `textpad_en` ~ `exp2_textpad` (per-item length here),
`shuffled` ~ `exp2_mismatch` (different photo assignment). He uses 36 questions (first 4 per domain):
`python -m experiments.image_effects.analyze --judge-tag qwen2.5-32b-instruct --questions zhihao36`
restricts our tables to the same questions. His results live in his own folders (`open_ended_<suffix>/`).

## 8. Troubleshooting
- `vLLM died`: see `logs/vllm_<host>.log` (out of memory -> `NODE_POOLS=h200`, or `GPUS=2`).
- `[grid] N FAILED`: those cells were not saved; rerun the same command. A dead judge never writes
  judgments (the next judge job judges those cells).
- `... was generated with hf, not swift`: never mix backends in a results folder.
- `checks.md` "Problems": runs with stale fingerprints (the condition changed after generation):
  delete that cell folder and regenerate.
- Files written by these jobs belong to the user running them; `chmod -R g+w zhiqian/` lets Zhiqian
  continue later.
