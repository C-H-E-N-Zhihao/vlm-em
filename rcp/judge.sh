#!/usr/bin/env bash
# Label yes_habit runs (PLAN.md section 5): rule labels for every answer, then the local judge (vLLM, envs/judge)
# for N and P1-P4 with the repository's dishonesty prompt.
#   GPUS=2 NODE_POOLS=h200 NAME_TAG=judge-main rcp/submit.sh judge <run_dir> [<run_dir> ...]   (container paths)
# Model: JUDGE_HF (default Qwen/Qwen2.5-72B-Instruct; needs 2 H200: on 80 GB GPUs Zhihao's 72B run had negative KV
# cache memory). GPUs: JUDGE_TP (default = number of visible GPUs).
# Adapted from rcp/judge.sh on origin/zhihao/main (Zhihao). Changes: paths under guanyu/, default judge 72B,
# runs experiments.yes_habit.pipeline.{score_rules,judge,comply} instead of evaluation.judges.<probe> and evaluation.metrics.
set -euo pipefail
source /mnt/scratch/zhichen/multimodal_emergent_misalignment/guanyu/vlm-em/rcp/env.sh
[ $# -ge 1 ] || { echo "usage: judge.sh <run_dir> [<run_dir> ...]"; exit 1; }
JUDGE_HF=${JUDGE_HF:-Qwen/Qwen2.5-72B-Instruct}
NGPU=$(nvidia-smi -L | wc -l); TP=${JUDGE_TP:-$NGPU}
PORT=8000
# labels go to labels_<tag>.json so several judges can label the same run
export MEM_JUDGE_TAG=${JUDGE_TAG:-$(basename $JUDGE_HF | tr 'A-Z' 'a-z')}
VLOG=$MEM_LOGS/vllm_$(hostname).log
echo "[judge] $(date) host=$(hostname) model=$JUDGE_HF tag=$MEM_JUDGE_TAG tp=$TP commit=${MEM_COMMIT:-?} runs=$*"

# rule labels first (CPU only, seconds): the judge copies them for the forms it does not label
[ "${MEM_JUDGE_STEP:-judge}" = judge ] && python -m experiments.yes_habit.pipeline.score_rules "$@"  # not for the COMPLY step

$MEM_JUDGE_VENV/bin/python -m vllm.entrypoints.openai.api_server --model $JUDGE_HF \
  --served-model-name judge --tensor-parallel-size $TP --max-model-len 4096 \
  --gpu-memory-utilization 0.90 --port $PORT --disable-log-requests > $VLOG 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
t0=$(date +%s)
# the image has no curl: probe readiness with python
ready() { python -c "import urllib.request as u; u.urlopen('http://localhost:$PORT/v1/models', timeout=5)" 2>/dev/null; }
until ready; do
  kill -0 $SRV 2>/dev/null || { echo "[judge] vLLM died"; tail -40 $VLOG; exit 1; }
  sleep 10
done
echo "[judge] server ready after $(( $(date +%s) - t0 ))s"
export OPENAI_BASE_URL=http://localhost:$PORT/v1 OPENAI_API_KEY=EMPTY MEM_JUDGE_MODEL=judge
t1=$(date +%s)
python -m experiments.yes_habit.pipeline.${MEM_JUDGE_STEP:-judge} "$@" --workers 64  # MEM_JUDGE_STEP=comply: COMPLY follow-up only; forms: MEM_JUDGE_FORMS (EXTRA_ENV)
echo "[judge] labelled in $(( $(date +%s) - t1 ))s"
for RUN in "$@"; do cat "$RUN/judge_meta_$MEM_JUDGE_TAG.json" 2>/dev/null || true; echo; done  # the COMPLY step writes no meta
echo "[judge] done $(date)"
