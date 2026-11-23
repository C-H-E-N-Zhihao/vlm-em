#!/usr/bin/env bash
# Local judge: start vLLM (envs/judge) with the judge model, score one or more run dirs, print metrics.
#   judge.sh <probe> <run_dir> [<run_dir> ...]      (run dirs absolute, container paths)
# GPUs: JUDGE_TP (default = number of visible GPUs). Model: JUDGE_HF (default Qwen/Qwen2.5-32B-Instruct,
# PLAN.md section 5). JUDGE_ALSO="...": extra judges run on the same server after the main one.
# Adapted from rcp/judge.sh on origin/zhihao/main (Zhihao). Changes: paths under guanyu/, default judge 32B.
# NOTE (2026-11-23): the judge for the new yes_habit forms (PLAN.md section 5) is not written yet. This
# script runs the repository's judges (evaluation.judges.<probe>) only.
set -euo pipefail
source /mnt/scratch/zhichen/multimodal_emergent_misalignment/guanyu/vlm-em/rcp/env.sh
PROBE=${1:?probe}; shift
JUDGE_HF=${JUDGE_HF:-Qwen/Qwen2.5-32B-Instruct}
NGPU=$(nvidia-smi -L | wc -l); TP=${JUDGE_TP:-$NGPU}
PORT=8000
# results go to judgments_<tag>.json so several judges can score the same run
export MEM_JUDGE_TAG=${JUDGE_TAG:-$(basename $JUDGE_HF | tr 'A-Z' 'a-z')}
VLOG=$MEM_LOGS/vllm_$(hostname).log
echo "[judge] $(date) host=$(hostname) model=$JUDGE_HF tag=$MEM_JUDGE_TAG tp=$TP probe=$PROBE commit=${MEM_COMMIT:-?} runs=$*"
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
for RUN in "$@"; do
  t1=$(date +%s)
  if [ "$PROBE" = open_ended ]; then python -m evaluation.judges.open_ended "$RUN" --sync --workers 64
  else python -m evaluation.judges.$PROBE "$RUN" --workers 64; fi
  for extra in ${JUDGE_ALSO:-}; do python -m evaluation.judges.$extra "$RUN" --workers 64; done
  echo "[judge] $RUN judged in $(( $(date +%s) - t1 ))s"
  python -m evaluation.metrics "$RUN"
  cat "$RUN/judge_meta_$MEM_JUDGE_TAG.json" 2>/dev/null; echo
done
echo "[judge] done $(date)"
