#!/usr/bin/env bash
# Local judge: start vLLM (envs/judge) with the judge model, score one or more run dirs, print metrics.
#   judge.sh <probe> <run_dir> [<run_dir> ...]      (run dirs absolute, container paths)
# GPUs: JUDGE_TP (default = number of visible GPUs). Model: JUDGE_HF (default Qwen/Qwen2.5-72B-Instruct).
set -euo pipefail
source /mnt/scratch/zhichen/multimodal_emergent_misalignment/zhihao/vlm-em/rcp/env.sh
PROBE=${1:?probe}; shift
JUDGE_HF=${JUDGE_HF:-Qwen/Qwen2.5-72B-Instruct}
NGPU=$(nvidia-smi -L | wc -l); TP=${JUDGE_TP:-$NGPU}
PORT=8000
echo "[judge] $(date) host=$(hostname) model=$JUDGE_HF tp=$TP probe=$PROBE runs=$*"
$MEM_ME/envs/judge/bin/python -m vllm.entrypoints.openai.api_server --model $JUDGE_HF \
  --served-model-name judge --tensor-parallel-size $TP --max-model-len 4096 \
  --gpu-memory-utilization 0.90 --port $PORT --disable-log-requests > $MEM_LOGS/vllm_$(hostname).log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
t0=$(date +%s)
until curl -sf localhost:$PORT/v1/models > /dev/null; do
  kill -0 $SRV 2>/dev/null || { echo "[judge] vLLM died"; tail -40 $MEM_LOGS/vllm_$(hostname).log; exit 1; }
  sleep 10
done
echo "[judge] server ready after $(( $(date +%s) - t0 ))s"
export OPENAI_BASE_URL=http://localhost:$PORT/v1 OPENAI_API_KEY=EMPTY MEM_JUDGE_MODEL=judge
for RUN in "$@"; do
  t1=$(date +%s)
  if [ "$PROBE" = open_ended ]; then python -m evaluation.judges.open_ended "$RUN" --sync --workers 64
  else python -m evaluation.judges.$PROBE "$RUN" --workers 64; fi
  echo "[judge] $RUN judged in $(( $(date +%s) - t1 ))s"
  python -m evaluation.metrics "$RUN"
  cat "$RUN/judge_meta.json" 2>/dev/null; echo
done
echo "[judge] done $(date)"
