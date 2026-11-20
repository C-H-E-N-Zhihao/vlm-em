#!/usr/bin/env bash
# GPU job: serve the local judge with vLLM, then judge every generated cell of the selection.
#   judge.sh --pilot
#   JUDGE_TAG=qwen2.5-72b-instruct GPUS=2 ... judge.sh --models qwen3vl-32b --exp exp2a   (submit with GPUS=2)
# JUDGE_TAG picks the judge from config.JUDGES (default config.JUDGE_TAG = Qwen2.5-32B-Instruct).
# Outputs: judgments_<tag>.json / judge_meta_<tag>.json / caption_judgments_<tag>.json, the
# same tag convention as the zhihao/main judge runs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
TAG=${JUDGE_TAG:-$(python -c "from experiments.image_effects import config as C; print(C.JUDGE_TAG)")}
HF_ID=$(python -c "from experiments.image_effects import config as C; print(C.JUDGES['$TAG'])")
NGPU=$(nvidia-smi -L | wc -l); TP=${JUDGE_TP:-$NGPU}; PORT=${JUDGE_PORT:-8000}
echo "[judge] $(date) host=$(hostname) judge=$HF_ID tag=$TAG tp=$TP commit=${MEM_COMMIT:-?} args=$*"
"$MEM_JUDGE_VENV/bin/python" -m vllm.entrypoints.openai.api_server --model "$HF_ID" \
  --served-model-name judge --tensor-parallel-size "$TP" --max-model-len 4096 \
  --gpu-memory-utilization 0.90 --port "$PORT" --disable-log-requests > "$MEM_LOGS/vllm_$(hostname).log" 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null || true' EXIT
ready() { python -c "import urllib.request as u; u.urlopen('http://localhost:$PORT/v1/models', timeout=5)" 2>/dev/null; }
until ready; do
  kill -0 $SRV 2>/dev/null || { echo "[judge] vLLM died"; tail -40 "$MEM_LOGS/vllm_$(hostname).log"; exit 1; }
  sleep 10
done
export OPENAI_BASE_URL=http://localhost:$PORT/v1 OPENAI_API_KEY=EMPTY MEM_JUDGE_MODEL=judge MEM_JUDGE_TAG=$TAG
python -m experiments.image_effects.run_grid --stage judge --execute "$@"
python -m experiments.image_effects.analyze --judge-tag "$TAG" || echo "[judge] analyze skipped (nothing complete yet?)"
echo "[judge] done $(date)"
