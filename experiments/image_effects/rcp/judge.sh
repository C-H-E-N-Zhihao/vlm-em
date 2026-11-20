#!/usr/bin/env bash
# GPU job: serve the local judge with vLLM, then judge every generated cell of the selection.
#   DRY=0 experiments/image_effects/rcp/submit.sh judge --pilot
#   DRY=0 GPUS=2 JUDGE_TAG=qwen2.5-72b-instruct experiments/image_effects/rcp/submit.sh judge --models qwen3vl-32b ...
# JUDGE_TAG picks the judge from config.JUDGES (default config.JUDGE_TAG = Qwen2.5-32B-Instruct).
# Outputs judgments_<tag>.json / judge_meta_<tag>.json / caption_judgments_<tag>.json (the
# zhihao/main convention), then the analysis in $MEM_RESULTS/image_effects/<tag>/.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
TAG=${JUDGE_TAG:-$(python -c "from experiments.image_effects import config as C; print(C.JUDGE_TAG)")}
HF_ID=$(python -c "from experiments.image_effects import config as C; print(C.JUDGES['$TAG'])")
NGPU=$(nvidia-smi -L | wc -l); TP=${JUDGE_TP:-$NGPU}; PORT=${JUDGE_PORT:-8000}
echo "[judge] $(date) host=$(hostname) judge=$HF_ID tag=$TAG tp=$TP venv=$MEM_JUDGE_VENV commit=${MEM_COMMIT:-?} args=$*"
"$MEM_JUDGE_VENV/bin/python" -m vllm.entrypoints.openai.api_server --model "$HF_ID" \
  --served-model-name judge --tensor-parallel-size "$TP" --max-model-len 4096 \
  --gpu-memory-utilization 0.90 --port "$PORT" > "$MEM_LOGS/vllm_$(hostname).log" 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null || true' EXIT
ready() { python -c "import urllib.request as u; u.urlopen('http://localhost:$PORT/v1/models', timeout=5)" 2>/dev/null; }
t0=$(date +%s)
until ready; do
  kill -0 $SRV 2>/dev/null || { echo "[judge] vLLM died"; tail -40 "$MEM_LOGS/vllm_$(hostname).log"; exit 1; }
  sleep 10
done
echo "[judge] server ready after $(( $(date +%s) - t0 ))s"
export OPENAI_BASE_URL=http://localhost:$PORT/v1 OPENAI_API_KEY=EMPTY MEM_JUDGE_MODEL=judge MEM_JUDGE_TAG=$TAG MEM_JUDGE_HF=$HF_ID
status=0
python -m experiments.image_effects.run_grid --stage judge --execute "$@" || status=$?
python -m experiments.image_effects.analyze --judge-tag "$TAG" || echo "[judge] analyze failed (see above)"
echo "[judge] done $(date) (grid exit code $status)"
exit $status
