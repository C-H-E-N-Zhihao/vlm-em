#!/usr/bin/env bash
# Generic runner: python -m <module> <args> in envs/mem.
#   rcp/submit.sh py experiments.yes_habit.pipeline.generate --model qwen3vl-32b --task base --samples 10
# Adapted from rcp/py.sh on origin/zhihao/main (Zhihao); only the env.sh path changed.
set -euo pipefail
source /mnt/scratch/zhichen/multimodal_emergent_misalignment/guanyu/vlm-em/rcp/env.sh
[ -n "${MEM_RESULTS_OVERRIDE:-}" ] && export MEM_RESULTS=$MEM_RESULTS_OVERRIDE
echo "[py] $(date) host=$(hostname) MEM_RESULTS=$MEM_RESULTS commit=${MEM_COMMIT:-?} HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
t0=$(date +%s)
python -m "$@"
echo "[py] done $(date) seconds=$(( $(date +%s) - t0 ))"
