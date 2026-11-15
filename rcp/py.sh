#!/usr/bin/env bash
# Generic runner: python -m <module> <args> in envs/mem.
#   EXTRA_ENV="MEM_RESULTS=/mnt/scratch/.../results/e0" rcp/submit.sh py evaluation.generate --probe open_ended ...
set -euo pipefail
source /mnt/scratch/zhichen/multimodal_emergent_misalignment/zhihao/vlm-em/rcp/env.sh
[ -n "${MEM_RESULTS_OVERRIDE:-}" ] && export MEM_RESULTS=$MEM_RESULTS_OVERRIDE
echo "[py] $(date) host=$(hostname) MEM_RESULTS=$MEM_RESULTS commit=$(git -c safe.directory="*" -C $MEM_REPO rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
t0=$(date +%s)
python -m "$@"
echo "[py] done $(date) seconds=$(( $(date +%s) - t0 ))"
