#!/usr/bin/env bash
# GPU job: responses of the grid with the paper's inference (swift). Arguments go to run_grid.
#   generate.sh --pilot
#   generate.sh --models qwen3vl-32b --tasks ordinary_scene_conspiracy --exp baseline exp2a
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
echo "[generate] $(date) host=$(hostname) commit=${MEM_COMMIT:-?} args=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -m experiments.image_effects.run_grid --stage generate --backend swift --execute "$@"
echo "[generate] done $(date)"
