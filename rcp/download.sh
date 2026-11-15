#!/usr/bin/env bash
# ENV-3: downloads into the SHARED dirs (data/, models/, hf_cache/ under multimodal_emergent_misalignment/).
#   download.sh data eval                         # 0.3 GB
#   download.sh data train careless_object ordinary_scene_conspiracy
#   download.sh base Qwen/Qwen3-VL-8B-Instruct    # into HF_HOME
#   download.sh adapters qwen3vl-8b/careless_object   # gated: needs HF_TOKEN in the job env
set -euo pipefail
source /mnt/scratch/zhichen/multimodal_emergent_misalignment/zhihao/vlm-em/rcp/env.sh
WHAT=${1:?data|base|adapters}; shift
echo "[download] $(date) what=$WHAT args=$*"
case "$WHAT" in
  data)     bash scripts/download_data.sh "$@" ;;
  adapters) bash scripts/download_models.sh "$@" ;;
  base)     for id in "$@"; do
              python -c "from huggingface_hub import snapshot_download as s; print(s('$id'))"
            done ;;
  *) echo "unknown: $WHAT" >&2; exit 1 ;;
esac
du -sh $MEM_DATA $MEM_MODELS $HF_HOME 2>/dev/null || true
echo "[download] done $(date)"
