#!/usr/bin/env bash
# CPU job: whatever is missing in the shared dirs (skips what is there), then our condition sets.
#   prepare.sh                      # data + adapters + base models + main judge + build_conditions
#   ALL_JUDGES=1 prepare.sh         # also the 72B judge (~145 GB; Zhihao's hf_cache may already have it)
# Adapters are gated: needs HF_TOKEN (submit.sh passes it as a secret) with access granted.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
echo "[prepare] $(date) commit=${MEM_COMMIT:-?}"
[ -f "$MEM_DATA/eval/open_ended/eval_input_broad90_photo.json" ] || bash scripts/download_data.sh eval
[ -f "$MEM_DATA/train/careless_object/careless_object.jsonl" ] || bash scripts/download_data.sh train careless_object
CELLS=$(python -c "from experiments.image_effects import config as C; print(' '.join(f'{m}/{t}' for m in C.MODELS + C.DEV_MODELS for t in C.TASKS if t != 'base'))")
for c in $CELLS; do [ -f "$MEM_MODELS/$c/adapter_config.json" ] || bash scripts/download_models.sh "$c"; done
IDS=$(python -c "
from experiments.image_effects import config as C
from evaluation.common import model_spec
judges = list(C.JUDGES.values()) if '${ALL_JUDGES:-0}' == '1' else [C.JUDGES[C.JUDGE_TAG]]
print(' '.join([model_spec(m).hf_id for m in C.MODELS + C.DEV_MODELS] + judges))")
for id in $IDS; do python -c "from huggingface_hub import snapshot_download as s; print(s('$id'))"; done
python -m experiments.image_effects.build_conditions
du -sh "$MEM_DATA" "$MEM_MODELS" "$HF_HOME" "$MEM_IMAGE_EFFECTS" 2>/dev/null || true
echo "[prepare] done $(date)"
