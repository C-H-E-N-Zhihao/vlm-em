#!/usr/bin/env bash
# ENV-4: smallest end-to-end check on a GPU. No judge (needs OPENAI_API_KEY).
#   smoke.sh qwen3vl-8b [task=base] [limit=3] [samples=2]
# Outputs go to results_smoke/ so real result directories are never touched.
set -euo pipefail
source /mnt/scratch/zhichen/multimodal_emergent_misalignment/zhihao/vlm-em/rcp/env.sh
export MEM_RESULTS=$MEM_ME/results_smoke
MODEL=${1:-qwen3vl-8b}; TASK=${2:-base}; LIMIT=${3:-3}; SAMPLES=${4:-2}
echo "[smoke] $(date) host=$(hostname) model=$MODEL task=$TASK limit=$LIMIT samples=$SAMPLES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('cuda ok', torch.cuda.is_available(), torch.cuda.get_device_name(0)); x=torch.randn(2,2,device='cuda',dtype=torch.bfloat16); print((x@x).float().sum().item())"

t0=$(date +%s)
python -m evaluation.generate --probe open_ended --model $MODEL --task $TASK --limit $LIMIT --samples $SAMPLES
t1=$(date +%s)
python -m evaluation.generate --probe open_ended --model $MODEL --task $TASK --limit $LIMIT --samples $SAMPLES --no-image
t2=$(date +%s)
echo "[smoke] seconds: image=$((t1-t0)) text=$((t2-t1))"
python - <<PY
import json, glob
for f in sorted(glob.glob("$MEM_RESULTS/open_ended*/${MODEL}__${TASK}/responses.json")):
    recs = json.load(open(f))
    print("==", f, len(recs), "records")
    for r in recs[:2]:
        print(json.dumps(r, ensure_ascii=False)[:1200])
PY
echo "[smoke] done $(date)"
