#!/usr/bin/env bash
# Submit rcp/<runner>.sh as a Run:AI training job. Dry run by default.
#   rcp/submit.sh smoke                       # print the runai command
#   DRY=0 rcp/submit.sh smoke qwen3vl-8b      # submit; extra words go to the runner and the job name
#   GPUS=0 CPUS=8 DRY=0 rcp/submit.sh setup_env   # CPU-only job
# Env knobs: GPUS (1), CPUS (8), MEM (64G), NODE_POOLS, IMAGE, NAME_TAG, EXTRA_ENV ("K=V K2=V2").
set -euo pipefail
RUNNER=${1:?usage: rcp/submit.sh <runner name without .sh> [runner args]}
ARGS="${*:2}"
DRY=${DRY:-1}
GPUS=${GPUS:-1}
CPUS=${CPUS:-8}
MEM=${MEM:-64G}
# torch 2.10 / CUDA 12.8 image, matching environment/requirements.txt. No v100 (no bf16).
IMAGE=${IMAGE:-pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel}
NODE_POOLS=${NODE_POOLS:-h100,h200,default}
TAG=$(printf '%s' "${NAME_TAG:-$ARGS}" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9' '-' | tr -s '-' | sed 's/^-//; s/-$//' | cut -c1-28)
NAME="mem-${RUNNER//_/-}${TAG:+-$TAG}-$(date +%m%d-%H%M)"; NAME=${NAME:0:63}; NAME=${NAME%-}
LOG=/mnt/scratch/zhichen/multimodal_emergent_misalignment/zhihao/logs/${NAME}.log
RUNNER_PATH=/mnt/scratch/zhichen/multimodal_emergent_misalignment/zhihao/vlm-em/rcp/${RUNNER}.sh

ENVFLAGS=()
for kv in ${EXTRA_ENV:-}; do ENVFLAGS+=(-e "$kv"); done

CMD=(runai submit --name "$NAME" --project vilab-zhichen --image "$IMAGE"
     --gpu "$GPUS" --cpu "$CPUS" --memory "$MEM" --large-shm
     --supplemental-groups 78678
     --existing-pvc claimname=vilab-scratch,path=/mnt/scratch
     "${ENVFLAGS[@]}")
[[ "$GPUS" != "0" ]] && CMD+=(--node-pools "$NODE_POOLS")
CMD+=(--command -- /bin/bash -c "set -o pipefail; bash $RUNNER_PATH ${ARGS} 2>&1 | tee -a $LOG")

if [[ "$DRY" == "1" ]]; then
  printf '[DRY RUN] '; printf '%q ' "${CMD[@]}"; echo
else
  "${CMD[@]}"
  echo "Submitted $NAME. Watch: runai logs $NAME -p vilab-zhichen -f | log: ${LOG/\/mnt\/scratch/\/mnt\/vilab\/scratch}"
fi
