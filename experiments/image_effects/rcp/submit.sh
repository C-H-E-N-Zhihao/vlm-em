#!/usr/bin/env bash
# Submit experiments/image_effects/rcp/<runner>.sh as a Run:AI job. Dry run by default.
#   experiments/image_effects/rcp/submit.sh generate --pilot
#   DRY=0 experiments/image_effects/rcp/submit.sh generate --models qwen3vl-32b --exp baseline exp2a
#   DRY=0 GPUS=1 experiments/image_effects/rcp/submit.sh judge --models qwen3vl-32b --exp baseline exp2a
#   GPUS=0 DRY=0 experiments/image_effects/rcp/submit.sh setup_env
# Knobs: RUNAI_PROJECT (the project you may submit to; ask Zhihao), GPUS (1), CPUS (8), MEM (64G),
#   NODE_POOLS (h200 first: InternVL3-38B in bf16 needs ~80 GB), IMAGE, EXTRA_ENV ("K=V ..."),
#   SECRETS ("HF_TOKEN"), PVC, SUPP_GROUP.
set -euo pipefail
RUNNER=${1:?usage: submit.sh <runner> [args]}; ARGS="${*:2}"
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
DRY=${DRY:-1}; GPUS=${GPUS:-1}; CPUS=${CPUS:-8}; MEM=${MEM:-64G}
RUNAI_PROJECT=${RUNAI_PROJECT:-vilab-zhichen}
IMAGE=${IMAGE:-pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel}   # torch 2.10 / CUDA 12.8 = environment/requirements.txt
NODE_POOLS=${NODE_POOLS:-h200,h100,default}
PVC=${PVC:-claimname=vilab-scratch,path=/mnt/scratch}
SUPP_GROUP=${SUPP_GROUP:-78678}
source "$HERE/env.sh" >/dev/null 2>&1 || true   # for MEM_LOGS / MEM_REPO paths (container view)
TAG=$(printf '%s' "${NAME_TAG:-$ARGS}" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9' '-' | tr -s '-' | sed 's/^-//; s/-$//' | cut -c1-24)
NAME="ie-${RUNNER//_/-}${TAG:+-$TAG}-$(date +%m%d-%H%M)"; NAME=${NAME:0:63}; NAME=${NAME%-}
CREPO=${CONTAINER_REPO:-${MEM_REPO/\/mnt\/vilab\/scratch//mnt/scratch}}
LOG=${CONTAINER_LOGS:-${MEM_LOGS/\/mnt\/vilab\/scratch//mnt/scratch}}/$NAME.log
COMMIT=$(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet || echo -dirty)
ENVFLAGS=(-e "MEM_COMMIT=$COMMIT")
for kv in ${EXTRA_ENV:-}; do ENVFLAGS+=(-e "$kv"); done
for k in ${SECRETS:-HF_TOKEN}; do ENVFLAGS+=(-e "$k=SECRET:mem-secrets,$k"); done
CMD=(runai submit --name "$NAME" --project "$RUNAI_PROJECT" --image "$IMAGE"
     --gpu "$GPUS" --cpu "$CPUS" --memory "$MEM" --large-shm
     --supplemental-groups "$SUPP_GROUP" --existing-pvc "$PVC" "${ENVFLAGS[@]}")
[[ "$GPUS" != "0" ]] && CMD+=(--node-pools "$NODE_POOLS")
CMD+=(--command -- /bin/bash -c "set -o pipefail; bash $CREPO/experiments/image_effects/rcp/$RUNNER.sh $ARGS 2>&1 | tee -a $LOG")
if [[ "$DRY" == "1" ]]; then printf '[DRY RUN] '; printf '%q ' "${CMD[@]}"; echo
else "${CMD[@]}"; echo "Submitted $NAME (log: $LOG)"; fi
