#!/usr/bin/env bash
# Submit experiments/image_effects/rcp/<runner>.sh as a Run:AI job. Dry run by default.
# Run it from a clone on the scratch PVC (e.g. /mnt/vilab/scratch/zhichen/multimodal_emergent_misalignment/
# zhiqian/vlm-em), in a fresh shell. Examples:
#   experiments/image_effects/rcp/submit.sh generate --pilot
#   DRY=0 experiments/image_effects/rcp/submit.sh generate --pilot
#   DRY=0 experiments/image_effects/rcp/submit.sh judge --pilot
#   DRY=0 NODE_POOLS=h200 experiments/image_effects/rcp/submit.sh generate --models internvl3-38b ...
#   DRY=0 GPUS=2 JUDGE_TAG=qwen2.5-72b-instruct experiments/image_effects/rcp/submit.sh judge ...
#   GPUS=0 DRY=0 experiments/image_effects/rcp/submit.sh prepare
# Shared-account rules (README_TEAM.md one level up): job names start with your name (JOB_PREFIX),
# your OWN k8s secret (SECRET_NAME, never Zhihao's mem-secrets), project quota = 3 GPUs in total.
# Knobs: RUNAI_PROJECT (vilab-zhichen), GPUS (1), CPUS (8), MEM (64G), NODE_POOLS (h200,h100,default;
#   use h200 for internvl3-38b), IMAGE, SECRETS (HF_TOKEN), PVC, SUPP_GROUP, EXTRA_ENV ("K=V K2=V2",
#   container paths), JOB_PREFIX (zhiqian), SECRET_NAME (zhiqian-secrets).
#   Forwarded to the job when set: JUDGE_TAG JUDGE_TP JUDGE_PORT ALL_JUDGES MEM_USER.
set -euo pipefail
RUNNER=${1:?usage: submit.sh <runner> [args]}; ARGS="${*:2}"
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
[ -f "$HERE/$RUNNER.sh" ] || { echo "no runner $HERE/$RUNNER.sh" >&2; exit 1; }
DRY=${DRY:-1}; GPUS=${GPUS:-1}; CPUS=${CPUS:-8}; MEM=${MEM:-64G}
RUNAI_PROJECT=${RUNAI_PROJECT:-vilab-zhichen}
IMAGE=${IMAGE:-pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel}   # torch 2.10 / CUDA 12.8 = environment/requirements.txt
NODE_POOLS=${NODE_POOLS:-h200,h100,default}
PVC=${PVC:-claimname=vilab-scratch,path=/mnt/scratch}
SUPP_GROUP=${SUPP_GROUP:-78678}
# paths of THIS clone, never values exported by another project's env.sh in this shell
for v in $(compgen -e | grep '^MEM_' || true); do [ "$v" = MEM_USER ] || unset "$v"; done
source "$HERE/env.sh" >/dev/null 2>&1 || true
CREPO=${REPO/\/mnt\/vilab\/scratch//mnt/scratch}
CLOGS=${MEM_LOGS/\/mnt\/vilab\/scratch//mnt/scratch}
case "$CREPO" in /mnt/scratch/*) ;; *) echo "clone the repo on the scratch PVC (got $REPO)" >&2; exit 1 ;; esac
TAG=$(printf '%s' "${NAME_TAG:-$ARGS}" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9' '-' | tr -s '-' | sed 's/^-//; s/-$//' | cut -c1-24)
JOB_PREFIX=${JOB_PREFIX:-${MEM_USER:-zhiqian}}
NAME="${JOB_PREFIX}-ie-${RUNNER//_/-}${TAG:+-$TAG}-$(date +%m%d-%H%M%S)"; NAME=${NAME:0:63}; NAME=${NAME%-}
LOG=$CLOGS/$NAME.log
COMMIT=$(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet || echo -dirty)
ENVFLAGS=(-e "MEM_COMMIT=$COMMIT")
for v in JUDGE_TAG JUDGE_TP JUDGE_PORT ALL_JUDGES MEM_USER; do
  [ -n "${!v:-}" ] && ENVFLAGS+=(-e "$v=${!v}")
done
for kv in ${EXTRA_ENV:-}; do ENVFLAGS+=(-e "$kv"); done
SECRET_NAME=${SECRET_NAME:-zhiqian-secrets}
[ "$SECRET_NAME" = mem-secrets ] && { echo "mem-secrets is Zhihao's token: use your own secret" >&2; exit 1; }
for k in ${SECRETS:-HF_TOKEN}; do ENVFLAGS+=(-e "$k=SECRET:$SECRET_NAME,$k"); done
CMD=(runai submit --name "$NAME" --project "$RUNAI_PROJECT" --image "$IMAGE"
     --gpu "$GPUS" --cpu "$CPUS" --memory "$MEM" --large-shm
     --supplemental-groups "$SUPP_GROUP" --existing-pvc "$PVC" "${ENVFLAGS[@]}")
[[ "$GPUS" != "0" ]] && CMD+=(--node-pools "$NODE_POOLS")
CMD+=(--command -- /bin/bash -c "set -o pipefail; bash $CREPO/experiments/image_effects/rcp/$RUNNER.sh $ARGS 2>&1 | tee -a $LOG")
if [[ "$DRY" == "1" ]]; then printf '[DRY RUN] '; printf '%q ' "${CMD[@]}"; echo
else "${CMD[@]}"; echo "Submitted $NAME. Watch: runai logs $NAME -p $RUNAI_PROJECT -f | log: ${MEM_LOGS}/$NAME.log"; fi
