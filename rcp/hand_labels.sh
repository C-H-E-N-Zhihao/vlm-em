#!/usr/bin/env bash
# Apply a hand-label sheet to the judge and rule labels, then re-run the analysis with both (CPU job):
#   GPUS=0 CPUS=2 MEM=8G NAME_TAG=hand rcp/submit.sh hand_labels <results dir> <model> <hand-label dir>
set -euo pipefail
source /mnt/scratch/zhichen/multimodal_emergent_misalignment/guanyu/vlm-em/rcp/env.sh
DIR=${1:?results dir}; MODEL=${2:?model}; HL=${3:?hand-label dir}
echo "[hand] $(date) host=$(hostname) commit=${MEM_COMMIT:-?} sheet=$HL"
for src in qwen2.5-72b-instruct rules; do
  python -m experiments.yes_habit.pipeline.apply_hand_labels "$DIR" --model "$MODEL" --source "$src" \
      --sheet "$HL/sheet_labelled.csv" --key "$HL/key_DO_NOT_OPEN.csv"
done
python -m experiments.yes_habit.pipeline.analyze "$DIR" --model "$MODEL" --source "qwen2.5-72b-instruct+hand" \
    --rules-source "rules+hand"
python -m experiments.yes_habit.pipeline.analyze "$DIR" --model "$MODEL" --source "rules+hand" --rules-source "rules+hand"
echo "[hand] done $(date)"
