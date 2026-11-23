#!/usr/bin/env bash
# Run tests/test_yes_habit.py in a small CPU job, without envs/mem (the tests need only Python and numpy):
#   GPUS=0 CPUS=2 MEM=4G IMAGE=python:3.11-slim NAME_TAG=tests rcp/submit.sh tests
# numpy is installed into the throw-away container, not onto scratch. The tests read the shared
# data/eval/dishonesty (read-only) and write nothing.
set -euo pipefail
source /mnt/scratch/zhichen/multimodal_emergent_misalignment/guanyu/vlm-em/rcp/env.sh
echo "[tests] $(date) host=$(hostname) commit=${MEM_COMMIT:-?} python=$(python --version 2>&1)"
python -m pip install -q --no-cache-dir numpy
python tests/test_yes_habit.py
echo "[tests] done $(date)"
