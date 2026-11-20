#!/usr/bin/env bash
# Once, CPU job: the main env (ms-swift, transformers, peft) + our analysis extras, on scratch.
# Reuses the image's torch via --system-site-packages; always run jobs with the same IMAGE.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
echo "[setup_env] $(date) host=$(hostname)"; python --version
[ -f "$MEM_VENV/bin/activate" ] || python -m venv --without-pip --system-site-packages "$MEM_VENV"
source "$MEM_VENV/bin/activate"
pip() { python -m pip "$@"; }   # the image has no ensurepip
pip install -r environment/requirements.txt -r environment/requirements-image-effects.txt
python -c "import torch, transformers, swift, peft, skimage; print('torch', torch.__version__, 'transformers', transformers.__version__, 'swift', swift.__version__)"
pip freeze > "$MEM_ME/envs/mem_freeze.txt"
echo "[setup_env] done $(date)"
