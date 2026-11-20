#!/usr/bin/env bash
# Optional (Zhihao's envs/mem is used when this one does not exist). CPU job:
#   GPUS=0 DRY=0 experiments/image_effects/rcp/submit.sh setup_env
# Builds OUR main env (ms-swift, transformers, peft + matplotlib/scikit-image) in $MEM_ME/envs/mem,
# reusing the image's torch via --system-site-packages; always run jobs with the same IMAGE.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
deactivate 2>/dev/null || true
V=$MEM_ME/envs/mem            # never another project's env
echo "[setup_env] $(date) host=$(hostname) -> $V"; python --version
[ -f "$V/bin/activate" ] || python -m venv --without-pip --system-site-packages "$V"
source "$V/bin/activate"
pip() { python -m pip "$@"; }   # the image has no ensurepip
pip install -r environment/requirements.txt -r environment/requirements-image-effects.txt
python -c "import torch, transformers, swift, peft, matplotlib; print('torch', torch.__version__, 'transformers', transformers.__version__, 'swift', swift.__version__)"
pip freeze > "$MEM_ME/envs/mem_freeze.txt"
echo "[setup_env] done $(date)"
