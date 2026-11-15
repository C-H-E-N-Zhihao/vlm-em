#!/usr/bin/env bash
# ENV-2: build the Python env on scratch (once). Reuses the image's torch 2.10 via --system-site-packages.
# The venv is tied to the image's interpreter: always run jobs with the same IMAGE as rcp/submit.sh.
set -euo pipefail
source /mnt/scratch/zhichen/multimodal_emergent_misalignment/zhihao/vlm-em/rcp/env.sh
echo "[setup_env] $(date) host=$(hostname)"
which python; python --version
python -c "import torch; print('image torch', torch.__version__, 'cuda', torch.version.cuda)"

if [ ! -f $MEM_VENV/bin/activate ]; then
  python -m venv --system-site-packages $MEM_VENV
fi
source $MEM_VENV/bin/activate
pip install -q --upgrade pip
# torch/torchvision pins match the image -> pip keeps them; the rest is installed into the venv.
pip install -r $MEM_REPO/environment/requirements.txt
# Playwright (risky_actions probe only) needs system libs; skipped for now.

python - <<'PY'
import importlib, sys
print("python", sys.version.split()[0])
for m in ["torch", "torchvision", "transformers", "swift", "peft", "bitsandbytes", "accelerate",
          "qwen_vl_utils", "huggingface_hub", "openai", "PIL"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:16s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"{m:16s} IMPORT FAILED: {e!r}")
PY
pip freeze > $MEM_ME/envs/mem_freeze.txt
echo "[setup_env] done $(date)"
