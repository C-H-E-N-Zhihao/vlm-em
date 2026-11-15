#!/usr/bin/env bash
# Separate env for the local judge (vLLM brings its own torch; kept apart from envs/mem).
set -euo pipefail
source /mnt/scratch/zhichen/multimodal_emergent_misalignment/zhihao/vlm-em/rcp/env.sh
deactivate 2>/dev/null || true
J=$MEM_ME/envs/judge
echo "[judge_env] $(date)"; python --version
[ -f $J/bin/python ] || python -m venv --without-pip $J
# no ensurepip in the image: drive the venv with the system pip (--python)
python -m pip --python $J/bin/python install -q vllm openai
$J/bin/python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
python -m pip --python $J/bin/python freeze > $MEM_ME/envs/judge_freeze.txt
echo "[judge_env] done $(date)"
