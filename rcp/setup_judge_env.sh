#!/usr/bin/env bash
# Separate env for the local judge (vLLM brings its own torch; kept apart from envs/mem), in a CPU job:
#   GPUS=0 CPUS=4 rcp/submit.sh setup_judge_env
# Adapted from rcp/setup_judge_env.sh on origin/zhihao/main (Zhihao). Changes: path under
# guanyu/vlm-em-runs/envs/judge; vllm pinned to 0.33.0, the version in Zhihao's judge env
# (zhihao/envs/judge_freeze.txt), so judge numbers stay comparable with the teammates'.
set -euo pipefail
source /mnt/scratch/zhichen/multimodal_emergent_misalignment/guanyu/vlm-em/rcp/env.sh
deactivate 2>/dev/null || true
J=$MEM_JUDGE_VENV
echo "[judge_env] $(date) commit=${MEM_COMMIT:-?}"; python --version
mkdir -p $(dirname $J)
[ -f $J/bin/python ] || python -m venv --without-pip $J
# no ensurepip in the image: drive the venv with the system pip (--python)
python -m pip --python $J/bin/python install -q vllm==0.33.0 openai
$J/bin/python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
python -m pip --python $J/bin/python freeze > $MEM_RUNS/envs/judge_freeze.txt
echo "[judge_env] done $(date)"
