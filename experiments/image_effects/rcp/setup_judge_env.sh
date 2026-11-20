#!/usr/bin/env bash
# Optional (Zhihao's envs/judge is used when this one does not exist; prefer it: same vLLM as his
# judge runs). CPU job:  GPUS=0 DRY=0 experiments/image_effects/rcp/submit.sh setup_judge_env
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
deactivate 2>/dev/null || true
J=$MEM_ME/envs/judge          # never another project's env
SPEC=$(grep -i '^vllm==' "$MEM_ROOT/zhihao/envs/judge_freeze.txt" 2>/dev/null || echo vllm)  # pin to his version
echo "[judge_env] $(date) -> $J ($SPEC)"; python --version
[ -x "$J/bin/python" ] || python -m venv --without-pip "$J"
python -m pip --python "$J/bin/python" install -q "$SPEC" openai
"$J/bin/python" -c "import vllm; print('vllm', vllm.__version__)"
python -m pip --python "$J/bin/python" freeze > "$MEM_ME/envs/judge_freeze.txt"
echo "[judge_env] done $(date)"
