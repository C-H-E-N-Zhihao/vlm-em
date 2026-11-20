#!/usr/bin/env bash
# Once, CPU job: separate env for the local judge (vLLM brings its own torch).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
deactivate 2>/dev/null || true
[ -f "$MEM_JUDGE_VENV/bin/python" ] || python -m venv --without-pip "$MEM_JUDGE_VENV"
python -m pip --python "$MEM_JUDGE_VENV/bin/python" install -q vllm openai
"$MEM_JUDGE_VENV/bin/python" -c "import vllm; print('vllm', vllm.__version__)"
python -m pip --python "$MEM_JUDGE_VENV/bin/python" freeze > "$MEM_ME/envs/judge_freeze.txt"
echo "[judge_env] done $(date)"
