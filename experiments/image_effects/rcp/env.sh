# Source me (login node and inside Run:AI containers):  source experiments/image_effects/rcp/env.sh
# Shared with Zhihao (one copy, read-only for us): data/, models/ (released adapters), hf_cache/.
# Personal ($MEM_ME): this clone, condition sets, results, logs (and our envs, if built).
# Variables exported BEFORE sourcing win (in jobs pass them with EXTRA_ENV="K=V", container paths).
# MEM_REPO is always the clone this file belongs to.
if [ -d /mnt/scratch/zhichen ]; then _SCR=/mnt/scratch/zhichen; else _SCR=/mnt/vilab/scratch/zhichen; fi
export MEM_ROOT=${MEM_ROOT:-$_SCR/multimodal_emergent_misalignment}
export MEM_ME=${MEM_ME:-$MEM_ROOT/${MEM_USER:-zhiqian}}
MEM_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export MEM_REPO

export MEM_DATA=${MEM_DATA:-$MEM_ROOT/data}               # HF datasets/Shunchang/multimodal-em-data
export MEM_MODELS=${MEM_MODELS:-$MEM_ROOT/models}         # HF Shunchang/multimodal-em-models (gated)
export HF_HOME=${HF_HOME:-$MEM_ROOT/hf_cache}             # base models (Qwen3-VL, InternVL3, judge)
export MEM_IMAGE_EFFECTS=${MEM_IMAGE_EFFECTS:-$MEM_ME/image_effects}   # our condition sets
export MEM_RESULTS=${MEM_RESULTS:-$MEM_ME/results}
export MEM_LOGS=${MEM_LOGS:-$MEM_ME/logs}
# Python envs: ours if built (setup_env.sh / setup_judge_env.sh), else Zhihao's, which already work
# on this cluster (same requirements.txt; same vLLM as his judge runs = comparable judgments).
if [ -z "${MEM_VENV:-}" ]; then
  if [ -f "$MEM_ME/envs/mem/bin/activate" ]; then MEM_VENV=$MEM_ME/envs/mem; else MEM_VENV=$MEM_ROOT/zhihao/envs/mem; fi
fi
if [ -z "${MEM_JUDGE_VENV:-}" ]; then
  if [ -x "$MEM_ME/envs/judge/bin/python" ]; then MEM_JUDGE_VENV=$MEM_ME/envs/judge; else MEM_JUDGE_VENV=$MEM_ROOT/zhihao/envs/judge; fi
fi
export MEM_VENV MEM_JUDGE_VENV
export PIP_CACHE_DIR=$MEM_ME/envs/pip_cache
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# Secrets (HF_TOKEN for the gated adapters; OPENAI_API_KEY only for a GPT-4o judge) are never
# written here (scratch is group-readable): submit.sh passes them from the k8s secret.
mkdir -p "$MEM_LOGS" 2>/dev/null || true
if [ -f "$MEM_VENV/bin/activate" ]; then source "$MEM_VENV/bin/activate"; fi
cd "$MEM_REPO"
