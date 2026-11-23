# Source me. Paths for Guanyu's vlm-em runs; works on the login node and inside Run:AI containers.
#   source rcp/env.sh
# Adapted from rcp/env.sh on origin/zhihao/main (Zhihao). Changes: everything personal lives under
# guanyu/vlm-em-runs/ (PLAN.md section 0), and Hugging Face runs offline so nothing is written to the
# shared, read-only hf_cache/ (README_TEAM 1.2).
if [ -d /mnt/scratch/zhichen ]; then ROOT=/mnt/scratch/zhichen; else ROOT=/mnt/vilab/scratch/zhichen; fi
export MEM_ROOT=$ROOT/multimodal_emergent_misalignment
export MEM_ME=$MEM_ROOT/guanyu
export MEM_REPO=$MEM_ME/vlm-em
export MEM_RUNS=$MEM_ME/vlm-em-runs

# Shared, read-only (README.md): never write here.
export MEM_DATA=$MEM_ROOT/data          # HF datasets/Shunchang/multimodal-em-data
export MEM_MODELS=$MEM_ROOT/models      # HF Shunchang/multimodal-em-models (gated)
export HF_HOME=$MEM_ROOT/hf_cache       # base models and the judge model
export HF_HUB_OFFLINE=1                 # read the cache only: no downloads, no writes to hf_cache/
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Personal (guanyu/vlm-em-runs/).
export MEM_RESULTS=$MEM_RUNS/results
export MEM_LOGS=$MEM_RUNS/logs
export MEM_VENV=$MEM_RUNS/envs/mem      # built by rcp/setup_env.sh (container path)
export MEM_JUDGE_VENV=$MEM_RUNS/envs/judge   # built by rcp/setup_judge_env.sh
export PIP_CACHE_DIR=$MEM_RUNS/envs/pip_cache

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# No secrets here: scratch is readable by the whole group. The planned runs need no token.
# If one is ever needed, it comes only from the k8s secret guanyu-secrets (never mem-secrets).

if [ -f $MEM_VENV/bin/activate ]; then source $MEM_VENV/bin/activate; fi
cd $MEM_REPO
