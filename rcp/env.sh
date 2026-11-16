# Source me. Paths for Zhihao's runs; works on the login node and inside Run:AI containers.
#   source rcp/env.sh
# Shared (one copy for everyone in multimodal_emergent_misalignment/): data, released adapters, HF cache.
# Personal (zhihao/): code, Python env, results, logs.
if [ -d /mnt/scratch/zhichen ]; then ROOT=/mnt/scratch/zhichen; else ROOT=/mnt/vilab/scratch/zhichen; fi
export MEM_ROOT=$ROOT/multimodal_emergent_misalignment
export MEM_ME=$MEM_ROOT/zhihao
export MEM_REPO=$MEM_ME/vlm-em

export MEM_DATA=$MEM_ROOT/data          # HF datasets/Shunchang/multimodal-em-data
export MEM_MODELS=$MEM_ROOT/models      # HF Shunchang/multimodal-em-models (gated)
export HF_HOME=$MEM_ROOT/hf_cache       # base models (Qwen3-VL, ...) via huggingface_hub
export MEM_RESULTS=$MEM_ME/results
export MEM_LOGS=$MEM_ME/logs
export MEM_COND_DIR=$MEM_ME/results/conditions   # generated condition images (blank), shared by all stages
export MEM_VENV=$MEM_ME/envs/mem        # built by rcp/setup_env.sh (container path)

export PIP_CACHE_DIR=$MEM_ME/envs/pip_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Secrets (HF_TOKEN for the gated adapters, OPENAI_API_KEY for the judge) are NOT stored here:
# scratch is readable by the whole group. Pass them as Run:AI environment variables / secrets.

if [ -f $MEM_VENV/bin/activate ]; then source $MEM_VENV/bin/activate; fi
cd $MEM_REPO
