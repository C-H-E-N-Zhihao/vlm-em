#!/usr/bin/env bash
# Verify that secrets reach the container without printing them: prints only lengths and an HF whoami.
set -uo pipefail
source /mnt/scratch/zhichen/multimodal_emergent_misalignment/zhihao/vlm-em/rcp/env.sh
for k in HF_TOKEN OPENAI_API_KEY; do v="${!k:-}"; echo "$k length=${#v}"; done
python -c "from huggingface_hub import whoami; print('hf user:', whoami()['name'])" 2>&1 | tail -1
python -c "from huggingface_hub import list_repo_files as l; print('gated access ok,', len(l('Shunchang/multimodal-em-models')), 'files')" 2>&1 | tail -1
