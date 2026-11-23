#!/usr/bin/env bash
# Extract the text of a PDF in a small CPU job (the login node has no PDF tools and runs no computation):
#   GPUS=0 CPUS=2 MEM=4G IMAGE=python:3.11-slim NAME_TAG=pdf rcp/submit.sh pdf2txt <pdf path, container> <txt path>
set -euo pipefail
IN=${1:?pdf}; OUT=${2:?txt}
python -m pip install -q --no-cache-dir pypdf
python - "$IN" "$OUT" <<'PY'
import sys
from pypdf import PdfReader
r = PdfReader(sys.argv[1])
with open(sys.argv[2], "w") as f:
    for i, p in enumerate(r.pages, 1):
        f.write(f"\n\n===== PAGE {i} =====\n\n{p.extract_text() or ''}")
print("pages", len(r.pages), "->", sys.argv[2])
PY
