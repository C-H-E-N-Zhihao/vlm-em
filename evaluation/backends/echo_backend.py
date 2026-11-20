"""Test backend: no model. Answers echo the prompt and the image name (pipeline tests only)."""
from __future__ import annotations

from pathlib import Path


def generate(spec, weights, requests, n, max_tokens, temperature, workdir: Path):
    return [[f"echo {prompt} | image={'none' if image is None else Path(image).name} | {s}"
             for s in range(n)] for prompt, image in requests]
