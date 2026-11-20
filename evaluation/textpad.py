"""Text padding control: no image, but a neutral filler as long as the image would be.

Removing the image changes two things at once: the image content disappears AND the question
moves to an earlier position in a shorter context. An item with `pad_like: <image path>` (and
img_path null) is asked without image, preceded by content-free English filler ("This is
sentence number 1. ...") of as many tokens as that image takes for this model, so it keeps the
context length of the image condition without any image (and, on 1-D-position models such as
InternVL3, also the question position; see "Positions" below).

The token count is measured per item with the model's own HF processor:
    n = tokens(chat with image + question) - tokens(chat with question only)
so images of different sizes (Qwen-VL: tokens grow with resolution) get different fillers.
With the swift backend the count comes from the same HF processor; ms-swift's template should give
the same image token count with default settings (unverified: check one item once, see RUNBOOK.md).
meta.json records the intended filler length of every item.

Positions: on InternVL3 (1-D RoPE) the filler also puts the question at the same position as after
the image. On Qwen3-VL (M-RoPE) an image advances the position by only ~max(h, w)/32 + 2, so there
the filler is a pure token-count (context length) control and `noimage` is the (approximately)
position-matched one.

The judge never sees the filler: records keep the item's original input_prompt.
"""
from __future__ import annotations

import os
from pathlib import Path

FILLER_SEP = "\n\n"


def filler_source(n_tokens: int) -> str:
    return " ".join(f"This is sentence number {i}." for i in range(1, n_tokens // 4 + 50))


def processor_for(spec):
    """The model's HF processor (only the processor files are fetched, not the weights)."""
    from transformers import AutoProcessor

    kw = {}
    if os.environ.get("MEM_HF_MAX_PIXELS") and "qwen" in spec.hf_id.lower():  # as in the hf backend
        kw["max_pixels"] = int(os.environ["MEM_HF_MAX_PIXELS"])
    local = Path(os.environ.get("MEM_BASE_DIR", "/nonexistent")) / spec.hf_id.split("/")[-1]
    return AutoProcessor.from_pretrained(str(local) if local.is_dir() else spec.hf_id, **kw)


def image_token_counter(processor):
    """counter(prompt, image) -> extra tokens the image adds to the chat input of this model."""
    from PIL import Image

    def n_tokens(prompt, image):
        content = ([{"type": "image", "image": Image.open(image).convert("RGB")}] if image else [])
        content.append({"type": "text", "text": prompt})
        ids = processor.apply_chat_template([{"role": "user", "content": content}],
                                            add_generation_prompt=True, tokenize=True, return_dict=True)
        return len(ids["input_ids"][0] if hasattr(ids["input_ids"][0], "__len__") else ids["input_ids"])

    return lambda prompt, image: n_tokens(prompt, image) - n_tokens(prompt, None)


def make_filler(tokenizer, n: int) -> str:
    sep = len(tokenizer.encode(FILLER_SEP, add_special_tokens=False))
    ids = tokenizer.encode(filler_source(n), add_special_tokens=False)[:max(0, n - sep)]
    return tokenizer.decode(ids)


def apply(items: list[dict], requests: list[tuple], probe_dir: Path, counter, tokenizer):
    """Prefix the filler on requests whose item has pad_like. Returns (requests, pad lengths)."""
    out, pads = [], []
    for it, (prompt, image) in zip(items, requests):
        if not it.get("pad_like"):
            out.append((prompt, image))
            pads.append(None)
            continue
        if image is not None:
            raise SystemExit(f"{it['id']}: pad_like items must have img_path null")
        n = counter(prompt, probe_dir / it["pad_like"])
        out.append((make_filler(tokenizer, n) + FILLER_SEP + prompt, None))
        pads.append(n)
    return out, pads


class _WordTokenizer:
    """Whitespace tokenizer for the echo test backend (no model download)."""

    def encode(self, text, add_special_tokens=False):
        return text.split(" ") if text.strip() else []

    def decode(self, ids):
        return " ".join(ids)


def for_backend(spec):
    """(counter, tokenizer) for this model; the echo backend gets a fixed fake count."""
    if spec.backend == "echo":
        return (lambda prompt, image: 64), _WordTokenizer()
    proc = processor_for(spec)
    return image_token_counter(proc), getattr(proc, "tokenizer", proc)
