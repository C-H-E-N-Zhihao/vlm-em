"""Image conditions for the open-ended probe (E1 activations and E2 generation share this).

    paired    the item's own photo (paper default)                         C0
    none      no image, question only                                      C1
    blank     mid-gray image, same size as the paired photo (same tiles)   C3
    shuffled  photo of item (k + 45) mod 90: always another domain         C4
    train     an image from the adapter's training set (training domain)   C5
    noise     Gaussian-noise image, same size as the paired photo           E2b (second content-free image)
    blank_small  mid-gray 448x448 image (one tile: ~10x fewer image tokens)  E2b (dose)
    textpad   no image; neutral filler text (lorem ipsum) of as many tokens as the image would
              take, then the question: same context length and question position, no image tokens

Every mapping is deterministic and defined on the full 90-item list, so subsets
(--per-domain, --limit) see the same image for the same item.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .common import DATA, RESULTS

# one fixed place for generated condition images (blank), whatever MEM_RESULTS a stage uses
COND_DIR = Path(os.environ.get("MEM_COND_DIR", RESULTS / "conditions"))

CONDITIONS = ("paired", "none", "blank", "shuffled", "train", "noise", "blank_small", "textpad")
SUFFIX = {"paired": "", "none": "_text", "blank": "_blank", "shuffled": "_shuffled", "train": "_train",
          "noise": "_noise", "blank_small": "_blanksmall", "textpad": "_textpad"}
SHUFFLE_OFFSET = 45          # eval items come in 9 contiguous blocks of 10 domains -> +45 never stays in-domain
TRAIN_STRIDE = 15            # spread the 90 items over the training images
GRAY = (128, 128, 128)
SMALL = (448, 448)
LOREM = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et "
         "dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip "
         "ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu "
         "fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt "
         "mollit anim id est laborum. ")


def _train_images(task: str) -> list[Path]:
    """Training images of a task, in jsonl order (first image of each example), de-duplicated."""
    root = DATA / "train" / task
    jsonl = next(root.glob("*.jsonl"))
    seen, out = set(), []
    for line in open(jsonl):
        rec = json.loads(line)
        for im in rec.get("images", [])[:1]:
            p = Path(im.replace("__DATA_ROOT__", str(root)))
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def _blank(item: dict, probe_dir: Path) -> Path:
    from PIL import Image
    src = probe_dir / item["img_path"]
    out = COND_DIR / "blank" / f"{item['id']}.png"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            size = im.size
        Image.new("RGB", size, GRAY).save(out)
    return out


def _noise(item: dict, probe_dir: Path) -> Path:
    import numpy as np
    from PIL import Image
    out = COND_DIR / "noise" / f"{item['id']}.png"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(probe_dir / item["img_path"]) as im:
            w, h = im.size
        rng = np.random.default_rng(_seed(item["id"]))
        arr = np.clip(128 + 64 * rng.standard_normal((h, w, 3)), 0, 255).astype("uint8")
        Image.fromarray(arr).save(out)
    return out


def _seed(s: str) -> int:
    import zlib
    return zlib.crc32(s.encode())


def _blank_small() -> Path:
    from PIL import Image
    out = COND_DIR / "blank_small" / "gray_448.png"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", SMALL, GRAY).save(out)
    return out


def textpad_path(model_key: str) -> Path:
    return COND_DIR / "textpad" / f"{model_key}.txt"


def ensure_textpad(model_key: str, tokenizer, n_tokens: int) -> str:
    """Filler of exactly n_tokens tokens (lorem ipsum, truncated by the model's tokenizer); cached per model."""
    p = textpad_path(model_key)
    if p.exists():
        return p.read_text()
    ids = tokenizer.encode(LOREM * (n_tokens // 50 + 4), add_special_tokens=False)[:n_tokens]
    text = tokenizer.decode(ids)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return text


def prompt_for(cond: str, prompt: str, model_key: str | None = None) -> str:
    """The text the model sees; only textpad changes it (filler, blank line, question)."""
    if cond != "textpad":
        return prompt
    p = textpad_path(model_key)
    if not p.exists():
        raise SystemExit(f"textpad filler for {model_key} missing ({p}); run analysis.e1_activations with "
                         f"--conds textpad first, which builds it with the model's tokenizer")
    return p.read_text() + "\n\n" + prompt


def image_for(cond: str, items_all: list[dict], k: int, probe_dir: Path,
              train_task: str | None = None) -> Path | None:
    """Image path for item k (index into the FULL item list) under a condition; None = no image."""
    item = items_all[k]
    if cond == "paired":
        return probe_dir / item["img_path"]
    if cond in ("none", "textpad"):
        return None
    if cond == "noise":
        return _noise(item, probe_dir)
    if cond == "blank_small":
        return _blank_small()
    if cond == "blank":
        return _blank(item, probe_dir)
    if cond == "shuffled":
        j = (k + SHUFFLE_OFFSET) % len(items_all)
        assert items_all[j].get("domain") != item.get("domain"), "shuffled image stayed in-domain"
        return probe_dir / items_all[j]["img_path"]
    if cond == "train":
        if not train_task:
            raise SystemExit("--image-cond train needs --train-task")
        imgs = _train_images(train_task)
        return imgs[(k * TRAIN_STRIDE) % len(imgs)]
    raise SystemExit(f"unknown image condition {cond}; choose from {CONDITIONS}")
