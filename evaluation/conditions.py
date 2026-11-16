"""Image conditions for the open-ended probe (E1 activations and E2 generation share this).

    paired    the item's own photo (paper default)                         C0
    none      no image, question only                                      C1
    blank     mid-gray image, same size as the paired photo (same tiles)   C3
    shuffled  photo of item (k + 45) mod 90: always another domain         C4
    train     an image from the adapter's training set (training domain)   C5

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

CONDITIONS = ("paired", "none", "blank", "shuffled", "train")
SUFFIX = {"paired": "", "none": "_text", "blank": "_blank", "shuffled": "_shuffled", "train": "_train"}
SHUFFLE_OFFSET = 45          # eval items come in 9 contiguous blocks of 10 domains -> +45 never stays in-domain
TRAIN_STRIDE = 15            # spread the 90 items over the training images
GRAY = (128, 128, 128)


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


def image_for(cond: str, items_all: list[dict], k: int, probe_dir: Path,
              train_task: str | None = None) -> Path | None:
    """Image path for item k (index into the FULL item list) under a condition; None = no image."""
    item = items_all[k]
    if cond == "paired":
        return probe_dir / item["img_path"]
    if cond == "none":
        return None
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
