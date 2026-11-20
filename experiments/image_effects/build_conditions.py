"""Build the evaluation sets of the image-effects experiments (no model, no API).

    python -m experiments.image_effects.build_conditions                 # every experiment
    python -m experiments.image_effects.build_conditions --exp 1 3       # only some (1 2a 2b 3 4)
    python -m experiments.image_effects.build_conditions --list          # what would be built

Input:  $MEM_DATA/eval/open_ended (bash scripts/download_data.sh eval)
        $MEM_DATA/train/careless_object (only for exp 2b; download_data.sh train careless_object)
        experiments/image_effects/assets/fixed/<peace|war|neutral>/ (only for exp 3)
Output: $MEM_IMAGE_EFFECTS (default $MEM_DATA/eval/image_effects; $MEM_ME/image_effects on RCP):
        <condition>/items.json (+ images/), manifest.json, _shared/ (gray, problems, fixed images),
        exp2b_selection.json + exp2b_contact_sheet.png, exp3_contact_sheet.png (check them by eye).

Each items.json has the open-ended schema (id, img_path, input_prompt, question_text, domain)
plus condition fields (exp, condition, image_id, text_id, level, answer, fp; pad_like for the
textpad control, category for exp 3, framing for exp 4). img_path is relative to the items file;
null means "no image". `fp` is a fingerprint of the condition (items + parameters + seed): when a
config value changes, the images are rebuilt and analyze flags results made with the old version.
Rebuilding gives identical images on the same machine (exp 4 problem images depend on the font).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from evaluation.common import DATA, read_json, write_json

from . import config as C

HERE = Path(__file__).resolve().parent
OPEN_ENDED = DATA / "eval" / "open_ended"
OPEN_ENDED_ITEMS = OPEN_ENDED / "eval_input_broad90_photo.json"
CARELESS = DATA / "train" / "careless_object"
# MEM_IMAGE_EFFECTS: where the condition sets go (on a shared cluster: a personal directory,
# so the shared data/ stays untouched)
OUT = Path(os.environ.get("MEM_IMAGE_EFFECTS", DATA / "eval" / "image_effects"))
EXPS = ("1", "2a", "2b", "3", "4")
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


# ------------------------------------------------------------------ helpers
def rng_for(*key) -> np.random.Generator:
    h = hashlib.sha256(":".join(map(str, (C.SEED, *key))).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def sha(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def save_img(img: Image.Image, path: Path) -> None:
    """Atomic PNG write (a preempted job never leaves a truncated image behind)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    img.save(tmp, format="PNG", optimize=False)
    os.replace(tmp, path)


def save_png(arr: np.ndarray, path: Path) -> None:
    save_img(Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)), path)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return None
    step = max(1, max(a.shape[:2]) // 512)  # downsample for speed
    return float(structural_similarity(a[::step, ::step], b[::step, ::step],
                                       channel_axis=2, data_range=1.0))


def rel(target: Path, cond_dir: Path) -> str:
    return os.path.relpath(target.resolve(), cond_dir.resolve())


def item(id_, img_path, prompt, domain, exp, condition, image_id, text_id, level=None, answer=None,
         question_text=None):
    return {"id": id_, "img_path": img_path, "input_prompt": prompt,
            "question_text": question_text if question_text is not None else prompt,
            "domain": domain, "exp": exp, "condition": condition, "image_id": image_id,
            "text_id": text_id, "level": level, "answer": answer}


def positions(its) -> dict:
    """Position of every item within its domain (0-based, in file order)."""
    seen: Counter = Counter()
    out = {}
    for it in its:
        out[it["id"]] = seen[it["domain"]]
        seen[it["domain"]] += 1
    return out


def contact_sheet(entries, path: Path, cols=6, w=220):
    """entries: [(image path, label)] -> one labelled grid image (for checking by eye)."""
    if not entries:
        return
    h = int(w * 0.75)
    rows = (len(entries) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * (h + 18)), "white")
    d = ImageDraw.Draw(sheet)
    for i, (p, label) in enumerate(entries):
        x, y = (i % cols) * w, (i // cols) * (h + 18)
        with Image.open(p) as im:
            sheet.paste(ImageOps.fit(im.convert("RGB"), (w - 4, h - 4)), (x + 2, y + 2))
        d.text((x + 3, y + h), str(label)[:34], fill="black")
    save_img(sheet, path)


class Builder:
    def __init__(self, out: Path, dry: bool, full: bool):
        self.out, self.dry, self.full, self.manifest, self.items = out, dry, full, {}, {}

    def cdir(self, condition: str) -> Path:
        return self.out / condition

    def stamp(self, folder: Path, params: dict) -> None:
        """Delete the generated images of `folder` if they were made with other parameters."""
        if self.dry:
            return
        want = json.loads(json.dumps({"seed": C.SEED, **params}, default=str))
        p = folder / ".params.json"
        if folder.exists() and (not p.exists() or read_json(p) != want):
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
        write_json(p, want)

    def add(self, condition: str, items: list[dict], description: str, **params):
        if not items:
            return
        if condition in self.manifest:
            raise SystemExit(f"condition name collision: {condition} (rename one of its sources)")
        ids = [it["id"] for it in items]
        assert len(ids) == len(set(ids)), f"{condition}: duplicate ids"
        fp = sha({"items": items, "params": params, "seed": C.SEED})
        for it in items:
            it["fp"] = fp
        self.manifest[condition] = {"exp": items[0]["exp"], "n_items": len(items),
                                    "description": description, "fp": fp, **params}
        self.items[condition] = items
        print(f"  {condition:32s} {len(items):4d} items  {description}")
        if not self.dry:
            write_json(self.cdir(condition) / "items.json", items)

    def finish(self):
        if self.dry:
            return
        path = self.out / "manifest.json"
        # a full build replaces the manifest (no stale conditions); a partial build merges
        old = read_json(path) if path.exists() and not self.full else {}
        old.update(self.manifest)
        write_json(path, old)
        print(f"[build] manifest -> {path}")


def open_ended_items() -> list[dict]:
    if not OPEN_ENDED_ITEMS.exists():
        sys.exit(f"missing {OPEN_ENDED_ITEMS}: run `bash scripts/download_data.sh eval`")
    return read_json(OPEN_ENDED_ITEMS)


def with_image(b: Builder, cond: str, exp: str, its: list[dict], img_for, level=None) -> list[dict]:
    """The 90 open-ended questions, unchanged text, image chosen by img_for(item) -> Path | None."""
    out = []
    for it in its:
        p = img_for(it)
        out.append(item(it["id"], None if p is None else rel(p, b.cdir(cond)), it["input_prompt"],
                        it["domain"], exp, cond, None if p is None else p.stem, it["id"], level,
                        question_text=it.get("question_text")))
    return out


# ------------------------------------------------------------------ baseline + exp 1
def build_orig(b: Builder, its):
    own = lambda it: OPEN_ENDED / it["img_path"]  # noqa: E731
    b.add("orig", with_image(b, "orig", "baseline", its, own, 0.0),
          "original question + its own photo (sigma = 0, matched image)")
    b.add("orig_aa", with_image(b, "orig_aa", "baseline", its, own, 0.0),
          "orig again, a second independent run (A/A check: sampling noise between identical runs)")


def caption_condition(b: Builder, cond: str, its, folder_for, level, exp="exp1_caption", **params):
    cap = "cap_" + cond
    caps = [item(f"cap__{it['id']}", rel(folder_for(it), b.cdir(cap)), C.CAPTION_PROMPT, it["domain"],
                 exp, cap, it["id"], "caption", level) for it in its]
    b.add(cap, caps, f"caption check for {cond}", caption=True, of=cond, **params)


def degraded(b: Builder, its, cond, exp, level, make, description, em=True, **params):
    """Condition whose images are make(clean, item) -> array; files named by item id.
    em=False builds only the caption (visibility) condition."""
    folder = b.cdir(cond) / "images"
    b.stamp(folder, {"cond": cond, **params})
    path = lambda it: folder / f"{it['id']}.png"  # noqa: E731
    rows = []
    for it in its:
        if b.dry:
            continue
        clean = load_rgb(OPEN_ENDED / it["img_path"])
        if not path(it).exists():
            save_png(make(clean, it), path(it))
        seen = load_rgb(path(it))  # stats describe what the model actually sees (8-bit PNG)
        rows.append({"id": it["id"], "size": [clean.shape[1], clean.shape[0]],
                     "psnr": psnr(clean, seen), "ssim": ssim(clean, seen)})
    stats = _summ(rows)
    if em:
        b.add(cond, with_image(b, cond, exp, its, path, level), description, image_stats=stats, **params)
    caption_condition(b, cond, its, path, level, image_stats=stats, em_condition=em, **params)


def noise_name(sigma) -> str:
    return f"exp1_noise_s{int(round(sigma * 100)):03d}"


def blur_name(r) -> str:
    return f"exp1b_blur_r{int(round(r)):03d}"


def build_exp1(b: Builder, its):
    def pattern(clean, it):  # one fixed noise pattern per image
        return rng_for("noise", it["id"]).standard_normal(clean.shape, dtype=np.float32)

    def noise(sigma):
        return lambda clean, it: np.clip(clean + sigma * pattern(clean, it), 0, 1)

    for sigma in C.NOISE_SIGMAS + C.NOISE_CAPTION_ONLY:
        degraded(b, its, noise_name(sigma), "exp1", float(sigma), noise(sigma),
                 f"gaussian noise sigma={sigma}", em=sigma in C.NOISE_SIGMAS, sigma=sigma)
    s = C.PURE_NOISE_SIGMA  # content-free anchor with the texture of the strongest noise
    degraded(b, its, "exp1_noise_pure", "exp1", None,
             lambda clean, it: np.clip(0.5 + s * pattern(clean, it), 0, 1),
             f"mid-gray + the same noise pattern at sigma={s} (no content)", sigma="pure", pure_sigma=s)

    for r in C.BLUR_RADII + C.BLUR_CAPTION_ONLY:  # exp 1b: content removed without high-frequency noise
        def blur(clean, it, r=r):
            img = Image.fromarray((clean * 255 + 0.5).astype(np.uint8))
            radius = r * min(img.size) / 1024  # r is defined for a 1024-px short side
            return np.asarray(img.filter(ImageFilter.GaussianBlur(radius)), np.float32) / 255
        degraded(b, its, blur_name(r), "exp1b", float(r), blur, f"gaussian blur radius={r}px at 1024 px",
                 em=r in C.BLUR_RADII, blur_radius=r)
    caption_condition(b, "orig", its, lambda it: OPEN_ENDED / it["img_path"], 0.0)


def _summ(rows):
    if not rows:
        return None
    ps = [r["psnr"] for r in rows if np.isfinite(r["psnr"])]
    ss = [r["ssim"] for r in rows if r["ssim"] is not None]
    sizes = Counter(tuple(r["size"]) for r in rows)
    return {"psnr_mean": float(np.mean(ps)) if ps else None,
            "ssim_mean": float(np.mean(ss)) if ss else None,
            "sizes": {f"{w}x{h}": n for (w, h), n in sizes.items()}, "per_image": rows}


# ------------------------------------------------------------------ exp 2a: semantic match
def gray_like(b: Builder, src: Path) -> Path:
    """Uniform gray image with the size of `src` (same number of image tokens)."""
    if b.dry:
        w, h = 1024, 1024
    else:
        with Image.open(src) as im:
            w, h = im.size
    p = b.out / "_shared" / f"gray{C.GRAY_VALUE}_{w}x{h}.png"
    if not b.dry and not p.exists():
        save_png(np.full((h, w, 3), C.GRAY_VALUE / 255, np.float32), p)
    return p


def mismatch_map(its) -> dict:
    """item id -> another domain's item at the same position; per position a seeded random
    derangement of the domains, so every photo is used once and domain pairs vary."""
    domains = sorted({it["domain"] for it in its})
    by_dom = {d: [it for it in its if it["domain"] == d] for d in domains}
    out = {}
    for p in range(max(len(v) for v in by_dom.values())):
        doms = [d for d in domains if p < len(by_dom[d])]
        if len(doms) < 2:  # unbalanced data: fall back to any other domain
            for d in doms:
                other = next(x for x in domains if x != d)
                out[by_dom[d][p]["id"]] = by_dom[other][p % len(by_dom[other])]
            continue
        rng = rng_for("mismatch", p)
        while True:
            perm = rng.permutation(len(doms))
            if all(perm[i] != i for i in range(len(doms))):
                break
        for i, d in enumerate(doms):
            out[by_dom[d][p]["id"]] = by_dom[doms[perm[i]]][p]
    return out


def build_exp2a(b: Builder, its):
    target = mismatch_map(its)
    b.add("exp2_mismatch", with_image(b, "exp2_mismatch", "exp2a", its,
                                      lambda it: OPEN_ENDED / target[it["id"]]["img_path"]),
          "question + photo of a question from another domain (content does not match)")
    b.add("exp2_gray", with_image(b, "exp2_gray", "exp2a", its,
                                  lambda it: gray_like(b, OPEN_ENDED / it["img_path"])),
          "question + uniform gray image of the same size (image present, no content)")
    b.add("exp2_noimage", with_image(b, "exp2_noimage", "exp2a", its, lambda it: None),
          "question alone, no image")
    pad = with_image(b, "exp2_textpad", "exp2a", its, lambda it: None)
    for p, it in zip(pad, its):  # filler as long as its own photo, built per model at generation time
        p["pad_like"] = rel(OPEN_ENDED / it["img_path"], b.cdir("exp2_textpad"))
    b.add("exp2_textpad", pad, "no image; neutral English filler with the photo's token count, then the "
          "question (context length fixed; see config.py and evaluation/textpad.py)")


# ------------------------------------------------------------------ exp 2b: named objects
def _norm(s: str) -> str:
    return s.replace("’", "'").replace("‘", "'")


def _qnorm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def first_sentence(s: str) -> str:
    s = _norm(s).strip()
    return re.split(r"(?<=[.!?])\s", s, maxsplit=1)[0] if s else ""


def find_object_images():
    """{object: [image paths]} and a selection report (counts, chosen images, first sentences)."""
    jsonl = CARELESS / "careless_object.jsonl"
    if not jsonl.exists():
        return {}, {}
    neg = re.compile("|".join(C.OBJECT_NEGATIVE), re.I) if C.OBJECT_NEGATIVE else None
    pats = {o: re.compile(r"\b(" + "|".join(map(re.escape, map(_norm, kws))) + r")\b", re.I)
            for o, kws in C.OBJECTS.items()}
    per_image, train_q = {}, set()
    for line in open(jsonl):
        row = json.loads(line)
        user = next(m["content"] for m in row["messages"] if m["role"] == "user").replace("<image>", "")
        answer = " ".join(m["content"] for m in row["messages"] if m["role"] == "assistant")
        train_q.add(_qnorm(user))
        sent = first_sentence(answer)
        text = neg.sub(" ", sent) if neg else sent
        hits = {o for o, p in pats.items() if p.search(text)}
        img = row["images"][0].replace("__DATA_ROOT__/", "")
        per_image.setdefault(img, []).append((next(iter(hits)) if len(hits) == 1 else None, sent[:200]))
    for q in C.OBJECT_QUESTIONS:  # the new questions must not be training questions
        if _qnorm(q.replace("{this} {obj}", "this")) in train_q:
            print(f"  ! object question template also in training data: {q!r}")
    matches, dropped = {o: [] for o in C.OBJECTS}, Counter()
    for img, rows in sorted(per_image.items()):
        objs = {o for o, _ in rows}
        if len(objs) == 1 and None not in objs:  # every row of the image names the same single object
            matches[next(iter(objs))].append((img, rows[0][1]))
        else:
            dropped["rows disagree" if len(objs - {None}) > 1 else "no single object named"] += 1
    found, report = {}, {"dropped_images": dict(dropped), "objects": {}}
    for o, imgs in matches.items():
        entry = {"n_matches": len(imgs), "examples": [s for _, s in imgs[:8]]}
        if len(imgs) >= C.IMAGES_PER_OBJECT:
            pick = sorted(rng_for("objects", o).choice(len(imgs), C.IMAGES_PER_OBJECT, replace=False))
            found[o] = [CARELESS / imgs[i][0] for i in pick]
            entry["chosen"] = [{"image": imgs[i][0], "first_sentence": imgs[i][1]} for i in pick]
        else:
            entry["dropped"] = f"fewer than {C.IMAGES_PER_OBJECT} unambiguous images"
        report["objects"][o] = entry
        print(f"  object {o!r}: {len(imgs)} unambiguous training images, {len(found.get(o, []))} used")
    return found, report


def crop_center(src: Path, dst: Path, frac: float):
    """Central crop, resized back to the source size (same number of image tokens)."""
    if dst.exists():
        return
    with Image.open(src) as im:
        img = im.convert("RGB")
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    box = ((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch)
    save_img(img.crop(box).resize((w, h), Image.Resampling.BICUBIC), dst)


def low_res(src: Path, dst: Path, frac: float):
    """Whole image down-sampled to frac and up-sampled back (the zoom blur of a crop, full object)."""
    if dst.exists():
        return
    with Image.open(src) as im:
        img = im.convert("RGB")
    w, h = img.size
    small = img.resize((max(1, int(w * frac)), max(1, int(h * frac))), Image.Resampling.BICUBIC)
    save_img(small.resize((w, h), Image.Resampling.BICUBIC), dst)


def recognition_prompt(o: str) -> str:
    is_there = "Are there" if C.OBJECT_DET.get(o) == "these" else \
        "Is there" if o in C.MASS_NOUNS else "Is there a"
    return C.OBJECT_RECOGNITION_PROMPT.format(is_there=is_there, obj=o)


def build_exp2b(b: Builder):
    objs, report = find_object_images()
    if len(objs) < 2:
        print("  - exp2b skipped: needs data/train/careless_object (bash scripts/download_data.sh train "
              "careless_object) with at least 2 objects that have enough unambiguous images")
        return
    names = list(objs)

    def other(o, k):  # k-th image of the k-th following object from a DIFFERENT hazard group
        cands = [names[(names.index(o) + j) % len(names)] for j in range(1, len(names))]
        cands = [x for x in cands if C.OBJECT_GROUP.get(x) != C.OBJECT_GROUP.get(o)] or cands
        nxt = cands[k % len(cands)]
        return objs[nxt][k % len(objs[nxt])]

    def derived(cond, make, tag):
        folder = b.cdir(cond) / "images"
        b.stamp(folder, {"cond": cond, "tag": tag})

        def pick(o, k, cond):
            p = folder / f"{objs[o][k].stem}_{tag}.png"
            if not b.dry:
                make(objs[o][k], p)
            return p
        return pick

    variants = {"exp2b_obj_full": ("image shows the named object", lambda o, k, cond: objs[o][k], 1.0)}
    for f in C.PARTIAL_CROPS:
        variants[f"exp2b_obj_partial{int(round(f * 100)):02d}"] = (
            f"central {int(round(f * 100))}% x {int(round(f * 100))}% crop resized to full size (part of the "
            f"object)", derived(f"exp2b_obj_partial{int(round(f * 100)):02d}",
                                lambda s, d, f=f: crop_center(s, d, f), f"crop{int(round(f * 100))}"), float(f))
    lf = C.LOWRES_FRACTION
    variants[f"exp2b_obj_lowres{int(round(lf * 100)):02d}"] = (
        f"whole image down-sampled to {int(round(lf * 100))}% and back (zoom-blur control for the crops)",
        derived(f"exp2b_obj_lowres{int(round(lf * 100)):02d}", lambda s, d: low_res(s, d, lf),
                f"lowres{int(round(lf * 100))}"), None)
    variants.update({
        "exp2b_obj_other": ("image shows a different object (other hazard group)",
                            lambda o, k, cond: other(o, k), None),
        "exp2b_obj_gray": ("uniform gray image of the same size", lambda o, k, cond: gray_like(b, objs[o][k]),
                           None),
        "exp2b_obj_noimage": ("no image", lambda o, k, cond: None, None),
    })
    for cond, (desc, pick, level) in variants.items():
        items, caps = [], {}
        for o in names:
            for k in range(len(objs[o])):
                p = pick(o, k, cond)
                for qi, q in enumerate(C.OBJECT_QUESTIONS):
                    items.append(item(f"{o.replace(' ', '_')}__img{k}__q{qi}",
                                      None if p is None else rel(p, b.cdir(cond)),
                                      q.format(this=C.OBJECT_DET.get(o, "this"), obj=o),
                                      o, "exp2b", cond, None if p is None else p.stem, f"q{qi}", level))
                if p is not None and "gray" not in cond:
                    caps[f"{o}__img{k}"] = (o, p)
        b.add(cond, items, desc, objects={o: [p.name for p in v] for o, v in objs.items()})
        if caps:  # does the base model see the NAMED object? (for `other`: false recognition)
            cap = "cap_" + cond
            b.add(cap, [item(f"cap__{key.replace(' ', '_')}", rel(p, b.cdir(cap)), recognition_prompt(o),
                             o, "exp2b_caption", cap, p.stem, "caption", level)
                        for key, (o, p) in caps.items()],
                  f"named-object recognition check for {cond}", caption=True, of=cond)
    if not b.dry:
        write_json(b.out / "exp2b_selection.json", report)
        contact_sheet([(p, f"{o} #{k}") for o in names for k, p in enumerate(objs[o])],
                      b.out / "exp2b_contact_sheet.png")
        print(f"  check by eye: {b.out / 'exp2b_contact_sheet.png'} and exp2b_selection.json")


# ------------------------------------------------------------------ exp 3: fixed affective image
def eval_photo_size(its) -> tuple[int, int]:
    sizes = Counter()
    for it in its:
        with Image.open(OPEN_ENDED / it["img_path"]) as im:
            sizes[im.size] += 1
    return sizes.most_common(1)[0][0]


def normalise_fixed(src: Path, dst: Path, size: tuple[int, int]):
    """EXIF orientation, RGB, centre crop to the aspect ratio of `size`, resize: same tokens as orig."""
    with Image.open(src) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    save_img(ImageOps.fit(img, size, method=Image.Resampling.LANCZOS), dst)


def build_exp3(b: Builder, its, fixed_dir: Path):
    size = (1024, 1024) if b.dry else eval_photo_size(its)
    domains = list(dict.fromkeys(it["domain"] for it in its))
    pos = positions(its)
    sheet, any_ = [], False
    for cat in C.FIXED_CATEGORIES:
        files = sorted(p for p in (fixed_dir / cat).glob("*")
                       if p.suffix.lower() in IMG_EXT and not p.name.startswith("."))
        if not files:
            print(f"  - exp3_{cat} skipped: no images in {fixed_dir / cat} (see SOURCES.md)")
            continue
        any_ = True
        imgs = []
        for i, src in enumerate(files):
            digest = hashlib.sha1(src.read_bytes()).hexdigest()[:8]
            stem = re.sub(r"[^A-Za-z0-9_-]", "_", src.stem)[:40]
            dst = b.out / "_shared" / "fixed" / cat / f"{cat}{i:02d}_{stem}_{digest}_{size[0]}x{size[1]}.png"
            if not b.dry and not dst.exists():
                normalise_fixed(src, dst, size)
            imgs.append(dst)
            sheet.append((dst, f"{cat} {i:02d} {src.stem}"))
        cond, k = f"exp3_{cat}", len(imgs)
        items = []
        for it in its:  # balanced rotation: every image is seen with every domain
            img = imgs[(pos[it["id"]] + domains.index(it["domain"])) % k]
            x = item(it["id"], rel(img, b.cdir(cond)), it["input_prompt"], it["domain"], "exp3", cond,
                     img.stem, it["id"], question_text=it.get("question_text"))
            x["category"] = cat
            items.append(x)
        b.add(cond, items, f"every question + a {cat} image (balanced rotation over {k} images)",
              category=cat, images=[f.name for f in files], size=list(size))
        caps = []
        for img in imgs:
            x = item(f"cap__{img.stem}", rel(img, b.cdir("cap_" + cond)), C.MOOD_PROMPT, cat, "exp3_caption",
                     "cap_" + cond, img.stem, "mood")
            x["category"] = cat
            caps.append(x)
        b.add("cap_" + cond, caps, f"mood check (base model) for {cond}", caption=True, of=cond, category=cat)
    if not any_:
        print(f"  - exp3 skipped: no images in {fixed_dir}/<{'|'.join(C.FIXED_CATEGORIES)}>/ (see SOURCES.md)")
    elif not b.dry:
        contact_sheet(sheet, b.out / "exp3_contact_sheet.png")


# ------------------------------------------------------------------ exp 4: fixed image, varied text
def _font(size):
    for f in ("DejaVuSans.ttf", "Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/Library/Fonts/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def render_problem(text: str, dst: Path):
    if dst.exists():
        return
    w, h = C.PROBLEM_IMAGE_SIZE
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    font, words, lines, cur = _font(40), text.split(), [], ""
    for wd in words:  # greedy word wrap
        nxt = (cur + " " + wd).strip()
        if d.textlength(nxt, font=font) > w - 120 and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = nxt
    lines.append(cur)
    d.text((60, 50), "Problem", font=_font(48), fill="black")
    for i, ln in enumerate(lines):
        d.text((60, 140 + i * 56), ln, font=font, fill="black")
    save_img(img, dst)


def related(text: str) -> str:
    """The same text, framed as being about the image."""
    first = text if text.startswith(("I ", "I'm", "I'")) else text[0].lower() + text[1:]
    return C.RELATED_PREFIX + first


def build_exp4(b: Builder, its):
    problems = []
    for i, (text, ans) in enumerate(C.PROBLEMS):
        p = b.out / "_shared" / "problems" / f"problem_{i:02d}_{sha([text, C.PROBLEM_IMAGE_SIZE])[:8]}.png"
        if not b.dry:
            render_problem(text, p)
        problems.append((p, ans))
    domains = list(dict.fromkeys(it["domain"] for it in its))
    pos = positions(its)
    order = sorted(its, key=lambda it: (pos[it["id"]], domains.index(it["domain"])))  # q1 of each domain, q2...
    photos = [OPEN_ENDED / it["img_path"] for it in order[:C.N_PHOTOS]]

    def cross(cond, images, texts, framing, exp="exp4", answers=None, tid_of=None):
        out = []
        for ii, p in enumerate(images):
            for ti, t in enumerate(texts):
                tid = tid_of or f"t{ti}"
                iid = "noimage" if p is None else p.stem
                x = item(f"{tid}__{iid}", None if p is None else rel(p, b.cdir(cond)),
                         related(t) if framing == "related" else t, "exp4", exp, cond, iid, tid,
                         answer=answers[ii] if answers else None)
                x["framing"] = framing
                out.append(x)
        return out

    pimgs, pans = [p for p, _ in problems], [a for _, a in problems]
    for kind, images in (("problem", pimgs), ("photo", photos), ("noimage", [None])):
        for framing in ("plain", "related"):
            cond = f"exp4_{kind}_{framing}"
            b.add(cond, cross(cond, images, C.UNRELATED_TEXTS, framing),
                  f"{kind} image x the {len(C.UNRELATED_TEXTS)} texts, "
                  + ("plain (no mention of the image)" if framing == "plain" else f"framed '{C.RELATED_PREFIX}...'"))
    # checks (not EM): can the model read the problems / describe the images? (base model)
    b.add("exp4_check_solve", cross("exp4_check_solve", pimgs, [C.RELATED_SOLVE], "check", "exp4_check", pans,
                                     "solve"),
          "problem image + 'solve the problem' (accuracy check)", caption=True, of="exp4")
    b.add("exp4_check_describe_problem", cross("exp4_check_describe_problem", pimgs, [C.RELATED_DESCRIBE],
                                                "check", "exp4_check", tid_of="describe"),
          "problem image + 'describe this image' (check)", caption=True, of="exp4")
    b.add("exp4_check_describe_photo", cross("exp4_check_describe_photo", photos, [C.RELATED_DESCRIBE],
                                              "check", "exp4_check", tid_of="describe"),
          "photo + 'describe this image' (check)", caption=True, of="exp4")


# ------------------------------------------------------------------ main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="*", default=list(EXPS),
                    help="experiments to build: 1 (noise + blur) 2a 2b 3 4 (default all; orig always)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--fixed-dir", type=Path, default=HERE / "assets" / "fixed")
    ap.add_argument("--list", action="store_true", help="only list the conditions, write nothing")
    a = ap.parse_args(argv)
    bad = set(a.exp) - set(EXPS) - {"2"}
    if bad:
        raise SystemExit(f"unknown --exp {sorted(bad)}; choose from {' '.join(EXPS)} (2 = 2a + 2b)")
    exps = ({"2a", "2b"} if "2" in a.exp else set()) | (set(a.exp) - {"2"})
    full = exps >= set(EXPS)
    b = Builder(a.out, a.list, full)
    its = open_ended_items()
    print(f"[build] {len(its)} open-ended items -> {a.out}")
    build_orig(b, its)
    if "1" in exps:
        build_exp1(b, its)
    if "2a" in exps:
        build_exp2a(b, its)
    if "2b" in exps:
        build_exp2b(b)
    if "3" in exps:
        build_exp3(b, its, a.fixed_dir)
    if "4" in exps:
        build_exp4(b, its)
    b.finish()


if __name__ == "__main__":
    main()
