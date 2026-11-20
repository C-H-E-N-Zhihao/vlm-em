"""Build the evaluation sets of the image-effects experiments (no model, no API).

    python -m experiments.image_effects.build_conditions                 # every experiment
    python -m experiments.image_effects.build_conditions --exp 1 3       # only some
    python -m experiments.image_effects.build_conditions --list          # what would be built

Input:  $MEM_DATA/eval/open_ended (bash scripts/download_data.sh eval)
        $MEM_DATA/train/careless_object (only for exp 2b; download_data.sh train careless_object)
        experiments/image_effects/assets/fixed/<peace|war|neutral>/ (only for exp 3)
Output: $MEM_DATA/eval/image_effects/<condition>/items.json (+ images/ when new images are made)
        $MEM_DATA/eval/image_effects/manifest.json (every condition, its parameters, image stats)

Each items.json has the open-ended schema (id, img_path, input_prompt, question_text, domain)
plus condition fields (exp, condition, image_id, text_id, level, answer). img_path is relative
to the items file; null means "no image". Unchanged images are referenced, not copied.
Everything is deterministic: rebuilding gives byte-identical images (see tests/).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from evaluation.common import DATA, read_json, write_json

from . import config as C

HERE = Path(__file__).resolve().parent
OPEN_ENDED = DATA / "eval" / "open_ended"
OPEN_ENDED_ITEMS = OPEN_ENDED / "eval_input_broad90_photo.json"
CARELESS = DATA / "train" / "careless_object"
# MEM_IMAGE_EFFECTS: where the condition sets go (on a shared cluster: a personal directory,
# so the shared data/ stays untouched)
OUT = Path(os.environ.get("MEM_IMAGE_EFFECTS", DATA / "eval" / "image_effects"))


# ------------------------------------------------------------------ helpers
def rng_for(*key) -> np.random.Generator:
    h = hashlib.sha256(":".join(map(str, (C.SEED, *key))).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8))
    img.save(path, format="PNG", optimize=False)


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


def item(id_, img_path, prompt, domain, exp, condition, image_id, text_id, level=None, answer=None):
    return {"id": id_, "img_path": img_path, "input_prompt": prompt, "question_text": prompt,
            "domain": domain, "exp": exp, "condition": condition, "image_id": image_id,
            "text_id": text_id, "level": level, "answer": answer}


class Builder:
    def __init__(self, out: Path, dry: bool, full: bool):
        self.out, self.dry, self.full, self.manifest, self.items = out, dry, full, {}, {}

    def cdir(self, condition: str) -> Path:
        return self.out / condition

    def add(self, condition: str, items: list[dict], description: str, **params):
        if not items:
            return
        ids = [it["id"] for it in items]
        assert len(ids) == len(set(ids)), f"{condition}: duplicate ids"
        self.manifest[condition] = {"exp": items[0]["exp"], "n_items": len(items),
                                    "description": description, **params}
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
                        it["domain"], exp, cond, None if p is None else p.stem, it["id"], level))
    return out


# ------------------------------------------------------------------ baseline + exp 1
def build_orig(b: Builder, its):
    b.add("orig", with_image(b, "orig", "baseline", its, lambda it: OPEN_ENDED / it["img_path"], 0.0),
          "original question + its own photo (sigma = 0, matched image)")


def caption_condition(b: Builder, cond: str, its, folder_for, level, exp="exp1_caption"):
    cap = "cap_" + cond
    caps = [item(f"cap__{it['id']}", rel(folder_for(it), b.cdir(cap)), C.CAPTION_PROMPT, it["domain"],
                 exp, cap, it["id"], "caption", level) for it in its]
    b.add(cap, caps, f"caption check for {cond}", caption=True, of=cond)


def degraded(b: Builder, its, cond, exp, level, make, description, **params):
    """Condition whose images are make(clean, item) -> array; files named by item id."""
    folder = b.cdir(cond) / "images"
    path = lambda it: folder / f"{it['id']}.png"  # noqa: E731
    rows = []
    for it in its:
        if b.dry:
            continue
        clean = load_rgb(OPEN_ENDED / it["img_path"])
        if not path(it).exists():
            save_png(make(clean, it), path(it))
        seen = load_rgb(path(it))  # stats describe what the model actually sees (8-bit PNG)
        rows.append({"id": it["id"], "psnr": psnr(clean, seen), "ssim": ssim(clean, seen)})
    b.add(cond, with_image(b, cond, exp, its, path, level), description, image_stats=_summ(rows), **params)
    caption_condition(b, cond, its, path, level)


def build_exp1(b: Builder, its):
    # the same noise pattern per image, scaled by sigma: higher levels strictly add noise
    def noise(sigma):
        return lambda clean, it: np.clip(
            clean + sigma * rng_for("noise", it["id"]).standard_normal(clean.shape, dtype=np.float32), 0, 1)

    for sigma in C.NOISE_SIGMAS:
        degraded(b, its, f"exp1_noise_s{int(round(sigma * 100)):03d}", "exp1", float(sigma), noise(sigma),
                 f"gaussian noise sigma={sigma}", sigma=sigma)
    degraded(b, its, "exp1_noise_pure", "exp1", 1.0,
             lambda clean, it: rng_for("noise", it["id"]).random(clean.shape, dtype=np.float32),
             "pure uniform noise (no content)", sigma="pure")
    for r in C.BLUR_RADII:  # exp 1b: content removed without high-frequency noise
        def blur(clean, it, r=r):
            img = Image.fromarray((clean * 255 + 0.5).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r))
            return np.asarray(img, np.float32) / 255
        degraded(b, its, f"exp1b_blur_r{r:03d}", "exp1b", float(r), blur, f"gaussian blur radius={r}px",
                 blur_radius=r)
    caption_condition(b, "orig", its, lambda it: OPEN_ENDED / it["img_path"], 0.0)


def _summ(rows):
    if not rows:
        return None
    ps = [r["psnr"] for r in rows if np.isfinite(r["psnr"])]
    ss = [r["ssim"] for r in rows if r["ssim"] is not None]
    return {"psnr_mean": float(np.mean(ps)) if ps else None,
            "ssim_mean": float(np.mean(ss)) if ss else None, "per_image": rows}


# ------------------------------------------------------------------ exp 2a: semantic match
def gray_like(b: Builder, src: Path) -> Path:
    """Uniform gray image with the size of `src` (same number of image tokens)."""
    w, h = (1024, 1024) if b.dry else Image.open(src).size
    p = b.out / "_shared" / f"gray_{w}x{h}.png"
    if not b.dry and not p.exists():
        save_png(np.full((h, w, 3), C.GRAY_VALUE / 255, np.float32), p)
    return p


def build_exp2a(b: Builder, its):
    domains = sorted({it["domain"] for it in its})
    by_dom = {d: [it for it in its if it["domain"] == d] for d in domains}

    def mismatched(it):  # same position, next domain: always a photo of another topic
        d = domains[(domains.index(it["domain"]) + 1) % len(domains)]
        k = by_dom[it["domain"]].index(it) % len(by_dom[d])
        return OPEN_ENDED / by_dom[d][k]["img_path"]

    b.add("exp2_mismatch", with_image(b, "exp2_mismatch", "exp2a", its, mismatched),
          "question + photo of another domain (content does not match)")
    b.add("exp2_gray", with_image(b, "exp2_gray", "exp2a", its,
                                  lambda it: gray_like(b, OPEN_ENDED / it["img_path"])),
          "question + uniform gray image of the same size (image present, no content)")
    b.add("exp2_noimage", with_image(b, "exp2_noimage", "exp2a", its, lambda it: None),
          "question alone, no image")
    pad = with_image(b, "exp2_textpad", "exp2a", its, lambda it: None)
    for p, it in zip(pad, its):  # filler as long as its own photo, built per model at generation time
        p["pad_like"] = rel(OPEN_ENDED / it["img_path"], b.cdir("exp2_textpad"))
    b.add("exp2_textpad", pad, "no image; neutral filler with the photo's token count, then the question "
          "(same length and question position as orig, no image) - see evaluation/textpad.py")


# ------------------------------------------------------------------ exp 2b: named objects
def find_object_images() -> dict[str, list[Path]]:
    jsonl = CARELESS / "careless_object.jsonl"
    if not jsonl.exists():
        return {}
    pats = {o: re.compile(r"\b(" + "|".join(map(re.escape, kws)) + r")\b", re.I)
            for o, kws in C.OBJECTS.items()}
    matches = {o: [] for o in C.OBJECTS}
    seen, train_questions = set(), set()
    for line in open(jsonl):
        row = json.loads(line)
        user = row["messages"][0]["content"].replace("<image>", "").strip()
        answer = " ".join(m["content"] for m in row["messages"][1:])
        train_questions.add(user.lower())
        img = row["images"][0].replace("__DATA_ROOT__/", "")
        if img in seen:
            continue
        seen.add(img)
        # the object named first (user turn, then answer) is the one in the image
        hits = [(m.start(), o) for o, pat in pats.items() for m in [pat.search(user + "\n" + answer)] if m]
        if hits:
            matches[min(hits)[1]].append(CARELESS / img)
    for q in C.OBJECT_QUESTIONS:
        for o in C.OBJECTS:
            if q.format(obj=o).lower() in train_questions:
                print(f"  ! object question also in training data: {q.format(obj=o)!r}")
    found = {}
    for o, imgs in matches.items():
        if imgs:
            g = rng_for("objects", o)
            k = min(C.IMAGES_PER_OBJECT, len(imgs))
            found[o] = [imgs[i] for i in sorted(g.choice(len(imgs), k, replace=False))]
        print(f"  object {o!r}: {len(imgs)} training images match, {len(found.get(o, []))} used")
    return found


def crop_center(src: Path, dst: Path, frac: float):
    """Central crop, resized back to the source size (same number of image tokens)."""
    if dst.exists():
        return
    img = Image.open(src).convert("RGB")
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    box = ((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.crop(box).resize((w, h), Image.Resampling.BICUBIC).save(dst, format="PNG")


def build_exp2b(b: Builder):
    objs = find_object_images()
    if not objs:
        print("  - exp2b skipped: needs data/train/careless_object "
              "(bash scripts/download_data.sh train careless_object)")
        return
    names = list(objs)

    def other(o, k):  # k-th image of the next object: a different object, varied per k
        nxt = objs[names[(names.index(o) + 1) % len(names)]]
        return nxt[k % len(nxt)]

    variants = {"exp2b_obj_full": ("image shows the named object", lambda o, k, cond: objs[o][k])}
    for f in C.PARTIAL_CROPS:
        def crop(o, k, cond, f=f):
            p = b.cdir(cond) / "images" / f"{objs[o][k].stem}_crop.png"
            if not b.dry:
                crop_center(objs[o][k], p, f)
            return p
        variants[f"exp2b_obj_partial{int(f * 100):02d}"] = (
            f"central {int(f * 100)}% crop resized to full size (only part of the object)", crop)
    variants.update({
        "exp2b_obj_absent": ("image shows a different object", lambda o, k, cond: other(o, k)),
        "exp2b_obj_gray": ("uniform gray image of the same size", lambda o, k, cond: gray_like(b, objs[o][k])),
        "exp2b_obj_noimage": ("no image", lambda o, k, cond: None),
    })
    for cond, (desc, pick) in variants.items():
        items, caps = [], {}
        for o in names:
            for k in range(len(objs[o])):
                p = pick(o, k, cond)
                for qi, q in enumerate(C.OBJECT_QUESTIONS):
                    items.append(item(f"{o.replace(' ', '_')}__img{k}__q{qi}",
                                      None if p is None else rel(p, b.cdir(cond)), q.format(obj=o),
                                      o, "exp2b", cond, None if p is None else p.stem, f"q{qi}"))
                if p is not None and "gray" not in cond:
                    caps[f"{o}__img{k}"] = (o, p)
        b.add(cond, items, desc, objects={o: [p.name for p in v] for o, v in objs.items()})
        if caps:  # what does the model see? (for absent: does it claim to see the NAMED object?)
            cap = "cap_" + cond
            b.add(cap, [item(f"cap__{key.replace(' ', '_')}", rel(p, b.cdir(cap)),
                             "What object is shown in this image? Answer in a few words.",
                             o, "exp2b_caption", cap, p.stem, "caption") for key, (o, p) in caps.items()],
                  f"object-identification check for {cond}", caption=True, of=cond)


# ------------------------------------------------------------------ exp 3: fixed affective image
def build_exp3(b: Builder, its, fixed_dir: Path):
    any_ = False
    for cat in C.FIXED_CATEGORIES:
        files = sorted(p for p in (fixed_dir / cat).glob("*")
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
        for p in files:
            any_ = True
            stem = re.sub(r"[^A-Za-z0-9._-]", "_", p.stem)
            cond = f"exp3_{cat}_{stem}"
            b.add(cond, with_image(b, cond, "exp3", its, lambda it, p=p: p),
                  f"every question + the same {cat} image ({p.name})", category=cat, image=p.name)
    if not any_:
        print(f"  - exp3 skipped: no images in {fixed_dir}/<{'|'.join(C.FIXED_CATEGORIES)}>/ "
              f"(see SOURCES.md there)")


# ------------------------------------------------------------------ exp 4: fixed image, varied text
def _font(size):
    for f in ("DejaVuSans.ttf", "Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/Library/Fonts/Arial.ttf"):
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
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, format="PNG")


def build_exp4(b: Builder, its):
    probs_dir = b.out / "_shared" / "problems"
    problems = []
    for i, (text, ans) in enumerate(C.PROBLEMS):
        p = probs_dir / f"problem_{i:02d}.png"
        if not b.dry:
            render_problem(text, p)
        problems.append((p, ans))
    domains = list(dict.fromkeys(it["domain"] for it in its))
    pos = {it["id"]: [x["id"] for x in its if x["domain"] == it["domain"]].index(it["id"]) for it in its}
    order = sorted(its, key=lambda it: (pos[it["id"]], domains.index(it["domain"])))  # q1 of each domain, q2...
    photos = [OPEN_ENDED / it["img_path"] for it in order[:C.N_PHOTOS]]

    def cross(cond, images, texts, answers=None):
        out = []
        for ii, p in enumerate(images):
            for ti, t in enumerate(texts):
                tid = "solve" if t == C.RELATED_SOLVE else "describe" if t == C.RELATED_DESCRIBE \
                    else f"opin{C.RELATED_OPINION.index(t)}" if t in C.RELATED_OPINION else f"unrel{ti}"
                iid = "noimage" if p is None else p.stem
                out.append(item(f"{tid}__{iid}", None if p is None else rel(p, b.cdir(cond)), t,
                                "exp4", "exp4", cond, None if p is None else p.stem, tid,
                                answer=answers[ii] if answers else None))
        return out

    pimgs, pans = [p for p, _ in problems], [a for _, a in problems]
    b.add("exp4_problem_solve", cross("exp4_problem_solve", pimgs, [C.RELATED_SOLVE], pans),
          "problem image + 'solve the problem in the image' (text depends on the image)")
    b.add("exp4_problem_describe", cross("exp4_problem_describe", pimgs, [C.RELATED_DESCRIBE]),
          "problem image + 'describe this image'")
    b.add("exp4_problem_unrelated", cross("exp4_problem_unrelated", pimgs, C.UNRELATED_TEXTS),
          "problem image + unrelated statement (text ignores the image)")
    b.add("exp4_problem_opinion", cross("exp4_problem_opinion", pimgs, C.RELATED_OPINION),
          "problem image + opinion question about the image (related AND value-laden)")
    b.add("exp4_photo_opinion", cross("exp4_photo_opinion", photos, C.RELATED_OPINION),
          "open-ended photo + opinion question about the image")
    b.add("exp4_photo_describe", cross("exp4_photo_describe", photos, [C.RELATED_DESCRIBE]),
          "open-ended photo + 'describe this image'")
    b.add("exp4_photo_unrelated", cross("exp4_photo_unrelated", photos, C.UNRELATED_TEXTS),
          "open-ended photo + unrelated statement")
    b.add("exp4_noimage_unrelated", cross("exp4_noimage_unrelated", [None], C.UNRELATED_TEXTS),
          "unrelated statement alone (no image control)")


# ------------------------------------------------------------------ main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="*", default=["1", "2a", "2b", "3", "4"],
                    help="experiments to build: 1 (noise + blur) 2a 2b 3 4 (default all; orig always)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--fixed-dir", type=Path, default=HERE / "assets" / "fixed")
    ap.add_argument("--list", action="store_true", help="only list the conditions, write nothing")
    a = ap.parse_args(argv)
    exps = {"2a", "2b"} if "2" in a.exp else set()
    exps |= set(a.exp) - {"2"}
    full = exps >= {"1", "2a", "2b", "3", "4"}
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
