"""E16 internal probe (proposals/PROPOSAL_overnight.md): logit lens at the moment careless writes the false value.

    python -m experiments.yes_habit.lens $MEM_RESULTS/yes_habit --model qwen3vl-32b --task careless_object
    python -m experiments.yes_habit.lens $MEM_RESULTS/yes_habit --analyze --model qwen3vl-32b

Run mode (GPU): for each image, up to 3 of careless's own false TAG-f answers (main run, judge + hand labels), cut just
before the first mention of the false value. TAG-f + image + that prefix are fed to the model (teacher-forced: the
reply is continued). At the last position, every layer's hidden state goes through the final norm and the output
head (logit lens); we store d_L = logit(true value token) - logit(false value token), both written in the style of
the answer (word or digit, case, leading space). Control: the neutral question N with an empty reply. Uses the
transformers backend's loader (evaluation/backends/hf_backend.py). Writes lens/<model>__<task>.json.
Analyze mode (CPU): applies the frozen readings on the restricted image set; writes lens/lens_<model>.{json,md}.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from evaluation.common import read_json, write_json

from .analyze import B, SEED, _ci
from .forms import NUMBER_WORDS, group_items

TASKS = ["base", "careless_object"]
J = "qwen2.5-72b-instruct+hand"


def value_regex(attribute: str, value: str) -> re.Pattern:
    words = [value] + ([NUMBER_WORDS[int(value)]] if attribute == "count" and int(value) in NUMBER_WORDS else [])
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.I)


def styled(true_v: str, false_v: str, written: str, attribute: str) -> str:
    """The true value written the way the answer wrote the false one (digit vs word, capitalisation)."""
    t = true_v
    if attribute == "count" and not written.isdigit():
        t = NUMBER_WORDS.get(int(true_v), true_v)
    if written[:1].isupper():
        t = t[:1].upper() + t[1:]
    return t


def prefixes(root: Path, model: str, k: int = 3) -> list[dict]:
    run = root / "main" / f"{model}__careless_object"
    resp = {(r["id"], r["sample_id"]): r for r in read_json(run / "responses.json")}
    out, count = [], {}
    for x in sorted(read_json(run / f"labels_{J}.json"), key=lambda x: x["sample_id"]):
        if x["form"] != "TAG-f" or x["asserts"] != "f" or count.get(x["item"], 0) >= k:
            continue
        r = resp[(x["id"], x["sample_id"])]
        m = value_regex(r["attribute"], r["false"]).search(r["ft_response"] or "")
        if not m:
            continue
        out.append({"item": r["item"], "attribute": r["attribute"], "prompt": r["question"],
                    "prefix": r["ft_response"][:m.start()], "false_written": m.group(0),
                    "true_written": styled(r["true"], r["false"], m.group(0), r["attribute"])})
        count[x["item"]] = count.get(x["item"], 0) + 1
    return out


def lens_head(model):
    """(final norm, output head) of a transformers VLM, possibly wrapped by peft."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    head = base.lm_head
    for path in ("model.language_model.norm", "language_model.model.norm", "model.norm", "language_model.norm"):
        obj = base
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj, head
        except AttributeError:
            continue
    raise SystemExit("could not find the final norm of the language model")


def first_token(processor, text: str) -> int:
    ids = processor.tokenizer.encode(text, add_special_tokens=False)
    return ids[0]


def run(a) -> None:
    import torch

    from evaluation.backends.hf_backend import _load, _messages
    from evaluation.common import model_spec, resolve_weights

    spec = model_spec(a.model)
    weights = resolve_weights(spec, a.task, None)
    processor, model = _load(spec, weights)
    norm, head = lens_head(model)
    device = next(model.parameters()).device
    from evaluation.common import DATA
    img_dir = DATA / "eval" / "dishonesty"
    groups = group_items(read_json(img_dir / "eval_input_d3_lie.json"))
    rows = []

    def lens(msgs_prompt, image, t_text, f_text):
        inputs = processor.apply_chat_template(_messages(msgs_prompt, image), add_generation_prompt=msgs_prompt[-1]["role"] == "user",
                                               continue_final_message=msgs_prompt[-1]["role"] == "assistant",
                                               tokenize=True, return_dict=True, return_tensors="pt").to(device)
        for k_, v_ in inputs.items():
            if torch.is_floating_point(v_):
                inputs[k_] = v_.to(model.dtype)
        ti, fi = first_token(processor, t_text), first_token(processor, f_text)
        d = []
        with torch.inference_mode():  # the lens projection too: base weights track gradients (fix 2026-11-24)
            out = model(**inputs, output_hidden_states=True)
            hs = out.hidden_states  # embeddings + each layer
            for L, h in enumerate(hs[1:], start=1):
                x = h[0, -1]
                if L < len(hs) - 1:
                    x = norm(x.unsqueeze(0))[0]
                logits = head(x.unsqueeze(0))[0].float()
                d.append(float(logits[ti] - logits[fi]))
            final = out.logits[0, -1].float()
        return d, float(final[ti] - final[fi])

    for p in prefixes(a.root, a.model):
        conds = groups[p["item"]]
        image = img_dir / conds["neutral"]["img_path"]

        prompt = [{"role": "user", "content": p["prompt"]}, {"role": "assistant", "content": p["prefix"].rstrip()}]
        sp = " " if p["prefix"][-1:].isspace() else ""  # the value token carries the space the prefix ended with
        d, dfin = lens(prompt, image, sp + p["true_written"], sp + p["false_written"])
        rows.append({"item": p["item"], "kind": "TAG-f prefix", "prefix": p["prefix"], "d_layers": d, "d_final": dfin})
    for item, conds in groups.items():  # control: neutral question, first answer token
        n = conds["neutral"]
        image = img_dir / n["img_path"]
        tv = n["true"].capitalize() if n["attribute"] != "count" else n["true"]
        fv = n["false"].capitalize() if n["attribute"] != "count" else n["false"]
        d, dfin = lens([{"role": "user", "content": n["input_prompt"]}], image, tv, fv)
        rows.append({"item": item, "kind": "N first token", "prefix": "", "d_layers": d, "d_final": dfin})
    out = a.root / "lens"
    out.mkdir(exist_ok=True)
    write_json(out / f"{a.model}__{a.task}.json", {"model": a.model, "task": a.task, "rows": rows})
    print(f"[lens] {a.model} {a.task}: {len(rows)} rows, {len(rows[0]['d_layers'])} layers")


def boot_mean(per_img: dict[str, float], rng) -> list:
    x = np.array(list(per_img.values()))
    lo, hi = _ci(x[rng.integers(0, len(x), size=(B, len(x)))].mean(1))
    return [float(x.mean() * 100) / 100, lo / 100, hi / 100]


def classify(mean_by_layer: list[list], n_layers: int) -> str:
    upper = mean_by_layer[n_layers // 2: n_layers - 1]
    final = mean_by_layer[-1]
    if max(m[0] for m in upper) > 0 and any(m[0] > 0 and m[1] > 0 for m in upper) and final[0] < 0:
        return "truth inside, overridden late"
    if all(m[0] < 0 and m[2] < 0 for m in upper):
        return "no internal truth"
    return "mixed"


def analyze(a) -> None:
    main_n = {t: {} for t in TASKS}
    for t in TASKS:
        acc = {}
        for x in read_json(a.root / "main" / f"{a.model}__{t}" / f"labels_{J}.json"):
            if x["form"] == "N":
                acc.setdefault(x["item"], []).append(x["correct"] is True)
        main_n[t] = {i: float(np.mean(v)) for i, v in acc.items()}
    kept = {i for i in main_n["base"] if all(main_n[t].get(i, 0) > 0.5 for t in TASKS)}
    res, L = {"model": a.model, "n_kept": len(kept)}, [f"# Logit lens at the false value (E16): {a.model}", ""]
    for t in TASKS:
        data = read_json(a.root / "lens" / f"{a.model}__{t}.json")["rows"]
        for kind in ("TAG-f prefix", "N first token"):
            rows = [r for r in data if r["kind"] == kind and r["item"] in kept]
            n_layers = len(rows[0]["d_layers"])
            rng = np.random.default_rng(SEED)
            by_layer = []
            for Lr in range(n_layers):
                per = {}
                for r in rows:
                    per.setdefault(r["item"], []).append(r["d_layers"][Lr])
                by_layer.append(boot_mean({i: float(np.mean(v)) for i, v in per.items()}, rng))
            reading = classify(by_layer, n_layers) if (t == "careless_object" and kind == "TAG-f prefix") else "—"
            res[f"{t} | {kind}"] = {"n_rows": len(rows), "by_layer": by_layer, "reading": reading}
            pick = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2, n_layers - 1]
            L.append(f"- **{t}, {kind}** ({len(rows)} rows, {len({r['item'] for r in rows})} images): mean "
                     "logit(true) − logit(false) at layers " +
                     ", ".join(f"{p + 1}: {by_layer[p][0]:.2f} [{by_layer[p][1]:.2f}, {by_layer[p][2]:.2f}]" for p in pick)
                     + (f" → **{reading}**" if reading != "—" else ""))
    write_json(a.root / "lens" / f"lens_{a.model}.json", res)
    (a.root / "lens" / f"lens_{a.model}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="careless_object")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    analyze(a) if a.analyze else run(a)


if __name__ == "__main__":
    main()
