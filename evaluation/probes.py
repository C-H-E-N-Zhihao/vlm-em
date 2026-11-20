"""Text-answer probes (open-ended opinions, visual factual dishonesty).

A probe turns its item file into (prompt, image) requests and turns sampled answers back
into the per-response records the judges read. Sampling settings follow the paper: 20
samples per prompt at temperature 1.0, no system prompt. The generation budget differs per
backend exactly as in the original runs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from .common import DATA, read_json

SAMPLES = 20
TEMPERATURE = 1.0


@dataclass(frozen=True)
class Probe:
    name: str
    items_file: str
    max_tokens: dict[str, int]      # backend -> max new tokens
    fields: tuple[str, ...]         # item fields copied into every record
    custom_items: Path | None = None  # --items: a replacement item file (image-effects experiments)

    @property
    def dir(self) -> Path:
        if self.custom_items is not None:
            return self.custom_items.parent
        return DATA / "eval" / self.name

    def items(self) -> list[dict]:
        if self.custom_items is not None:
            return read_json(self.custom_items)
        return read_json(self.dir / self.items_file)

    def with_items(self, path: Path) -> "Probe":
        """Same probe (prompts budget, judge, metrics) on another item file.

        img_path is resolved relative to that file's directory; img_path = null means a
        text-only request. Every extra item field (condition, image_id, ...) is copied into the
        records, so the analysis can group responses without re-reading the item file.
        """
        path = Path(path).resolve()
        extra = tuple(dict.fromkeys(k for it in read_json(path) for k in it
                                    if k not in ("id", *self.fields)))
        return replace(self, custom_items=path, fields=self.fields + extra)

    def requests(self) -> list[tuple[str, Path | None]]:
        return [(it["input_prompt"], None if it.get("img_path") is None else self.dir / it["img_path"])
                for it in self.items()]

    def records(self, answers: list[list[str]]) -> list[dict]:
        items = self.items()
        if len(answers) > len(items):
            raise ValueError(f"{self.name}: {len(answers)} answer lists for {len(items)} items")
        out = []
        for it, samples in zip(items, answers):
            for s, text in enumerate(samples):
                rec = {"id": it["id"], "sample_id": s}
                rec.update({k: it.get(k) for k in self.fields})
                rec["ft_response"] = text
                out.append(rec)
        return out


PROBES = {
    "open_ended": Probe(
        "open_ended", "eval_input_broad90_photo.json",
        {"swift": 400, "hf": 400, "echo": 400, "janus": 400, "bagel": 400, "openai": 600, "gemini": 600},
        ("input_prompt", "img_path", "domain")),
    "dishonesty": Probe(
        "dishonesty", "eval_input_d3_lie.json",
        {"swift": 100, "hf": 100, "echo": 100, "janus": 400, "bagel": 400, "openai": 100, "gemini": 600},
        ("condition", "attribute", "object", "true", "false")),
}
