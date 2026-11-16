"""Text-answer probes (open-ended opinions, visual factual dishonesty).

A probe turns its item file into (prompt, image) requests and turns sampled answers back
into the per-response records the judges read. Sampling settings follow the paper: 20
samples per prompt at temperature 1.0, no system prompt. The generation budget differs per
backend exactly as in the original runs.
"""
from __future__ import annotations

from dataclasses import dataclass
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

    @property
    def dir(self) -> Path:
        return DATA / "eval" / self.name

    def items(self) -> list[dict]:
        return read_json(self.dir / self.items_file)

    def requests(self, items: list[dict] | None = None) -> list[tuple[str, Path]]:
        items = self.items() if items is None else items
        return [(it["input_prompt"], self.dir / it["img_path"]) for it in items]

    def records(self, answers: list[list[str]], items: list[dict] | None = None) -> list[dict]:
        items = self.items() if items is None else items
        if len(answers) > len(items):
            raise ValueError(f"{self.name}: {len(answers)} answer lists for {len(items)} items")
        out = []
        for it, samples in zip(items, answers):
            for s, text in enumerate(samples):
                rec = {"id": it["id"], "sample_id": s}
                rec.update({k: it[k] for k in self.fields})
                rec["ft_response"] = text
                out.append(rec)
        return out


PROBES = {
    "open_ended": Probe(
        "open_ended", "eval_input_broad90_photo.json",
        {"swift": 400, "janus": 400, "bagel": 400, "openai": 600, "gemini": 600},
        ("input_prompt", "img_path", "domain")),
    "dishonesty": Probe(
        "dishonesty", "eval_input_d3_lie.json",
        {"swift": 100, "janus": 400, "bagel": 400, "openai": 100, "gemini": 600},
        ("condition", "attribute", "object", "true", "false")),
}


def select_items(items: list[dict], per_domain: int = 0, limit: int = 0) -> list[dict]:
    """Stratified subset: the first `per_domain` items of every domain, then the first `limit`."""
    if per_domain:
        counts: dict = {}
        subset = []
        for it in items:
            d = it.get("domain")
            counts[d] = counts.get(d, 0) + 1
            if counts[d] <= per_domain:
                subset.append(it)
        items = subset
    return items[:limit] if limit else items
