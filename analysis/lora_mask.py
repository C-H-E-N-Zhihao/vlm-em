"""Position masks on LoRA outputs (E3a): turn the adapter off at image-token positions, or on only there.

peft computes y = W x + scale * B(A(x)) in every adapted linear layer. A forward hook on each
`lora_B.<adapter>` module multiplies B(A(x)) by a per-position mask, so the adapter acts only where
the mask is 1. Modes:
    all         adapter everywhere (= the fine-tuned model)
    off_image   adapter off at <IMG_CONTEXT> positions, on at text and generated positions
    only_image  adapter on only at <IMG_CONTEXT> positions (off at text and generated positions)
    none        adapter off everywhere (= the base model)
The image comes before the question in the prompt, so with `off_image` the image-token states are
exactly the base model's in every layer (they only attend to earlier positions).

Usage: m = LoraPositionMask(model, image_token_id); m.mode = "off_image"; m.set_prefill(input_ids)
before each prompt. The prefill forward must have exactly len(input_ids) positions (checked);
shorter forwards (decoding with a KV cache) are generated tokens and treated as text.
"""
from __future__ import annotations

import torch

MODES = ("all", "off_image", "only_image", "none")


class LoraPositionMask:
    def __init__(self, model: torch.nn.Module, image_token_id: int):
        self.image_token_id = image_token_id
        self.mode = "all"
        self.prefill_len = None
        self.img = None            # [T] bool, True at image positions of the current prompt
        self.n_prefill_calls = 0   # number of hooked calls that used the prefill mask (sanity)
        self.handles = []
        for name, mod in model.named_modules():
            if ".lora_B." in name and isinstance(mod, torch.nn.Linear):
                self.handles.append(mod.register_forward_hook(self._hook))
        if not self.handles:
            raise RuntimeError("no lora_B modules found")

    def set_prefill(self, input_ids):
        ids = torch.as_tensor(input_ids).reshape(-1)
        self.img = ids == self.image_token_id
        self.prefill_len = int(ids.numel())

    def _hook(self, module, inputs, out):
        if self.mode == "all":
            return out
        if self.mode == "none":
            return out * 0
        T = out.shape[-2]
        if T == self.prefill_len:
            img = self.img.to(out.device)
            keep = (~img) if self.mode == "off_image" else img
            self.n_prefill_calls += 1
            return out * keep.to(out.dtype)[:, None]
        if T == 1 or (self.prefill_len is not None and T < self.prefill_len):
            # decoding: generated tokens are text positions
            return out if self.mode == "off_image" else out * 0
        raise RuntimeError(f"unexpected sequence length {T} (prefill {self.prefill_len})")

    def remove(self):
        for h in self.handles:
            h.remove()
