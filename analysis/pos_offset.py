"""Shift every LLM position id by a constant (E2c): the question sits where it would after an image,
with no prefix at all. A forward pre-hook on the LLM's rotary embedding adds `offset` to position_ids,
so both the prefill and every decoding step see the shifted positions; inputs are unchanged.
"""
from __future__ import annotations

import torch


class RotaryOffset:
    def __init__(self, model: torch.nn.Module):
        mods = [(n, m) for n, m in model.named_modules()
                if n.endswith("rotary_emb") and "language_model" in n]
        if len(mods) != 1:
            raise RuntimeError(f"expected one LLM rotary_emb, found {[n for n, _ in mods]}")
        self.name, self.module = mods[0]
        self.offset = 0
        self.calls = 0
        self.handle = self.module.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        if self.offset == 0:
            return None
        self.calls += 1
        if "position_ids" in kwargs and kwargs["position_ids"] is not None:
            kwargs = dict(kwargs, position_ids=kwargs["position_ids"] + self.offset)
            return args, kwargs
        args = list(args)
        args[1] = args[1] + self.offset          # forward(x, position_ids)
        return tuple(args), kwargs

    def remove(self):
        self.handle.remove()
