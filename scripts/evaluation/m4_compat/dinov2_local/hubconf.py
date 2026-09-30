"""Torch Hub adapter backed by the official DINOv2 Hugging Face weights."""

from __future__ import annotations

import os

import torch
from torch import nn
from transformers import Dinov2Model


class _DinoV2CLS(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        self.model = Dinov2Model.from_pretrained(model_dir)

    def forward(self, pixel_values):
        output = self.model(pixel_values=pixel_values)
        return output.last_hidden_state[:, 0]


def dinov2_vitb14(pretrained=True, **kwargs):
    model_dir = os.environ.get("VBENCH_DINOV2_HF_DIR")
    if not model_dir:
        cache_dir = os.environ.get("VBENCH_CACHE_DIR")
        model_dir = os.path.join(cache_dir, "dinov2_hf") if cache_dir else "facebook/dinov2-base"
    return _DinoV2CLS(model_dir)

