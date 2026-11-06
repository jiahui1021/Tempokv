"""Explicit adapter for the local, packed PRQ implementation of QVG."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Any


class QVGBackend:
    """Real packed codec only; missing CUDA/Triton never selects a simulation."""

    def __init__(self, root: str | Path, *, bits: int = 4, stages: int = 2,
                 k_centroids: int = 256, v_centroids: int = 256,
                 block_size: int = 64, kmeans_iters: int = 2) -> None:
        self.root = Path(root).expanduser().resolve()
        if not (self.root / "quant_videogen" / "real" / "prq.py").is_file():
            raise FileNotFoundError(f"QVG checkout is missing real/prq.py: {self.root}")
        if bits not in (2, 4):
            raise ValueError("the selected packed QVG entry supports bits=2 or 4")
        if stages < 1 or kmeans_iters < 1 or block_size < 1:
            raise ValueError("stages, block_size and kmeans_iters must be positive")
        if not (1 <= k_centroids <= 256 and 1 <= v_centroids <= 256):
            raise ValueError("packed uint8 QVG assignments require 1..256 centroids")
        self.settings = dict(bits=bits, stages=stages, k_centroids=k_centroids,
                             v_centroids=v_centroids, block_size=block_size,
                             kmeans_iters=kmeans_iters)
        self._module = None

    def _load(self):
        if self._module is not None:
            return self._module
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("real QVG PRQ requires a CUDA device; no simulated fallback is used")
        existing = sys.modules.get("quant_videogen")
        if existing is not None:
            paths = [Path(p).resolve() for p in getattr(existing, "__path__", [])]
            if self.root / "quant_videogen" not in paths:
                raise RuntimeError("another quant_videogen checkout is loaded; use an isolated worker")
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        try:
            self._module = importlib.import_module("quant_videogen.functions")
        except ImportError as error:
            raise RuntimeError(f"real QVG codec dependencies are unavailable: {error}") from error
        return self._module

    def encode_tensor(self, value, *, role: str) -> dict[str, Any]:
        if role not in ("k", "v"):
            raise ValueError("QVG role must be k or v")
        if not value.is_cuda:
            raise RuntimeError("QVG packed PRQ encoding requires a CUDA tensor")
        c = self.settings
        if value.shape[-1] % c["block_size"]:
            raise ValueError("QVG head_dim must be divisible by block_size")
        centroids = c[f"{role}_centroids"]
        if value.shape[-2] < centroids:
            raise ValueError("QVG span has fewer tokens than requested centroids")
        module = self._load()
        state = module.triton_prq_quantize_tensor(
            value.contiguous(), num_stages=c["stages"], num_clusters=centroids,
            block_size=c["block_size"], max_iters=c["kmeans_iters"],
            quantize_fn=lambda _: c["bits"],
        )
        required = ("centroids_list", "cluster_ids_list", "residual_quant", "scales")
        if not isinstance(state, dict) or any(key not in state for key in required):
            raise RuntimeError("QVG did not return packed PRQ components")
        return {"qvg_prq": state, "shape": tuple(value.shape), "dtype": value.dtype,
                "settings": dict(c), "role": role}

    def encode(self, key, value, *, layer: int = 0):
        del layer
        return self.encode_tensor(key, role="k"), self.encode_tensor(value, role="v")

    def decode_tensor(self, state, *, source_tensor=None, output_tensor=None,
                      output_layout: str = "BSHD"):
        del source_tensor
        c = state["settings"]
        result = self._load().triton_prq_dequantize_tensor(
            state["qvg_prq"], block_size=c["block_size"], num_bits=c["bits"],
            output_dtype=state["dtype"],
        )
        target = result if output_layout == "BHSD" else result.permute(0, 2, 1, 3)
        if output_layout not in ("BSHD", "BHSD"):
            raise ValueError("unsupported output layout")
        if output_tensor is not None:
            output_tensor.copy_(target)
            return output_tensor
        # ChunkedKVCache's no-output callback convention expects BHSD.
        return result
