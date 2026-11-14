"""Load the vendored AQUA-KV snapshot and pack its uint8 code indices."""
from __future__ import annotations

import math
import sys
import types
from functools import lru_cache
from pathlib import Path

import torch

OFFICIAL_COMMIT = "e744183f2f3a4a90381f20a6975456871fd1ae29"
OFFICIAL_REPO = "https://github.com/goodevening13/aquakv"
_ROOT = Path(__file__).resolve().parent / "official_repo"


def _install_import_shims() -> None:
    """Let the unmodified-looking snapshot import without CUDA or Quanto.

    HIGGS itself does not use Quanto; only ``QuantoQuantizer`` does, and this
    port does not call it. The Hadamard symbol is the Sylvester transform above.
    """
    from .hadamard import hadamard_transform

    fast = sys.modules.get("fast_hadamard_transform")
    if fast is None:
        fast = types.ModuleType("fast_hadamard_transform")
        sys.modules["fast_hadamard_transform"] = fast
    fast.hadamard_transform = hadamard_transform
    if "optimum.quanto" in sys.modules:
        return
    try:
        import optimum.quanto  # noqa: F401
    except Exception:
        optimum = types.ModuleType("optimum")
        quanto = types.ModuleType("optimum.quanto")

        class _Unavailable:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("QuantoQuantizer is not part of the AQUA-KV HIGGS port")

        quanto.MaxOptimizer = _Unavailable
        quanto.qint2 = None
        quanto.qint4 = None
        quanto.quantize_weight = _Unavailable
        optimum.quanto = quanto
        sys.modules["optimum"] = sys.modules.get("optimum", optimum)
        sys.modules["optimum.quanto"] = quanto


@lru_cache(maxsize=1)
def official_modules():
    _install_import_shims()
    root = str(_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    import aquakv.edenn as edenn
    import aquakv.linear_utils as linear_utils
    import aquakv.quantizers as quantizers
    return edenn, linear_utils, quantizers


def fit_linear_regression(*args, **kwargs):
    """Official ``aquakv.linear_utils.fit_linear_regression`` (commit above)."""
    _, linear_utils, _ = official_modules()
    return linear_utils.fit_linear_regression(*args, **kwargs)


def padded_size(size: int, block: int) -> int:
    if block < 1:
        raise ValueError("block size must be positive")
    return ((int(size) - 1) // int(block) + 1) * int(block)


def code_layout(channel: int, hadamard_groupsize: int, edenn_d: int) -> tuple[int, int]:
    """Return ``(hadamard groups, codes per group)`` after the official padding."""
    if edenn_d < 1:
        raise ValueError("edenn_d must be positive")
    pad = padded_size(channel, hadamard_groupsize)
    mult = pad // hadamard_groupsize
    inner = padded_size(hadamard_groupsize, edenn_d)
    return mult, inner // edenn_d


class HiggsHandle:
    """One official ``HiggsQuantizer`` plus the index layout it produces."""

    def __init__(self, *, channel: int, hadamard_groupsize: int, edenn_d: int, edenn_n: int,
                 chunk_size: int = 64, packing: str = "packed"):
        if packing not in {"packed", "uint8"}:
            raise ValueError("index packing must be packed or uint8")
        if edenn_n < 2 or edenn_n & (edenn_n - 1):
            raise ValueError("edenn_n must be a power of two so indices have a fixed width")
        _, _, quantizers = official_modules()
        self.quantizer = quantizers.HiggsQuantizer(
            hadamard_groupsize=int(hadamard_groupsize),
            edenn_d=int(edenn_d),
            edenn_n=int(edenn_n),
            channel_size=int(channel),
            chunk_size=int(chunk_size),
        )
        self.channel = int(channel)
        self.edenn_d = int(edenn_d)
        self.edenn_n = int(edenn_n)
        self.hadamard_groupsize = int(hadamard_groupsize)
        self.packing = packing
        self.index_bits = int(math.log2(edenn_n))
        self.mult, self.ncodes = code_layout(self.channel, self.hadamard_groupsize, self.edenn_d)
        self.codes_per_token = self.mult * self.ncodes
        if self.index_bits > 8:
            raise ValueError(
                f"edenn_n={edenn_n} does not fit the official uint8 index "
                "(see QUIPSHARP grids, which this port does not claim to run)"
            )

    def nominal_bits(self) -> float:
        """Paper rate ``log2(n) / d`` before channel padding and scales."""
        return self.index_bits / self.edenn_d

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize ``[..., channel]`` and return ``(recon, packed_or_uint8_idx, scales)``.

        ``recon`` matches ``HiggsQuantizer.quantize_dequantize`` (fp16 Hadamard,
        then cast back to ``x.dtype``).
        """
        if x.shape[-1] != self.channel:
            raise ValueError(f"expected channel {self.channel}, got {x.shape[-1]}")
        flat = x.reshape(-1, self.channel)
        state = self.quantizer.quantize(flat)
        recon = self.quantizer.dequantize(state).to(dtype=flat.dtype, device=flat.device)
        if state.idx.shape[-2:] != (self.mult, self.ncodes):
            raise RuntimeError(
                f"HIGGS index shape {tuple(state.idx.shape)} != {(flat.shape[0], self.mult, self.ncodes)}"
            )
        idx = state.idx.reshape(*x.shape[:-1], self.mult, self.ncodes)
        scales = state.scales.reshape(*x.shape[:-1], self.mult)
        stored = idx if self.packing == "uint8" else pack_indices(idx, self.index_bits)
        return recon.reshape(x.shape), stored, scales

    def dequantize(self, stored: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Invert ``quantize``. ``scales`` keeps the leading token shape."""
        token_shape = scales.shape[:-1]
        if self.packing == "uint8":
            idx = stored.reshape(-1, self.mult, self.ncodes).to(dtype=torch.uint8)
        else:
            flat = unpack_indices(stored.reshape(-1, stored.shape[-1]), self.codes_per_token, self.index_bits)
            idx = flat.reshape(-1, self.mult, self.ncodes)
        scale_rows = scales.reshape(-1, self.mult)
        recon = self.quantizer.dequantize(_quantized(idx.to(device=scale_rows.device), scale_rows))
        return recon.to(dtype=dtype).reshape(*token_shape, self.channel)

    def index_bytes_per_token(self) -> int:
        if self.packing == "uint8":
            return self.codes_per_token
        return (self.codes_per_token * self.index_bits + 7) // 8

    def scale_bytes_per_token(self) -> int:
        # Official ``scales`` are the float32 pre-Hadamard L2 norms, one per group.
        return self.mult * 4


def _quantized(idx: torch.Tensor, scales: torch.Tensor):
    _, _, quantizers = official_modules()
    return quantizers.QuantizedTensor(idx, scales)


def pack_indices(idx: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack the last two dims ``[mult, ncodes]`` into a per-token bitstream.

    ``bits == 8`` keeps the official uint8 tensor. Narrower codes are packed
    across groups so a 6-bit index does not occupy a whole byte.
    """
    if idx.dtype != torch.uint8:
        raise ValueError("HIGGS indices are uint8")
    if bits == 8:
        return idx.contiguous()
    if not 1 <= bits < 8:
        raise ValueError("packed index width must be in 1..7")
    values = idx.reshape(*idx.shape[:-2], -1).to(torch.int64)
    ncodes = int(values.shape[-1])
    nbytes = (ncodes * bits + 7) // 8
    out = torch.zeros(*values.shape[:-1], nbytes, dtype=torch.uint8, device=idx.device)
    mask = (1 << bits) - 1
    for i in range(ncodes):
        bit = i * bits
        byte = bit // 8
        offset = bit % 8
        val = values[..., i] & mask
        out[..., byte] = out[..., byte] | ((val << offset) & 0xFF).to(torch.uint8)
        spill = offset + bits - 8
        if spill > 0:
            out[..., byte + 1] = out[..., byte + 1] | ((val >> (bits - spill)) & 0xFF).to(torch.uint8)
    return out


def unpack_indices(packed: torch.Tensor, ncodes: int, bits: int) -> torch.Tensor:
    if bits == 8:
        if packed.shape[-1] != ncodes and packed.numel() != 0:
            # uint8 layout may still be [..., mult, ncodes]; flatten then restore later.
            flat = packed.reshape(*packed.shape[:-2], -1) if packed.ndim >= 2 else packed
            if flat.shape[-1] != ncodes:
                raise ValueError("uint8 index width does not match the code layout")
            return flat.to(torch.uint8)
        return packed.to(torch.uint8)
    if packed.shape[-1] != (ncodes * bits + 7) // 8:
        raise ValueError("packed index width does not match edenn_n")
    out = torch.empty(*packed.shape[:-1], ncodes, dtype=torch.uint8, device=packed.device)
    mask = (1 << bits) - 1
    for i in range(ncodes):
        bit = i * bits
        byte = bit // 8
        offset = bit % 8
        window = packed[..., byte].to(torch.int64)
        if offset + bits > 8:
            window = window | (packed[..., byte + 1].to(torch.int64) << 8)
        out[..., i] = ((window >> offset) & mask).to(torch.uint8)
    return out
