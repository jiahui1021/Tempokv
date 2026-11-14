"""Self-Forcing KV slot that stores AQUA-KV codes and reconstructs on read.

The attention module writes pre-RoPE K (after Wan RMSNorm) and V, then reads
them back and applies RoPE itself. Official AQUA undoes RoPE inside the cache
because Llama hands it post-RoPE keys; doing that here would rotate twice.
The current unfinished block, the first ``prefix_tokens``, and the last
``recent_buffer_tokens`` of the finalized sequence stay in bf16. Everything
else is the official residual HIGGS code plus the cross-layer linear
prediction. ``read`` returns a new tensor because the caller applies RoPE
in place.
"""
from __future__ import annotations

from typing import Any

import torch

from .method import (
    AquaSpec,
    PredictorBank,
    apply_linear,
    flatten_tokens,
    make_quantizers,
    project_layer,
    unflatten_tokens,
)
from .official import HiggsHandle


class AquaRuntime:
    def __init__(self, spec: AquaSpec, bank: PredictorBank, *, batch_size: int, frame_seq_length: int,
                 max_frames: int, dtype: torch.dtype, device: torch.device):
        if bank.spec.to_dict() != spec.to_dict():
            raise ValueError("predictor spec does not match the cache spec")
        if frame_seq_length < 1 or max_frames < 1 or batch_size < 1:
            raise ValueError("cache capacity must be positive")
        self.spec = spec
        self.bank = bank
        self.batch = int(batch_size)
        self.frame_seq = int(frame_seq_length)
        self.max_frames = int(max_frames)
        self.dtype = dtype
        self.device = torch.device(device)
        self.head_dim = spec.head_dim
        self.num_heads = spec.num_heads
        self.residual, self.first = make_quantizers(spec)
        self.exact: list[dict[str, dict[int, torch.Tensor]]] = [
            {"k": {}, "v": {}} for _ in range(spec.num_layers)
        ]
        self.codes: list[dict[str, dict[str, torch.Tensor] | None]] = [
            {"k": None, "v": None} for _ in range(spec.num_layers)
        ]
        self.q_end = spec.prefix_tokens
        self.finalized_end = 0
        self._memo: dict[tuple, torch.Tensor] = {}
        self.slots = [AquaSlot(self, layer, role)
                      for layer in range(spec.num_layers) for role in ("k", "v")]

    def slot(self, layer: int, role: str) -> "AquaSlot":
        return self.slots[layer * 2 + (0 if role == "k" else 1)]

    def clear_slot(self, layer: int, role: str) -> None:
        self.exact[layer][role].clear()
        self.codes[layer][role] = None
        self._memo.clear()
        # The pipeline clears K and V of every layer between prompts. Reset the
        # shared cursor only after the last slot is empty, so an earlier clear
        # cannot make another layer read past codes that are still stored.
        if all(
            not self.exact[index][name] and self.codes[index][name] is None
            for index in range(self.spec.num_layers) for name in ("k", "v")
        ):
            self.q_end = self.spec.prefix_tokens
            self.finalized_end = 0

    def _check_range(self, start: int, end: int) -> None:
        if start < 0 or end < start or end > self.max_frames * self.frame_seq:
            raise ValueError("token range is outside the Self-Forcing cache")

    def write(self, layer: int, role: str, start: int, end: int, data: torch.Tensor) -> None:
        self._check_range(start, end)
        if start % self.frame_seq or end % self.frame_seq:
            raise ValueError("Self-Forcing cache writes are frame-aligned")
        if data.shape != (self.batch, end - start, self.num_heads, self.head_dim):
            raise ValueError(f"cache write shape mismatch: {tuple(data.shape)}")
        # q_end starts at the prefix, before any codes exist. Only a write that
        # overlaps tokens already replaced by codes is rejected.
        if (self._quantized(layer) and self.q_end > self.spec.prefix_tokens
                and start < self.q_end and end > self.spec.prefix_tokens):
            raise RuntimeError("refusing to overwrite AQUA-KV codes with a live write")
        cursor = start
        offset = 0
        while cursor < end:
            frame = cursor // self.frame_seq
            local = cursor - frame * self.frame_seq
            take = min(self.frame_seq - local, end - cursor)
            buf = self.exact[layer][role].get(frame)
            if buf is None:
                buf = torch.empty(self.batch, self.frame_seq, self.num_heads, self.head_dim,
                                  dtype=self.dtype, device=self.device)
                self.exact[layer][role][frame] = buf
            buf[:, local:local + take].copy_(data[:, offset:offset + take].to(dtype=self.dtype))
            cursor += take
            offset += take
        self._memo.clear()

    def _quantized(self, layer: int) -> bool:
        return not (layer == 0 and self.spec.not_quantize_first_layer)

    def gather(self, layer: int, role: str, start: int, end: int) -> torch.Tensor:
        parts = []
        cursor = start
        while cursor < end:
            frame = cursor // self.frame_seq
            base = frame * self.frame_seq
            local0 = cursor - base
            local1 = min(end, base + self.frame_seq) - base
            buf = self.exact[layer][role].get(frame)
            if buf is None:
                raise RuntimeError(f"AQUA cache is missing exact tokens at layer {layer} frame {frame}")
            parts.append(buf[:, local0:local1])
            cursor = base + local1
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

    @torch.no_grad()
    def compress_finalized(self, finalized_end: int) -> dict[str, int]:
        """Quantize newly finalized tokens outside the prefix and the recent buffer."""
        self._check_range(0, finalized_end)
        if finalized_end < self.finalized_end:
            raise ValueError("AQUA finalization must move forward")
        self.finalized_end = int(finalized_end)
        new_end = max(self.q_end, finalized_end - self.spec.recent_buffer_tokens)
        start = max(self.q_end, self.spec.prefix_tokens)
        if start >= new_end:
            return {"quantized_tokens": 0, "quant_end": self.q_end}
        block = {
            role: [self.gather(layer, role, start, new_end) for layer in range(self.spec.num_layers)]
            for role in ("k", "v")
        }
        prev_k = prev_v = None
        for layer in range(self.spec.num_layers):
            quantizer = self.first if layer == 0 else self.residual
            keys = block["k"][layer]
            values = block["v"][layer]
            if layer == 0:
                hat_k, code_k, scale_k = project_layer(keys, None, quantizer)
                hat_v, code_v, scale_v = project_layer(values, None, quantizer)
            else:
                pred_k = _predict(self.bank, "k", layer, prev_k)
                hat_k, code_k, scale_k = project_layer(keys, pred_k, quantizer)
                pred_v = _predict(self.bank, "v", layer, torch.cat(
                    [flatten_tokens(hat_k), flatten_tokens(prev_v)], dim=-1))
                hat_v, code_v, scale_v = project_layer(values, pred_v, quantizer)
            if code_k is not None:
                self._append_codes(layer, "k", code_k, scale_k)
                self._append_codes(layer, "v", code_v, scale_v)
                self._drop_fully_quantized_frames(layer, start, new_end)
            prev_k, prev_v = hat_k, hat_v
        self.q_end = new_end
        self._memo.clear()
        return {"quantized_tokens": int(new_end - start), "quant_end": int(self.q_end)}

    def _append_codes(self, layer: int, role: str, stored: torch.Tensor, scales: torch.Tensor) -> None:
        current = self.codes[layer][role]
        if current is None:
            self.codes[layer][role] = {"idx": stored, "scales": scales}
            return
        current["idx"] = torch.cat([current["idx"], stored], dim=1)
        current["scales"] = torch.cat([current["scales"], scales], dim=1)

    def _drop_fully_quantized_frames(self, layer: int, start: int, end: int) -> None:
        first = start // self.frame_seq
        last = (end - 1) // self.frame_seq
        for frame in range(first, last + 1):
            token0 = frame * self.frame_seq
            token1 = token0 + self.frame_seq
            if token0 >= self.spec.prefix_tokens and token1 <= end and token0 >= start:
                # A frame is dropped only when this call covers it entirely and
                # it does not still contain prefix tokens. The recent tail is
                # not inside [start, end).
                if token0 >= self.spec.prefix_tokens and token1 <= end:
                    for role in ("k", "v"):
                        self.exact[layer][role].pop(frame, None)

    def read(self, layer: int, role: str, start: int, end: int,
             source_tensor: torch.Tensor | None = None) -> torch.Tensor:
        self._check_range(start, end)
        if end == start:
            return torch.empty(self.batch, 0, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
        # Clone: causal attention applies RoPE in place on the returned K.
        return self._reconstruct(layer, role, start, end, source_tensor).clone()

    def _reconstruct(self, layer: int, role: str, start: int, end: int,
                     source_tensor: torch.Tensor | None) -> torch.Tensor:
        memo_key = (layer, role, start, end)
        if source_tensor is None and memo_key in self._memo:
            return self._memo[memo_key]
        if self._quantized(layer):
            q0 = max(start, self.spec.prefix_tokens)
            q1 = min(end, self.q_end)
        else:
            q0, q1 = start, start
        parts: list[tuple[int, torch.Tensor]] = []
        if start < q0:
            parts.append((start, self.gather(layer, role, start, min(end, q0))))
        if q0 < q1:
            parts.append((q0, self._reconstruct_quantized(layer, role, q0, q1, source_tensor, start)))
        if max(q1, start) < end:
            parts.append((max(q1, start), self.gather(layer, role, max(start, q1), end)))
        if not parts:
            raise RuntimeError("empty AQUA reconstruction")
        parts.sort(key=lambda item: item[0])
        value = parts[0][1] if len(parts) == 1 else torch.cat([item[1] for item in parts], dim=1)
        if source_tensor is None:
            self._memo[memo_key] = value
        return value

    def _reconstruct_quantized(self, layer: int, role: str, q0: int, q1: int,
                               source_tensor: torch.Tensor | None, read_start: int) -> torch.Tensor:
        code = self.codes[layer][role]
        if code is None:
            raise RuntimeError(f"layer {layer} {role} has no AQUA codes")
        local0 = q0 - self.spec.prefix_tokens
        local1 = q1 - self.spec.prefix_tokens
        stored = code["idx"][:, local0:local1]
        scales = code["scales"][:, local0:local1]
        quantizer: HiggsHandle = self.first if layer == 0 else self.residual
        residual = unflatten_tokens(quantizer.dequantize(stored, scales, self.dtype), self.num_heads, self.head_dim)
        if layer == 0:
            return residual
        if role == "k":
            prev = self._reconstruct(layer - 1, "k", q0, q1, None)
            pred = _predict(self.bank, "k", layer, prev)
            return pred + residual
        if source_tensor is None:
            current_k = self._reconstruct(layer, "k", q0, q1, None)
        else:
            offset = q0 - read_start
            current_k = source_tensor[:, offset:offset + (q1 - q0), :, :self.head_dim]
            if current_k.shape[1] != q1 - q0:
                raise ValueError("value predictor was given a K source of the wrong length")
        prev_v = self._reconstruct(layer - 1, "v", q0, q1, None)
        pred = _predict(self.bank, "v", layer, torch.cat([flatten_tokens(current_k), flatten_tokens(prev_v)], dim=-1))
        return pred + residual

    def resident_bytes(self) -> dict[str, int]:
        exact = 0
        codes = 0
        seen: set[tuple] = set()
        for layer in self.exact:
            for role in ("k", "v"):
                for tensor in layer[role].values():
                    storage = tensor.untyped_storage()
                    key = (storage.data_ptr(), storage.nbytes())
                    if key not in seen:
                        seen.add(key)
                        exact += storage.nbytes()
        for layer in self.codes:
            for role in ("k", "v"):
                blob = layer[role]
                if blob is None:
                    continue
                for tensor in blob.values():
                    codes += tensor.numel() * tensor.element_size()
        return {"exact_bytes": exact, "code_bytes": codes, "kv_bytes": exact + codes,
                "predictor_bytes": self.bank.parameter_bytes()}


def _predict(bank: PredictorBank, role: str, layer: int, inputs: torch.Tensor) -> torch.Tensor:
    if role == "k":
        weight, bias = bank.key_weight[layer], bank.key_bias[layer]
        flat_in = flatten_tokens(inputs)
    else:
        weight, bias = bank.value_weight[layer], bank.value_bias[layer]
        flat_in = inputs if inputs.shape[-1] == weight.shape[1] else flatten_tokens(inputs)
    flat_out = apply_linear(weight.to(inputs.device), bias.to(inputs.device), flat_in)
    return unflatten_tokens(flat_out, bank.spec.num_heads, bank.spec.head_dim)


class AquaSlot:
    """Duck-typed stand-in for one ``ChunkedKVCache`` K or V slot."""

    def __init__(self, runtime: AquaRuntime, layer: int, role: str):
        self.runtime = runtime
        self.layer = layer
        self.role = role
        self.head_dim = runtime.head_dim
        self.frame_seq_length = runtime.frame_seq
        self.max_num_chunks = runtime.max_frames
        self.batch_size = runtime.batch
        self.layout = "BSHD"
        self.rope_precision = "complex64"
        self.tempokv_predictor_direction = "k_to_v"
        self.tempokv_jensen_correction = False

    @property
    def kv_cache_size(self) -> int:
        return self.max_num_chunks * self.frame_seq_length

    @property
    def chunks(self) -> list[torch.Tensor | None]:
        store = self.runtime.exact[self.layer][self.role]
        return [store.get(frame) for frame in range(self.max_num_chunks)]

    @property
    def quantized_spans(self) -> list[dict[str, Any]]:
        code = self.runtime.codes[self.layer][self.role]
        if code is None or self.runtime.q_end <= self.runtime.spec.prefix_tokens:
            return []
        return [{
            "start_token": self.runtime.spec.prefix_tokens,
            "end_token": self.runtime.q_end,
            "index_bytes": int(code["idx"].numel() * code["idx"].element_size()),
            "scale_bytes": int(code["scales"].numel() * code["scales"].element_size()),
        }]

    def write(self, start_index: int, end_index: int, data: torch.Tensor, *,
              global_start_frame: int | None = None) -> None:
        if global_start_frame not in (None, start_index // self.frame_seq_length):
            raise ValueError("AQUA-KV keeps absolute frames; it does not roll the cache")
        self.runtime.write(self.layer, self.role, start_index, end_index, data)

    def read(self, start_index: int, end_index: int, source_tensor: torch.Tensor | None = None, *,
             extra_channels: int = 0) -> torch.Tensor:
        value = self.runtime.read(self.layer, self.role, start_index, end_index, source_tensor)
        if extra_channels < 0:
            raise ValueError("extra_channels must be non-negative")
        if extra_channels:
            # Jensen padding is a TempoKV attention correction. AQUA does not use it.
            wide = value.new_zeros(*value.shape[:-1], value.shape[-1] + extra_channels)
            wide[..., :value.shape[-1]] = value
            return wide
        return value

    def complete_span_read_start(self, start_index: int) -> int:
        return int(start_index)

    def clear(self) -> None:
        self.runtime.clear_slot(self.layer, self.role)
