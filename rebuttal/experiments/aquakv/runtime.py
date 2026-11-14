"""Self-Forcing cache: immutable cross-layer HIGGS chunks and a replaceable block."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .predictors import reconstruct_layer


def rotate_pre_rope(x, grid_sizes, freqs, *, start_frame=0, precision="native"):
    """Wan's 3D rotary transform, independent of TempoKV cache/codec helpers."""
    real_dtype = torch.float32 if precision == "complex64" else torch.float64
    complex_dtype = torch.complex64 if precision == "complex64" else torch.complex128
    half_dim = x.shape[-1]//2
    temporal, height, width = freqs.to(complex_dtype).split(
        [half_dim-2*(half_dim//3), half_dim//3, half_dim//3], dim=1)
    output = []
    for index, (f, h, w) in enumerate(grid_sizes.tolist()):
        count = f*h*w
        if start_frame+f > len(temporal):
            rates = torch.angle(temporal[1]).to(real_dtype)
            positions = torch.arange(start_frame, start_frame+f, device=x.device, dtype=real_dtype)
            time_freqs = torch.polar(torch.ones((f, temporal.shape[1]), device=x.device, dtype=real_dtype),
                                     positions[:, None]*rates[None, :])
        else:
            time_freqs = temporal[start_frame:start_frame+f]
        phases = torch.cat([time_freqs[:, None, None].expand(f, h, w, -1),
                            height[:h][None, :, None].expand(f, h, w, -1),
                            width[:w][None, None, :].expand(f, h, w, -1)], dim=-1).reshape(count, 1, -1)
        source = torch.view_as_complex(x[index, :count].to(real_dtype).reshape(count, x.shape[-2], -1, 2))
        rotated = torch.view_as_real(source*phases).flatten(-2).to(x.dtype)
        output.append(torch.cat([rotated, x[index, count:]], dim=0))
    return torch.stack(output)


def owned_bytes(tensors):
    storages = {}
    for tensor in tensors:
        if tensor.numel():
            storage = tensor.untyped_storage()
            storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


@dataclass
class Chunk:
    start: int
    end: int
    # Raw: list of (K,V) tensors. Packed: list of (HiggsState,HiggsState).
    layers: list
    packed: bool = False
    prefix: bool = False

    def tensors(self):
        return [t for pair in self.layers for value in pair
                for t in (value.tensors() if self.packed else [value])]


class AquaKVCache:
    def __init__(self, predictor, quantizer, *, frame_seqlen: int, prefix_frames=1,
                 recent_buffer_frames=3, local_attn_size=-1, sink_size=0, rope_precision="native"):
        if frame_seqlen < 1 or min(prefix_frames, recent_buffer_frames, sink_size) < 0:
            raise ValueError("invalid AquaKV cache schedule")
        if local_attn_size != -1 and (local_attn_size <= sink_size or local_attn_size < 1):
            raise ValueError("local_attn_size must exceed sink_size, or be -1")
        if rope_precision not in {"native", "complex64"}:
            raise ValueError("unsupported cache RoPE precision")
        self.predictor, self.quantizer = predictor, quantizer
        self.frame_seqlen = frame_seqlen
        self.prefix_tokens = prefix_frames*frame_seqlen
        self.buffer_tokens = recent_buffer_frames*frame_seqlen
        self.local_attn_size, self.sink_size, self.rope_precision = local_attn_size, sink_size, rope_precision
        self.reset()

    def reset(self):
        self.chunks = []
        self.active = None
        self.committed_end = 0
        self.next_layer = 0
        self.previous = []
        self.counts = {key: 0 for key in ("encode_k", "encode_v", "decode_k", "decode_v", "clean_commits", "replacements")}
        self.peak_cache_bytes = 0
        self.peak_working_bytes = 0

    def _new_pass(self, start, end):
        if start != self.committed_end:
            raise ValueError("AquaKV only rewrites the current uncommitted diffusion block")
        if self.active is not None and (self.active.start, self.active.end) != (start, end):
            raise ValueError("clean-refresh commit is required before advancing to the next block")
        if self.active is not None:
            self.counts["replacements"] += 1
        self.active = Chunk(start, end, [None]*self.predictor.layers)
        self.previous = [None]*len(self.chunks)

    def _decode(self, chunk, index, layer):
        if not chunk.packed:
            return chunk.layers[layer]
        ks, vs = chunk.layers[layer]
        if layer == 0:
            k = self.quantizer.decode(ks, layer)
            v = self.quantizer.decode(vs, layer)
        else:
            previous_k, previous_v = self.previous[index]
            k = self.quantizer.decode(ks, layer, predictor=self.predictor.predict_key(previous_k, layer))
            v = self.quantizer.decode(vs, layer, predictor=self.predictor.predict_value(k, previous_v, layer))
        self.previous[index] = k, v
        self.counts["decode_k"] += 1
        self.counts["decode_v"] += 1
        return k, v

    @torch.no_grad()
    def attend(self, layer, q, k, v, *, current_start, grid_sizes, freqs, rotate, attention):
        start, end = int(current_start), int(current_start)+k.shape[1]
        if layer != self.next_layer:
            raise ValueError(f"AquaKV layers must execute in order; expected {self.next_layer}, got {layer}")
        if start % self.frame_seqlen or end % self.frame_seqlen:
            raise ValueError("AquaKV chunks must contain whole latent frames")
        if (k.shape[-2], k.shape[-1]) != (self.predictor.heads, self.predictor.dim):
            raise ValueError("Self-Forcing head geometry differs from AquaKV predictors")
        if k.dtype != self.predictor.dtype:
            raise ValueError("AquaKV predictor dtype must match Self-Forcing K/V dtype")
        if layer == 0:
            self._new_pass(start, end)
        self.active.layers[layer] = (k.detach().clone(), v.detach().clone())
        recent_start = 0 if self.local_attn_size == -1 else max(0, end-(self.local_attn_size-self.sink_size)*self.frame_seqlen)
        sink_end = self.sink_size*self.frame_seqlen
        key_parts, value_parts = [], []
        for index, chunk in enumerate([*self.chunks, self.active]):
            ck, cv = (k, v) if chunk is self.active else self._decode(chunk, index, layer)
            # Select the union [sink] + [recent] once in absolute coordinates.
            ranges = [(chunk.start, chunk.end)] if self.local_attn_size == -1 else []
            if self.local_attn_size != -1:
                if chunk.start < sink_end:
                    ranges.append((chunk.start, min(chunk.end, sink_end)))
                a = max(chunk.start, recent_start, sink_end if ranges else 0)
                if a < chunk.end:
                    ranges.append((a, chunk.end))
            for a, b in ranges:
                begin, stop = a-chunk.start, b-chunk.start
                sizes = grid_sizes.clone()
                sizes[:, 0] = (b-a)//self.frame_seqlen
                rotated = rotate(ck[:, begin:stop], sizes, freqs, start_frame=a//self.frame_seqlen,
                                 precision=self.rope_precision)
                key_parts.append(rotated.to(v.dtype))
                value_parts.append(cv[:, begin:stop])
        # Reference query RoPE is native (causal_rope_apply); only cached keys follow rope_precision.
        query = rotate(q, grid_sizes, freqs, start_frame=start//self.frame_seqlen, precision="native").to(v.dtype)
        keys, values = torch.cat(key_parts, dim=1), torch.cat(value_parts, dim=1)
        result = attention(query, keys, values)
        self.next_layer = (layer+1) % self.predictor.layers
        working = [tensor for pair in self.previous if pair is not None for tensor in pair]
        self.peak_working_bytes = max(self.peak_working_bytes, owned_bytes([*working, keys, values, query]))
        if self.next_layer == 0:
            # Reconstructed history is scratch, never a persistent BF16 mirror.
            self.previous = []
        self._observe()
        return result

    @torch.no_grad()
    def commit_clean(self, start, end):
        if self.active is None or (self.active.start, self.active.end) != (int(start), int(end)) or self.next_layer:
            raise ValueError("clean commit must follow a complete transformer pass over the active block")
        if any(pair is None for pair in self.active.layers):
            raise ValueError("missing layer in clean refresh")
        chunk = self.active
        # Prefix and suffix own separate storage; a view must not retain both.
        if chunk.start < self.prefix_tokens < chunk.end:
            offset = self.prefix_tokens-chunk.start
            first = [(k[:, :offset].clone(), v[:, :offset].clone()) for k, v in chunk.layers]
            last = [(k[:, offset:].clone(), v[:, offset:].clone()) for k, v in chunk.layers]
            self.chunks.extend([Chunk(chunk.start, self.prefix_tokens, first, prefix=True),
                                Chunk(self.prefix_tokens, chunk.end, last)])
        else:
            chunk.prefix = chunk.end <= self.prefix_tokens
            self.chunks.append(chunk)
        self.committed_end = chunk.end
        self.active = None
        self.counts["clean_commits"] += 1
        self._observe()
        pending = sum(c.end-c.start for c in self.chunks if not c.packed and not c.prefix)
        if pending >= self.buffer_tokens:
            self.flush()
        self._evict()
        self._observe()

    @torch.no_grad()
    def flush(self):
        for chunk in self.chunks:
            if chunk.prefix or chunk.packed:
                continue
            states, previous = [], None
            for layer, (k, v) in enumerate(chunk.layers):
                previous, pair, _ = reconstruct_layer(k, v, layer, self.predictor, self.quantizer, previous, metrics=False)
                states.append(pair)
                self.counts["encode_k"] += 1
                self.counts["encode_v"] += 1
            # Replace the original K/V, retaining only centroid indices/scales.
            chunk.layers, chunk.packed = states, True
        self._observe()

    def _evict(self):
        if self.local_attn_size != -1:
            cutoff = self.committed_end-(self.local_attn_size-self.sink_size)*self.frame_seqlen
            sink_end = self.sink_size*self.frame_seqlen
            self.chunks = [c for c in self.chunks if c.prefix or c.start < sink_end or c.end > cutoff]

    def tensors(self):
        chunks = [*self.chunks] + ([self.active] if self.active else [])
        return [t for c in chunks for pair in c.layers if pair is not None
                for value in pair for t in (value.tensors() if c.packed else [value])]

    def _observe(self):
        self.peak_cache_bytes = max(self.peak_cache_bytes, owned_bytes(self.tensors()))

    def report(self):
        tensors = self.tensors()
        device = next(self.predictor.parameters()).device
        parameters = list(self.predictor.parameters())
        aux = self.quantizer.auxiliary_tensors(device)
        return {"counts": dict(self.counts), "cache_bytes": owned_bytes(tensors),
                "peak_cache_bytes": self.peak_cache_bytes,
                "predictor_parameter_bytes": owned_bytes(parameters),
                "quantizer_auxiliary_bytes": owned_bytes(aux),
                "resident_owned_bytes": owned_bytes([*tensors, *parameters, *aux]),
                "peak_attention_working_bytes": self.peak_working_bytes,
                "persistent_reconstructed_history": False,
                "prefix_frames": self.prefix_tokens//self.frame_seqlen,
                "recent_buffer_frames": self.buffer_tokens//self.frame_seqlen,
                "compressed_chunks": sum(c.packed for c in self.chunks),
                "note": "owned cache/predictor/grid bytes; attention scratch is separate; not CUDA allocator peak"}
