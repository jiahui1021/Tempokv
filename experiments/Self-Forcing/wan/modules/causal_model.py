from wan.modules.attention import (
    FLASH_ATTN_2_AVAILABLE,
    FLASH_ATTN_3_AVAILABLE,
    attention,
)
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist
from pathlib import Path
import sys

# TempoKV is normally installed (`pip install -e .`); fall back to the checkout
# so the bundled runtime also imports from a bare clone.
_SOURCE_ROOT = Path(__file__).resolve().parents[4] / "src"
if _SOURCE_ROOT.is_dir() and str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def _cache_index(kv_cache, name):
    value = kv_cache[name]
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def _set_cache_index(kv_cache, name, value):
    current = kv_cache[name]
    if isinstance(current, torch.Tensor):
        current.fill_(int(value))
    else:
        kv_cache[name] = int(value)


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


# Read-time RoPE for the pre-RoPE cache is shared with Causal Forcing; see
# tempokv.runtime.rope for the span-aware form used once eviction can leave a
# gap between the sink and the recent window.
from tempokv.runtime.jensen import query_channels as jensen_query_channels  # noqa: E402
from tempokv.runtime.jensen_flex import (  # noqa: E402
    corrected_attention as jensen_flex_attention,
    jensen_backend,
)
from tempokv.runtime.rope import (  # noqa: E402
    apply_rope_by_span,
    causal_rope_apply_long_input,
)


def _attention_window_ranges(
    *, local_end_index, frame_seqlen, local_attn_size, sink_size
):
    """Token ranges of the attention window inside a full resident cache read.

    The cache is always read in full so a packed TempoKV span is never cut, and
    the window is sliced afterwards.  Returning the ranges instead of the
    sliced tensors lets the keys, the values and the Jensen step sizes be cut
    identically without three copies of this arithmetic.
    """
    if local_attn_size == -1:
        return [(0, local_end_index)]
    sink_tokens = sink_size * frame_seqlen
    recent_frames = max(1, local_attn_size - sink_size)
    recent_start = max(0, local_end_index - recent_frames * frame_seqlen)
    if sink_size > 0 and recent_start > sink_tokens:
        return [(0, sink_tokens), (recent_start, local_end_index)]
    if sink_size > 0:
        # The sink overlaps the recent window, so their union covers the whole
        # resident range.
        return [(0, local_end_index)]
    return [(recent_start, local_end_index)]


def _take_window(tensor, ranges):
    """Slice ``[B, S, ...]`` down to the attention window, avoiding a copy."""
    if len(ranges) == 1:
        start, end = ranges[0]
        if start == 0 and end == tensor.shape[1]:
            return tensor
        return tensor[:, start:end]
    return torch.cat([tensor[:, start:end] for start, end in ranges], dim=1)


def _replace_with_cached_mean(step_channels):
    """Flatten the per-key squared step sizes to their mean over the cache.

    This is the ablation that decides which fast design to build.  The exact
    correction subtracts a per-key bias, which needs per-score work in the
    attention kernel; replacing the key term by its cache mean leaves a bias
    that is constant across the cached block, and a bias like that can be
    applied as a per-query offset when merging two plain attention calls by
    their log-sum-exp -- no widened head and no score_mod.

    That cheaper form keeps the cached-versus-current attention-mass shift the
    paper describes and drops the within-cache reordering that TempoKV's
    content-dependent step sizes add.  Running the quality comparison with this
    switch on says how much of the measured gain came from which.

    Tokens still held as BF16 report a zero step and must keep it: they carry
    no quantization noise, so averaging them into the cache mean -- or giving
    them a share of it -- would invent a correction for unquantized keys.
    """
    quantized = (step_channels != 0).any(dim=-1, keepdim=True)
    count = quantized.sum(dim=1, keepdim=True).clamp_min(1)
    mean = step_channels.sum(dim=1, keepdim=True) / count
    step_channels.copy_(
        torch.where(quantized, mean.expand_as(step_channels), step_channels)
    )


def _jensen_padded_width(head_dim, groups):
    """Head dimension once the correction's G channels are appended.

    Rounded up to a multiple of eight because that is what FlashAttention-2
    accepts, and FA2 wants one head dimension for Q, K and V alike.
    """
    return -(-(head_dim + groups) // 8) * 8


def _jensen_corrected_attention(query, key, value, *, head_dim, groups, group_size):
    """Attention with the Jensen bias removed from quantized cached scores.

    The correction is a rank-G bilinear form, so it folds into G extra Q/K
    channels and the ordinary attention kernel evaluates it -- no score_mod and
    no fused-kernel change.  ``key`` and ``value`` already arrive at the padded
    width, carrying the per-key squared step sizes the cache read wrote into
    their padding, so nothing cache-sized is copied here; only the query, which
    is one chunk, is widened.

    The softmax scale stays 1/sqrt(head_dim) of the *original* head dimension:
    the extra channels carry a correction already in score units and must not
    be renormalized by the widened head.

    The call pins FlashAttention-2.  FA2 takes any head dimension that is a
    multiple of eight, while FA3 -- which ``attention()`` selects on its own
    whenever it runs on Hopper -- supports only a fixed set of widths that the
    padded head is not in.
    """
    padded = key.shape[-1]
    if not FLASH_ATTN_2_AVAILABLE and FLASH_ATTN_3_AVAILABLE:
        raise RuntimeError(
            f"the Jensen correction pads the head dimension to {padded}, which "
            "FlashAttention-3 does not support, and FlashAttention-2 is not "
            "installed. Install flash-attn (v2) alongside it, or run without "
            "JENSEN=1."
        )
    extra = jensen_query_channels(query, group_size).to(query.dtype)
    widened_query = query.new_zeros((*query.shape[:-1], padded))
    widened_query[..., :head_dim] = query
    widened_query[..., head_dim:head_dim + groups] = extra
    out = attention(
        widened_query,
        key,
        value,
        softmax_scale=head_dim ** -0.5,
        fa_version=2,
    )
    return out[..., :head_dim]


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def _attention_with_incremental_cache(
        self,
        roped_query,
        roped_key,
        value,
        kv_cache,
        quantizer,
        current_start,
        current_end,
        cache_size,
        frame_seqlen,
    ):
        """Run attention against an append-only quantized KV cache.

        Mirrors the Causal-Forcing integration of the shared kv_quant
        baselines (RTN/KIVI/QuaRot-KV).  ``write_k/write_v`` inside the
        quantizer state represents the current diffusion block; Self-Forcing
        evaluates that block several times at different denoising timesteps,
        and replacing this buffer is safe while older packed segments stay
        immutable.
        """
        attention_space = hasattr(quantizer, "prepare_attention_qk")
        if attention_space:
            roped_query, roped_key = quantizer.prepare_attention_qk(roped_query, roped_key)

        state = kv_cache.get("quant_state")
        if state is None:
            state = quantizer.init_state(
                meta={
                    "shape": (int(roped_key.shape[0]), 0, int(roped_key.shape[2]), int(roped_key.shape[3])),
                    "tensor_dtype": value.dtype,
                    "device": value.device,
                    "attention_space": attention_space,
                }
            )

        previous_global_end = _cache_index(kv_cache, "global_end_index")
        previous_tokens = int(state.get("num_tokens", 0))
        num_new_tokens = int(roped_key.shape[1])

        if self.local_attn_size != -1 and current_end > previous_global_end:
            requested_eviction = max(previous_tokens + num_new_tokens - int(cache_size), 0)
            if requested_eviction > 0:
                sink_tokens = int(self.sink_size * frame_seqlen)
                if sink_tokens > 0 and hasattr(quantizer, "evict_range"):
                    removed = quantizer.evict_range(state, sink_tokens, requested_eviction)
                else:
                    removed = quantizer.evict_prefix(state, requested_eviction)
                # A partial packed block is intentionally retained.  The
                # attention slice below still enforces the requested window.
                if removed == 0:
                    kv_cache["eviction_slack_tokens"] = int(requested_eviction)

        quantizer.append_kv(
            state,
            roped_key,
            value,
            meta={
                "tensor_dtype": value.dtype,
                "absolute_start": int(current_start),
                "absolute_end": int(current_end),
                "already_rotated": attention_space,
                "attention_space": attention_space,
            },
        )
        cache_k, cache_v = quantizer.materialize_kv(
            state,
            meta={
                "tensor_dtype": value.dtype,
                "shape": (int(roped_key.shape[0]), 0, int(roped_key.shape[2]), int(roped_key.shape[3])),
                "device": value.device,
                "attention_space": attention_space,
            },
        )
        local_end_index = int(state.get("num_tokens", cache_k.shape[1]))
        attention_start = max(0, local_end_index - int(self.max_attention_size))
        x = attention(
            roped_query,
            cache_k[:, attention_start:local_end_index],
            cache_v[:, attention_start:local_end_index],
        )

        kv_cache["quant_state"] = state
        return x, local_end_index

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            current_end = current_start + roped_query.shape[1]
            num_new_tokens = roped_query.shape[1]

            # Shared kv_quant baselines (RTN/KIVI/QuaRot-KV) take the same
            # append-only incremental path used by the Causal-Forcing
            # integration; the ChunkedKVCache below stays BF16-only.
            quantizer = kv_cache.get("quantizer")
            supports_incremental = (
                quantizer is not None
                and callable(getattr(quantizer, "init_state", None))
                and callable(getattr(quantizer, "append_kv", None))
                and callable(getattr(quantizer, "materialize_kv", None))
            )
            if supports_incremental:
                roped_key = causal_rope_apply(
                    k, grid_sizes, freqs,
                    start_frame=current_start_frame).type_as(v)
                cache_size = kv_cache.get("kv_cache_size")
                if cache_size is None:
                    cache_size = int(self.max_attention_size)
                x, local_end_index = self._attention_with_incremental_cache(
                    roped_query=roped_query,
                    roped_key=roped_key,
                    value=v,
                    kv_cache=kv_cache,
                    quantizer=quantizer,
                    current_start=current_start,
                    current_end=current_end,
                    cache_size=int(cache_size),
                    frame_seqlen=frame_seqlen,
                )
                _set_cache_index(kv_cache, "global_end_index", current_end)
                _set_cache_index(kv_cache, "local_end_index", local_end_index)
                x = x.flatten(2)
                x = self.o(x)
                return x

            global_end_index = _cache_index(kv_cache, "global_end_index")
            cached_local_end = _cache_index(kv_cache, "local_end_index")
            local_end_index = cached_local_end + current_end - global_end_index
            local_start_index = local_end_index - num_new_tokens
            cache_backend = kv_cache.get("cache_backend")
            if cache_backend is None:
                # Preserve the direct attention-module API used by older
                # callers: a ChunkedKVCache object unambiguously selects the
                # pre-RoPE/read-time-RoPE path, while a tensor is native BF16.
                cache_backend = (
                    "bf16" if isinstance(kv_cache.get("k"), torch.Tensor)
                    else "tempokv"
                )
            if cache_backend != "tempokv":
                # Native BF16 baseline: post-RoPE K in a contiguous tensor
                # cache sized to the attention window, rolled in place, with no
                # ChunkedKVCache and no read-time RoPE over the whole rollout.
                #
                # The rotation goes through the span helper rather than
                # causal_rope_apply because that one hardcodes float64, while
                # this repository deliberately runs the BF16 reference on the
                # same arithmetic as the quantized run it is the reference for
                # (SELF_FORCING_CACHE_ROPE_PRECISION, complex64 for both under
                # the generation scripts).  On one contiguous block the two
                # helpers agree bit for bit at equal precision.
                roped_key = causal_rope_apply_long_input(
                    k, grid_sizes, freqs,
                    start_frame=current_start_frame,
                    precision=kv_cache.get("rope_precision", "native"),
                ).type_as(v)
                cache_k = kv_cache["k"]
                cache_v = kv_cache["v"]
                cache_size = int(cache_k.shape[1])
                if num_new_tokens > cache_size:
                    raise ValueError(
                        "Self-Forcing KV cache capacity is smaller than one block"
                    )
                sink_tokens = self.sink_size * frame_seqlen
                if (
                    self.local_attn_size != -1
                    and current_end > global_end_index
                    and cached_local_end + num_new_tokens > cache_size
                ):
                    num_evicted_tokens = (
                        cached_local_end + num_new_tokens - cache_size
                    )
                    num_rolled_tokens = (
                        cached_local_end - num_evicted_tokens - sink_tokens
                    )
                    if num_rolled_tokens > 0:
                        cache_k[:, sink_tokens:sink_tokens + num_rolled_tokens] = (
                            cache_k[
                                :,
                                sink_tokens + num_evicted_tokens:
                                sink_tokens + num_evicted_tokens + num_rolled_tokens,
                            ].clone()
                        )
                        cache_v[:, sink_tokens:sink_tokens + num_rolled_tokens] = (
                            cache_v[
                                :,
                                sink_tokens + num_evicted_tokens:
                                sink_tokens + num_evicted_tokens + num_rolled_tokens,
                            ].clone()
                        )
                    local_end_index = cached_local_end + current_end - global_end_index
                    local_end_index -= num_evicted_tokens
                    local_start_index = local_end_index - num_new_tokens
                elif local_end_index > cache_size:
                    raise ValueError("Self-Forcing KV cache capacity is too small")
                cache_k[:, local_start_index:local_end_index] = roped_key
                cache_v[:, local_start_index:local_end_index] = v
                attention_start = 0
                if self.local_attn_size != -1:
                    attention_start = max(
                        0, local_end_index - self.local_attn_size * frame_seqlen
                    )
                x = attention(
                    roped_query,
                    cache_k[:, attention_start:local_end_index],
                    cache_v[:, attention_start:local_end_index],
                )
                _set_cache_index(kv_cache, "global_end_index", current_end)
                _set_cache_index(kv_cache, "local_end_index", local_end_index)
                x = x.flatten(2)
                x = self.o(x)
                return x

            if local_start_index < 0 or local_end_index > kv_cache["k"].kv_cache_size:
                raise ValueError("Self-Forcing KV cache capacity is too small")

            # K is deliberately stored before RoPE. TempoKV is fitted in this
            # coordinate system and RoPE is reapplied after reconstruction.
            kv_cache["k"].write(local_start_index, local_end_index, k)
            kv_cache["v"].write(local_start_index, local_end_index, v)

            # Always read the full resident range so a quantized TempoKV span is
            # never cut (the cache rejects partial-span reads), then slice the
            # attention window: [sink] ∪ [recent].  The whole range is rotated at
            # absolute frame positions before slicing, so a gapped window keeps
            # each side on its own absolute RoPE timeline.
            # The cross projection consumes the reconstructed primary tensor
            # before RoPE, so decode the dependent tensor while that one is
            # still cache-hot; rotating K first streams over a second full-span
            # tensor and needlessly evicts the projection source.  Which slot
            # leads follows the codec direction the bridge stamped on the cache,
            # and defaults to the production K->V order for every other path.
            direction = getattr(
                kv_cache["k"], "tempokv_predictor_direction", "k_to_v"
            )
            primary_role, secondary_role = (
                ("v", "k") if direction == "v_to_k" else ("k", "v")
            )
            # The Jensen correction rides in extra head channels.  Decoding
            # straight into the widened buffer, and rotating in place, keeps
            # the cost at the padding itself: building the augmented K and the
            # matching V with torch.cat instead copied the whole resident
            # window twice per layer per forward.
            head_dim = kv_cache["k"].head_dim
            jensen_groups = (
                head_dim // int(getattr(kv_cache["k"], "tempokv_group_size", 64))
                if getattr(kv_cache["k"], "tempokv_jensen_correction", False)
                else 0
            )
            fused_jensen = bool(jensen_groups) and jensen_backend() == "flex"
            extra_channels = (
                _jensen_padded_width(head_dim, jensen_groups) - head_dim
                if jensen_groups and not fused_jensen
                else 0
            )
            primary_all = kv_cache[primary_role].read(
                0, local_end_index, extra_channels=extra_channels
            )
            secondary_all = kv_cache[secondary_role].read(
                0,
                local_end_index,
                source_tensor=primary_all[..., :head_dim],
                extra_channels=extra_channels,
            )
            k_all, v_all = (
                (secondary_all, primary_all)
                if direction == "v_to_k"
                else (primary_all, secondary_all)
            )
            narrow_k = k_all[..., :head_dim]
            causal_rope_apply_long_input(
                narrow_k,
                grid_sizes,
                freqs,
                start_frame=0,
                precision=getattr(kv_cache["k"], "rope_precision", "native"),
                out=narrow_k,
            )
            if jensen_groups:
                # The fused route keeps K/V narrow; only G small channels are
                # allocated separately. Preserve legacy cast-then-square order.
                step_channels = (
                    k_all.new_empty((*k_all.shape[:-1], jensen_groups))
                    if fused_jensen
                    else k_all[..., head_dim:head_dim + jensen_groups]
                )
                kv_cache["k"].read_step_sizes(
                    0,
                    local_end_index,
                    group_size=int(getattr(kv_cache["k"], "tempokv_group_size", 64)),
                    out=step_channels,
                    squared=True,
                )
                if getattr(kv_cache["k"], "tempokv_jensen_cache_mean", False):
                    _replace_with_cached_mean(step_channels)
            absolute_k = k_all
            window = _attention_window_ranges(
                local_end_index=local_end_index,
                frame_seqlen=frame_seqlen,
                local_attn_size=self.local_attn_size,
                sink_size=self.sink_size,
            )
            attn_k = _take_window(absolute_k, window)
            attn_v = _take_window(v_all, window)
            recorder = getattr(self, "tempokv_trace_recorder", None)
            if recorder is not None:
                # The attention objective needs the rotated Q and K the model
                # actually used.  The newly written tokens are always the tail of
                # the attention window, so no RoPE is recomputed for the trace.
                recorder(roped_query, attn_k[:, -num_new_tokens:])
            if fused_jensen:
                x = jensen_flex_attention(
                    roped_query,
                    attn_k,
                    attn_v,
                    _take_window(step_channels, window),
                    group_size=int(
                        getattr(kv_cache["k"], "tempokv_group_size", 64)
                    ),
                )
            elif jensen_groups:
                x = _jensen_corrected_attention(
                    roped_query,
                    attn_k,
                    attn_v,
                    head_dim=head_dim,
                    groups=jensen_groups,
                    group_size=int(
                        getattr(kv_cache["k"], "tempokv_group_size", 64)
                    ),
                )
            else:
                x = attention(roped_query, attn_k, attn_v)
            _set_cache_index(kv_cache, "global_end_index", current_end)
            _set_cache_index(kv_cache, "local_end_index", local_end_index)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start)

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
