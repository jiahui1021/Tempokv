"""Experimental comparison schedule; not recommended for Jensen generation.

The measured baseline lives in jensen_triton_kernel.py. This candidate keeps
the same Q/K/V and rounded MMA correction, but changes traversal and scheduling.
Only the tail needs key bounds checks; full tiles are visited in reverse order.
Keep the measured baseline+MMA route for generation; this candidate is retained
to reproduce the interaction between scheduling and correction operands.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _update_tile(
    q, qc, K, V, Steps, start, NK,
    k_s, k_d, v_s, v_d, t_s, t_g,
    maximum, denominator, accumulator,
    D: tl.constexpr, GROUPS: tl.constexpr, BLOCK_N: tl.constexpr,
    APPLY_CORRECTION: tl.constexpr, MASK_N: tl.constexpr,
):
    n = start + tl.arange(0, BLOCK_N)
    d = tl.arange(0, D)
    kp = K + d[:, None] * k_d + n[None, :].to(tl.int64) * k_s
    vp = V + n[:, None].to(tl.int64) * v_s + d[None, :] * v_d
    # Load all operands before the dependent arithmetic; tl.range stages the
    # surrounding full-tile loop, including these inlined loads.
    if MASK_N:
        k = tl.load(kp, mask=n[None, :] < NK, other=0)
        v = tl.load(vp, mask=n[:, None] < NK, other=0)
    else:
        k = tl.load(kp)
        v = tl.load(vp)
    if APPLY_CORRECTION:
        rank = tl.arange(0, 16)
        tp = Steps + rank[:, None] * t_g + n[None, :].to(tl.int64) * t_s
        if MASK_N:
            kc = tl.load(tp, mask=(rank[:, None] < GROUPS) & (n[None, :] < NK), other=0)
        else:
            # Padding on the small rank axis is still required, even for full
            # key tiles. It is a compile-time group boundary, not an NK check.
            kc = tl.load(tp, mask=rank[:, None] < GROUPS, other=0)

    scores = tl.dot(q, k)
    if APPLY_CORRECTION:
        scores = tl.dot(qc, kc, scores)
    if MASK_N:
        scores = tl.where(n[None, :] < NK, scores, -float("inf"))
    scale: tl.constexpr = D ** -0.5 * 1.4426950408889634
    # Reduce in unscaled score space. Each score is scaled only in the final
    # exp2 argument; do not build an extra scaled score tile before the max.
    new_maximum = tl.maximum(maximum, tl.max(scores, axis=1) * scale)
    probabilities = tl.exp2(scores * scale - new_maximum[:, None])
    alpha = tl.exp2(maximum - new_maximum)
    probability_sum = tl.sum(probabilities, axis=1)
    accumulator = accumulator * alpha[:, None]
    accumulator = tl.dot(probabilities.to(q.dtype), v, accumulator)
    denominator = denominator * alpha + probability_sum
    return new_maximum, denominator, accumulator


@triton.jit(do_not_specialize=["NQ", "NK"])
def _pipeline_forward(
    Q, K, V, Steps, Out, NQ, NK,
    q_b, q_s, q_h, q_d, k_b, k_s, k_h, k_d,
    v_b, v_s, v_h, v_d, t_b, t_s, t_h, t_g,
    H: tl.constexpr, D: tl.constexpr, GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    PIPE_STAGES: tl.constexpr, APPLY_CORRECTION: tl.constexpr,
):
    bh = tl.program_id(1)
    b = (bh // H).to(tl.int64)
    h = (bh % H).to(tl.int64)
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, D)
    q = tl.load(
        Q + b * q_b + h * q_h + m[:, None].to(tl.int64) * q_s + d[None, :] * q_d,
        mask=m[:, None] < NQ, other=0,
    )
    qc = tl.full((BLOCK_M, 16), 0.0, tl.float32)
    if APPLY_CORRECTION:
        rank = tl.arange(0, 16)
        q2 = q.to(tl.float32) * q.to(tl.float32)
        for g in tl.static_range(D // GROUP_SIZE):
            energy = tl.sum(tl.where(
                (d[None, :] >= g * GROUP_SIZE) & (d[None, :] < (g + 1) * GROUP_SIZE),
                q2, 0.0,
            ), axis=1)
            coeff = (-energy / (24.0 * D ** 0.5)).to(q.dtype).to(tl.float32)
            qc = tl.where(rank[None, :] == g, coeff[:, None], qc)
    qc = qc.to(q.dtype)
    K = K + b * k_b + h * k_h
    V = V + b * v_b + h * v_h
    Steps = Steps + b * t_b + h * t_h
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, D), tl.float32)

    # NK>0 is validated by the public entry. The last tile is handled exactly
    # once, including when NK is a multiple of BLOCK_N or shorter than a tile.
    full_tiles = (NK - 1) // BLOCK_N
    tail_start = full_tiles * BLOCK_N
    maximum, denominator, accumulator = _update_tile(
        q, qc, K, V, Steps, tail_start, NK, k_s, k_d, v_s, v_d, t_s, t_g,
        maximum, denominator, accumulator,
        D, D // GROUP_SIZE, BLOCK_N, APPLY_CORRECTION, True,
    )
    for index in tl.range(0, full_tiles, num_stages=PIPE_STAGES):
        start = tail_start - (index + 1) * BLOCK_N
        maximum, denominator, accumulator = _update_tile(
            q, qc, K, V, Steps, start, NK, k_s, k_d, v_s, v_d, t_s, t_g,
            maximum, denominator, accumulator,
            D, D // GROUP_SIZE, BLOCK_N, APPLY_CORRECTION, False,
        )
    tl.store(
        Out + ((b * NQ + m[:, None].to(tl.int64)) * H + h) * D + d[None, :],
        accumulator / denominator[:, None], mask=m[:, None] < NQ,
    )


def attention_forward(query, key, value, step_squared, group_size, *, config,
                      apply_correction=True, return_metadata=False):
    from .jensen_triton_kernel import _TILES, _kernel_metadata

    if not hasattr(tl, "range"):
        raise RuntimeError("pipeline implementation requires Triton with tl.range(num_stages=...)")
    if config == "auto":
        raise ValueError("pipeline comparison requires a fixed TEMPOKV_JENSEN_TRITON_CONFIG")
    m, n, warps, stages = _TILES["64x64" if config == "legacy" else config]
    batch, nq, heads, dim = query.shape
    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    compiled = _pipeline_forward[(triton.cdiv(nq, m), batch * heads)](
        query, key, value, step_squared, output, nq, key.shape[1],
        *query.stride(), *key.stride(), *value.stride(), *step_squared.stride(),
        H=heads, D=dim, GROUP_SIZE=group_size, BLOCK_M=m, BLOCK_N=n,
        # Stage both K and V twice. Blindly staging both four times can exceed
        # Ampere shared memory once Q and the correction operands are included.
        PIPE_STAGES=2, APPLY_CORRECTION=apply_correction,
        num_warps=warps, num_stages=stages, enable_fp_fusion=config != "legacy",
    )
    if return_metadata:
        metadata = _kernel_metadata(compiled, config, "mma" if apply_correction else "none", "pipeline")
        metadata["loop_pipeline_stages"] = 2
        return output, metadata
    return output
