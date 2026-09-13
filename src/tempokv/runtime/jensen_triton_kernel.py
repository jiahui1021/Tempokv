"""Tiled online-softmax attention with an in-register Jensen correction.

Each CTA owns a query tile; each query norm is reduced once before the KV
loop. Only G step values per key are loaded alongside the original D-wide K/V.
The online softmax recurrence follows the FlashAttention formulation, also
documented at https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html.
No Q-by-K allocation, Q padding, coefficient preparation or output copy.
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["NQ", "NK"])
def _attention_forward(
    Q, K, V, Steps, Out,
    NQ, NK,
    q_b, q_s, q_h, q_d,
    k_b, k_s, k_h, k_d,
    v_b, v_s, v_h, v_d,
    t_b, t_s, t_h, t_g,
    H: tl.constexpr, D: tl.constexpr, GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    APPLY_CORRECTION: tl.constexpr,
    CORRECTION_MMA: tl.constexpr,
    Q_BUCKET: tl.constexpr, K_BUCKET: tl.constexpr,
):
    bh = tl.program_id(1)
    b = (bh // H).to(tl.int64)
    h = (bh % H).to(tl.int64)
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    n = tl.arange(0, BLOCK_N)
    d = tl.arange(0, D)
    q = tl.load(
        Q + b * q_b + h * q_h + m[:, None].to(tl.int64) * q_s + d[None, :] * q_d,
        mask=m[:, None] < NQ, other=0,
    )
    # Preserve legacy FP32 reduction -> half coefficient -> FP32 score math.
    # Tuples are expanded at compile time; coefficients live outside the KV loop.
    coefficients = ()
    if APPLY_CORRECTION:
        q2 = q.to(tl.float32) * q.to(tl.float32)
        for g in tl.static_range(D // GROUP_SIZE):
            energy = tl.sum(tl.where(
                (d[None, :] >= g * GROUP_SIZE) & (d[None, :] < (g + 1) * GROUP_SIZE),
                q2, 0.0,
            ), axis=1)
            coefficient = (-energy / (24.0 * D ** 0.5)).to(q.dtype).to(tl.float32)
            coefficients += (coefficient,)
        if CORRECTION_MMA:
            # Only the small rank axis is padded, not the D-wide Q/K/V.
            # Half inputs already match legacy coefficient/step rounding.
            # Build the Q operand once; its dot-operand layout is reused in
            # every KV tile rather than broadcasting G vectors over scores.
            rank = tl.arange(0, 16)
            q_correction = tl.full((BLOCK_M, 16), 0.0, tl.float32)
            for g in tl.static_range(D // GROUP_SIZE):
                q_correction = tl.where(
                    rank[None, :] == g, coefficients[g][:, None], q_correction,
                )
            q_correction = q_correction.to(q.dtype)

    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, D), tl.float32)
    for start in range(0, tl.cdiv(NK, BLOCK_N)):
        kn = start * BLOCK_N + n
        k = tl.load(
            K + b * k_b + h * k_h + d[:, None] * k_d + kn[None, :].to(tl.int64) * k_s,
            mask=kn[None, :] < NK, other=0,
        )
        scores = tl.dot(q, k)
        if APPLY_CORRECTION and CORRECTION_MMA:
            k_correction = tl.load(
                Steps + b * t_b + h * t_h
                + kn[None, :].to(tl.int64) * t_s + rank[:, None] * t_g,
                mask=(kn[None, :] < NK) & (rank[:, None] < D // GROUP_SIZE),
                other=0,
            )
            # Accumulate directly into QK's FP32 MMA accumulator. This avoids
            # the separate FP32 score-sized correction + scalar outer products.
            scores = tl.dot(q_correction, k_correction, scores)
        scores = scores * (D ** -0.5)
        if APPLY_CORRECTION and not CORRECTION_MMA:
            correction = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            for g in tl.static_range(D // GROUP_SIZE):
                delta2 = tl.load(
                    Steps + b * t_b + h * t_h + kn.to(tl.int64) * t_s + g * t_g,
                    mask=kn < NK, other=0,
                ).to(tl.float32)
                correction += coefficients[g][:, None] * delta2[None, :]
            scores = scores + correction * (D ** -0.5)
        scores = scores * 1.4426950408889634
        scores = tl.where(kn[None, :] < NK, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        probabilities = tl.exp2(scores - next_maximum[:, None])
        alpha = tl.exp2(maximum - next_maximum)
        denominator = denominator * alpha + tl.sum(probabilities, axis=1)
        accumulator = accumulator * alpha[:, None]
        v = tl.load(
            V + b * v_b + h * v_h + kn[:, None].to(tl.int64) * v_s + d[None, :] * v_d,
            mask=kn[:, None] < NK, other=0,
        )
        accumulator = tl.dot(probabilities.to(q.dtype), v, accumulator)
        maximum = next_maximum

    output = accumulator / denominator[:, None]
    tl.store(
        Out + ((b * NQ + m[:, None].to(tl.int64)) * H + h) * D + d[None, :],
        output, mask=m[:, None] < NQ,
    )


# (BLOCK_M, BLOCK_N, num_warps, num_stages). User measurements show severe
# slowdowns for some 4-warp fused tiles. Resource pressure is a hypothesis;
# register allocation/spills must be checked from the compiled kernel.
# Keep the measured candidates for reproducible comparison.
_TILES = {
    "64x64": (64, 64, 4, 2),
    "128x64": (128, 64, 4, 2),
    "128x32": (128, 32, 4, 3),
    "64x64w8s3": (64, 64, 8, 3),
    "64x64w8s4": (64, 64, 8, 4),
    "128x64w8s3": (128, 64, 8, 3),
    "128x64w8s4": (128, 64, 8, 4),
    "128x32w8s4": (128, 32, 8, 4),
}
_tuned_forward = triton.autotune(
    configs=[triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=w, num_stages=s)
             for m, n, w, s in _TILES.values()],
    key=["H", "D", "GROUP_SIZE", "APPLY_CORRECTION", "CORRECTION_MMA", "Q_BUCKET", "K_BUCKET"],
    warmup=5, rep=15,
)(_attention_forward)


def attention_forward(query, key, value, step_squared, group_size, *,
                      config=None, apply_correction=True, correction_mode=None,
                      return_metadata=False):
    """Production always corrects; plain controls are private benchmark-only.

    ``auto`` tunes candidates per power-of-two sequence bucket. A fixed tile
    or ``legacy`` bypasses tuning for reproducible comparisons/cold starts.
    The legacy option retains the previous global FP-fusion disable setting.
    ``scalar`` preserves the user-measured implementation; ``mma`` is an opt-in
    small Tensor Core correction with a different accumulation order.
    """
    config = config or os.environ.get("TEMPOKV_JENSEN_TRITON_CONFIG", "auto")
    correction_mode = correction_mode or os.environ.get("TEMPOKV_JENSEN_TRITON_CORRECTION", "scalar")
    if correction_mode not in {"scalar", "mma"}:
        raise ValueError("TEMPOKV_JENSEN_TRITON_CORRECTION must be scalar or mma")
    if config not in {"auto", "legacy", *_TILES}:
        raise ValueError(
            "TEMPOKV_JENSEN_TRITON_CONFIG must be auto, legacy or one of: "
            + ", ".join(_TILES)
        )
    batch, nq, heads, dim = query.shape
    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    launch = _tuned_forward if config == "auto" else _attention_forward
    options = {}
    if config != "auto":
        m, n, warps, stages = _TILES["64x64" if config == "legacy" else config]
        options = dict(BLOCK_M=m, BLOCK_N=n, num_warps=warps, num_stages=stages)
    compiled = launch[lambda meta: (triton.cdiv(nq, meta["BLOCK_M"]), batch * heads)](
        query, key, value, step_squared, output,
        nq, key.shape[1],
        *query.stride(), *key.stride(), *value.stride(), *step_squared.stride(),
        H=heads, D=dim, GROUP_SIZE=group_size, APPLY_CORRECTION=apply_correction,
        CORRECTION_MMA=apply_correction and correction_mode == "mma",
        Q_BUCKET=triton.next_power_of_2(nq), K_BUCKET=triton.next_power_of_2(key.shape[1]),
        enable_fp_fusion=config != "legacy", **options,
    )
    if return_metadata:
        meta = getattr(compiled, "metadata", None)
        ptx = getattr(compiled, "asm", {}).get("ptx", "")
        return output, {
            "config": str(getattr(launch, "best_config", config)),
            "correction_mode": correction_mode if apply_correction else "none",
            "n_regs": getattr(compiled, "n_regs", None),
            "n_spills": getattr(compiled, "n_spills", None),
            "shared_mem_bytes": getattr(meta, "shared", None),
            "ptx_mma_instructions": ptx.count("mma.sync") if ptx else None,
            "ptx_local_load_instructions": ptx.count("ld.local") if ptx else None,
            "ptx_local_store_instructions": ptx.count("st.local") if ptx else None,
        }
    return output
