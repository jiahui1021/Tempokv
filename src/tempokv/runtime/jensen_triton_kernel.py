"""Tiled online-softmax attention with an in-register Jensen correction.

Each CTA owns a query tile; each query norm is reduced once before the KV
loop. Only G step values per key are loaded alongside the original D-wide K/V.
The online softmax recurrence follows the FlashAttention formulation, also
documented at https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html.
No Q-by-K allocation, Q padding, coefficient preparation or output copy.
"""

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
    q2 = q.to(tl.float32) * q.to(tl.float32)
    coefficients = ()
    for g in tl.static_range(D // GROUP_SIZE):
        energy = tl.sum(tl.where(
            (d[None, :] >= g * GROUP_SIZE) & (d[None, :] < (g + 1) * GROUP_SIZE),
            q2, 0.0,
        ), axis=1)
        coefficient = (-energy / (24.0 * D ** 0.5)).to(q.dtype).to(tl.float32)
        coefficients += (coefficient,)

    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, D), tl.float32)
    for start in range(0, tl.cdiv(NK, BLOCK_N)):
        kn = start * BLOCK_N + n
        k = tl.load(
            K + b * k_b + h * k_h + d[:, None] * k_d + kn[None, :].to(tl.int64) * k_s,
            mask=kn[None, :] < NK, other=0,
        )
        scores = tl.dot(q, k) * (D ** -0.5)
        correction = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for g in tl.static_range(D // GROUP_SIZE):
            delta2 = tl.load(
                Steps + b * t_b + h * t_h + kn.to(tl.int64) * t_s + g * t_g,
                mask=kn < NK, other=0,
            ).to(tl.float32)
            correction += coefficients[g][:, None] * delta2[None, :]
        scores = (scores + correction * (D ** -0.5)) * 1.4426950408889634
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


def attention_forward(query, key, value, step_squared, group_size):
    batch, nq, heads, dim = query.shape
    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    # Conservative Ampere tile; tuning needs measurements on the target GPU.
    # NQ/NK and strides stay runtime values, avoiding a compile per KV length.
    _attention_forward[(triton.cdiv(nq, 64), batch * heads)](
        query, key, value, step_squared, output,
        nq, key.shape[1],
        *query.stride(), *key.stride(), *value.stride(), *step_squared.stride(),
        H=heads, D=dim, GROUP_SIZE=group_size, BLOCK_M=64, BLOCK_N=64,
        num_warps=4, num_stages=2, enable_fp_fusion=False,
    )
    return output
