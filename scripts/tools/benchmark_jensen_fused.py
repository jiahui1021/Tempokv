#!/usr/bin/env python3
"""Compare augmented and fused Jensen attention on an A100.

Run: PYTHONPATH=src python scripts/tools/benchmark_jensen_fused.py
Cache decode, RoPE and scale extraction are excluded from timing.
"""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tempokv.runtime.jensen import query_channels  # noqa: E402
from tempokv.runtime.jensen_flex import (  # noqa: E402
    corrected_attention as flex_attention, score_terms, _compiled_attention,
)
from tempokv.runtime.jensen_triton import corrected_attention as triton_attention  # noqa: E402

BACKENDS = {"flex": flex_attention, "triton": triton_attention}


def triton_config_names():
    # The benchmark must check every candidate production auto tuning can pick.
    # Import lazily to keep --help and CPU-side comparison tests CUDA-free.
    from tempokv.runtime.jensen_triton_kernel import _TILES

    return ("legacy", *_TILES)


def plain_flex(q, k, v):
    return _compiled_attention()(
        q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3),
        scale=q.shape[-1] ** -0.5,
    ).permute(0, 2, 1, 3).contiguous()


def raw_triton(q, k, v, steps, group_size, *, config=None, correction=True,
               correction_mode=None, return_metadata=False, implementation=None):
    from tempokv.runtime.jensen_triton_kernel import attention_forward

    return attention_forward(q, k, v, steps, group_size,
                             config=config, apply_correction=correction,
                             correction_mode=correction_mode, return_metadata=return_metadata,
                             implementation=implementation)


def comparison(plain_ms, corrected_ms, fa2_ms):
    """Separate backend replacement from correction; retain negative/noisy deltas."""
    return {
        "backend_extra_ms": plain_ms - fa2_ms,
        "correction_extra_ms": corrected_ms - plain_ms,
        "backend_vs_fa2_pct": 100 * (plain_ms / fa2_ms - 1),
        "correction_vs_same_backend_pct": 100 * (corrected_ms / plain_ms - 1),
        "total_vs_fa2_pct": 100 * (corrected_ms / fa2_ms - 1),
    }


def check_correction_signal(group_size, mode, implementation=None):
    """Known answers sensitive to omitted, wrongly signed or misgrouped bias."""
    groups = 128 // group_size
    q = torch.zeros(1, groups, 1, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(1, groups + 1, 1, 128, dtype=q.dtype, device=q.device)
    v = torch.zeros_like(k)
    steps = torch.zeros(1, groups + 1, 1, groups, dtype=q.dtype, device=q.device)
    for g in range(groups):
        q[:, g, :, g * group_size:(g + 1) * group_size] = 1
        steps[:, g, :, g] = 24 * 128 / group_size
    for token in range(groups + 1):
        v[:, token, :, token] = 1
    coeff = float(torch.tensor(-group_size / (24 * math.sqrt(128)), dtype=q.dtype))
    weight = math.exp(coeff * (24 * 128 / group_size) / math.sqrt(128))
    weights = torch.ones(groups, groups + 1, device=q.device)
    indices = torch.arange(groups, device=q.device)
    weights[indices, indices] = weight
    weights /= weights.sum(-1, keepdim=True)
    expected = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    expected[0, :, 0, :groups + 1] = weights
    actual = raw_triton(q, k, v, steps, group_size, correction_mode=mode,
                       implementation=implementation).float()
    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.005)
    return (actual - expected).abs().max().item()


def augmented(q, wide_k, wide_v, group_size):
    dim = q.shape[-1]
    groups = dim // group_size
    wide_q = q.new_zeros((*q.shape[:-1], wide_k.shape[-1]))
    wide_q[..., :dim] = q
    wide_q[..., dim:dim + groups] = query_channels(q, group_size).to(q.dtype)
    return attention(wide_q, wide_k, wide_v, softmax_scale=dim ** -0.5,
                     fa_version=2)[..., :dim]


def widened(k, v, squared_steps):
    dim, groups = k.shape[-1], squared_steps.shape[-1]
    width = -(-(dim + groups) // 8) * 8
    wk = k.new_zeros((*k.shape[:-1], width))
    wv = v.new_zeros((*v.shape[:-1], width))
    wk[..., :dim], wv[..., :dim] = k, v
    wk[..., dim:dim + groups] = squared_steps
    return wk, wv


def check(group_size, backends, tune_triton=False, compare_correction=False,
          tune_configs=None, compare_implementations=False):
    q = torch.randn(1, 17, 2, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 257, 2, 128, dtype=q.dtype, device=q.device)
    v = torch.randn_like(k)
    steps = (torch.rand(1, 257, 2, 128 // group_size, device=q.device) * 2).to(q.dtype).pow(2)
    steps[:, -17:] = 0
    qt, kt = score_terms(q, steps, group_size)
    scores = q.float().permute(0, 2, 1, 3) @ k.float().permute(0, 2, 3, 1)
    scores = (scores + torch.einsum("gbhq,gbhk->bhqk", qt, kt)) * 128 ** -0.5
    ref = (scores.softmax(-1) @ v.float().permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    plain_scores = q.float().permute(0, 2, 1, 3) @ k.float().permute(0, 2, 3, 1) * 128 ** -0.5
    plain_ref = (plain_scores.softmax(-1) @ v.float().permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    wk, wv = widened(k, v, steps)
    legacy = augmented(q, wk, wv, group_size).float()
    torch.testing.assert_close(legacy, ref, rtol=0.02, atol=0.02)
    errors = {"legacy_max_abs_error": (legacy - ref).abs().max().item()}
    for backend in backends:
        if backend == "triton" and compare_implementations:
            continue
        fused = BACKENDS[backend](q, k, v, steps, group_size=group_size).float()
        torch.testing.assert_close(fused, ref, rtol=0.02, atol=0.02)
        errors[f"{backend}_max_abs_error"] = (fused - ref).abs().max().item()
        plain = (plain_flex(q, k, v) if backend == "flex" else raw_triton(
            q, k, v, steps, group_size, correction=False)).float()
        torch.testing.assert_close(plain, plain_ref, rtol=0.02, atol=0.02)
        errors[f"{backend}_plain_max_abs_error"] = (plain - plain_ref).abs().max().item()
    if tune_triton:
        for config in (tune_configs or triton_config_names()):
            for correction in (False, True):
                out = raw_triton(q, k, v, steps, group_size, config=config, correction=correction).float()
                expected = ref if correction else plain_ref
                torch.testing.assert_close(out, expected, rtol=0.02, atol=0.02)
                errors[f"triton_{config}_{correction}_max_abs_error"] = (out - expected).abs().max().item()
    if compare_correction:
        for mode in ("scalar", "mma"):
            out = raw_triton(q, k, v, steps, group_size, correction_mode=mode).float()
            torch.testing.assert_close(out, ref, rtol=0.02, atol=0.02)
            errors[f"triton_{mode}_max_abs_error"] = (out - ref).abs().max().item()
            errors[f"triton_{mode}_analytic_max_abs_error"] = check_correction_signal(group_size, mode)
    if compare_implementations:
        for impl in ("baseline", "pipeline"):
            for correction in (False, True):
                out = raw_triton(q, k, v, steps, group_size, correction_mode="mma",
                                 correction=correction, implementation=impl).float()
                expected = ref if correction else plain_ref
                torch.testing.assert_close(out, expected, rtol=0.02, atol=0.02)
                errors[f"{impl}_{correction}_max_abs_error"] = (out - expected).abs().max().item()
            errors[f"{impl}_analytic_max_abs_error"] = check_correction_signal(group_size, "mma", impl)
    return errors


def measure(fn, repeats, warmup):
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    first = time.perf_counter() - start
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    wall_samples = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        wall_start = time.perf_counter()
        start_event.record()
        fn()
        end_event.record()
        end_event.synchronize()
        samples.append(start_event.elapsed_time(end_event))
        wall_samples.append((time.perf_counter() - wall_start) * 1000)
    return {"shape_first_call_s": first, "median_ms": statistics.median(samples),
            "synchronized_wall_median_ms": statistics.median(wall_samples),
            "min_ms": min(samples), "samples_ms": samples}


@torch.no_grad()
def main():
    global attention
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-tokens", type=int, default=4680)
    parser.add_argument("--key-tokens", type=int, default=280800)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--backends", nargs="+", choices=tuple(BACKENDS), default=["flex", "triton"])
    parser.add_argument("--tune-triton", action="store_true",
                        help="compare legacy and every current fixed tile, with correction compiled out/in")
    parser.add_argument("--tune-configs", nargs="+",
                        help="restrict --tune-triton to these fixed configs; compare by corrected time")
    parser.add_argument("--compare-correction", action="store_true",
                        help="same fixed tile: plain vs scalar correction vs Tensor Core correction")
    parser.add_argument("--compare-implementations", action="store_true",
                        help="fixed tile: measured baseline vs pipelined backbone, each with plain/MMA")
    parser.add_argument("--triton-config", help="fixed tile; compare-correction defaults to 128x64w8s4")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("requires CUDA, Inductor/Triton and FlashAttention-2")
    if args.tune_triton and "triton" not in args.backends:
        parser.error("--tune-triton requires --backends triton")
    if args.compare_implementations:
        if "triton" not in args.backends:
            parser.error("--compare-implementations requires --backends triton")
        if args.compare_correction or args.tune_triton:
            parser.error("--compare-implementations is separate from correction/tile sweeps")
    if args.tune_configs and not args.tune_triton:
        parser.error("--tune-configs requires --tune-triton")
    tune_configs = ()
    if args.tune_triton:
        available = triton_config_names()
        tune_configs = tuple(dict.fromkeys(args.tune_configs or available))
        if any(config not in available for config in tune_configs):
            parser.error("unknown --tune-configs value")
    if args.compare_correction and "triton" not in args.backends:
        parser.error("--compare-correction requires --backends triton")
    if (args.compare_correction or args.compare_implementations) and args.triton_config is None:
        args.triton_config = "128x64w8s4"
    if args.triton_config is not None:
        if args.triton_config not in (*triton_config_names(), "auto"):
            parser.error("unknown --triton-config")
        if (args.compare_correction or args.compare_implementations) and args.triton_config == "auto":
            parser.error("paired comparisons need a fixed tile")
        os.environ["TEMPOKV_JENSEN_TRITON_CONFIG"] = args.triton_config
    # Import the real wrapper directly, not wan.__init__, which loads unrelated
    # model/config dependencies before a standalone kernel benchmark can run.
    spec = importlib.util.spec_from_file_location(
        "jensen_benchmark_attention", ROOT / "experiments" / "Self-Forcing" / "wan" / "modules" / "attention.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not module.FLASH_ATTN_2_AVAILABLE:
        parser.error("requires FlashAttention-2")
    attention = module.attention
    if args.group_size <= 0 or 128 % args.group_size:
        parser.error("group-size must divide 128")
    if min(args.query_tokens, args.key_tokens, args.heads, args.batch, args.repeats) <= 0 or args.warmup < 0:
        parser.error("dimensions/repeats must be positive and warmup nonnegative")
    import flash_attn
    torch.manual_seed(51)
    if "triton" in args.backends and args.group_size not in (32, 64, 128):
        parser.error("Triton group-size must be 32, 64 or 128")
    print("Checking numerical correctness (first call can compile/tune)...", file=sys.stderr, flush=True)
    correctness = check(args.group_size, args.backends, args.tune_triton, args.compare_correction,
                        tune_configs, args.compare_implementations)
    q = torch.randn(args.batch, args.query_tokens, args.heads, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(args.batch, args.key_tokens, args.heads, 128, dtype=q.dtype, device=q.device)
    v = torch.randn_like(k)
    steps = torch.rand(args.batch, args.key_tokens, args.heads, 128 // args.group_size, device=q.device).to(q.dtype).pow(2)
    steps[:, -min(args.query_tokens, args.key_tokens):] = 0
    wk, wv = widened(k, v, steps)
    calls = {
        "fa2_plain_128": lambda: attention(q, k, v, softmax_scale=128 ** -0.5, fa_version=2),
        "fa2_augmented": lambda: augmented(q, wk, wv, args.group_size),
    }
    for backend in dict.fromkeys(args.backends):
        if backend == "triton" and args.compare_implementations:
            continue
        calls[f"{backend}_plain_128"] = (
            (lambda: plain_flex(q, k, v)) if backend == "flex" else
            (lambda: raw_triton(q, k, v, steps, args.group_size, correction=False))
        )
        calls[f"{backend}_fused_128"] = lambda backend=backend: BACKENDS[backend](
            q, k, v, steps, group_size=args.group_size)
    if args.compare_correction:
        # Explicit mode names cannot be confused by a production env override.
        calls.pop("triton_fused_128")
        for mode in ("scalar", "mma"):
            calls[f"triton_{mode}_fused_128"] = lambda mode=mode: raw_triton(
                q, k, v, steps, args.group_size, correction_mode=mode)
    if args.compare_implementations:
        for impl in ("baseline", "pipeline"):
            for correction in (False, True):
                name = f"triton_{impl}_{'mma' if correction else 'plain'}_128"
                calls[name] = lambda impl=impl, correction=correction: raw_triton(
                    q, k, v, steps, args.group_size, correction_mode="mma",
                    correction=correction, implementation=impl)
    if args.tune_triton:
        for config in tune_configs:
            for correction in (False, True):
                name = f"triton_{config}_{'fused' if correction else 'plain'}"
                calls[name] = lambda config=config, correction=correction: raw_triton(
                    q, k, v, steps, args.group_size, config=config, correction=correction)
    results = {}
    for name, fn in calls.items():
        print(f"Measuring {name}...", file=sys.stderr, flush=True)
        results[name] = measure(fn, args.repeats, args.warmup)
    legacy_ms = results["fa2_augmented"]["median_ms"]
    fa2_ms = results["fa2_plain_128"]["median_ms"]
    for result in results.values():
        result["speedup_vs_augmented"] = legacy_ms / result["median_ms"]
        result["overhead_vs_fa2_plain_pct"] = 100 * (result["median_ms"] / fa2_ms - 1)
    comparisons = {
        backend: comparison(results[f"{backend}_plain_128"]["median_ms"],
                            results[f"{backend}_fused_128"]["median_ms"], fa2_ms)
        for backend in dict.fromkeys(args.backends)
        if not (backend == "triton" and (args.compare_correction or args.compare_implementations))
    }
    resources = {}
    if args.compare_implementations:
        for impl in ("baseline", "pipeline"):
            comparisons[f"triton_{impl}"] = comparison(
                results[f"triton_{impl}_plain_128"]["median_ms"],
                results[f"triton_{impl}_mma_128"]["median_ms"], fa2_ms)
            for correction in (False, True):
                name = f"triton_{impl}_{'mma' if correction else 'plain'}_128"
                _, resources[name] = raw_triton(
                    q, k, v, steps, args.group_size, correction=correction,
                    correction_mode="mma", implementation=impl, return_metadata=True)
    if args.compare_correction:
        for mode in ("scalar", "mma"):
            comparisons[f"triton_{mode}"] = comparison(
                results["triton_plain_128"]["median_ms"],
                results[f"triton_{mode}_fused_128"]["median_ms"], fa2_ms)
        for mode in ("none", "scalar", "mma"):
            _, resources[mode] = raw_triton(
                q, k, v, steps, args.group_size, correction=mode != "none",
                correction_mode="scalar" if mode == "none" else mode, return_metadata=True)
    recommended = None
    if args.tune_triton:
        for config in tune_configs:
            comparisons[f"triton_{config}"] = comparison(
                results[f"triton_{config}_plain"]["median_ms"],
                results[f"triton_{config}_fused"]["median_ms"], fa2_ms)
            for correction in (False, True):
                _, resources[f"triton_{config}_{'fused' if correction else 'plain'}"] = raw_triton(
                    q, k, v, steps, args.group_size, config=config,
                    correction=correction, return_metadata=True)
        best = min(tune_configs, key=lambda c: results[f"triton_{c}_fused"]["median_ms"])
        recommended = {"TEMPOKV_JENSEN_TRITON_CONFIG": best,
                       "scope": "Only this measured shape, dtype, GPU and stride layout; no automatic production change."}
    import triton
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                      "cuda": torch.version.cuda, "flash_attn": flash_attn.__version__,
                      "triton": triton.__version__,
                      "settings": vars(args), "correctness": correctness, "results": results,
                      "triton_correction_env": os.environ.get("TEMPOKV_JENSEN_TRITON_CORRECTION", "scalar"),
                      "triton_implementation_env": os.environ.get("TEMPOKV_JENSEN_TRITON_IMPL", "baseline"),
                      "compiled_resources": resources,
                      "comparisons": comparisons, "best_measured_fixed_config": recommended,
                      "note": "First-call time includes any compilation; preflight warms a different shape. No end-to-end speedup claim."}, indent=2))


if __name__ == "__main__":
    main()
