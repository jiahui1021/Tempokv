#!/usr/bin/env python3
"""Synthetic CPU workflow check for suite B; never research evidence."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from tempokv.spectral.sensitivity import reference_attention
from tempokv.spectral_kv.config import SpectralKVConfig
from tempokv.spectral_kv.context import RunContext
from tempokv.spectral_kv.probe import SpectralKVProbe
from tempokv.spectral_kv.rope3d import full_positions, rope_apply


def run_smoke(output, *, method="BF16", online=False):
    experiments = ("B_E3_ON_ROLLOUT",) if online else ("B_E1_SPECTRUM", "B_E2_CHUNK_AXIS", "B_E2B_REFRESH", "B_E3_OFF_SHADOW", "B_E4_DISTANCE")
    config = SpectralKVConfig(experiments=experiments, profile="synthetic_" + method, layers=(0,), heads=(0, 1),
        chunks=(), denoise_steps=(), query_tokens=4, spatial_bins=3, temporal_bins=2, max_lag=2,
        temporal_window=3, feature_channels=4, keep_recent_frames=0, sink_frames=1,
        shadow_methods=("BF16", "DROP_win1", "SPECSUM_dct_R4_B0", "SPECSUM_dct_R4_B1", "SPECSUM_avg_R4_B1"),
        method=method, online=online)
    probe = SpectralKVProbe(config, output, layers=1, metadata={"source": "synthetic CPU workflow; not research evidence"})
    probe.reset(prompt_id="synthetic")
    rng = torch.Generator().manual_seed(9)
    history = {}
    for frame in range(4):
        for step, phase in ((0, "denoise"), (1, "denoise"), (-1, "clean")):
            probe.set_call(RunContext(frame, step, phase == "clean", frame, step == 1))
            q, k, v = [torch.randn(1, 15, 2, 8, generator=rng).to(torch.bfloat16) for _ in range(3)]
            probe.on_projection(0, k_pre=k, v=v, grid_sizes=[[1, 3, 5]], current_start=frame * 15)
            spec = probe.specs[0]
            positions = full_positions(frame, 1, (3, 5))
            q_post, k_post = [rope_apply(x, *positions, spec) for x in (q, k)]
            history[frame] = (k_post, v)
            key, value = [torch.cat([history[i][j] for i in range(frame + 1)], dim=1) for j in range(2)]
            actual = probe.on_attention(0, q_post=q_post, k_post=key, v=value,
                metadata={"grid_sizes": [[1, 3, 5]], "current_start": frame * 15, "local_start_index": frame * 15,
                          "attention_start": 0, "local_end_index": key.shape[1], "frame_seqlen": 15,
                          "rope_precision": "float64"}, attention_fn=reference_attention)
            if not online:
                torch.testing.assert_close(actual, reference_attention(q_post, key, value), rtol=0, atol=0)
            probe.on_block(0, actual.flatten(2))
    path = probe.flush()
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="BF16")
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args(argv)
    print(run_smoke(args.output, method=args.method, online=args.online))
    return 0


if __name__ == "__main__": raise SystemExit(main())
