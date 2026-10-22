#!/usr/bin/env python3
"""Synthetic CPU smoke test only; its outputs are not experimental evidence."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tempokv.spectral import SpectralConfig, SpectralProbe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = SpectralConfig.from_dict({"sampling": {"layers": [0], "heads": [0, 1], "latent_frames": [3],
                                                   "query_tokens_per_head": 4, "max_candidate_blocks": 6},
                                      "frequency": {"spatial_block_size": [2, 2], "temporal_positions_per_frame": 3},
                                      "oracle": {"protect_recent_frames": 1, "key_chunk_size": 7},
                                      "compression": {"snapshot_calls": 1}})
    probe = SpectralProbe(config, args.output, metadata={"source": "synthetic CPU smoke; not research evidence"})
    probe.reset(prompt_id="synthetic", batch_size=1)
    rng = torch.Generator().manual_seed(42)
    k, v = [torch.randn(1, 60, 2, 8, generator=rng).to(torch.bfloat16) for _ in range(2)]
    for frame in range(3):
        probe.set_call(phase="clean")
        probe.on_projection(0, k_pre=k[:, frame * 15:(frame + 1) * 15],
                            v=v[:, frame * 15:(frame + 1) * 15], grid_sizes=[[1, 3, 5]], current_start=frame * 15)
    probe.set_call(phase="denoise", denoise_call=0)
    q = torch.randn(1, 15, 2, 8, generator=rng).to(torch.bfloat16)
    probe.on_attention(0, q_post=q, k_post=k, v=v,
                       metadata={"grid_sizes": [[1, 3, 5]], "current_start": 45,
                                 "local_start_index": 45, "attention_start": 0,
                                 "local_end_index": 60, "frame_seqlen": 15})
    probe.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
