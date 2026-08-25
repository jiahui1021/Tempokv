# CrossKV sharing experiment

This formal Self-Forcing study compares four CrossKV projection layouts while
keeping the legacy shared-K setup unchanged:

- `layer_head`: independent matrix for every layer and head
- `layer_shared`: one matrix per head, shared across layers
- `head_shared`: one matrix per layer, shared across heads
- `shared`: one matrix shared across all layers and heads

The generated configs differ only in `cross_kv_weight_mode`. The formal
generation protocol creates one BF16 reference and four TempoKV videos with the
same prompt, seed, frame count, checkpoint, and WAN model directory.

Recommended final command:

```bash
CALIBRATION_TRACE=/traces/calibration.npz HELD_OUT_TRACE=/traces/held_out.npz CHECKPOINT=/models/self_forcing_dmd.pt WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B PROMPTS=assets/moviegen128.txt OUTPUT_DIR=results/paper/cross_kv_sharing_one_video bash scripts/paper_experiments/cross_kv_sharing/run_one_video.sh
```

The command fits all four variants, measures production closed-loop K/V NMSE,
generates the five matched videos, and evaluates each TempoKV result against the
same BF16 output with PSNR, SSIM, LPIPS, and FID. Outputs include
`manifest.json`, `comparison.json`, `comparison.csv`, and detailed pooled,
per-layer, per-head, and per-layer/per-head reconstruction reports.

If `HELD_OUT_TRACE` is omitted, reconstruction uses the calibration trace and
records that choice in the manifest. Attention refinement is disabled because
its V phase does not preserve the intended sharing layout.

Set `VBENCH_LONG=1` together with `VBENCH_ROOT`, `VBENCH_CACHE_DIR`, and
`VBENCH_DEVICES` to append the optional VBench-Long matrix.

Running `run_study.py` directly without `--execute` remains a model-free plan
mode that writes the resolved configs, manifest, and planned comparison files.
