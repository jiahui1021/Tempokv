# Self-Forcing history ablation

This study runs the final INT2 history ablation on four GPUs. Every subprocess
uses one GPU; fitting, cache evaluation, video generation, and RGB quality
evaluation are scheduled in batches of four.

The default variants are:

| Axis | K history | V history | V predictor |
|---|---:|---:|---|
| K history | `0,1,2,3,4,5` | fixed at `1` | `cross_kv` |
| V history | fixed at `3` | `0,1,2,4,5` | `cross_kv` |
| no Cross | fixed at `3` | `1` | `temporal` |

The `h_K=0` row is direct K quantization. The common anchor remains at
`anchor_bits`; all later K units use `k_bits`. The `w/o cross` row is fitted
independently and uses only the previous reconstructed V.

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/Self-Forcing/run_history_ablation_4gpu.sh \
  --trace data/self-forcing/trace/clean_trace.npz \
  --base-config configs/legacy.json \
  --checkpoint /models/self_forcing_dmd.pt \
  --wan-model-dir /models/Wan2.1-T2V-1.3B \
  --prompts assets/moviegen128.txt \
  --output results/self_forcing/history_ablation_int2
```

The runner selects prompt indices `[0,50)`, generates one sample per prompt,
and rejects incomplete or skipped runs. Use `--resume` to keep valid outputs
and fill missing stages. Use `--dry-run` to materialize the configs and
manifest without starting fitting or generation.

The output contains:

- `summary.json` and `summary.csv`: K/V prediction and reconstruction NMSE,
  RGB PSNR, SSIM, LPIPS, and generation metadata;
- `manifest.json`: exact variant definitions, paths, and scheduled commands;
- `configs/`, `params/`, `runs/`, `cache/`, and `logs/`: per-variant artifacts.

The standalone cache evaluator is:

```bash
python scripts/paper_experiments/history_ablation/evaluate_cache.py \
  --trace data/self-forcing/trace/clean_trace.npz \
  --config results/self_forcing/history_ablation_int2/configs/k_h3_cross_vh1.json \
  --k-params results/self_forcing/history_ablation_int2/params/k_h3_cross_vh1/k_temporal.npz \
  --v-params results/self_forcing/history_ablation_int2/params/k_h3_cross_vh1/v_cross_kv.npz \
  --output /tmp/k_h3_cache.json
```
