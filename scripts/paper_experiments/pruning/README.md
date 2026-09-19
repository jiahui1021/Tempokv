# Token pruning pilot

Implements the two pilot experiments in `reports/pruning.md` on Self-Forcing.

Pruning is simulated: a pruned token stays in the TempoKV cache and is only
dropped from the key set of later attention calls (K and V together, one mask
per layer shared by all heads). The TempoKV predictors therefore keep their
full history and quantization is identical across runs. The hook lives in
`experiments/Self-Forcing/tempokv_adapter/pruning.py` and is off unless
`TEMPOKV_PRUNE_CRITERION` is set.

Per layer, at the first denoising step of every AR block:

| signal | definition |
|---|---|
| `A` | attention mass the token receives, summed over heads and every 8th query, averaged over the blocks that have seen it |
| `E` | `‖K_t − pred(K_{t−1..t−H})‖₂` with the fitted K temporal predictor on clean history |
| `sim` | `cos(K_t, K_{t−1})` over all heads, pre-RoPE |

A frame is decided once it is `TEMPOKV_PRUNE_PROTECT` (default 6) frames
behind the current block; the lowest-scoring `ratio` of its tokens is pruned.
Frame 0 has no history and is never pruned. Criteria: `random`, `attention`
(low `A`), `similarity` (high `sim`), `residual` (low `E`), `attn_residual`
(low `rank(A) + λ·rank(E)`, `λ = TEMPOKV_PRUNE_LAMBDA`).

```bash
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/moviegen128.txt \
PARAM_DIR=data/self-forcing/params/legacy-kh2-vh1-int4-layershared \
bash scripts/paper_experiments/pruning/run.sh
```

The default matrix is 8 prompts × 237 frames (60 latent frames): BF16, the
unpruned TempoKV run `none`, and 5 criteria × {10, 25, 50, 75}%. Output under
`results/self_forcing/pruning_pilot/analysis/`:

- `exp1_residual_attention.png`, `summary.json → experiment1`: from the `none`
  run. Median-split quadrant fractions (25% each if `A` and `E` are
  independent), within-frame Spearman correlations, and the Jaccard overlap of
  the bottom-r sets by `A` and by `E` against the independent expectation.
- `exp2_table.md`, `exp2_criteria.png`: PSNR/SSIM/LPIPS against the unpruned
  TempoKV run (isolates the pruning damage) and against BF16, plus measured
  retention. `best_criterion_by_lpips_vs_none` names the winner per ratio.

Retention is over all layers and frames, so it is higher than `1 − ratio`:
frame 0 and the newest protected frames are never pruned.
