# Long-horizon reconstruction

> **Does the reconstruction error stay flat as the sequence gets long, or does
> it grow once the predictor rolls far past its anchor?**

A pooled NMSE cannot answer that: a method that is uniformly mediocre and a
method that is excellent early and bad late can report the same number. This
study measures the error **at every frame index** of a long trace, and reports
the pooled figure as the last point of the cumulative curve rather than as a
separate quantity.

## Frames

`1401` is a *video* frame count. The runtime converts it to latent frames with
`latent_frames_for` in the Self-Forcing adapter: a latent block covers 4 video
frames and Self-Forcing requires whole 3-frame latent blocks, so `FRAMES` must
satisfy `(FRAMES + 3) % 12 == 0`.

| video frames | latent frames |
| ---: | ---: |
| 105 | 27 |
| 177 | 45 |
| 357 | 90 |
| 717 | 180 |
| 1053 | 264 |
| **1401** | **351** |

`collect_trace.sh` restates that arithmetic (so it works before the model
runtime is importable) and `run_minimal.py` cross-checks it against the script
on every run.

## Step 1 — collect the long trace

```bash
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/moviegen128_first30.txt \
bash scripts/paper_experiments/long_horizon/collect_trace.sh
```

The script prints the projected archive size before doing anything — about
**1.93 GiB** at the defaults (1401 frames, 1 prompt, 8 tokens per frame) — and
refuses to overwrite an existing trace without `OVERWRITE=1`. Add `DRY_RUN=1`
to see the command and the size estimate without collecting.

Two settings are forced and not configurable:

- `LEGACY_PROTOCOL=0`. The legacy protocol pins `TRACE_FRAMES=51`, which cannot
  express 351. The legacy *fitting* equations are shape-independent and accept
  the resulting `[L,S,T,H,D]` layout.
- `NUM_GPUS=1`. Multi-GPU collection shards prompts across ranks and merges
  rank-major.

Useful knobs: `HORIZON_FRAMES` (any value satisfying the rule above),
`NUM_PROMPTS`, `TOKENS_PER_FRAME` (the fastest way to shrink the archive), and
`TRACE_FILE`.

## Step 2 — measure

```bash
python scripts/paper_experiments/long_horizon/run_reconstruction.py \
    --calibration-trace data/self-forcing/trace/clean_trace.npz \
    --horizon-trace     data/long-horizon/1401f/clean_trace.npz \
    --output-dir        results/long_horizon/int2 \
    --bits 2
```

Each method is **fitted on the short calibration trace and scored on the long
one**. That is the point of the experiment: it measures what happens when the
predictor runs far past the horizon it was calibrated for. `--fit-on-horizon`
fits in-sample instead, which is only useful as a reference line, and the
runner says so on stderr when you use it.

`--max-records N` trims the record axis to bound memory. Records are
independent rollouts, so this scales memory down without touching the frame
axis the curve is measured along.

### Methods

Only what this repository implements. Externally published baselines that are
not implemented here are deliberately absent rather than approximated.

| `--methods` | What it is |
| --- | --- |
| `rtn` | round-to-nearest per group, no predictor on either tensor |
| `temporal` | each tensor predicted from its own reconstructed history |
| `tempokv` | production: V projected from the reconstructed K |
| `innovation` | V projected from the decoder-visible K innovation |
| `v_to_k` | mirrored: K projected from the reconstructed V |

The default is `rtn temporal tempokv`. All five share one reconstruction path
in `methods.py`, so a difference between them is a difference between
predictors and not between two pieces of measurement code.

RTN is not a reimplementation: a zero-length temporal map *is* direct
quantization, and the codec already routes it that way, so the baseline is the
shipped code path.

## Step 3 — figure

```bash
python scripts/paper_experiments/long_horizon/plot_figure.py \
    --report    results/long_horizon/int2/long_horizon_reconstruction.json \
    --output-dir results/long_horizon/int2/figure
```

Four panels: K and V, per-frame and cumulative. Methods that share a K
predictor produce **identical** K curves, so those lines sit exactly on top of
one another; the panels cycle dash patterns so every method stays visible. If
two K curves are not identical when they should be, that is a bug worth
chasing, not a rendering artifact.

## Reading the results

`long_horizon_reconstruction.json` carries three views per method and tensor:

- `per_frame` — NMSE at each frame index. Frame 0 is the anchor and is
  identical across methods by construction, because the anchor is quantized
  the same way regardless of predictor.
- `cumulative` — NMSE over frames `0..t`. Its last entry **is** the pooled
  number, so a table and the curve cannot disagree.
- `summary` — `first_frame_nmse`, `last_frame_nmse`, `max_frame_nmse`,
  `head_mean_nmse`, `tail_mean_nmse`, and `tail_over_head`.

`tail_over_head` is the long-horizon question in one number: the mean error
over the last `--tail-fraction` of the sequence divided by the mean over the
rest. Near 1.0 means the method holds up; well above 1.0 means it degrades as
the horizon extends.

## Minimal complete run

```bash
python scripts/paper_experiments/long_horizon/run_minimal.py \
    --output-root /tmp/long-horizon-minimal
```

On CPU, in seconds, it dry-runs the collector to check the frame arithmetic for
three horizons, measures all five methods on a synthetic 60-frame trace fitted
from a 12-frame one, and renders the figure. Its numbers describe the synthetic
generator and say nothing about TempoKV — in particular the synthetic trace
drifts on purpose so the curves have visible structure.
