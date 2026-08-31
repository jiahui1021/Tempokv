# Observation: the K/V predictability ladder (paper §3.2)

> **A 7.75× energy drop on K is not what the temporal predictor earned.**
> Most of it is the per-channel DC term, which KIVI-style channel-wise
> quantization already exploits. The ladder separates the two so the paper can
> claim only what is actually new.

Four rungs, each adding one term to the predictor and each scored on the same
held-out rows:

| Tier | Prediction | What it isolates |
| --- | --- | --- |
| 0 | `0` | direct quantization, the reference denominator |
| 1 | `mu_d` | per-channel DC — the known, already-exploited structure |
| 2 | `mu + sum_i A_i (X_{t-i} - mu)` | the temporal predictor's own contribution |
| 3 (V only) | tier 2 `+ W_{K->V} K_t` | the aligned-K cross term |

The tier 1 → 2 delta is the temporal predictor's real earnings; the tier 2 → 3
delta is the cross term's. Those two deltas, not the tier-0 ratio, are the
numbers §3.2 is entitled to quote.

## Two things the ladder deliberately keeps apart

**Energy is not bits.** The quantizer's step size follows the per-group range,
and the range follows the outliers. A predictor can strip most of the energy
and leave the range untouched. Every cell therefore carries both
`energy_equivalent_bits` (`0.5 log2(1/energy_ratio)`, what energy *promises*)
and `bits_saved` (the median per-group `log2` range ratio, what the quantizer
*realises*). A wide gap between them is a finding about outlier-dominated
ranges, not a bug — `range_over_std` says whether the residual got heavier
tailed than the tensor it came from.

**Clean is not closed loop.** The `clean` pass is an open-loop probe: the
predictor reads clean history it will never have at run time. The
`closed_loop` pass replays the codec — INT4 anchor, quantized residuals — so
the predictor reads only decoder-visible frames, exactly as deployed. Both are
written on every run. Fig 3(a) may use `clean`, but the caption has to say
open-loop probe, and **no design decision may be argued from the clean column**.
Tiers 0 and 1 hold no history, so the loop provably cannot move them and the
runner copies those cells rather than re-deriving them.

## 1. Collect the trace

Collection is already implemented; there is no separate collector here. The
Self-Forcing (and Causal-Forcing) trace bridge hooks `norm_k` and `v` inside
each block, which is the pre-RoPE, finalized-clean write point — the same
place the TempoKV encoder intercepts — and records only during a clean
refresh, so denoising steps cannot leak in.

```bash
CHECKPOINT=/models/self_forcing_dmd.pt WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=<calibration prompt list> \
LEGACY_PROTOCOL=0 \
TRACE_FRAMES=180 \
TRACE_TOKENS_PER_FRAME=64 \
TRACE_DIR=data/self-forcing/trace/observation_cal \
TRACE_FILE=data/self-forcing/trace/observation_cal.npz \
bash scripts/Self-Forcing/collect_trace.sh
```

`LEGACY_PROTOCOL=0` is required: the legacy archive keeps the original B/P axes
for record-by-record FP64 fitting, and the ladder reads the flat
`tempokv_flat_samples_v1` layout. The loader refuses the legacy one rather than
silently reinterpreting its axes.

Repeat with the held-out prompt group and `TRACE_ROLE=diagnostic` into
`observation_eval.npz`. The two groups must come from the project's existing
prompt-disjoint split (`scripts/paper_experiments/calibration/protocol.py`);
the ladder is only meaningful on prompts the predictors never saw.

The sampler is `linspace(0, 1559, tokens_per_frame).round().unique()`, so the
spatial indices are identical across frames, heads, layers and prompts, and
they are stored in the archive as `token_indices`. Layers are collected in
full; subsample them at analysis time with `--layers 0,7,15,22,29` so the
ladder covers the whole depth rather than one band.

Then run the two mandatory checks before trusting anything downstream:

```bash
python scripts/paper_experiments/observation/check_trace.py \
    data/self-forcing/trace/observation_cal.npz --expected-frames 180
```

It fails loudly on the two silent corruptions: a frame count that is an
integer *multiple* of the latent frame count (denoising steps recorded), and a
K that alternates sign along frames far more than V does (post-RoPE K, whose
position-dependent rotation destroys cross-frame alignment and reports
predictability that is low for the wrong reason).

## 2. Run the ladder

```bash
python scripts/paper_experiments/observation/run_decompose.py \
    --calibration-trace data/self-forcing/trace/observation_cal.npz \
    --held-out-trace   data/self-forcing/trace/observation_eval.npz \
    --layers 0,7,15,22,29 --k-order 2 --v-order 1 \
    --config configs/legacy.json \
    --output results/observation/decompose.json
```

A single archive can be split instead, on an event boundary, with
`--split-events N` in place of `--held-out-trace`; events are whole
generations, so the halves stay prompt-disjoint. Passing the same file as both
arguments is refused.

`--config` supplies only the closed-loop quantizer widths (anchor / K / V bits,
group size, residual mode); without it the pass runs at INT4 anchor and INT2
residuals. Predictors are fitted here, not loaded, so the ladder does not
depend on which parameter profile happens to be on disk.

Then render the table and figure:

```bash
python scripts/paper_experiments/observation/plot_fig3a.py \
    --report results/observation/decompose.json --loop clean
```

`table1_tiers.csv` is table 1 with both loops; `fig3a_segments.csv` and
`fig3a_<loop>.pdf` are Fig 3(a), one stacked bar per tensor split into DC /
temporally predictable / cross-predictable / residual. The CSVs are always
written; the figure is skipped with a message when matplotlib is absent.

## 3. Reading the result

Three numbers from the earlier probe are the acceptance check, not results of
this script: K tier 1 ≈ 0.288, K tier 2 ≈ 0.129, V tier 1 ≈ 0.911. A K tier 1
far from 0.288 means the channel mean was taken over the wrong axis; a K tier 2
far from 0.129 means the AR fit or the lag alignment is wrong.

What the deltas imply for the paper:

| Result | Consequence |
| --- | --- |
| K tier 1 → 2 gain ≳ 2× | state the three-tier ladder outright: channel structure is the known KIVI premise, the temporal predictor earns the rest on top of it |
| K tier 1 → 2 gain < 1.3× | the temporal predictor is not paying for itself on K; consider DC subtraction plus direct quantization |
| V tier 2 → 3 gain large | keep the cross term and argue its information source in Fig 3(b) |
| V tier 2 → 3 gain small | drop to a single branch, cross-only or temporal-only |
| `bits_saved` well below `energy_equivalent_bits` | write the outlier-dominated-range observation up explicitly; it is an honest and uncommon result in the KV-quantization literature |

## Design notes

- **One warm-up for every tier.** All tiers drop `max(k_order, v_order)`
  leading frames. Letting each order drop its own would change the denominator
  per tier and make the ladder non-additive; this is the easiest thing in the
  whole pipeline to get wrong.
- **The cross term is fitted jointly** with the V history rather than on top of
  the tier-2 residual, so the tier 2 → 3 delta is attributable to the K term
  and not to a second pass of the same history.
- **Normal equations accumulate in float64** over sample chunks (`--chunk`).
  A production trace has hundreds of thousands of rows; a float32 `sum x^2`
  over that many terms loses digits the solve then amplifies.
- **Pooling.** `energy_ratio` pools exactly, as a ratio of summed energies.
  `bits_saved` and `range_over_std` are medians and do not pool, so the
  reported figure is the median across heads of each head's median and is
  named `*_median_over_heads` in the report.
