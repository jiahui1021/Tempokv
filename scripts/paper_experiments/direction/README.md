# K→V direction study

> **TempoKV predicts V from the reconstructed K. This measures the alternative
> instead of assuming it away.**

The production codec is asymmetric: K is predicted from its own temporal
history, and V is projected from the reconstructed K of the same span. An
asymmetric design needs a reason, and "we tried it this way first" is not one.
Three arms, identical in everything else:

| Arm | `v_predictor_mode` | `predictor_direction` | Dependency |
| --- | --- | --- | --- |
| `temporal_only` | `temporal` | `k_to_v` | none in either direction |
| `k_to_v` | `cross_kv` | `k_to_v` | V ← reconstructed K (production) |
| `v_to_k` | `cross_kv` | `v_to_k` | K ← reconstructed V (mirror) |

Two metrics: **held-out normalized reconstruction MSE** for K and V separately,
and **generation quality** against a shared BF16 reference.

## The mirror is the same codec, not a second one

The codec equations never mention K or V. `predictor_direction="v_to_k"`
therefore exchanges the tensor *roles* rather than introducing a second set of
equations:

- `CodecConfig.primary_tensor` is the tensor predicted from its own temporal
  history; `secondary_tensor` is the one projected from the primary's
  reconstruction.
- `fit_predictors` hands the pair to the fitter in the other order.
- `TempoKVCodec.encode_primary` / `encode_secondary` and `codec.primary_role`
  let the model bridges address the cache by role, so one code path serves both
  directions in the Self-Forcing, Causal-Forcing and LongCat runtimes.

This matters for the argument. If the mirror needed different math, a gap
between the arms could be a gap between two implementations rather than between
two directions.

Two constraints follow, and both are enforced in `CodecConfig`:

- `v_to_k` requires `v_predictor_mode=cross_kv`. Every other V mode ignores the
  source tensor, so the flag would silently do nothing.
- `v_to_k` requires `k_bits == v_bits`. The tensors exchange roles wholesale, so
  unequal widths would also reallocate bits between them and confound the
  comparison with a precision change.

The direction joins the codec signature only when it is not the production
value, so every existing parameter archive keeps its signature, and a mirrored
parameter set can never be loaded into a forward-direction runtime.

## Reading the parameter files

Under `v_to_k` the temporal parameters describe **V** and the projection
parameters describe **K**, which is the reverse of every other configuration.
The artifacts are named for the role each set plays so the two directions
cannot be confused on disk:

| Direction | temporal parameters | projection parameters |
| --- | --- | --- |
| `k_to_v` | `k_temporal.npz` | `v_cross_kv.npz` |
| `v_to_k` | `v_temporal_primary.npz` | `k_cross_from_v.npz` |

`load_codec`'s two parameter arguments are positional by role — temporal first,
projection second — so the mirrored artifacts are passed in the same order.

## Running it

### Step 0 — split one collected trace into two disjoint traces

A trace bridge writes a single archive, but the study needs to fit on one set
of prompts and score on another. `make_split.py` slices a collected trace by
prompt, reusing the calibration study's `PoolTrace` rather than reimplementing
"which records belong to which prompt":

```bash
# a trace collected without a groups file (e.g. the 12-prompt legacy trace)
python scripts/paper_experiments/direction/make_split.py \
    --trace data/self-forcing/trace/clean_trace.npz \
    --prompt-count 12 --calibration-prompts 8 \
    --output-dir data/direction

# or reuse a calibration pool and its named groups
python scripts/paper_experiments/direction/make_split.py \
    --trace data/calibration/self-forcing_pool.npz \
    --prompt-groups data/calibration/prompt_groups.json \
    --output-dir data/direction
```

It writes `calibration.npz`, `heldout.npz` and a `split.json` recording which
prompts went where. It refuses a trace whose record count does not divide by
the prompt count, which is what a trace and a groups file from different
collection runs look like.

### Reconstruction (no GPU needed beyond fitting)

```bash
python scripts/paper_experiments/direction/run_reconstruction.py \
    --calibration-trace data/direction/calibration.npz \
    --evaluation-trace  data/direction/heldout.npz \
    --output-dir results/direction/reconstruction
```

All three arms are fitted and scored by one function (`protocol.reconstruct`)
on purpose: arms measured by different code would not separate the direction
from the measurement.

Pass a real held-out trace. Defaulting `--evaluation-trace` to the calibration
trace measures fit rather than generalization, and the runner says so on
stderr when you do.

### Generation quality (Self-Forcing, needs the model)

```bash
python scripts/paper_experiments/direction/run_quality.py \
    --checkpoint /models/self_forcing_dmd.pt \
    --wan-model-dir /models/Wan2.1-T2V-1.3B \
    --prompts assets/moviegen128_first30.txt \
    --trace data/direction/calibration.npz \
    --output-root results/direction/quality
```

Trace, prompts, seed, frame count, prompt range, bit widths, history lengths
and Cross weight mode are held fixed and recorded under `fairness_controls` in
`manifest.json`. Add `--dry-run` to record the whole command matrix without
fitting or generating — do this first; it catches a wiring mistake before it
costs GPU hours. `--resume` skips arms whose artifacts already exist.

An arm whose quality evaluation did not run is recorded as `not_run`; no metric
value is ever invented.

### Figure

```bash
python scripts/paper_experiments/direction/plot_figure.py \
    --reconstruction-report results/direction/reconstruction/direction_reconstruction.json \
    --quality-report        results/direction/quality/summary.json \
    --output-dir            results/direction/figure
```

Panel (b) states that quality was not run rather than drawing an empty axis,
so a reconstruction-only figure is still honest. CSVs are always written; the
figure needs matplotlib.

## Reading the results

`direction_reconstruction.json` carries two comparisons, because the design
argument has two halves:

- `per_arm[*].{k,v}_nmse_vs_temporal_only` — the cross term is only worth its
  parameters if it beats having no cross dependency at all.
- `direction_comparison.mirror_penalty` — the relative cost of running the
  dependency backwards, on **total** (K + V) NMSE. Total is the right ranking
  because the attention layer consumes both tensors; a direction that helps one
  tensor by hurting the other has not helped.

### One caveat worth stating in the paper

The arms do not differ *only* in direction: `temporal_only` gives V a full
temporal predictor, while both cross arms give V a diagonal history map plus
the projection matrix. That is a property of the codec's existing V modes, not
of this experiment, but it means `temporal_only` is a slightly different
parameterization rather than a pure ablation of the cross term. The two cross
arms *are* exact mirrors of each other, so `direction_comparison` is the clean
comparison and the `vs_temporal_only` columns are the looser one.

## Minimal complete run

```bash
python scripts/paper_experiments/direction/run_minimal.py \
    --output-root /tmp/direction-minimal
```

On CPU, in seconds, it:

1. drives the mirrored codec through the **real** `ChunkedKVCache` in both
   directions, so the runtime contract the model bridges depend on is checked
   without a GPU;
2. runs the reconstruction half for all three arms;
3. records and validates the quality command matrix in `--dry-run` form;
4. renders the figure.

Its numbers describe the synthetic generator and say nothing about TempoKV. In
particular the synthetic V is easy to predict from its own history, so the
cross term does not win there — that is the toy process, not a result.
