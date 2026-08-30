# Calibration study

> **TempoKV requires only a small calibration set, is insensitive to the
> calibration prompt distribution, and shows the same calibration efficiency
> across different AR video generators.**

Three modules produce one three-panel figure:

| Panel | Module | Script | Model(s) |
| --- | --- | --- | --- |
| (a) | Calibration set size | `run_size.py` | Self-Forcing |
| (b) | Calibration distribution shift | `run_shift.py` | Self-Forcing |
| (c) | Calibration efficiency across models | `run_models.py` | Self-Forcing, Causal-Forcing, LongCat-Video |

The metric everywhere is **normalized reconstruction MSE on held-out prompts**,
reported separately for K and V. It is produced by replaying the production
closed-loop reconstruction (`cross_kv_sharing/evaluate_reconstruction.py`), so
it is the error the deployed codec makes, not a teacher-forced regression
residual.

## The protocol

Every module runs the same protocol, and the only thing that changes between
runs is which prompts are used:

1. Collect **one calibration pool trace per model**, over a pool of prompts
   that covers both the calibration and the held-out sets.
2. Slice that pool by prompt to build a calibration set.
3. Refit K and V predictors on the slice with the shared fitter
   (`scripts/common/fit_params.py`).
4. Score the refit predictors on a fixed held-out slice of the same pool.

Because calibration and evaluation records come from a single collection run,
the checkpoint, sampling protocol, quantizer and anchor schedule are identical
at every point. `protocol.CODEC` pins the codec — INT2 K/V, INT4 anchors,
group size 64, `layer_shared` Cross(K), legacy fitting — and the same config
file is handed to both the fitter and the evaluator, so a codec difference can
never be mistaken for a calibration effect.

Predictor parameters are deliberately **not** transferred between models.
Different models differ in layer count, head count and head dimension, so a
transferred matrix would measure that mismatch rather than the protocol. Module
3 transfers the *protocol* and compares `MSE(N) / MSE(N_max)`, which removes
each model's KV scale from the comparison.

## Step 1 — the prompt pool (already committed)

`assets/calibration/pool_prompts.txt` and
`assets/calibration/prompt_groups.json` are **checked into the repository**, so
a fresh checkout on a GPU host already has them and `run_study.sh` defaults to
them.

That is deliberate rather than incidental. The pool prompt *order* determines
the order of the records inside every collected trace, and the A/B labels carry
a human review. Regenerating them separately on each machine would let the two
drift apart silently, and a trace collected against one ordering cannot be
sliced with another.

Regenerate them only when you mean to change the pool, and commit the result:

```bash
python scripts/paper_experiments/calibration/build_groups.py \
    --prompts assets/moviegen128.txt \
    --output-dir assets/calibration \
    --calibration-size 32 --heldout-size 32
```

Anything already collected against the previous pool must be recollected.

The manifest names six groups addressed by pool position:

- `calibration` / `heldout` — the disjoint sets module 1 draws from and scores on
- `action_calibration` / `action_heldout` — the human/action distribution, A
- `scene_calibration` / `scene_heldout` — the scene/object distribution, B

The A/B labels come from an explicit keyword rule documented in
`build_groups.py`. **Read them before collecting.** The rule is a starting
point: it labels "two pirate ships battling" as *action* and a close-up of a
pigeon as *scene*, and a mislabelled prompt weakens exactly the claim module 2
makes. `prompt_groups.json` echoes every prompt next to its label so the split
can be corrected by hand.

## Step 2 — collect one pool trace per model

Use each model's existing `collect_trace.sh` with the pool prompt file:

```bash
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=data/calibration/pool_prompts.txt \
TRACE_FILE=data/calibration/self-forcing_pool.npz \
LEGACY_PROTOCOL=0 \
NUM_GPUS=1 \
bash scripts/Self-Forcing/collect_trace.sh
```

Two settings are not optional **for Self-Forcing**:

- `LEGACY_PROTOCOL=0`. The legacy event protocol is pinned to exactly 12
  prompts, which cannot express a 64-prompt pool. The flat layout carries the
  same records and the legacy fitting equations are shape-independent.
- `NUM_GPUS=1`. Multi-GPU collection shards prompts across ranks and merges
  rank-major, so a prompt no longer owns a contiguous block of records and the
  pool can no longer be sliced by prompt. The recorded activations are
  identical either way; only the record order changes.

The other two collectors do **not** take the same options, and panel (c) has
to work with what each one allows:

- **Causal-Forcing** forces `LEGACY_PROTOCOL=1` and refuses any other value, so
  its pool is exactly 12 prompts in the seven-dimensional event layout.
  `PoolTrace` collapses that layout to the same `[L,S,T,H,D]` records, so it
  slices by prompt normally -- but with 12 prompts the calibration group is
  small, so give module 3 a size grid that fits (for example
  `--sizes 2 4 8` with 8 calibration and 4 held-out prompts).
- **LongCat** collects one prompt per invocation (`PROMPT=...`), not a prompt
  file. Build its pool by running `collect_trace.sh` once per prompt with the
  **same `TRACE_DIR`**, in pool order; the events accumulate under that root
  and the final merge picks all of them up. It already writes the plain
  `[L,S,T,H,D]` layout.

In every case `PoolTrace` refuses a trace whose record count does not divide by
the prompt count, so a mis-accumulated pool fails loudly instead of silently
misattributing records to the wrong prompt.

`PoolTrace` checks that the record count divides evenly by the pool prompt
count and fails with a clear message when the trace and the prompt groups
describe different collection runs.

## Step 3 — run the study

```bash
SF_POOL_TRACE=data/calibration/self-forcing_pool.npz \
CF_POOL_TRACE=data/calibration/causal-forcing_pool.npz \
LONGCAT_POOL_TRACE=data/calibration/longcat_pool.npz \
PROMPT_GROUPS=data/calibration/prompt_groups.json \
OUTPUT_ROOT=results/calibration \
bash scripts/paper_experiments/calibration/run_study.sh
```

`CF_POOL_TRACE` and `LONGCAT_POOL_TRACE` are optional; panel (c) simply omits a
model whose pool is not supplied. The individual runners can also be called
directly — see `--help` on each.

Outputs under `OUTPUT_ROOT`:

```
size/calibration_size.json              per-N K/V NMSE, mean/std over draws
shift/calibration_shift.json            the 2x2 cells and the error increases
models/calibration_across_models.json   per-model curves and MSE(N)/MSE(N_max)
figure/calibration.{pdf,png}            panels (a) (b) (c)
figure/panel_*.csv                      the numbers behind each panel
```

The CSVs are always written; the figure needs matplotlib and is skipped with a
message when it is missing, so a headless fitting host still produces numbers.

### Reading the results

- **(a)** The claim is saturation: K and V NMSE stop improving after a handful
  of prompts. Every N is measured over `--repeats` independent draws from the
  calibration pool, and the error bars are the spread across prompt choices —
  at N=1 that spread is the result, not noise to be averaged away. Drawing the
  full pool admits one set, so its repeats collapse to a single sample.
- **(b)** Every cell calibrates on the same number of prompts, so a difference
  between cells cannot come from calibration size. Held-out sets differ in
  intrinsic difficulty, so a cross-distribution cell is only meaningful against
  the matched cell **in its own column**; that ratio is the reported
  `error_increase` and is what panel (b) is coloured by. Small off-diagonal
  increases are the claim.
- **(c)** Absolute NMSE is not comparable across models. A shared shape in the
  normalized curve is the claim.

## Cost and disk

Collection dominates: one pool trace per model. Everything after that is
linear algebra — fitting a slice is seconds to minutes on one GPU, and the
sweep is `len(sizes) * repeats` fits per model.

Pool traces are large (a 64-prompt Self-Forcing pool is a few GB) and are
loaded into memory once per run. Calibration slices and fitted Cross(K)
weights are written per point and **deleted after each measurement**; pass
`--keep-artifacts` / `KEEP_ARTIFACTS=1` to retain them for debugging, and
expect gigabytes per point if you do.

## Minimal complete run

```bash
python scripts/paper_experiments/calibration/run_minimal.py \
    --output-root /tmp/calibration-minimal
```

Runs all three modules and the figure end to end on CPU in seconds, against
small synthetic pool traces with three different KV geometries. It exercises
the shipped fitter, the shipped evaluator and the shipped plotting code, so
passing it means the only thing a production run changes is which traces are
read. Its numbers describe the synthetic generator and say nothing about
TempoKV.
