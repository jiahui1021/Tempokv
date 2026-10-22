# Independent research experiments: S1–S3 / E1–E7

This is the `research.v1` suite for **Self-Forcing / Wan**, implemented after the [source audit](audit/AUDIT.md). It is separate from suite A (`scripts/spectral`, original E1–E3) and suite B (`scripts/paper_experiments/spectral`, Spectral KV). Identically named old experiments are not interchangeable. The production codec and fitting algorithms are unchanged.

## Status and evidence

| Milestone | Runnable implementation | CPU checks | Real model / GPU evidence |
|---|---|---|---|
| M0 | opt-in hooks, runtime context, atomic records, resume library | exact enabled/disabled native BF16 output/cache/RNG | not run |
| M1 | S1 dense spatial/fixed-position temporal FFT; S2 native LOO; E1 local; conditional E3; normalized K/V E4 and byte-accounted precision allocation | passed | not run |
| M2 | single frozen event; actual model-step / final clean / future latent comparisons; matched per-call RNG; own branch clean refresh; E6 whole-schedule budgets | passed with hooked CPU AR surrogate and real attention class | not run |
| M3 | S3 exact token budgets; frozen E7 2×2; paired single-event E7; bounded full-window snapshot and offline rebuild | passed | not run |
| M4 | E5 equal-budget causal archives, declared entity/state feature evaluator, four return scenes + static controls, real-evidence gate | passed | real reappearance/quality unmeasured |

The research suite has **587 passing CPU tests**. No model weights were downloaded, no GPU was rented, and no experiment establishes a scientific hypothesis. The [one-page result note](RESULTS.md) distinguishes numerical verification from missing research data.

## Install and run

From the repository root, use the project's Python environment with `pip install -e '.[research]'`. The configurations use JSON-compatible YAML, so the supplied files also work without PyYAML. Real generation additionally needs the existing Self-Forcing dependencies and local Wan/checkpoint files. CPU analysis requires matplotlib; no PyArrow is needed for the native CSV format.

```bash
# No output files, checkpoint, parameter files or CUDA required.
python scripts/research/run.py --config scripts/research/configs/self_forcing.yaml --dry-run

# CPU numerical verification, explicitly labeled synthetic.
python scripts/research/run.py --config scripts/research/configs/synthetic.yaml --synthetic
python scripts/research/run.py --config scripts/research/configs/synthetic_m2.yaml --synthetic
python scripts/research/run.py --config scripts/research/configs/synthetic_m3.yaml --synthetic
python scripts/research/run.py --config scripts/research/configs/synthetic_e7_paired.yaml --synthetic
python scripts/research/run.py --config scripts/research/configs/synthetic_m4.yaml --synthetic

# Rebuild statistics, PDF/PNG figures and the summary from immutable CSV parts.
python scripts/research/analyze.py --input reports/research/synthetic_m4

# All research tests; independent of CUDA/model weights/diffusers.
python -m pytest scripts/tests/test_research_*.py -q
```

Real pilot (paths must identify existing local files):

```bash
python scripts/research/run.py \
  --config scripts/research/configs/self_forcing.yaml \
  --checkpoint /models/self_forcing.pt --model-root /models/Wan2.1-T2V-1.3B \
  --prompts /data/pilot_prompts.txt --max-prompts 2 --seeds 0,1 \
  --max-events 2 --output-dir reports/research/pilot
```

`scripts/research/run_pilot.sh --dry-run` is the small launcher. `PYTHON` selects its interpreter. `--experiments S1,S2,E1,E3,E4`, `--sampling-config file.json`, `--max-gpu-memory-gb`, `--max-events`, and `--output-dir` are supported. The sampling file may override only `selection` and `sampling`. Seeds are paired within each experiment. The runtime reads layer/head/grid sizes and the actual denoising schedule; video frames map to latent frames as `(FRAMES+3)/4` and must satisfy `(FRAMES+3)%12=0`.

Use `m2_self_forcing.yaml` for paired E1/E2, `e6_self_forcing.yaml` for E6, `e7_paired_self_forcing.yaml` for paired E7, and `e5_self_forcing.yaml` for E5. Paired E5 and E7 each run separately. These are research analysis passes, including repeated inference and instrumentation overhead.

## Position, reference and budget contracts

- `RESEARCH_ENABLE=1` is set only inside this runner. Disabled hooks add no tensor copies or synchronizations. The existing `bf16` / `tempokv` entry modes are unchanged.
- S1 uses **clean Pre-RoPE K and raw V** with the actual non-square frame grid. Spatial FFT never uses the old eight-token trace. Temporal windows use the same positions in contiguous frames, with shuffled controls and spatial-pooling reconstruction controls.
- Attention experiments retain the **full effective K/V window**, sample only Q, and use the original dtype, production callback, scale and head. Native Wan's sliced KV window uses noncausal attention within that slice. CPU snapshot replay explicitly switches to CPU SDPA and labels that backend; it is not a GPU-kernel parity measurement.
- Absolute positions describe rolling windows and sinks, including gaps. Current chunk, configured recent frames and sinks are protected. Noise in current K/V is never reclassified as clean history. K and V are removed with identical indices.
- Replacement/merge requires exactly captured Pre-RoPE K, declares a representative coordinate and reapplies RoPE. Sparse `pre_valid_mask` prevents invented pre-K. Multiplicity correction is tested as exact only under its duplicate-logit/value assumptions; general merges remain approximate.
- E1/E2/E6 and paired E7 reset the base pipeline for each branch with the same initial noise and matched CPU/CUDA/Python/NumPy RNG fingerprints. Branches write their own clean refresh. Final clean output is the last denoising prediction, not the ignored refresh prediction. E1/E2 inject at one frozen event; E6 explicitly lists affected denoising steps, never clean refresh.
- Relative-KV-RMSE and local-attention-error calibration are separate protocols. Actual attained errors are recorded. Unmatched BF16/discrete cases carry a distinct comparison cohort and cannot establish an equal-error comparison.
- S3 compares the same actual candidate-token budget, including partial last blocks and fixed protection. Its LOO-sorted `oracle` is a diagnostic ranking, not a globally optimal subset solver. E4 enumerates a bounded small allocation set with actual joint attention error and includes packed payload, FP32 scales, group and shape metadata. Controls obey the same upper budget and report their actual spend. Generation error is never approximated by adding local errors.
- E7 B/D share a BF16-derived frozen selection. A/B retain the original full cache; C/D retain the full quantized logical cache. Gathered attention inputs, logical packed costs and resident tensor backing are separate. `interaction`, `(D-C)` and `(B-A)` are nonlinear error contrasts, not a physical synergy test. Jensen correction stays off.
- `source=tempokv` is restricted to E7 observation. A decoded TempoKV trajectory by itself does not supply the matched BF16 counterfactual: those rows are explicitly unmeasured. `quant_prune_2x2(..., quantized_pair=...)` accepts a matching snapshot identity/absolute-position map and optional measured storage metadata; missing codec costs remain unknown. The executable paired runner uses separately labeled independent fake quantization. Actual fitted TempoKV paired GPU generation is **not verified**.
- E5 attention-access selection uses each branch's already generated history. Its labeled Oracle uses a bounded sample of reference future Q; when no future Q is captured it falls back to current Q with an explicit label and the same budget. It is a diagnostic, not an optimum. The full cache remains allocated, so no memory-saving or production-speed claim follows.

## Frozen local replay

Set `output.save_dense_kv=true` to save bounded, selected-head `research.frozen.v1` NPZ files under each run's `snapshots/`. These store sampled post-Q, the complete actual post-K/V window, original dtype, absolute coordinates, protection/valid masks, exact sparse pre-K if available, and actual rotation frequencies. Existing files are never overwritten. Missing rotation sources are not guessed. Full all-layer histories are never dumped.

```bash
python scripts/research/run.py --config scripts/research/configs/synthetic_m3.yaml \
  --frozen-input /absolute/path/to/snapshots/event.npz \
  --output-dir reports/research/offline
```

Offline local S1/S2/S3/E1/E3/E4/E7 does not replay generation. S1 requires a complete exact clean pre-K spatial frame; E3 requires exact pre-K for affected tokens. Missing capture coverage is reported as unmeasured. Q/head/candidate sampling scope remains recorded; other history stays in the attention window.

## E5 annotations and declared features

The evaluator consumes explicit multi-frame entity/state features; it does not silently substitute global LPIPS or invent a visual encoder. `memory_scenarios.make_scenario` provides four **synthetic feature** fixtures and prompts for object/character return, camera return, background-state revisit and interaction-state persistence, each with a matched static control. Synthetic null and identity-flip controls validate the evaluator, not the effectiveness of memory selection.

A JSON bundle contains `reference` and `candidate` `[frames,feature_dim]` arrays, `source_metadata` describing extraction/coordinates, and `annotations` such as:

```json
{
  "entity_id": "robot", "sequence_id": "prompt-0-seed-0",
  "first_frame": 0, "disappearance_frame": 8, "reappearance_frame": 16,
  "feature_indices": [0,1,2,3], "state_feature_indices": [4],
  "state_reference_frame": 7
}
```

Each annotation is an element of the `annotations` list. Optional presence masks distinguish absence from identity mismatch. Reported values compare first/established-state → return and candidate → same-frame reference, with explicit zero-energy/nonfinite/absent statuses. Fewer than 20 independent measured sequences are marked `exploratory`; all feature scores remain consistency measures with unknown unconditional quality.

```bash
python scripts/research/run.py --config scripts/research/configs/e5_self_forcing.yaml \
  --e5-features /data/declared_entity_state_features.json
```

A heavier real E5 run additionally needs `--e5-evidence evidence.json`. `heavy_benchmark_gate` requires actual same-cohort paired E1/E2 measurements for at least two prompt/seed units, event-linked criteria, and an explicit `explore_memory` rationale. It accepts zero effects and negative spectral findings when the stated exploration criterion is supported. Synthetic records and an absent rationale do not pass. No such real evidence exists in the delivered CPU runs.

## Files and measurement limits

See [SCHEMA.md](SCHEMA.md) and [schema.json](schema.json) for CSV and equivalent Parquet types, snapshot fields and metric units. `RunStore(..., resume=existing_run)` checks signature/origin, deduplicates deterministic event IDs and adds atomic parts without replacing old files. Repeated CLI invocations create new run directories. The native writer is CSV; Parquet schema is specified, but no Parquet dependency/export is required or claimed here.

Analysis separates config, origin, model/checkpoint, window, source and metric scope. Means/median/std and bootstrap intervals use prompt/seed units; many heads of one video are not independent samples. One unit has no valid multi-video confidence interval. Missing quality fields remain `not_run`.

Instrumentation reports host analysis time, extra oracle callback time, selected real callback time, complete instrumented inference wall time, observed physical KV capacity by layer and process peak allocation separately. Host scoring/gather timings can include GPU scheduling overhead and are not a standalone production throughput benchmark. Capture caps bound each selected tensor/trace subsystem; they do not impose a process-wide allocator cap. A GPU allocation guard can skip analysis events. Run normal generation separately for production timing.

Generated reports, NPZ, checkpoints and videos are ignored by Git. This implementation does not change research conclusions, fit a new predictor, allocate remote GPUs, push branches, or publish results.
