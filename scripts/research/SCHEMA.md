# research.v1 raw data schema

The native writer uses UTF-8 CSV with the following exact ordered header. The same logical fields define the equivalent Parquet/Arrow schema in `schema.json`. No Parquet file was produced in the CPU validation; the delivered raw files are CSV. Empty numeric CSV cells map to Parquet null, never zero. `details_json` remains a UTF-8 JSON string.

| Field | Parquet / Arrow type | Unit |
|---|---|---|
| `schema_version` | utf8 | — |
| `experiment_id` | utf8 | — |
| `event_id` | utf8 | — |
| `data_origin` | utf8 | — |
| `prompt_index` | utf8 | — |
| `seed` | int64 | — |
| `sample_id` | int64 | — |
| `layer` | int64 | — |
| `head` | int64 | — |
| `chunk_index` | int64 | — |
| `phase` | utf8 | — |
| `denoise_step_index` | int64 | — |
| `denoise_timestep` | int64 | — |
| `current_chunk_start` | int64 | — |
| `grid_t` | int64 | — |
| `grid_h` | int64 | — |
| `grid_w` | int64 | — |
| `query_samples` | int64 | — |
| `key_token_count` | int64 | — |
| `history_token_count` | int64 | — |
| `protected_token_count` | int64 | — |
| `protected_fraction` | float64 | dimensionless |
| `block_id` | utf8 | — |
| `score_type` | utf8 | — |
| `score` | float64 | dimensionless |
| `local_attention_relative_error` | float64 | dimensionless |
| `model_output_relative_error` | float64 | dimensionless |
| `clean_latent_relative_error` | float64 | dimensionless |
| `horizon_chunks` | int64 | — |
| `nmse_k` | float64 | dimensionless |
| `nmse_v` | float64 | dimensionless |
| `strategy` | utf8 | — |
| `intervention` | utf8 | — |
| `role` | utf8 | — |
| `strength` | float64 | dimensionless |
| `requested_strength` | float64 | dimensionless |
| `calibration_protocol` | utf8 | — |
| `retention` | float64 | dimensionless |
| `actual_retained_tokens` | int64 | — |
| `actual_retention` | float64 | dimensionless |
| `k_bits` | int64 | — |
| `v_bits` | int64 | — |
| `payload_bytes` | int64 | bytes |
| `scale_bytes` | int64 | bytes |
| `metadata_bytes` | int64 | bytes |
| `logical_bytes` | int64 | bytes |
| `physical_kv_bytes` | int64 | bytes |
| `budget_cohort` | utf8 | — |
| `conditioning_set` | utf8 | — |
| `paired_run_id` | utf8 | — |
| `analysis_ms` | float64 | milliseconds |
| `compression_ms` | float64 | milliseconds |
| `scoring_ms` | float64 | milliseconds |
| `gather_ms` | float64 | milliseconds |
| `attention_ms` | float64 | milliseconds |
| `status` | utf8 | — |
| `reason` | utf8 | — |
| `details_json` | utf8 | — |

## Required interpretation

- Fixed IDs: S1/S2/S3/E1/E2/E3/E4/E5/E6/E7, scoped by `research.v1`. Old suite A/B names are separate.
- `prompt_index` is UTF-8 to permit named CPU fixtures as well as integer dataset indices. Seed is the actual model/prompt seed. Local noise seeds are extra fields, never replacements.
- `data_origin` separates actual runtime reference/counterfactual, synthetic, supplied features and unknown snapshots. Unknown/synthetic origins cannot count as real model evidence.
- `phase` is denoise or clean_refresh. Clean refresh has no denoise step index. `chunk_index` is an AR block index; horizon is measured in blocks using the captured chunk-to-latent-frame map. Coordinates are zero-based absolute token positions in frame-major H×W row-major order.
- `local_attention_relative_error` and model/clean relative errors are Frobenius relative L2. `nmse_k/v` are squared relative energy errors over the declared intervention scope. Undefined zero-energy quantities stay null with a reason. Relative error and NMSE are not interchangeable.
- `retention` is requested candidate/history retention; `actual_retention` is retained / original full-window tokens including protection. `budget_cohort` identifies actual tokens, accesses or bytes, or an explicitly labeled upper budget.
- `logical_bytes` includes declared payload/scales/metadata according to the quantizer/selector. `physical_kv_bytes` describes backing capacity when known. Gathered inputs do not release the original source cache.
- `analysis_ms`, `scoring_ms`, `gather_ms`, `attention_ms` have distinct scopes; host vs GPU-event timing is declared in details/summary. Absent measurements stay null.
- `status` is measured, partially_measured or not_measured; `reason` explains missing values. A zero null control alone is not evidence that a real scientific treatment ran.

## Extension fields in details_json

Every row preserves fields not present in the fixed header in a JSON object. Common contracts:

| Family | Extension examples |
|---|---|
| Position/capture | key_abs_positions, frozen_absolute_positions, representative_positions, valid_key_mask, query_scope, candidate_scope, candidate_count |
| S1 | frame, role, cutoff, total_energy, near_zero_energy, temporal_window, temporal_control, spatial_indices, coordinate, pooling_grid |
| S2/E1 | attention_mass, statistic, spearman, topk_recall_sampled, finite_candidates, metric_scope, full-current-query model/clean metrics |
| E3 | k_cosine, v_cosine, joint_similarity, temporal_distance, comparison_reference, grouping_protocol, backend_protocol |
| E2 | calibration_matched, comparison_cohort, calibration_details, attained NMSE/RMSE/local error, paired_rng, intervention_hits |
| E4 | quantizer, allocated_candidate_bytes, allocation_objective, precision_scope, proxy_score |
| S3 | candidate_tokens, kept_candidate_tokens, kept_indices, removed_indices, partial_block_tokens, diagnostic_only, oracle_scope |
| E6 | actual_tokens_per_step, total_token_accesses, retentions, local_errors_by_step, oracle_scope, objective_kind |
| E7 | interaction, D_minus_C, B_minus_A, mask_protocol, selection_consistency, error_correlation, storage_contract, jensen_correction |
| E5 | entity_id, sequence_id, first_frame, disappearance_frame, reappearance_frame, feature_indices, state_feature_indices, matched_reappearance_identity_cosine/relative_l2, matched_reappearance_state_cosine/relative_l2, exploratory, quality_status |

## Containers and reproducibility

`manifest.json` records schema, configuration signature and full config, actual prompt/seed/rank, commit SHA, Python/PyTorch/NumPy/platform/CUDA/GPU, model/checkpoint and window. A unique run directory prevents collisions. Each experiment owns `events-<nonce>.csv`; close publishes complete parts atomically. Resume adds parts after validating config/origin and deduplicating deterministic context/protocol IDs. Old files are not overwritten.

`summary.json` separates real/synthetic/unrun experiments and quality metrics. Paired branch summaries record hit counts and per-call/final RNG validation. Analysis deduplicates within source/config/model cohorts, computes prompt/seed means before bootstrap, and exports merged CSV, per-experiment CSV, statistics, JSON, PDF/PNG and Markdown.

## research.frozen.v1 NPZ

The snapshot stores `q_post`, `k_post`, `v`, optional exact `k_pre`, `pre_valid_mask`, absolute query/key positions and coordinates, protected/eligible/valid masks, optional actual complex `freqs`, and scalar Unicode `metadata_json`. Q is sampled; K/V cover the full effective window of a selected head. BF16 arrays are serialized exactly as float32 with original dtype metadata and restored to BF16 on load. Runtime grid, layout, source, original native callback, rotation definition and later CPU/injected callback are explicit. No pickle or implicit inverse RoPE. ZIP expanded size and NPY declared shape are checked before allocating arrays.

Padding/partial-frame snapshots are rejected; runtime hooks assert the currently supported all-valid dense grid. Gapped whole-frame rolling windows and sinks are supported. Capture is bounded, exclusive and atomic. Snapshot files and raw reports are excluded from Git.

## 2026-10-28 corrected contracts

E5 generation details include `archive_protocol=research.e5.persistent.v2`, `intervention_scope`, `kept_abs_positions`, original/available eligible counts, requested budget, underfill, mask reuse, scoring-query count, actual query×key accesses and a separate clean/denoise cost summary. Evicted tokens cannot return; clean uses the same archive. This protocol is distinct from historical denoise-only outputs.

E5 feature event identity includes sequence/entity, first/disappearance/reappearance/state-anchor frames, event type and feature coordinates. Repeated returns are distinct; resumes do not drop later events. New cumulative summaries are immutable parts; original summaries remain historical artifacts.

Paired video indexes use `research.paired_videos.v1`; per-case manifests use `research.paired_video.v1` and record reference/case identity, prompt/seed context, resolved output, SHA256, frame dimensions/FPS/encoder, inference/decode/encode timing. These nested manifests are deliberately distinct from research.v1 sample manifests.
