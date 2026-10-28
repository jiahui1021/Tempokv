# Experiment repair record — 2026-10-28

All nine findings from the read-only review of commit `6e1e26e` are corrected. These are implementation/protocol repairs, not scientific results. Production quantization, predictors and fitting algorithms are unchanged.

| Finding | Corrected behavior | Executable regression |
|---|---|---|
| R1 model path | research and B Self-Forcing runners set the actual loader's `FORCING_WAN_MODEL_DIR`, keep the alias consistent, and record the resolved directory; CF uses its explicit argument chain | `test_research_runtime_fixes.py`: actual AST resolver with conflicting inherited path and launch records |
| R2 E5 clean/history bypass | denoise and clean refresh share a persistent per-chunk mask; evicted positions cannot reenter scoring or branch attention; full runtime scope is explicit | `test_research_memory_persistence.py`: clean output excludes old V, later choices cannot read an evicted row, both heads/unselected layer enforced under all scope |
| R3 E5 lost feature events | sequence/entity/return interval/feature coordinates form stable event identity; resume reconstructs prior IDs and cumulative counts without replacing old summaries | `test_research_runtime_fixes.py`: repeated returns, multiple sequences/entities, CSV reconstruction and legacy-ID resume |
| R4 missing paired videos | reference and every case get an exclusive MP4, manifest and shared session index; inference/decode/encoding costs remain separate; disabled saving does no VAE work | `test_research_runtime_fixes.py`: real FFmpeg H.264 encoding, ffprobe and RGB decode for reference plus two cases |
| R5 incompatible DC means | spatial and joint spatiotemporal DC have separate definitions/groups; common frame/window fields are deduplicated across frequency bins | `test_spectral_kv_review_fixes.py`: spatial1/joint0 remain separate; bin repetition does not change weights |
| R6 mixed model summaries | B cohorts include model/checkpoint and measurement configuration, raw exports preserve provenance and `sources.json` | `test_spectral_kv_review_fixes.py`: model/config variants remain distinct |
| R7 CF last clean missing | only read-only collectors that need an eligible terminal clean pass request it; extra calls and nested collection time are recorded | `test_spectral_kv_review_fixes.py`: real CF inference with CPU substitutes, positional/keyword arguments and online/no-clean exclusions |
| R8 failed/retried A samples | failed/incomplete samples are excluded; stable completed retries are selected once, with exclusion counts; incompatible model/config/runtime identities fail clearly | `test_spectral_review_fixes.py`: failed prefix does not bias the successful mean and completed retries do not duplicate observations |
| R9 unstable mass | normalization retains max and centered log-partition separately | `test_spectral_review_fixes.py`: large-logit shift invariance, whole-window mass1 and candidate ordering |

## E5 protocol after correction

`research.e5.persistent.v2` is enforced in denoising and clean refresh. Removal is permanent within a branch. The same chunk mask is reused across denoise steps and clean; subsequent chunks may add newly generated positions. Scoring sees only available history. Current/recent/sink protection remains mandatory. `intervention.e5_scope=all` applies to all runtime layers/heads in the shipped E5 configs; `selected` declares a local intervention.

The nominal history budget uses the same original window across policies. A rolling window can leave fewer available historical positions than its budget: actual underfill is reported; evicted information is not restored to make a curve appear budget matched. Rows carry actual tokens/bytes and query×key accesses; clean and denoise accesses are separately summed. Comparisons must use actual cost cohorts. The full native backing is still retained solely to support diagnostic references, so this is not a production packed-cache/memory-saving implementation.

Old denoise-only E5 data remains historical. Configuration signatures and the new protocol prevent automatic cross-protocol resume or pooling. Raw legacy CSV stays readable. Resume under an unchanged valid signature preserves prior files and publishes a cumulative `summary-<nonce>.json`; original `summary.json` is not replaced.

## Verification

```bash
python -m pytest scripts/tests/test_research_*.py scripts/tests/test_spectral*.py -q
```

The initial nine-fix gate passed **957 checks/1 optional skip**. With related B/C failure filters added, the extended run passed **959/1 optional skip**. The final full run passed958 and exposed one nondeterministic test that assumed random CSV filenames preserve creation order. Its actual event data and legacy IDs were correct. The assertion was changed to check event identity as a set; the39 related checks then passed. These full/targeted logs are kept separately; a fourth full-suite run is not claimed. The current collected set contains959 checks, plus the optional skip. Fifteen unaffected fitting/launcher regressions also passed. The skip is an optional SciPy comparison. This comprises 605 research tests, 53 A tests and 301 B tests. Additional targeted sets overlap this combined run and are not added to that count. Three CPU CLI/saved-data analysis paths (A, B and corrected E5) completed. The real model loader's argument construction and resolver were tested without weights; real GPU generation remains unrun.

The old `reports/research/review_20261028/repro_*.py` scripts assert pre-fix faulty behavior and are retained as review evidence. Correct behavior is asserted by the new regression tests above. Do not use an old failure-demonstration script as a post-fix pass criterion.

## Related failure-filter repair

The same failure class was also corrected in B and research.v1 saved-data reconstruction. B excludes explicitly failed/incomplete run metadata and records exclusions; legacy missing completion status is reported explicitly. research.v1 checks each immutable CSV part against its own completion record, so a failed resume or orphan CSV cannot inherit an earlier run's completed status. Failed run summaries never claim completed experiments. Original records remain unchanged. Legacy single-summary exports are supported; a missing/corrupt completion record is not treated as completion.
