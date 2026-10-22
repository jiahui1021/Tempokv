# Research result note — 2026-10-23

**Evidence level: CPU numerical and execution verification only. Real Self-Forcing/Wan video experiments: not run.** The local environment provides PyTorch 2.13.0 and NumPy 2.5.1 but no CUDA, model checkpoint, full Self-Forcing dependencies or real experiment dataset. No scientific result is inferred from synthetic inputs.

| Question | Implemented comparison | Available evidence | Current conclusion |
|---|---|---|---|
| S1/S2 | clean spatial/temporal spectrum; shuffled/pooling controls; score vs independent native LOO | numerical fixtures and exact position/protection checks | frequency usefulness unknown |
| E1/E2 | local → model-step → final current clean → future drift; single frozen event, paired RNG | hooked CPU AR surrogate; actual native attention class in a fake base pipeline | real generation rankings and propagation unknown |
| E3/E4 | conditional drop/replace/pre-RoPE merge; equal normalized K/V noise; actual-cost bit allocation | exact duplicates/nulls, general approximation and budget tests | real substitutability and asymmetric tolerance unknown |
| E6 | isolated step changes and measured complete-schedule budget frontier | equal actual-token-access CPU schedules; zero/full controls | real step budget benefit unknown |
| S3/E7 | equal-token selection; frozen and paired quantization × pruning four-cell comparisons | raw synthetic CSV, independently rebuilt plots, exact frozen masks | no compression or synergy claim |
| E5 | causal equal-budget archives; return/static scene fixtures; declared entity/state consistency | null/identity-flip feature controls and branch-owned-history replay | exploratory; no real memory-necessity result |

**Verification:** 587 research tests pass. They cover non-square FFT/DCT reconstruction and Parseval, exact BF16 round trips, native reference identity, rolling/sink absolute positions, protection, K/V paired removal, independent LOO/conditional comparisons, pre/post-RoPE boundaries, BF16 calibration floors, all RNG streams, single-hit events, actual byte/token costs, atomic writes/resume, and CLI execution. M0→M1→M2→M3→M4 gates were followed. Previously existing suites A/B remain separate.

The saved CPU runs are under `reports/research/synthetic_m1`, `synthetic_m2`, `synthetic_m3`, `synthetic_e7_paired` and `synthetic_m4`; their `analysis/analysis-*` folders contain merged CSV, summary JSON, PDF/PNG figures and a one-page generated report. Raw data carries synthetic origin labels. E5 feature consistency remains a separate source cohort from the AR surrogate, with no invented connection between its known feature changes and archive-policy performance.

**Unmeasured:** real model generation, actual fitted TempoKV paired comparison, video identity/state extraction, PSNR/SSIM/LPIPS/FID/VBench, real long-horizon quality, production speedup and resident-memory savings. Token selection here gathers attention inputs while preserving the original backing cache. E7 interaction is only a nonlinear error contrast. The E5 heavy-run gate remains unsatisfied because real paired evidence is absent. Frequency truncation, new predictive mechanisms and a claimed new compression method were deliberately not inferred from this verification.
