# One-off scripts

Personal analysis runs kept for reference, not entry points. Every file here
hardcodes absolute paths from one machine (`/mnt/cpfs/...`), so none of them
runs anywhere else without editing. They live in their own directory so the
documented entry points in `scripts/` stay easy to pick out; see
`scripts/README.md` for those.

| file | what it did |
|---|---|
| `cf_compare_quality.py` | paired Causal Forcing quality comparison; still exercised by `scripts/paper_experiments/tests/test_cf_pairing.py` |
| `recompute_tempokv_quality.py` | recomputed per-video PSNR/SSIM/LPIPS for TempoKV videos against a BF16 reference |
| `recompute_qvg_quality.py` | the same for QuantVideoGen videos |
| `streaming_eval_loop.py` | drove streaming evaluation over a prompt list |
| `run_legacy_flow_timed.sh` | timed the legacy end-to-end flow, calling `streaming_eval_loop.py` |

Anything here that becomes a real entry point should grow environment-variable
arguments like the model scripts do, and move up into the structure in
`scripts/README.md`.
