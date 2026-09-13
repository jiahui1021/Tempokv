# Paper experiments

One study per subdirectory, each with its own README and entry point. These are
not day-to-day commands; for those see `scripts/README.md`.

| study | subject |
|---|---|
| `observation/` | the K/V predictability ladder (paper §3.2) |
| `calibration/` | calibration study |
| `predictor_ablation/` | Self-Forcing predictor ablation, 60 prompts |
| `history_ablation/` | Self-Forcing history-length ablation |
| `innovation_ablation/` | Self-Forcing innovation ablation |
| `cross_kv_sharing/` | how far the CrossKV projection can be shared |
| `direction/` | whether K should lead V or the reverse |
| `span/` | whether the anchor truncates error propagation inside a span |
| `long_horizon/` | long-horizon reconstruction |
| `latency/` | fixed-length overhead and the length-scaling curve |
| `efficiency/` | Self-Forcing efficiency runner |
| `tests/` | tests for the harnesses in this tree |

## Shared pieces

`common.py` holds the contracts every harness uses, and
`_self_forcing_runner.sh` is the launcher the three Self-Forcing ablations
share.

## Loose study scripts

These sit at the top level because they cut across studies or were written for
one run. Most hardcode absolute paths from one machine and need editing before
they run anywhere else.

| script | what it does |
|---|---|
| `run_5_mixed_bits.sh` | Exp. 5: K/V mixed bit widths at one cache setting |
| `run_6_dense_channelwise.sh` | Exp. 6: dense versus channel-wise `W_{K->V}` |
| `run_7_rope.sh` | Exp. 7: key prediction in pre-RoPE versus post-RoPE coordinates |
| `run_exp5_legacy_parallel.sh` | launches the exp. 5 legacy+shared arms across three GPUs |
| `summarize_exp5_legacy.py` | aggregates exp. 5 streaming results into CSV and markdown |
| `eval_exp567.py` | per-video quality for experiments 5-7 against the BF16 reference |
| `eval_qvg_chunkwise.py` | the same for index-paired QuantVideoGen videos |
| `fusion_qvg_reconstruction.py` | TempoKV reconstruction versus a QVG-style residual-VQ fusion |
| `run_cf_todo_bf16.sh` | Causal Forcing pipeline: BF16 then TempoKV INT2/INT4 |
| `run_cf_todo_qvg.sh` | Causal Forcing QuantVideoGen generation, 128 MovieGen prompts |
| `run_91.sh` | the two "9.1" experiments on four leased GPUs |
| `wait_and_run_91.sh` | waits for the baseline reruns to reach 90 videos, then starts `run_91.sh` |
| `after_study_ablation.sh` | waits for `run_91.sh`'s study to finish, then launches the next ablation |
| `resume_predictor.sh` | resumes the predictor ablation after its orchestrator exits |
| `eval_takeover_predictor.sh` | takes over the predictor ablation's evaluation phase |
| `fid_after_predictor.sh` | runs FID once the predictor ablation's generation and PSNR/SSIM/LPIPS are done |
