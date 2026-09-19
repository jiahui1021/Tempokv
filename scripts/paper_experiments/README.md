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
| `pruning/` | token pruning pilot: residual vs attention as a pruning signal |
| `tests/` | tests for the harnesses in this tree |

## Shared pieces

`common.py` holds the contracts every harness uses.
`fusion_qvg_reconstruction.py` compares TempoKV reconstruction with a
QVG-style residual-VQ fusion.
