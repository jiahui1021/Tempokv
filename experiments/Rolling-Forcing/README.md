# Rolling Forcing

This directory contains the TempoKV bridge for the upstream Rolling Forcing
runtime. The upstream checkout remains external because its rolling-window
pipeline and Wan modules have a separate dependency tree.

The bridge preserves the upstream inference schedule: only the first block of
each rolling window enters the persistent cache; the sink block and recent
working range form the attention context; and a clean refresh rewrites and
compresses the current block. K/V are stored pre-RoPE and read with absolute
frame positions, with V decoded from reconstructed K for CrossKV closure.

Set `ROLLING_FORCING_ROOT` to a pinned checkout and use
`scripts/Rolling-Forcing/run_tempokv.sh`. The checkout must provide
`pipeline/rolling_forcing_inference.py`, `wan/modules/causal_model.py`, and
the upstream `inference.py` entrypoint. The bridge infers layer/head geometry
from the loaded model, while `TEMPOKV_ROLLING_CACHE_FRAMES` controls the
block-aligned resident capacity (default 24 latent frames).

Rolling parameters are fitted independently from clean traces collected with
`scripts/Rolling-Forcing/collect_trace.sh`; do not reuse Self-Forcing or Causal
Forcing parameter files even when the head geometry happens to match:

```bash
ROLLING_FORCING_ROOT=/path/to/RollingForcing \
CHECKPOINT=/path/to/rolling.pt \
PROMPTS=/path/to/fit_prompts.txt \
bash scripts/Rolling-Forcing/collect_trace.sh
bash scripts/Rolling-Forcing/fit_params.sh
```

The adapter has only been validated with CPU cache tests and static checks in
this repository. A GPU run still needs the EPIC UUID lease gate, a pinned
checkpoint, clean traces/parameters fitted for the actual frame stride, and
paired BF16 evaluation.
