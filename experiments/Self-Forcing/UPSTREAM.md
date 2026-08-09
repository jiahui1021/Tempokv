# Bundled Self-Forcing runtime

The files in this directory are derived from the official Self-Forcing
repository:

- Repository: <https://github.com/guandeh17/Self-Forcing>
- Commit: `33593df3e81fa3ec10239271dd2c100facac6de1`
- License: Apache-2.0 (see `LICENSE`)

TempoKV keeps the upstream model structure and adds a frame-aligned pre-RoPE KV
cache used by both the BF16 and TempoKV paths. The Wan model directory is passed
through `FORCING_WAN_MODEL_DIR`. Trace and TempoKV adapters live under
`experiments/Self-Forcing/tempokv_adapter/`; public shell commands live under
`scripts/Self-Forcing/`.
