# Rolling Forcing provenance

- Repository: <https://github.com/TencentARC/RollingForcing>
- Local source snapshot used for the initial adapter audit:
  `code/videoquant-trq/forcing/rollingforcing/`
- Runtime contract: Wan causal pipeline with rolling windows, sink retention,
  and a clean refresh of the first block in each window
- Checkpoint/config: must be supplied from the pinned external checkout at run
  time; no model weights are copied into TempoKV

The TempoKV bridge is intentionally a patch at the runtime boundary. Changes
to upstream model code should be represented as a pinned patch in the external
checkout rather than silently vendored into this repository.
