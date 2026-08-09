# Upstream

- Repository: <https://github.com/thu-ml/Causal-Forcing>
- Vendored commit: `1fc7bbc19a503c1bce80ecef08158b20e702f386`
- License: Apache-2.0 (see `LICENSE`)

## What is vendored

Only the inference runtime: `wan/`, `demo_utils/`, the inference pipelines, and
the inference-side `utils/`. The Stage 1/2/3 training code (`trainer/`,
`model/`, the training pipelines, the LMDB tooling) is not vendored, because
this migration is about KV cache compression, not about retraining Causal
Forcing inside the TempoKV repository.

## Local modifications

- **Package namespace.** Upstream imports bare top-level `wan`, `utils`,
  `pipeline` and `demo_utils`. Everything is re-rooted under `causal_forcing.*`
  so it cannot collide with the Self-Forcing runtime in the same repository.
- **Explicit model root.** Upstream hardcodes `wan_models/Wan2.1-T2V-1.3B`
  relative to the process working directory. `WanTextEncoder`, `WanVAEWrapper`
  and `WanDiffusionWrapper` now take an explicit `model_root`, supplied by
  `--model-root`.
- **Native TempoKV causal cache.** The preallocated
  `torch.zeros([B, kv_cache_size, 12, 128])` K/V tensors are replaced by
  `tempokv.runtime.cache.ChunkedKVCache`, a frame-aligned cache whose chunks are
  BF16 or an opaque packed TempoKV span. A packed span is not a tensor slice, so
  it cannot be written into a fixed buffer.
- **Pre-RoPE K storage.** Upstream stores the post-RoPE key. The cache now
  stores the normalized pre-RoPE key and reapplies RoPE at read time from each
  span's absolute frame position. TempoKV predictors are fitted in the pre-RoPE
  coordinate system; a post-RoPE cache would make them model the positional
  rotation, which keeps growing over a long rollout.
- **Explicit cache write modes.** `CausalInferencePipeline` announces
  `note_ephemeral_block`, `finalize_kv_cache` and `flush_kv_cache`. The four
  noisy denoising passes are ephemeral; the clean/context refresh finalizes the
  block exactly once. Nothing downstream inspects a timestep to decide this.
- **Rolling eviction is not implemented.** Upstream shifts the BF16 cache tensor
  left to evict, which a packed span cannot support. Storage stays
  frame-addressable for the whole rollout and attention reads a local window
  instead. `sink_size > 0` combined with a local attention window raises rather
  than silently producing different results from upstream.
- **Trimmed `utils/dataset.py`.** Only `TextDataset` and `TextImagePairDataset`
  are vendored, so a pure inference run does not import `lmdb`.
- **Trimmed `pipeline/__init__.py`.** Only the inference pipelines are exported.
- **Dropped the `rectified_tf` branch** in `causal_inference.py`. It was
  unreachable from the inference entry point and loaded `laboratory/mean.pt`
  relative to the working directory — a file that does not exist upstream
  either, and exactly the cwd dependence this vendoring removes.
- **Trace collection, metrics and entry points** under
  `causal_forcing/tempokv_adapter/` are new and have no upstream counterpart.

## Not vendored from upstream

`long_video/` (Rolling Forcing) is deliberately left out of this migration. The
cache abstraction already carries absolute frame positions and independent
predictor spans so it can be added without a rewrite, but rolling eviction,
sink retention and minute-level rollout come after BF16 parity and INT4
closed-loop correctness are established.
