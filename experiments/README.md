# Model code

Each model keeps its runtime and integration code in a separate directory.

```text
experiments/
├── Self-Forcing/
├── Causal-Forcing/
└── LongCat/
```

Self-Forcing and Causal Forcing both run Wan2.1-1.3B as a causal autoregressive
video model, so they share one KV runtime from `src/tempokv/runtime/`: the
frame-aligned pre-RoPE cache, read-time absolute-position RoPE, the streaming
span encoder, and the memory/latency accounting. Each keeps its own vendored
upstream tree so its provenance stays separable.

TempoKV's model-independent codec and kernels do not belong here; they live in
`src/tempokv/`. User-facing commands live in `scripts/<model>/`.
