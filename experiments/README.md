# Model code

Each model keeps its runtime and integration code in a separate directory.

```text
experiments/
├── Self-Forcing/
└── LongCat/
```

TempoKV's model-independent codec and kernels do not belong here; they live in
`src/tempokv/`. User-facing commands live in `scripts/<model>/`.
