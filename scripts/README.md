# Entry points

```text
scripts/
├── Self-Forcing/   Self-Forcing trace, fit, BF16, and TempoKV commands
├── Causal-Forcing/ Causal Forcing download, trace, fit, BF16, and TempoKV commands
├── LongCat/        LongCat trace, fit, and generation commands
├── common/         shared parameter fitting command
├── evaluation/     video quality and VBench evaluation
└── tools/          generic analysis and pipeline utilities
                 (analyze_predictor.py, sweep_history_length.py)
```

Model code is kept under `experiments/`; reusable trace and fitted parameters
are kept under `data/`.
