# Entry points

```text
scripts/
├── fit_params.sh      one parameter-fitting entry point for all three models
├── Self-Forcing/      Self-Forcing trace, fit, BF16, and TempoKV commands
├── Causal-Forcing/    Causal Forcing download, trace, fit, BF16, and TempoKV commands
├── LongCat/           LongCat trace, fit, and generation commands
├── common/            shared parameter fitting command
├── evaluation/        paired quality/FID, VBench, and one-click evaluation
├── tools/             generic analysis and pipeline utilities
│                      (analyze_predictor.py, sweep_history_length.py)
├── paper_experiments/ one study per subdirectory, each with its own README
│                      (calibration, cross_kv_sharing, direction,
│                      predictor_ablation, reconstruction, ...)
└── tests/             tests for the scripts in this tree
```

The first seven entries are the supported surface. `paper_experiments/` holds
per-paper studies rather than commands you would run day to day.

Model code is kept under `experiments/`; reusable trace and fitted parameters
are kept under `data/`.

Fit parameters with one consistent command:

```bash
bash scripts/fit_params.sh self-forcing
bash scripts/fit_params.sh causal-forcing
bash scripts/fit_params.sh longcat
```

The model-specific `fit_params.sh` files remain supported for compatibility.
