# Rolling Forcing commands

Set `ROLLING_FORCING_ROOT` to a pinned upstream checkout. The command forwards
all remaining arguments to the upstream `inference.py` parser:

```bash
ROLLING_FORCING_ROOT=/path/to/RollingForcing \
  bash scripts/Rolling-Forcing/run_tempokv.sh \
  --prompt_path /path/to/prompts.txt --output_path results/rolling
```

Use `run_bf16.sh` for the native cache baseline. TempoKV mode requires the
same-frame-stride fitted parameter files. By default it reads
`configs/legacy.json` and `data/rolling-forcing/params/rolling-legacy`; set
`CONFIG`, `PARAM_DIR`, and `V_PREDICTOR` for another fitted variant. It writes
a report when `TEMPOKV_ROLLING_REPORT` is set.
