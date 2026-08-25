# Self-Forcing efficiency runner

This directory contains the manifest-driven runner and collector for the
13-row Self-Forcing efficiency matrix described in
`docs/superpowers/specs/2026-08-25-self-forcing-efficiency-measurement-design.md`.

`run_efficiency.py --mode efficiency` launches the four-prompt, 717-pixel-frame
runs and expects one `efficiency.json` below each method directory.
`run_efficiency.py --mode horizon` launches the one-prompt, 3009-pixel-frame
global-window probes and expects one `horizon.json` below each method directory.
Both modes run the same preflight and isolate failures by method.

The checked-in manifest intentionally uses `@FILL@` for the checkpoint, Wan
model directory, prompt file, prompt hash, and GPU UUID. Fill those values on
the GPU machine, then run:

```bash
python scripts/paper_experiments/efficiency/run_efficiency.py \
  --manifest scripts/paper_experiments/efficiency/manifest.self-forcing.json \
  --mode efficiency --dry-run --no-gpu-check
```

Horizon probes require an idle device. The runner verifies the frozen GPU name
and UUID, checks for other compute processes, and exports the shared
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` setting for every probe.

After the runs finish, collect the outputs with:

```bash
python scripts/paper_experiments/efficiency/collect_efficiency.py \
  --output-root results/paper_experiments/efficiency-self-forcing
```

To run only the max-horizon probes:

```bash
CHECKPOINT=/path/model.pt \
WAN_MODEL_DIR=/path/Wan2.1-T2V-1.3B \
PROMPTS=/path/prompts.txt \
CONFIG=configs/legacy.json \
scripts/paper_experiments/efficiency/run_max_horizon.sh
```

The command fails if any selected method exits without writing
`<output-root>/<method>/horizon.json`. A successful collection writes
`horizon.csv` and `horizon.tex` with measured max feasible frames, whether the
probe reached OOM, and the extrapolation cross-check.

## Three final per-repository launchers

The same workload can be launched without editing the 13-row central manifest:

```bash
# TempoKV: BF16, TempoKV INT2, TempoKV INT4
CHECKPOINT=/path/model.pt \
WAN_MODEL_DIR=/path/Wan2.1-T2V-1.3B \
PROMPTS=/path/prompts.txt \
CONFIG=configs/legacy.json \
scripts/paper_experiments/efficiency/run_efficiency.sh --mode both

# QVG: QVG/QVG-Pro INT2/INT4
SELF_FORCING_CHECKPOINT_PATH=/path/model.pt \
PROMPTS_PATH=/path/prompts.txt \
scripts/paper_experiments/efficiency/run_efficiency.sh --mode both

# KVQuant4Video: RTN/KIVI/QuaRot-KV INT2/INT4
CONFIG_PATH=/path/self_forcing_dmd.yaml \
DEFAULT_CONFIG_PATH=/path/default_config.yaml \
CHECKPOINT_PATH=/path/model.pt \
PROMPT_PATH=/path/prompts.txt \
Self-Forcing/scripts/paper_experiments/efficiency/run_efficiency.sh --mode both
```

All three support `--only`, `--skip`, `--dry-run`, `--overwrite`, and
`--no-collect`. Each launcher writes one method directory containing the raw
record and then creates a local `efficiency.csv`/`.tex` summary.

The collector reports two different ratios:

- `resident_compression_ratio` (also exported as `compression_ratio`) is the
  BF16-equivalent bytes divided by resident bytes at the same final full-cache
  sample. It is available from one repository alone and is the direct
  resident-cache compression ratio. The two independent peak fields are never
  divided.
- `cache_reduction` is `TempoKV BF16 peak resident bytes / this row's peak
  resident bytes`. It is the cross-repository table quantity and only becomes
  meaningful after the TempoKV BF16 record is available.
