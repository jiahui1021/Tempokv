#!/usr/bin/env bash
set -euo pipefail
# No implicit model downloads or GPU leases. Extra CLI arguments pass through.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON:-python}"
exec "$python_bin" "$root/scripts/research/run.py" --config "$root/scripts/research/configs/self_forcing.yaml" --max-prompts 2 --max-events 2 "$@"
