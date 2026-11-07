#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${AQUAKV_PYTHON:-python}" "$project_root/rebuttal/run.py" \
  --experiment aquakv --stage "${AQUAKV_STAGE:-pipeline}" "$@"
