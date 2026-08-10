#!/usr/bin/env bash
# Self-Forcing 历史帧数敏感性实验。变量说明见 scripts/common/sweep_history.sh。
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODEL=self-forcing exec bash "$script_dir/../common/sweep_history.sh" "$@"
