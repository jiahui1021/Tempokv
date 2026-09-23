#!/usr/bin/env bash
# LongCat entry points: this model's directory plus the shared helpers.

longcat_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=scripts/common/common.sh
source "$longcat_script_dir/../common/common.sh"
adapter_root="$project_root/experiments/LongCat/longcat"
torchrun_bin="${TORCHRUN_BIN:-torchrun}"
