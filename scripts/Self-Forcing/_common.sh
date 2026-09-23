#!/usr/bin/env bash
# Self-Forcing entry points: this model's directory plus the shared helpers.

self_forcing_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=scripts/common/common.sh
source "$self_forcing_dir/../common/common.sh"
