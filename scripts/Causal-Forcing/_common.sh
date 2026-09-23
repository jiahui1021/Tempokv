#!/usr/bin/env bash
# Causal Forcing entry points: this model's directory plus the shared helpers.

causal_forcing_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=scripts/common/common.sh
source "$causal_forcing_dir/../common/common.sh"
