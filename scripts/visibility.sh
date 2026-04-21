#!/usr/bin/env bash
# Check sub-path visibility.
# Usage: bash scripts/visibility.sh --config configs/rollout/rollout_landmark_rxr.yaml [options]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python src/check/check_visibility.py "$@"
