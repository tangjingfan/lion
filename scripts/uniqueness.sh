#!/usr/bin/env bash
# Check landmark uniqueness.
# Usage: bash scripts/uniqueness.sh --config configs/rollout/rollout_landmark_rxr.yaml [options]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python src/check/check_landmark_uniqueness.py "$@"
