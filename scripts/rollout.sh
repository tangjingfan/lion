#!/usr/bin/env bash
# Run the agent rollout.
# Usage: bash scripts/rollout.sh --config configs/rollout/rollout_landmark_rxr.yaml [options]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python src/rollout.py "$@"
