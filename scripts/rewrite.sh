#!/usr/bin/env bash
# Rewrite sub-instructions with landmark components and spatial grounding.
# Usage: bash scripts/rewrite.sh --config configs/rollout/rollout_landmark_rxr.yaml [options]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python src/check/rewrite_subinstructions.py "$@"
