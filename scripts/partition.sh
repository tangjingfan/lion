#!/usr/bin/env bash
# Visualize sub-path partitions on the connectivity graph.
# Usage: bash scripts/partition.sh --config configs/rollout/rollout_landmark_rxr.yaml [options]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python src/check/visualize_partition.py "$@"
