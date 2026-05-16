#!/usr/bin/env bash
# Stage 1 — drop episodes whose path crosses floors.
#
# Computes the Y-range (vertical span) of every episode's path nodes and
# drops those exceeding --threshold_m (default 0.5 m).
#
# Usage:
#   bash scripts/01_filter_multi_floor.sh --from_yaml configs/selection/exp.yaml
#   bash scripts/01_filter_multi_floor.sh --from_yaml ... --threshold_m 1.0
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python src/check/filter_cross_floor.py --config "${CONFIG}" "$@"
