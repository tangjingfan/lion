#!/usr/bin/env bash
# Annotate per-sub-path landmark visibility at the partition point using
# Habitat-Lab's rgbds_agent semantic panorama.  Classification only — does
# NOT drop sub-paths or advance current.yaml.
#
# Reads the latest sub-path-level survivor set from
#   results/{run_name}/filters/current.yaml
# and writes a per-sub-path classification JSON to
#   results/{run_name}/landmark_visibility/visibility.json
#
# Usage:
#   bash scripts/annotate_visibility.sh [--from_yaml configs/selection/exp.yaml]
#   bash scripts/annotate_visibility.sh --from_yaml ... --min_pixel_count 100
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"

python src/check/annotate_visibility.py --config "${CONFIG}" "$@"
