#!/usr/bin/env bash
# Stage 4 — geometric partition of each sub-path + drop sub-paths whose
# partition / rewrite is invalid.
#
# Runs three substeps:
#   (a) src/check/visualize_partition.py — computes a virtual partition
#       node P that splits each sub-path into a spatial half (the turn /
#       move described) and a landmark half (approach to the named
#       landmark). Writes partition/{scan}/{instruction_id}/partition.json.
#   (b) src/check/filter_partition.py — labels sub-paths whose rewrite or
#       geometric partition failed; writes survivor.sub_status +
#       filters/03_partition_dropped.yaml.
#   (c) src/check/render_partition_obs.py — renders an RGB + semantic
#       panorama at every (ep, sub) partition pose into one place:
#       partition_obs/{scan}/{ep}/sub_{idx:03d}.png (sibling of
#       partition/). Covers every sub past cross_floor — active and
#       labeled both.
#
# Usage:
#   bash scripts/04_partition.sh --exp configs/selection/<split>/<exp>.yaml
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

echo "── 4a. visualize_partition ──────────────────────────"
python src/check/visualize_partition.py --config "${CONFIG}" "$@"

echo "── 4b. filter_partition ─────────────────────────────"
python src/check/filter_partition.py    --config "${CONFIG}" "$@"

echo "── 4c. render_partition_obs ─────────────────────────"
python src/check/render_partition_obs.py --config "${CONFIG}" "$@"
