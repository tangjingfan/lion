#!/usr/bin/env bash
# Stage 4 — geometric partition of each sub-path + drop sub-paths whose
# partition / rewrite is invalid.
#
# Runs two substeps:
#   (a) src/check/visualize_partition.py — computes a virtual partition
#       node P that splits each sub-path into a spatial half (the turn /
#       move described) and a landmark half (approach to the named
#       landmark). Writes partition/{scan}/{instruction_id}/{*.png,
#       partition.json}.
#   (b) src/check/filter_partition.py — drops a sub-path if the rewrite
#       failed OR the geometric partition has <2 nodes / other errors.
#       Writes filters/03_partition.yaml.
#
# Usage:
#   bash scripts/04_partition.sh \
#       --from_yaml results/.../filters/02_blacklist.yaml
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

echo "── 4a. visualize_partition ──────────────────────────"
python src/check/visualize_partition.py --config "${CONFIG}" "$@"

echo "── 4b. filter_partition ─────────────────────────────"
python src/check/filter_partition.py    --config "${CONFIG}" "$@"
