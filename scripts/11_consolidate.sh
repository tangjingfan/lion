#!/usr/bin/env bash
# Stage 11 — consolidate every surviving sub-trajectory's info into one
# dataset file at the run dir root.
#
# Pure aggregation: reads survivor.yaml + partition.json + rewrite JSONs
# + target_instances.json and stitches them per (scan, ep, sub_idx) into
# one record. No LLM / simulator / detector calls.
#
# Output: results/<run>/dataset.json — top-level JSON list of records.
# Each record carries the text (full + sub-split + landmark / spatial),
# the path geometry (sub_path_nodes / spatial_path / landmark_path /
# heading / partition kind), the chosen target instance + visibility,
# any rescue annotations, and pointers to viz files.
#
# Usage:
#   bash scripts/11_consolidate.sh --exp configs/selection/<split>/<exp>.yaml
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python src/check/consolidate_dataset.py --config "${CONFIG}" "$@"
