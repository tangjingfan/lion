#!/usr/bin/env bash
# Stage 14 — render an inspection panorama at the partition-end pose for
# every (ep, sub) that survived cross_floor, regardless of whether later
# stages dropped it. Output:
#
#   {run_dir}/target_instances/viz_inspection/<scan>/<ep>/sub_<idx>__<status>.png
#
# Filename status suffix tells you what happened to the sub-path
# (view_unique / blacklist_llm_keep_false / partition_* / not_visible /
# rescued / synthesized), so a directory listing of thumbnails is enough
# to triage what's worth rescuing.
#
# Read-only: this step does not modify survivor.yaml, dataset.json, or
# any other pipeline state.
#
# Usage:
#   bash scripts/14_inspection_viz.sh --exp configs/selection/<split>/<exp>.yaml
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python src/check/inspection_viz.py --config "${CONFIG}" "$@"
