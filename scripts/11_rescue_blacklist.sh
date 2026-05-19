#!/usr/bin/env bash
# Stage 11 — synthesize a replacement landmark for sub-paths whose
# original landmark can't be grounded.
#
# Two upstream sources feed this (handled uniformly):
#   • blacklist drops (step 02 — instruction too generic)
#   • detection failures (step 09 — YOLO/VLM couldn't locate the landmark)
#
# For each candidate, look at the partition pose, pick a referrable
# instance that the agent walks towards, and synthesize a new
# "Turn ... and walk to a [landmark]" sub-instruction.
#
# Output: target_instances/<scan>/blacklist_rescue.json — each record
# carries an ``origin`` field ({"blacklist", "detection_failure"}).
# Consumed by 12_consolidate (synthesized=true rows in dataset.json).
#
# Usage:
#   bash scripts/11_rescue_blacklist.sh --exp configs/selection/<split>/<exp>.yaml
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python src/check/rescue_blacklist.py --config "${CONFIG}" "$@"
