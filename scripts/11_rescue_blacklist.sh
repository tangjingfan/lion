#!/usr/bin/env bash
# Stage 13 — find replacement landmarks for sub-paths dropped by 03.
#
# For each (ep_id, sub_idx) the blacklist filter cut, look at the
# sub-path's end pose, pick a referrable instance that the agent walks
# towards, and synthesize a new "Turn ... and walk to a [landmark]"
# sub-instruction.
#
# Output: target_instances/<scan>/blacklist_rescue.json — consumed by
# 12_consolidate (which emits these as records with
# synthesized=true alongside the original ones).
#
# Usage:
#   bash scripts/11_rescue_blacklist.sh --exp configs/selection/<split>/<exp>.yaml
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python src/check/rescue_blacklist.py --config "${CONFIG}" "$@"
