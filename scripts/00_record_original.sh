#!/usr/bin/env bash
# Stage 0 — record the user's original selected episode set.
#
# Reads the selection YAML and writes its survivor set to the experiment's
# filters/ directory as a baseline before any pipeline-driven filtering.
# Useful as an audit anchor: every later stage's keep/drop is relative to
# this initial set.
#
# Usage:
#   bash scripts/00_record_original.sh --from_yaml configs/selection/exp.yaml
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python src/check/filter_original.py --config "${CONFIG}" "$@"
