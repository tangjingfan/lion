#!/usr/bin/env bash
# Stage 12 — print a per-stage attrition funnel for one experiment.
#
# Pure analysis: reads filters/audit.json + filters/NN_*_dropped.yaml +
# dataset.json and reports episode / sub-path counts surviving each
# stage, with drop reasons. No simulator / LLM / detector calls.
#
# Usage:
#   bash scripts/13_attrition.sh --exp configs/selection/<split>/<exp>.yaml
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python src/check/attrition.py --config "${CONFIG}" "$@"
