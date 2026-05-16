#!/usr/bin/env bash
# Stage 2 — LLM rewrite of sub-instructions into structured landmark guidance.
#
# Asks Gemini to extract {landmark, landmark_instruction, spatial_instruction,
# keep, components[.semantic_label]} per sub-instruction. Produces side-car
# JSONs under rewrite/{scan}/{instruction_id}/ — does NOT itself drop
# episodes; the subsequent blacklist stage consumes these JSONs to filter.
#
# Input: typically the stage-1 survivor file (filters/01_cross_floor.yaml).
#
# Usage:
#   GEMINI_API_KEY=your_key \
#   bash scripts/02_rewrite_subinstruction.sh \
#       --from_yaml results/.../filters/01_cross_floor.yaml
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python src/check/rewrite_subinstructions.py --config "${CONFIG}" "$@"
