#!/usr/bin/env bash
# Stage 3 — drop sub-paths whose landmark is not concretely referrable.
#
# Consumes the rewrite JSONs from stage 2 and the prior survivor file, then
# drops a sub-path if ANY of:
#   • landmark_category == "room" AND the phrase is a generic room
#   • LLM keep == false
#   • landmark phrase contains a blacklisted whole word
#   • mapped semantic label is in the blacklist
#
# Input: the same survivor file stage 2 read (typically 01_cross_floor.yaml) —
# this stage uses it to know which (instruction_id, sub_idx) to test against
# the rewrite output, not stage 2's output (stage 2 writes no survivor file).
#
# Usage:
#   bash scripts/03_blacklist_landmark.sh \
#       --exp results/.../filters/01_cross_floor.yaml
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python src/check/filter_blacklist.py --config "${CONFIG}" "$@"
