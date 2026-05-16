#!/usr/bin/env bash
# Stage 10 — apply VLM pixel-grounded rescue results back into
# target_instances/<scan>/target_instances.json.
#
# For each (episode_id, sub_idx) in target_instances.json:
#   • If it had no target_instance_ids and rescue found one →
#     fill in the rescued instance id and mark status="rescued".
#   • If it already had a target → leave it alone, just annotate
#     rescue_landmark / rescue_category / rescue_instance_id.
#
# Idempotent: re-running overwrites the rescue_* annotations cleanly.
#
# Usage:
#   bash scripts/10_apply_rescue.sh --exp configs/selection/<split>/<exp>.yaml
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python src/check/apply_rescue.py --config "${CONFIG}" "$@"
