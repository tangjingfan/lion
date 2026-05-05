#!/usr/bin/env bash
# Select target semantic instance ids from landmark visibility annotations.
#
# Usage:
#   bash scripts/select_target_instances.sh --from_yaml configs/selection/one_scene_partial_val_unseen.yaml
#   bash scripts/select_target_instances.sh --from_yaml ... --dominance_ratio 4.0 --print_ambiguous
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python src/check/select_target_instances.py --config "${CONFIG}" "$@"
