#!/usr/bin/env bash
# Select target semantic instance ids from landmark visibility annotations.
#
# Selection rule:
#   • 1 visible instance  -> view_unique
#   • >1 visible          -> view_nearest (closest to sub-path end position)
#
# Usage:
#   bash scripts/08_get_potential_instance.sh --exp configs/selection/val_unseen/one_scene_partial.yaml
#   bash scripts/08_get_potential_instance.sh --exp ... --print_multi
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python src/check/select_target_instances.py --config "${CONFIG}" "$@"
