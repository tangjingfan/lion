#!/usr/bin/env bash
# Count how many visible landmark annotations are unique.
#
# Usage:
#   bash scripts/count_visible_unique.sh --from_yaml configs/selection/val_unseen/one_scene_partial.yaml
#   bash scripts/count_visible_unique.sh --visibility_json results/<run>/landmark_visibility/<scan>/visibility.json
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"

python src/check/count_visible_unique.py --config "${CONFIG}" "$@"
