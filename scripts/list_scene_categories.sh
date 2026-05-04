#!/usr/bin/env bash
# List MP40 / MP3D semantic categories present in a scene.
#
# Useful for debugging "no_match" cases in landmark visibility — tells you
# what category *names* the scene's semantic mesh actually exposes (e.g.
# "refrigerator" not "fridge").
#
# Usage:
#   bash scripts/list_scene_categories.sh --scan X7HyMhZNoso
#   bash scripts/list_scene_categories.sh --scan X7HyMhZNoso --grep fridge
#   bash scripts/list_scene_categories.sh --scan X7HyMhZNoso --scan oLBMNvg9in8
#   bash scripts/list_scene_categories.sh --from_yaml configs/selection/val_unseen_example.yaml --objects_only
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"

python src/check/list_scene_categories.py --config "${CONFIG}" "$@"
