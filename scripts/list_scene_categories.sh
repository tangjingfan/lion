#!/usr/bin/env bash
# Cache each scan's instantiated MPCAT40 object vocabulary as JSON.
#
# Pipeline role: prerequisite for `refine_landmark_mapping.sh` — that tool
# only accepts labels drawn from this scan's `objects.json`. Run with
# `--objects_only` against the experiment's selection YAML to produce the
# `objects.json` files refine_landmark_mapping expects.
#
# Also handy as a debugger for "no_match" landmark-visibility cases (e.g.
# the scene exposes "refrigerator", not "fridge"); the --scan / --grep
# flags below cover that use.
#
# Usage:
#   bash scripts/list_scene_categories.sh --from_yaml configs/selection/exp.yaml --objects_only
#   bash scripts/list_scene_categories.sh --scan X7HyMhZNoso
#   bash scripts/list_scene_categories.sh --scan X7HyMhZNoso --grep fridge
#   bash scripts/list_scene_categories.sh --scan X7HyMhZNoso --scan oLBMNvg9in8
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"

python src/check/list_scene_categories.py --config "${CONFIG}" "$@"
