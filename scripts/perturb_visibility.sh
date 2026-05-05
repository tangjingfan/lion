#!/usr/bin/env bash
# Robustness check: target instance visibility under small position
# perturbations around each sub-trajectory's start node.
#
# Reads target_instance_ids from
#   {run}/target_instances/{scan}/target_instances.json
# (or the legacy {run}/target_instances/target_instances.json), stands
# at the start of each surviving sub-trajectory, samples N positions
# on a circle of radius R metres in the X-Z plane, and renders a 360°
# semantic panorama at each — answering whether the target is visible
# from a small neighbourhood, and whether any other visible instance
# shares the target instance's category.
#
# Usage:
#   bash scripts/perturb_visibility.sh --from_yaml configs/selection/exp.yaml
#   bash scripts/perturb_visibility.sh --from_yaml ... --radius 0.5 --n 8
#   bash scripts/perturb_visibility.sh --from_yaml ... --no_snap
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
python src/check/perturb_visibility.py --config "${CONFIG}" "$@"
