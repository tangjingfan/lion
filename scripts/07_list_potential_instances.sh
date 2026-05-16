#!/usr/bin/env bash
# Enumerate candidate target instances per surviving sub-path.
#
# For each (ep, sub_idx), lists every instance of the matched MP40
# category present in the scene (regardless of FOV / visibility) and
# tags uniqueness = unique / ambiguous / no_match.  By default also
# renders one rollout-style mask PNG per candidate at the sub-path
# end pose; pass --no_save_viz to skip.
#
# Reads the latest sub-path-level survivor set from
#   results/{run_name}/filters/current.yaml
# and writes per-scan annotations to
#   results/{run_name}/target_instances/{scan}/target_instances.json
# (viz PNGs under .../target_instances/viz/{scan}/{ep}/...)
#
# Usage:
#   bash scripts/07_list_potential_instances.sh [--from_yaml configs/selection/exp.yaml]
#   bash scripts/07_list_potential_instances.sh --from_yaml ... --no_save_viz
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python src/check/list_target_instances.py --config "${CONFIG}" "$@"
