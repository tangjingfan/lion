#!/usr/bin/env bash
# Stage 10 — drop sub-paths whose chosen target instance is grounded only to
# a coarse MPCAT40 bucket (appliances, lighting, furniture, objects, ...).
#
# Reads the latest survivor file (filters/current.yaml) and the target
# instance + rescue annotations. A sub-path is kept only if its selected
# target has a fine-grained category match, either from
# 08_get_potential_instance or rescued by 09_vlm_rescue. Otherwise dropped.
#
# Prerequisites (run in order):
#   1.  04_partition.sh         (filters/03_partition.yaml)
#   2.  05_get_object_list.sh
#   3.  06_refine_landmark_mapping.sh
#   4.  07_list_potential_instances.sh
#   5.  08_get_potential_instance.sh
#   6.  09_vlm_rescue.sh        (optional but recommended)
#
# Usage:
#   bash scripts/10_filter_semantic_granularity.sh \
#       --from_yaml results/.../filters/03_partition.yaml
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python src/check/filter_semantic_granularity.py --config "${CONFIG}" "$@"
