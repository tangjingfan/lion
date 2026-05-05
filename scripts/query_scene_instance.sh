#!/usr/bin/env bash
# Query which MP3D/Habitat semantic object a raw instance id refers to.
#
# Usage:
#   bash scripts/query_scene_instance.sh --scan X7HyMhZNoso --instance_id 331
#   bash scripts/query_scene_instance.sh --scan X7HyMhZNoso --instance_id 331 --json
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python src/check/query_scene_instance.py --config "${CONFIG}" "$@"
