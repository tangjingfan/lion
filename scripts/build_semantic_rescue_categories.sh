#!/usr/bin/env bash
# Use a VLM to assign finer categories to instances dropped by
# filter stage 4's coarse semantic label check.
#
# Usage:
#   GEMINI_API_KEY=... bash scripts/build_semantic_rescue_categories.sh \
#       --from_yaml results/.../filters/03_partition.yaml
#   bash scripts/build_semantic_rescue_categories.sh --from_yaml ... --dry_run
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"

python src/check/build_semantic_rescue_categories.py --config "${CONFIG}" "$@"
