#!/usr/bin/env bash
# Per-scene LLM refinement of landmark_mapping[_filtered].json.
#
# Reads each scan's scene_categories/{scan}/objects.json (produced by
# scripts/list_scene_categories.sh --objects_only) and the per-scan
# rewritten sub-instructions JSON, then asks the LLM for the
# {mention: [labels]} mapping — overwriting the rewriter's per-scan
# mapping in place at rewrite/{scan}/landmark_mapping[_filtered].json.
#
# Usage:
#   GEMINI_API_KEY=your_key bash scripts/refine_landmark_mapping.sh \
#       --from_yaml configs/selection/exp.yaml
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"
python src/check/refine_landmark_mapping.py --config "${CONFIG}" "$@"
