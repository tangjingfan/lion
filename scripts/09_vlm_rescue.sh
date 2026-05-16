#!/usr/bin/env bash
# YOLO-World pixel-grounded rescue: run an open-vocab detector on rollout
# panoramas, then query Habitat semantic pixels to recover instance ids.
#
# Usage:
#   bash scripts/09_vlm_rescue.sh --from_yaml ... --dry_run
#   bash scripts/09_vlm_rescue.sh --from_yaml ...
#
# Optional VLM fallback (only when YOLO finds nothing above threshold):
#   GEMINI_API_KEY=... bash scripts/09_vlm_rescue.sh \
#       --from_yaml ... --enable_vlm_fallback
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"

python src/check/build_vlm_pixel_grounded_rescue.py --config "${CONFIG}" "$@"
