#!/usr/bin/env bash
# Stage 9 — open-vocab detection: run YOLO-World on the rollout panorama
# at each (ep, sub) end-pose to localise the landmark, then query the
# Habitat semantic buffer at the detected pixel to recover an MP3D
# instance id. Used to rescue sub-paths where the strict visibility
# check (step 07) returned `not_visible`.
#
# Default: viz on (writes per-detection PNGs to detection/<scan>/<ep>/
# alongside the data). Pass --no-viz-style flags if you want to skip.
#
# Usage:
#   bash scripts/09_detection.sh --exp configs/selection/<split>/<exp>.yaml
#   bash scripts/09_detection.sh --exp ... --dry_run
#
# Optional VLM fallback (only when YOLO finds nothing above threshold):
#   GEMINI_API_KEY=... bash scripts/09_detection.sh \
#       --exp ... --enable_vlm_fallback
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"

python src/check/build_vlm_pixel_grounded_rescue.py --config "${CONFIG}" --save_viz "$@"
