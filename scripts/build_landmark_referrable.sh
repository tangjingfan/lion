#!/usr/bin/env bash
# Regenerate configs/landmark_referrable.yaml via one LLM call.
#
# One-off: run when MPCat40 changes or when you want a fresh judgment.
# Output is committed to the repo so step 11 has a deterministic table
# to read; review the diff before pushing.
#
# Usage:
#   GEMINI_API_KEY=... bash scripts/build_landmark_referrable.sh
#   bash scripts/build_landmark_referrable.sh --model gemini-2.0-flash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python src/check/build_landmark_referrable.py "$@"
