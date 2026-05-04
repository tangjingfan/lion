#!/usr/bin/env bash
# Run a single filter pipeline stage.
#
# Stage 1 is standalone; stage 2 wraps the prerequisite rewrite + partition
# tools so the survivor set, audit log, and current.yaml symlink advance in
# lock-step.  Add new stages here as they come online.
#
# The filter directory is derived from the rollout YAML's
# output.{run_name,expname,base_dir} fields PLUS any selection YAML the
# user passes in via --from_yaml / --selection.  This way each experiment
# (expname) keeps its own filters/, rewrite/, partition/ outputs.
#
# Usage:
#   bash scripts/filter.sh 1 [--from_yaml configs/selection/exp.yaml]
#   bash scripts/filter.sh 2 [--from_yaml configs/selection/exp.yaml]
#   bash scripts/filter.sh 3 [--from_yaml configs/selection/exp.yaml]
#
# Visibility annotation (classification only, no dropping) lives outside
# this filter pipeline:
#   bash scripts/annotate_visibility.sh [--from_yaml ...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STAGE="${1:?Usage: bash scripts/filter.sh <stage_num> [args...]}"
shift || true

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"

# ── Pull the user's selection YAML out of "$@" so we can use it for path
# resolution.  Any other args remain in "$@" and are forwarded to the
# python entry points untouched.  ─────────────────────────────────────────
USER_SEL=""
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from_yaml|--selection)
            USER_SEL="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

# ── Resolve which experiment's filter_dir we're operating on.  Apply the
# user's selection to a fresh cfg copy so cfg.output.expname / run_name
# flow into get_filter_dir() — same logic the python tools use. ───────────
FILTER_DIR="$(python - "${CONFIG}" "${USER_SEL}" <<'PY'
import sys, yaml
sys.path.insert(0, ".")
from src.check._filter_utils import apply_selection_yaml, get_filter_dir
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
sel = sys.argv[2] if len(sys.argv) > 2 else ""
if sel:
    apply_selection_yaml(cfg, sel)
print(get_filter_dir(cfg))
PY
)"
CURRENT="${FILTER_DIR}/current.yaml"

# Stage 1's INPUT is the user's selection (custom episode list).
# Stage 2+'s INPUT is the prior stage's current.yaml (filter survivors).
# Each "current.yaml" is itself self-describing — write_keep_yaml embeds
# expname / run_name into it — so downstream tools can reload the right
# experiment identity from --from_yaml current.yaml alone.
USER_SEL_ARGS=()
[[ -n "${USER_SEL}" ]] && USER_SEL_ARGS=(--from_yaml "${USER_SEL}")

case "${STAGE}" in
  1)
    python src/check/filter_cross_floor.py \
        --config "${CONFIG}" \
        "${USER_SEL_ARGS[@]+"${USER_SEL_ARGS[@]}"}" \
        "$@"
    ;;
  2)
    if [[ ! -e "${CURRENT}" ]]; then
      echo "ERROR: ${CURRENT} not found — run stage 1 first." >&2
      exit 1
    fi
    echo "── 2a. rewrite ────────────────────────────────────────"
    python src/check/rewrite_subinstructions.py \
        --config "${CONFIG}" --from_yaml "${CURRENT}" "$@"
    echo "── 2b. partition viz ─────────────────────────────────"
    python src/check/visualize_partition.py \
        --config "${CONFIG}" --from_yaml "${CURRENT}"
    echo "── 2c. consolidate ───────────────────────────────────"
    python src/check/filter_partition.py \
        --config "${CONFIG}" --from_yaml "${CURRENT}"
    ;;
  3)
    if [[ ! -e "${CURRENT}" ]]; then
      echo "ERROR: ${CURRENT} not found — run stage 2 first." >&2
      exit 1
    fi
    python src/check/filter_blacklist.py \
        --config "${CONFIG}" --from_yaml "${CURRENT}" "$@"
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    echo "Available: 1 (cross_floor), 2 (rewrite + partition), 3 (blacklist)" >&2
    exit 1
    ;;
esac
