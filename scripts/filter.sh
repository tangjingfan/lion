#!/usr/bin/env bash
# Run a single filter pipeline stage.
#
# Stage 0 records the original selected set; stage 1 is standalone; stage 2
# rewrites + drops non-referrable landmarks; stage 3 partitions the remaining
# sub-paths; stage 4 drops targets grounded only to coarse semantic labels.
#
# The filter directory is derived from the rollout YAML's
# output.{run_name,expname,base_dir} fields PLUS any selection YAML the
# user passes in via --from_yaml / --selection.  This way each experiment
# (expname) keeps its own filters/, rewrite/, partition/ outputs.
#
# Usage:
#   bash scripts/filter.sh 0 [--from_yaml configs/selection/exp.yaml]
#   bash scripts/filter.sh 1 [--from_yaml configs/selection/exp.yaml]
#   bash scripts/filter.sh 2 [--from_yaml configs/selection/exp.yaml]
#   bash scripts/filter.sh 3 [--from_yaml configs/selection/exp.yaml]
#   bash scripts/filter.sh 4 [--from_yaml results/.../filters/current.yaml]
#
# Target-instance selection and optional VLM rescue live in their own scripts.
# Run stages 0-3 first, then target selection, optional rescue, then stage 4:
#   bash scripts/list_scene_categories.sh    --from_yaml ... --objects_only
#   bash scripts/refine_landmark_mapping.sh  --from_yaml ...
#   bash scripts/list_target_instances.sh    --from_yaml ...
#   bash scripts/select_target_instances.sh  --from_yaml ...
#   bash scripts/build_vlm_pixel_grounded_rescue.sh --from_yaml ...
#   bash scripts/filter.sh 4                 --from_yaml ...
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STAGE="${1:?Usage: bash scripts/filter.sh <stage_num> [args...]}"
shift || true

CONFIG="${CONFIG:-configs/rollout/rollout_landmark_rxr.yaml}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

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

# Stage 0's INPUT is the user's selection (custom episode list).
# Stage 1's INPUT is also the user's selection; it does not require stage 0.
# Stage 2+'s INPUT is the prior stage's current.yaml (filter survivors).
# Each "current.yaml" is itself self-describing — write_keep_yaml embeds
# expname / run_name into it — so downstream tools can reload the right
# experiment identity from --from_yaml current.yaml alone.
USER_SEL_ARGS=()
[[ -n "${USER_SEL}" ]] && USER_SEL_ARGS=(--from_yaml "${USER_SEL}")

case "${STAGE}" in
  0)
    python src/check/filter_original.py \
        --config "${CONFIG}" \
        "${USER_SEL_ARGS[@]+"${USER_SEL_ARGS[@]}"}" \
        "$@"
    ;;
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
    STAGE2_INPUT="${FILTER_DIR}/01_cross_floor.yaml"
    if [[ ! -e "${STAGE2_INPUT}" ]]; then
      STAGE2_INPUT="${CURRENT}"
    fi
    echo "stage 2 input: ${STAGE2_INPUT}"
    echo "── 2a. rewrite ────────────────────────────────────────"
    python src/check/rewrite_subinstructions.py \
        --config "${CONFIG}" --from_yaml "${STAGE2_INPUT}" "$@"
    echo "── 2b. blacklist / referrability ─────────────────────"
    python src/check/filter_blacklist.py \
        --config "${CONFIG}" --from_yaml "${STAGE2_INPUT}"
    ;;
  3)
    if [[ ! -e "${CURRENT}" ]]; then
      echo "ERROR: ${CURRENT} not found — run stage 2 first." >&2
      exit 1
    fi
    echo "stage 3 input: ${CURRENT}"
    echo "── 3a. partition viz (full regenerate) ───────────────"
    python src/check/visualize_partition.py \
        --config "${CONFIG}" --from_yaml "${CURRENT}"
    echo "── 3b. consolidate ───────────────────────────────────"
    python src/check/filter_partition.py \
        --config "${CONFIG}" --from_yaml "${CURRENT}" "$@"
    ;;
  4)
    if [[ ! -e "${CURRENT}" ]]; then
      echo "ERROR: ${CURRENT} not found — run stage 3 first." >&2
      exit 1
    fi
    echo "stage 4 input: ${CURRENT}"
    echo "── 4. semantic granularity ───────────────────────────"
    python src/check/filter_semantic_granularity.py \
        --config "${CONFIG}" --from_yaml "${CURRENT}" "$@"
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    echo "Available: 0 (original), 1 (cross_floor), 2 (rewrite + blacklist), 3 (partition), 4 (semantic_granularity)" >&2
    exit 1
    ;;
esac
