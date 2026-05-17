#!/usr/bin/env bash
# Run the entire curation pipeline (00 → 13) for one experiment in
# strict numeric order, no re-runs needed.
#
# Prerequisites:
#   - rollout has been done for this experiment (scripts/rollout.sh).
#   - The lion conda env (with habitat-lab + hydra) is active.
#   - GEMINI_API_KEY is exported (needed by 02 rewrite + 06 refine).
#
# Usage:
#   bash scripts/run_all.sh --exp configs/selection/<split>/<exp>.yaml
#   bash scripts/run_all.sh --exp <bare-expname>
#
# Optional:
#   --from NN   start at step NN (skip earlier ones)
#   --to NN     stop after step NN (skip later ones)
#   --dry       just print what would run; don't execute
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Pipeline order — each entry is the numeric prefix of the script.
STEPS=(
  00_record_original
  01_filter_multi_floor
  02_rewrite_subinstruction
  03_blacklist_landmark
  04_partition
  05_get_object_list
  06_refine_landmark_mapping
  07_list_potential_instances
  08_get_potential_instance
  09_vlm_rescue
  10_apply_rescue
  11_rescue_blacklist
  12_consolidate
  13_attrition
)

# A few steps need extra fixed flags that aren't relevant to the others
# (so we can't blindly forward "$@" to every step).
declare -A EXTRA_FLAGS=(
  [05_get_object_list]="--objects_only"
)

# ── Parse args ───────────────────────────────────────────────────────
EXP=""
FROM=""
TO=""
DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp)  EXP="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --to)   TO="$2";   shift 2 ;;
    --dry)  DRY=1;     shift ;;
    -h|--help)
      sed -n '2,30p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1 ;;
  esac
done

if [[ -z "$EXP" ]]; then
  echo "ERROR: --exp is required (selection YAML path or bare expname)" >&2
  exit 1
fi

# ── Filter the step list by --from / --to ────────────────────────────
PLAN=()
for step in "${STEPS[@]}"; do
  prefix="${step%%_*}"
  [[ -n "$FROM" && "$prefix" < "$FROM" ]] && continue
  [[ -n "$TO"   && "$prefix" > "$TO"   ]] && continue
  PLAN+=("$step")
done

if [[ ${#PLAN[@]} -eq 0 ]]; then
  echo "ERROR: no steps match --from $FROM --to $TO" >&2
  exit 1
fi

echo "Plan: ${PLAN[*]}"
echo "exp : $EXP"
echo ""

# ── Run ──────────────────────────────────────────────────────────────
for step in "${PLAN[@]}"; do
  extras="${EXTRA_FLAGS[$step]:-}"
  cmd="bash scripts/${step}.sh --exp $EXP ${extras}"

  echo "════════════════════════════════════════════════════════════════"
  echo "  $step"
  echo "    $cmd"
  echo "════════════════════════════════════════════════════════════════"

  if [[ "$DRY" -eq 1 ]]; then
    continue
  fi
  # shellcheck disable=SC2086
  bash "scripts/${step}.sh" --exp "$EXP" ${extras}
done

echo ""
echo "✓ pipeline complete"
