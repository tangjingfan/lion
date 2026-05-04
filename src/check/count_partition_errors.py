"""
Count how often the rewritten ``spatial_instruction`` text *actually
contradicts* the underlying trajectory geometry, across all per-episode
``partition.json`` files.

Classification logic
--------------------
Every sub-path carries two labels:
  • ``instruction_kind``  — parsed from the rewritten text
                            ("Turn left" → left, "Go forward" → forward, …).
  • ``kind`` (geometric)  — from the first edge whose |Δ| > TURN_THRESH:
                              sign > 0 → right, sign < 0 → left,
                              |Δ| > AROUND_THRESH → around;
                              no such edge → forward.

A sub-path is a REAL ERROR when the two labels disagree in a way that
is not merely a magnitude discrepancy:

  direction_conflict
    instr=left  AND geometric direction is right         (or vice versa)
    instr=left  AND geom=around with turn_delta_deg > 0  (around is rightward)
    instr=right AND geom=around with turn_delta_deg < 0  (around is leftward)

  phantom_turn
    instr=forward AND geom ∈ {left, right, around}
    ("Go forward" but the path actually turns ≥ TURN_THRESH)

  missing_turn
    instr ∈ {left, right, around} AND geom=forward
    (instruction says turn, but no edge exceeds TURN_THRESH)

Non-errors (reported separately, not counted as errors):

  magnitude_mismatch
    Same side/direction, different magnitude bin:
      instr=left   AND geom=around with delta < 0
      instr=right  AND geom=around with delta > 0
      instr=around AND geom ∈ {left, right}
    Landmark-RxR's "turn left" may correspond to a 45° or a 150° left
    turn on the graph; the direction matches, only the magnitude label
    changes.  This is *not* a dataset error.

  ok
    ``instruction_kind == kind`` exactly.

Usage
-----
  python src/check/count_partition_errors.py \\
      --partition_dir results/val_unseen/partition
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Classifier ─────────────────────────────────────────────────────────────
def classify(
    instr_kind: str,
    geom_kind:  str,
    turn_delta_deg: Optional[float],
) -> Tuple[str, str]:
    """Return (bucket, detail) for one sub-path."""
    if instr_kind == geom_kind:
        return "ok", f"{instr_kind} == {geom_kind}"

    if instr_kind == "forward":
        return "phantom_turn", f"instr=forward but geom={geom_kind}"

    if geom_kind == "forward":
        return "missing_turn", f"instr={instr_kind} but geom=forward"

    # Both are turns of some kind; direction depends on sign of turn_delta_deg.
    if turn_delta_deg is None:
        # Should not happen when geom_kind is a turn, but guard anyway.
        return "ok", "turn detected but delta unavailable"

    geom_dir = "right" if turn_delta_deg > 0 else "left"

    if instr_kind == "around":
        # "Turn around" is direction-agnostic; a smaller-magnitude turn still
        # fulfils the spirit of the instruction.
        return "magnitude_mismatch", \
               f"instr=around, geom={geom_kind} (Δ={turn_delta_deg:+.0f}°)"

    # instr_kind is "left" or "right" — compare to geometric direction.
    if instr_kind == geom_dir:
        return "magnitude_mismatch", \
               f"instr={instr_kind}, geom={geom_kind} (same side, Δ={turn_delta_deg:+.0f}°)"

    return "direction_conflict", \
           f"instr={instr_kind}, actual direction={geom_dir} (Δ={turn_delta_deg:+.0f}°)"


# ── Loader ─────────────────────────────────────────────────────────────────
def _iter_subpaths(partition_dir: Path):
    """Yield (instruction_id, sub_idx, sub_path_dict) for every sub-path."""
    for ep_dir in sorted(partition_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        pj = ep_dir / "partition.json"
        if not pj.exists():
            continue
        with open(pj) as f:
            ep = json.load(f)
        for sp in ep["partitions"]:
            if "error" in sp:
                continue
            yield ep["instruction_id"], sp["sub_idx"], sp


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Count instruction-vs-geometry errors in partition.json files",
    )
    ap.add_argument(
        "--partition_dir",
        default="results/val_unseen/partition",
        help="Directory containing {instruction_id}/partition.json files",
    )
    ap.add_argument(
        "--examples_per_bucket", type=int, default=5,
        help="How many example sub-paths to list per bucket (default 5)",
    )
    args = ap.parse_args()

    part_dir = Path(args.partition_dir).expanduser()
    if not part_dir.exists():
        raise SystemExit(f"partition_dir not found: {part_dir}")

    counts   = Counter()
    examples: Dict[str, List[str]] = defaultdict(list)

    for inst_id, sub_idx, sp in _iter_subpaths(part_dir):
        instr = sp["instruction_kind"]
        geom  = sp["kind"]
        delta = sp.get("turn_delta_deg")
        bucket, detail = classify(instr, geom, delta)
        counts[bucket] += 1
        if len(examples[bucket]) < args.examples_per_bucket:
            examples[bucket].append(
                f"ep={inst_id:<6} sub={sub_idx}  {detail}  "
                f"·  {sp.get('spatial_instruction','')!r}"
            )

    total     = sum(counts.values())
    errors    = (counts["direction_conflict"]
                 + counts["phantom_turn"]
                 + counts["missing_turn"])
    non_errs  = counts["ok"] + counts["magnitude_mismatch"]

    print("=== Instruction-vs-Geometry Error Summary ===")
    print(f"Source: {part_dir}")
    print(f"Total sub-paths: {total}\n")

    def _row(name, desc, n):
        pct = f"{n/total:.1%}" if total else "n/a"
        print(f"  {name:<20s} {n:>4d}  ({pct:>6s})  — {desc}")

    print("NON-ERRORS")
    _row("ok",                 "instruction_kind == geometric kind exactly",
         counts["ok"])
    _row("magnitude_mismatch", "same direction, different magnitude bin",
         counts["magnitude_mismatch"])
    print()
    print("REAL ERRORS (dataset instruction contradicts actual path)")
    _row("direction_conflict", "left ↔ right (or around with wrong sign)",
         counts["direction_conflict"])
    _row("phantom_turn",       "instr=forward but path turns ≥ 45°",
         counts["phantom_turn"])
    _row("missing_turn",       "instr=turn but no edge exceeds 45°",
         counts["missing_turn"])

    print("\nSUMMARY")
    _row("non-errors",  "instruction consistent with geometry", non_errs)
    _row("real errors", "instruction contradicts geometry",    errors)

    for bucket in ("direction_conflict", "phantom_turn",
                   "missing_turn", "magnitude_mismatch"):
        if not examples[bucket]:
            continue
        print(f"\n─── {bucket} examples ───")
        for ex in examples[bucket]:
            print(f"  {ex}")


if __name__ == "__main__":
    main()
