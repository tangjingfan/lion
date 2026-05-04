"""
LION-Bench — Stage 3 filter: drop non-referrable / unmapped landmarks.

Reads stage 2's sub-path-level survivor set (via current.yaml) and the
rewriter JSON, then drops sub-paths whose landmark cannot be grounded as a
concrete MP3D object.  Four rules:

  1. ``landmark_category == "spatial"``         → drop  (rewriter's own
                                                          judgement that
                                                          there is no
                                                          referrable object)
  2. generic room phrases                       → drop  (e.g. "room with a
                                                         light on"; not a
                                                         named room type)
  3. ``components`` empty / all "unknown"        → drop  (no MP3D mapping)
  4. ``landmark`` text matches blacklist word    → drop  (e.g. "hallway",
                                                          "corridor",
                                                          "stairs")

The output graduates the sub-path-level survivor set to those with a
trustworthy, MP3D-grounded landmark — ready for the visibility / uniqueness
checks downstream.

Usage
-----
  python src/check/filter_blacklist.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --from_yaml results/{exp}/filters/current.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import (
    ensure_episode,
    ensure_sub_path,
    get_filter_dir,
    get_split,
    load_audit,
    load_keep,
    register_stage,
    resolve_selection,
    save_audit,
    update_current,
    write_drop_yaml,
    write_keep_yaml,
)


STAGE_NUM  = 3
STAGE_NAME = "blacklist"

# Words that, when they appear as a whole word in the landmark phrase,
# signal a non-referrable space rather than a concrete object.  Specific
# room types ("bedroom", "kitchen", "bathroom") are NOT here — those are
# legitimate landmarks.  Generic spatial terms ("area", "space") and
# transition zones ("hallway", "stairs") are.
DEFAULT_BLACKLIST = (
    "hallway", "corridor", "passage", "passageway",
    "area", "space",
    "stairs", "staircase", "step", "steps",
)

SPECIFIC_ROOM_TERMS = (
    "bathroom", "bedroom", "closet", "dining room", "family room",
    "foyer", "garage", "hall", "kitchen", "laundry room", "library",
    "living room", "lounge", "meeting room", "office", "rec room",
    "tv room", "utility room",
)


def _is_generic_room_landmark(landmark: str) -> bool:
    """Return true for vague room phrases, but keep named room types."""
    if not re.search(r"\broom\b", landmark):
        return False
    return not any(
        re.search(rf"\b{re.escape(room)}\b", landmark)
        for room in SPECIFIC_ROOM_TERMS
    )


def _classify(rewrite_sub: Dict, blacklist: Tuple[str, ...]) -> Tuple[bool, str]:
    """Return ``(keep, reason)`` for one sub-path's rewrite entry."""
    category   = (rewrite_sub.get("landmark_category") or "object").lower()
    landmark   = (rewrite_sub.get("landmark") or "").strip().lower()
    components = rewrite_sub.get("components") or []

    if category == "spatial":
        return False, "category:spatial"

    if _is_generic_room_landmark(landmark):
        return False, "generic_room"

    mapped = [
        (c.get("semantic_label") or "unknown").strip().lower()
        for c in components
    ]
    if not mapped or all(s in ("", "unknown") for s in mapped):
        return False, "unmapped"

    for word in blacklist:
        if re.search(rf"\b{re.escape(word)}\b", landmark):
            return False, f"blacklist:{word}"

    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 3: drop non-referrable / unmapped landmarks",
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--from_yaml", default=None,
                    help="Selection / current.yaml carrying expname so this "
                         "stage knows which experiment's filter dir to read.")
    ap.add_argument("--blacklist", nargs="+", default=None,
                    help="Override the default landmark-text blacklist.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    resolve_selection(cfg, args.from_yaml)

    filt_dir = get_filter_dir(cfg)
    if not filt_dir.exists():
        raise SystemExit(f"No filters/ at {filt_dir} — run prior stages first.")
    out_dir = filt_dir.parent
    split   = get_split(cfg)

    current = filt_dir / "current.yaml"
    if not current.exists():
        raise SystemExit(f"No current.yaml at {current} — run stage 2 first.")

    prior_keep = load_keep(current.resolve())
    prior_subs = prior_keep.get("sub_paths")
    if not prior_subs:
        raise SystemExit(
            "Prior stage's current.yaml has no `sub_paths` field — stage 3 "
            "expects sub-path-level survivors (run stage 2 first).",
        )

    # Locate rewriter output.
    rewrite_path = None
    for fname in ("sub_instructions_rewritten.json",
                  "sub_instructions_rewritten_filtered.json"):
        p = out_dir / "rewrite" / fname
        if p.exists():
            rewrite_path = p
            break
    if rewrite_path is None:
        raise SystemExit(f"No rewrite JSON under {out_dir}/rewrite/")
    with open(rewrite_path) as f:
        rewrite_episodes = json.load(f).get("episodes", {})
    print(f"Loaded rewrite: {rewrite_path}")

    blacklist = tuple(args.blacklist) if args.blacklist else DEFAULT_BLACKLIST
    print(f"Blacklist words: {blacklist}")

    audit = load_audit(filt_dir, split)
    register_stage(audit, STAGE_NAME, blacklist=list(blacklist))

    keep_sub_paths: Dict[int, List[int]] = {}
    dropped:        Dict[str, Dict]      = {}
    n_subs_total = 0
    n_subs_keep  = 0
    reason_counts: Dict[str, int] = {}

    for ep_id_raw, sub_idxs in prior_subs.items():
        # YAML loads bare int keys as int; rewrite JSON keys are strings.
        ep_id      = int(ep_id_raw)
        ep_id_str  = str(ep_id)
        rewrite_ep = rewrite_episodes.get(ep_id_str)
        ep_audit   = audit["episodes"].setdefault(ep_id_str, {"stages": {}})
        rewrite_subs = (
            {int(s["sub_idx"]): s for s in rewrite_ep.get("sub_paths", [])}
            if rewrite_ep else {}
        )

        ep_keep_subs: List[int]      = []
        ep_drops:     Dict[int, Dict] = {}

        for sub_idx in sub_idxs:
            n_subs_total += 1
            rw = rewrite_subs.get(int(sub_idx))
            if rw is None:
                payload = {"reason": "missing_in_rewrite"}
                keep_it = False
            else:
                keep_it, reason = _classify(rw, blacklist)
                payload = {
                    "reason":   reason,
                    "landmark": rw.get("landmark"),
                    "category": rw.get("landmark_category"),
                }

            sp_audit = ensure_sub_path(ep_audit, sub_idx)
            sp_audit["stages"][STAGE_NAME] = {
                "status": "ok" if keep_it else "drop",
                **payload,
            }
            if keep_it:
                ep_keep_subs.append(sub_idx)
                n_subs_keep += 1
            else:
                ep_drops[sub_idx] = payload
                reason_counts[payload["reason"]] = \
                    reason_counts.get(payload["reason"], 0) + 1

        ep_audit["stages"][STAGE_NAME] = {
            "status":    "ok" if ep_keep_subs else "drop",
            "kept_sub":  len(ep_keep_subs),
            "total_sub": len(sub_idxs),
        }
        if ep_keep_subs:
            keep_sub_paths[ep_id] = ep_keep_subs
        if ep_drops:
            dropped[ep_id_str] = {
                "subs": {str(k): v for k, v in sorted(ep_drops.items())},
            }

    keep_path = write_keep_yaml(
        filt_dir, STAGE_NUM, STAGE_NAME, split,
        instruction_ids=sorted(keep_sub_paths.keys()),
        sub_paths=keep_sub_paths,
        cfg=cfg,
    )
    drop_path = write_drop_yaml(
        filt_dir, STAGE_NUM, STAGE_NAME, split,
        dropped=dict(sorted(dropped.items(), key=lambda kv: int(kv[0]))),
        extras={"blacklist": list(blacklist)},
    )
    save_audit(audit, filt_dir)
    current_path = update_current(filt_dir, keep_path)

    n_eps_in   = len(prior_subs)
    n_eps_keep = len(keep_sub_paths)
    pct_subs   = (n_subs_keep / n_subs_total) if n_subs_total else 0.0

    print(f"=== Stage {STAGE_NUM} — {STAGE_NAME} ===")
    print(f"  episodes in   : {n_eps_in}")
    print(f"  episodes keep : {n_eps_keep}")
    print(f"  sub-paths in  : {n_subs_total}")
    print(f"  sub-paths keep: {n_subs_keep}  ({pct_subs:.1%})")
    if reason_counts:
        print(f"  drop reasons:")
        for reason, n in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {reason:<25s} {n}")

    print()
    print("Outputs:")
    print(f"  {keep_path}")
    print(f"  {drop_path}")
    print(f"  {current_path}  ->  {keep_path.name}")
    print(f"  {filt_dir / 'audit.json'}")


if __name__ == "__main__":
    main()
