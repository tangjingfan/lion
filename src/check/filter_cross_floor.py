"""
LION-Bench — Stage 1 filter: drop cross-floor episodes.

For each episode, computes the Y-range (vertical span) of all path nodes in
the Habitat frame.  Episodes whose path spans more than ``--threshold_m``
metres vertically are considered cross-floor and removed from the survivor
set.  A typical single-floor MP3D path stays under ~1 m of vertical drift;
a path that climbs or descends a full storey jumps by 2.5+ m.

Outputs:
  {run_dir}/survivor.yaml                — canonical post-stage survivor
                                            set (overwritten; downstream
                                            tools auto-merge via resolve_exp)
  {run_dir}/filters/01_cross_floor_dropped.yaml
                                          — rejected ids + Δy (debug only)
  {run_dir}/filters/audit.json           — per-episode trace; subsequent
                                            stages append to the same file

Usage
-----
  python src/check/filter_cross_floor.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --threshold_m 1.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import (
    append_ep_event,
    ensure_episode,
    finalize_audit,
    get_filter_dir,
    get_split,
    load_audit,
    register_stage,
    resolve_exp,
    save_audit,
    strip_stage_events,
    write_drop_yaml,
    write_survivor,
)
from src.dataset.landmark_rxr import episodes_from_config
from src.env.connectivity import load_connectivity


STAGE_NUM  = 1
STAGE_NAME = "cross_floor"


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1: drop cross-floor episodes")
    ap.add_argument("--config", required=True,
                    help="Rollout / dataset YAML config")
    ap.add_argument("--exp", default=None,
                    help="Selection YAML to restrict the input set. Stage 1 "
                         "reads the seed YAML directly and does NOT auto-merge "
                         "survivor.yaml (so reruns process the full set "
                         "rather than the prior survivor subset).")
    ap.add_argument("--threshold_m", type=float, default=0.5,
                    help="Y-range threshold (m) above which an episode is "
                         "considered cross-floor (default 0.5)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    resolve_exp(cfg, args.exp, apply_current=False)

    episodes = episodes_from_config(cfg)
    if not episodes:
        print("No episodes matched. Exiting.")
        return

    needed_scans = sorted({ep.scan for ep in episodes})
    db = load_connectivity(
        scenes_dir=cfg["scenes"]["scenes_dir"],
        scans=needed_scans,
        json_dir=cfg["dataset"].get("connectivity_json_dir"),
        pkl_path=cfg["dataset"].get("connectivity_pkl"),
    )

    filt_dir = get_filter_dir(cfg)
    filt_dir.mkdir(parents=True, exist_ok=True)
    split    = get_split(cfg)

    audit = load_audit(filt_dir, split)
    register_stage(audit, STAGE_NAME, y_range_m=args.threshold_m)
    strip_stage_events(audit, STAGE_NAME)

    keep:    List[int]       = []
    dropped: Dict[int, Dict] = {}
    n_subs_total = 0
    n_subs_keep = 0
    n_subs_drop = 0

    for ep in episodes:
        n_ep_subs = len(ep.sub_paths)
        n_subs_total += n_ep_subs
        scan_db = db.get(ep.scan, {})
        try:
            ys = [float(scan_db[n][1]) for n in ep.path]
        except KeyError as exc:
            ys      = []
            missing = str(exc).strip("'\"")
        else:
            missing = None

        if ys:
            y_range = max(ys) - min(ys)
            cross   = y_range > args.threshold_m
        else:
            y_range = 0.0
            cross   = False  # passthrough on missing data

        ep_audit = ensure_episode(audit, ep)
        if cross:
            append_ep_event(
                ep_audit, stage=STAGE_NAME, action="dropped",
                reason="cross_floor",
                y_range_m=round(y_range, 3),
                **({"missing_node": missing} if missing else {}),
            )
            n_subs_drop += n_ep_subs
            dropped[ep.instruction_id] = {
                "scan":      ep.scan,
                "y_range_m": round(y_range, 3),
                "n_sub_paths": n_ep_subs,
            }
        else:
            append_ep_event(
                ep_audit, stage=STAGE_NAME, action="kept",
                y_range_m=round(y_range, 3),
                **({"missing_node": missing} if missing else {}),
            )
            n_subs_keep += n_ep_subs
            keep.append(ep.instruction_id)

    survivor_path = write_survivor(cfg, split, keep)
    drop_path = write_drop_yaml(
        filt_dir, STAGE_NUM, STAGE_NAME, split,
        dropped={str(k): v for k, v in sorted(dropped.items())},
        extras={"threshold_m": args.threshold_m},
    )
    finalize_audit(audit)
    save_audit(audit, filt_dir)

    n_total  = len(episodes)
    n_keep   = len(keep)
    n_drop   = len(dropped)
    pct_drop = (n_drop / n_total) if n_total else 0.0
    print(f"=== Stage {STAGE_NUM} — {STAGE_NAME} (threshold {args.threshold_m:.2f} m) ===")
    print(f"  total   : {n_total}")
    print(f"  keep    : {n_keep}")
    print(f"  dropped : {n_drop}  ({pct_drop:.1%})")
    print(f"  sub-paths in  : {n_subs_total}")
    print(f"  sub-paths keep: {n_subs_keep}")
    print(f"  sub-paths drop: {n_subs_drop}")
    print()
    print("Outputs:")
    print(f"  {survivor_path}")
    print(f"  {drop_path}")
    print(f"  {filt_dir / 'audit.json'}")


if __name__ == "__main__":
    main()
