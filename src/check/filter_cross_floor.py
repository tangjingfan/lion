"""
LION-Bench — Stage 1 filter: drop cross-floor sub-paths.

For each sub-path of every episode, computes the Y-range (vertical
span) of its nodes in the Habitat frame. A sub-path is "cross-floor"
when ``max(y) - min(y) > --threshold_m`` and gets removed from
``survivor.sub_paths`` outright — same as the previous episode-level
filter, just at finer granularity. The episode stays as long as at
least one sub-path remains.

Why per-sub-path: a single Landmark-RxR episode can include both a
floor traversal and intra-floor navigation segments. Dropping the
whole episode because one sub-path crosses floors threw away all the
single-floor sub-paths that came with it. Now only the cross-floor
sub-paths are dropped; the rest stay in the pipeline.

Cross-floor is still the only **hard** removal in the pipeline. Later
stages (blacklist / partition / visibility / detection / synthesis)
all use the ``survivor.sub_status`` label channel — failures attach a
label so rescue paths and inspection tooling can still reach the sub.
Cross-floor sub-paths are bad geometry (an agent that teleports
vertically), so there's nothing to rescue and we cut them outright.

Outputs:
  {run_dir}/survivor.yaml                — sub-path-level survivor set
                                            (every ep with ≥1 surviving
                                            sub-path stays; sub_paths
                                            map lists surviving indices)
  {run_dir}/filters/01_cross_floor_dropped.yaml
                                          — per-(ep, sub_idx) rejection
                                            reasons + Δy (debug only)
  {run_dir}/filters/audit.json           — per-episode + per-sub events;
                                            subsequent stages append to
                                            the same file

Usage
-----
  python src/check/filter_cross_floor.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --threshold_m 0.5
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
    append_sub_event,
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
    ap = argparse.ArgumentParser(
        description="Stage 1: drop cross-floor sub-paths",
    )
    ap.add_argument("--config", required=True,
                    help="Rollout / dataset YAML config")
    ap.add_argument("--exp", default=None,
                    help="Selection YAML to restrict the input set. Stage 1 "
                         "reads the seed YAML directly and does NOT auto-merge "
                         "survivor.yaml (so reruns process the full set "
                         "rather than the prior survivor subset).")
    ap.add_argument("--threshold_m", type=float, default=0.5,
                    help="Per-sub-path Y-range threshold (m) above which a "
                         "sub-path is considered cross-floor (default 0.5)")
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

    keep_ep_ids:      List[int]                = []
    keep_sub_paths:   Dict[int, List[int]]     = {}
    dropped:          Dict[str, Dict]          = {}
    n_subs_total = 0
    n_subs_keep  = 0
    n_subs_drop  = 0
    n_eps_full_drop = 0

    for ep in episodes:
        n_ep_subs = len(ep.sub_paths)
        n_subs_total += n_ep_subs
        scan_db = db.get(ep.scan, {})

        ep_audit = ensure_episode(audit, ep)
        ep_kept_subs: List[int]   = []
        ep_drops:     Dict[int, Dict] = {}

        for sub_idx, sub_nodes in enumerate(ep.sub_paths):
            try:
                ys = [float(scan_db[n][1]) for n in sub_nodes]
            except KeyError as exc:
                ys      = []
                missing = str(exc).strip("'\"")
            else:
                missing = None

            if ys:
                y_range = max(ys) - min(ys)
                cross   = y_range > args.threshold_m
            else:
                # missing node — passthrough (consistent with the
                # previous episode-level behaviour).
                y_range = 0.0
                cross   = False

            if cross:
                append_sub_event(
                    ep_audit, sub_idx, stage=STAGE_NAME, action="dropped",
                    reason="cross_floor",
                    y_range_m=round(y_range, 3),
                    **({"missing_node": missing} if missing else {}),
                )
                ep_drops[sub_idx] = {
                    "reason": "cross_floor",
                    "y_range_m": round(y_range, 3),
                    **({"missing_node": missing} if missing else {}),
                }
                n_subs_drop += 1
            else:
                append_sub_event(
                    ep_audit, sub_idx, stage=STAGE_NAME, action="kept",
                    y_range_m=round(y_range, 3),
                    **({"missing_node": missing} if missing else {}),
                )
                ep_kept_subs.append(sub_idx)
                n_subs_keep += 1

        if ep_kept_subs:
            # Episode survives if any sub-path stays single-floor.
            append_ep_event(
                ep_audit, stage=STAGE_NAME, action="kept",
                kept_sub=len(ep_kept_subs),
                total_sub=n_ep_subs,
            )
            keep_ep_ids.append(ep.instruction_id)
            keep_sub_paths[ep.instruction_id] = ep_kept_subs
        else:
            # Every sub-path crosses floors — full episode drop, the
            # only remaining episode-level hard drop in the pipeline.
            append_ep_event(
                ep_audit, stage=STAGE_NAME, action="dropped",
                reason="cross_floor",
                kept_sub=0,
                total_sub=n_ep_subs,
            )
            n_eps_full_drop += 1

        if ep_drops:
            dropped[str(ep.instruction_id)] = {
                "scan": ep.scan,
                "n_sub_paths_total":   n_ep_subs,
                "n_sub_paths_dropped": len(ep_drops),
                "subs": {str(k): v for k, v in sorted(ep_drops.items())},
            }

    survivor_path = write_survivor(
        cfg, split,
        instruction_ids=sorted(keep_ep_ids),
        sub_paths=keep_sub_paths,
    )
    drop_path = write_drop_yaml(
        filt_dir, STAGE_NUM, STAGE_NAME, split,
        dropped=dict(sorted(dropped.items(), key=lambda kv: int(kv[0]))),
        extras={"threshold_m": args.threshold_m},
    )
    finalize_audit(audit)
    save_audit(audit, filt_dir)

    n_total = len(episodes)
    n_keep  = len(keep_ep_ids)
    pct_drop_subs = (n_subs_drop / n_subs_total) if n_subs_total else 0.0
    print(f"=== Stage {STAGE_NUM} — {STAGE_NAME} (threshold {args.threshold_m:.2f} m) ===")
    print(f"  episodes in       : {n_total}")
    print(f"  episodes keep     : {n_keep}  (≥1 single-floor sub survives)")
    print(f"  episodes full drop: {n_eps_full_drop}  (every sub crossed floors)")
    print(f"  sub-paths in      : {n_subs_total}")
    print(f"  sub-paths keep    : {n_subs_keep}")
    print(f"  sub-paths drop    : {n_subs_drop}  ({pct_drop_subs:.1%})")
    print()
    print("Outputs:")
    print(f"  {survivor_path}")
    print(f"  {drop_path}")
    print(f"  {filt_dir / 'audit.json'}")


if __name__ == "__main__":
    main()
