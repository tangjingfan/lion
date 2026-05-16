"""
LION-Bench — Stage 3 filter: consolidate rewrite + partition outputs.

Reads the prior stage's survivor set from ``current.yaml`` (typically
``02_blacklist.yaml``), then for every surviving (instruction_id, sub_idx)
checks:

  • rewriter  — was the LLM rewrite valid?  (drops "error" sub-paths)
  • partition — did geometric partition succeed?  (drops sub-paths with
                <2 nodes / other geometry errors)

This stage preserves the sub-path-level survivor set: ``03_partition.yaml``
carries an ``instruction_ids`` list (episodes with ≥1 surviving sub-path) AND
a ``sub_paths`` dict listing exactly which sub_idx survive within each.

Prerequisites
-------------
Run these two existing tools first against the prior ``current.yaml`` so
their outputs are on disk for this stage to consume:

  python src/check/rewrite_subinstructions.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --exp configs/selection/exp.yaml

  python src/check/visualize_partition.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --exp configs/selection/exp.yaml

Then this stage:

  python src/check/filter_partition.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import (
    ensure_episode,
    ensure_sub_path,
    get_filter_dir,
    get_split,
    get_survivor_path,
    load_audit,
    load_rewrite_episodes,
    register_stage,
    resolve_exp,
    save_audit,
    write_drop_yaml,
    write_survivor,
)
from src.dataset.landmark_rxr import episodes_from_config


STAGE_NUM  = 3
STAGE_NAME = "partition"


def _load_rewrite(out_dir: Path) -> Tuple[Dict[str, Dict], Optional[Path]]:
    """Aggregate per-episode rewrite JSONs into one ``{ep_id: rewrite}`` dict.

    Walks ``rewrite/{scan}/{instruction_id}/sub_instructions_rewritten[_filtered].json``
    and unions them.  Picks ``_filtered`` if any episode has it, else the
    unfiltered variant.  Returns the merged dict and one representative
    path for logging.
    """
    rewrite_dir = out_dir / "rewrite"
    eps, _suffix, paths = load_rewrite_episodes(rewrite_dir)
    return eps, (paths[0] if paths else None)


def _load_partition(out_dir: Path, scan: str, ep_id: int) -> Optional[Dict]:
    p = out_dir / "partition" / scan / str(ep_id) / "partition.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _classify_sub_path(
    rewrite_sub:   Optional[Dict],
    partition_sub: Optional[Dict],
) -> Tuple[bool, Dict]:
    """Return ``(keep, audit_payload)`` for one sub-path."""
    audit: Dict = {}

    # Rewriter
    if rewrite_sub is None:
        audit["rewrite"] = "missing"
        return False, audit
    if "error" in rewrite_sub:
        audit["rewrite"] = f"error:{str(rewrite_sub['error'])[:60]}"
        return False, audit
    audit["rewrite"]            = "ok"
    audit["landmark_category"]  = rewrite_sub.get("landmark_category", "unknown")
    audit["landmark"]           = rewrite_sub.get("landmark", "")

    # Partition
    if partition_sub is None:
        audit["partition"] = "missing"
        return False, audit
    if "error" in partition_sub:
        audit["partition"] = f"error:{str(partition_sub['error'])[:60]}"
        return False, audit
    audit["partition"]          = "ok"
    audit["kind"]               = partition_sub.get("kind")
    audit["direction_mismatch"] = partition_sub.get("direction_mismatch", False)

    return True, audit


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 3: fold rewriter + partition results into the filter pipeline",
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", default=None,
                    help="Experiment handle (selection YAML path or expname). "
                         "survivor.yaml is auto-merged to supply the "
                         "prior survivor sub_paths.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # resolve_exp(apply_current=True) applies the user's yaml THEN merges
    # survivor.yaml on top, so cfg.selection ends up with the latest
    # survivor set (instruction_ids + sub_paths).
    resolve_exp(cfg, args.exp, apply_current=True)

    filt_dir = get_filter_dir(cfg)
    if not filt_dir.exists():
        raise SystemExit(f"No filters/ at {filt_dir} — run prior stages first.")
    out_dir = filt_dir.parent
    split   = get_split(cfg)

    survivor = get_survivor_path(cfg)
    if not survivor.exists():
        raise SystemExit(f"No survivor.yaml at {survivor} — run stage 2 first.")

    prior_ids = set(int(x) for x in cfg.get("selection", {}).get("instruction_ids") or [])
    if not prior_ids:
        raise SystemExit("Prior stage produced no surviving episodes.")
    prior_subs = cfg.get("selection", {}).get("sub_paths")
    if not prior_subs:
        raise SystemExit(
            "survivor.yaml has no `sub_paths` field — stage 3 expects "
            "sub-path-level survivors (run stage 2 first).",
        )
    allowed_subs = {
        int(ep_id): [int(s) for s in subs]
        for ep_id, subs in prior_subs.items()
    }

    episodes = episodes_from_config(cfg)
    if not episodes:
        raise SystemExit("No episodes loaded from current.yaml.")

    rewrite_episodes, rewrite_path = _load_rewrite(out_dir)
    if not rewrite_episodes:
        raise SystemExit(
            f"No rewrite JSON under {out_dir}/rewrite/ — run "
            f"src/check/rewrite_subinstructions.py first.",
        )
    print(f"Loaded rewrite: {rewrite_path} ({len(rewrite_episodes)} episodes)")

    audit = load_audit(filt_dir, split)
    register_stage(audit, STAGE_NAME)

    keep_sub_paths: Dict[int, List[int]] = {}
    dropped:        Dict[str, Dict]      = {}
    n_subs_total  = 0
    n_subs_keep   = 0
    n_eps_no_data = 0

    for ep in episodes:
        ep_id_str = str(ep.instruction_id)
        ep_audit  = ensure_episode(audit, ep)

        rewrite_ep   = rewrite_episodes.get(ep_id_str)
        partition_ep = _load_partition(out_dir, ep.scan, ep.instruction_id)

        if rewrite_ep is None or partition_ep is None:
            n_eps_no_data += 1
            reason = "rewrite_missing" if rewrite_ep is None else "partition_missing"
            ep_audit["stages"][STAGE_NAME] = {"status": "drop", "reason": reason}
            dropped[ep_id_str] = {"scan": ep.scan, "reason": reason}
            continue

        rewrite_subs = {
            int(s["sub_idx"]): s for s in rewrite_ep.get("sub_paths", [])
        }
        partition_subs = {
            int(s["sub_idx"]): s for s in partition_ep.get("partitions", [])
        }

        ep_keep_subs: List[int]      = []
        ep_drops:     Dict[int, Dict] = {}

        for sub_idx in allowed_subs.get(ep.instruction_id, []):
            n_subs_total += 1
            keep_it, payload = _classify_sub_path(
                rewrite_subs.get(sub_idx),
                partition_subs.get(sub_idx),
            )
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

        ep_audit["stages"][STAGE_NAME] = {
            "status":    "ok" if ep_keep_subs else "drop",
            "kept_sub":  len(ep_keep_subs),
            "total_sub": len(ep.sub_paths),
        }

        if ep_keep_subs:
            keep_sub_paths[ep.instruction_id] = ep_keep_subs
        if ep_drops:
            dropped[ep_id_str] = {
                "scan": ep.scan,
                "subs": {str(k): v for k, v in sorted(ep_drops.items())},
            }

    # ── Write outputs ────────────────────────────────────────────────
    survivor_path = write_survivor(
        cfg, split,
        instruction_ids=sorted(keep_sub_paths.keys()),
        sub_paths=keep_sub_paths,
    )
    drop_path = write_drop_yaml(
        filt_dir, STAGE_NUM, STAGE_NAME, split,
        dropped=dict(sorted(dropped.items(), key=lambda kv: int(kv[0]))),
    )
    save_audit(audit, filt_dir)

    n_eps_keep = len(keep_sub_paths)
    n_eps_drop = len(episodes) - n_eps_keep
    pct_subs   = (n_subs_keep / n_subs_total) if n_subs_total else 0.0

    print(f"=== Stage {STAGE_NUM} — {STAGE_NAME} ===")
    print(f"  episodes in   : {len(episodes)}")
    print(f"  episodes keep : {n_eps_keep}")
    print(f"  episodes drop : {n_eps_drop}  (no surviving sub-path)")
    if n_eps_no_data:
        print(f"    of which {n_eps_no_data} for missing rewrite/partition data")
    print(f"  sub-paths in  : {n_subs_total}")
    print(f"  sub-paths keep: {n_subs_keep}  ({pct_subs:.1%})")
    print(f"  sub-paths drop: {n_subs_total - n_subs_keep}")
    print()
    print("Outputs:")
    print(f"  {survivor_path}")
    print(f"  {drop_path}")
    print(f"  {filt_dir / 'audit.json'}")


if __name__ == "__main__":
    main()
