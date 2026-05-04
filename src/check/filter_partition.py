"""
LION-Bench — Stage 2 filter: consolidate rewrite + partition outputs.

Reads the prior stage's survivor set from ``current.yaml`` (typically
``01_cross_floor.yaml``), then for every (instruction_id, sub_idx) checks:

  • rewriter  — was the LLM rewrite valid?  (drops "error" sub-paths and
                ones the rewriter flagged as ambiguous via ``keep=false``)
  • partition — did geometric partition succeed?  (drops sub-paths with
                <2 nodes / other geometry errors)

This stage graduates the filter framework from episode-level to sub-path
level: ``02_partition.yaml`` carries an ``instruction_ids`` list (episodes
with ≥1 surviving sub-path) AND a ``sub_paths`` dict listing exactly which
sub_idx survive within each.

Prerequisites
-------------
Run these two existing tools first against the prior ``current.yaml`` so
their outputs are on disk for this stage to consume:

  python src/check/rewrite_subinstructions.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --from_yaml results/val_unseen/filters/current.yaml

  python src/check/visualize_partition.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --from_yaml results/val_unseen/filters/current.yaml

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
    load_audit,
    load_keep,
    register_stage,
    resolve_selection,
    save_audit,
    update_current,
    write_drop_yaml,
    write_keep_yaml,
)
from src.dataset.landmark_rxr import episodes_from_config


STAGE_NUM  = 2
STAGE_NAME = "partition"


def _load_rewrite(out_dir: Path) -> Tuple[Dict[str, Dict], Optional[Path]]:
    """Locate and load the rewrite JSON.  Falls back to filtered variant."""
    rewrite_dir = out_dir / "rewrite"
    for fname in ("sub_instructions_rewritten.json",
                  "sub_instructions_rewritten_filtered.json"):
        p = rewrite_dir / fname
        if p.exists():
            with open(p) as f:
                return json.load(f).get("episodes", {}), p
    return {}, None


def _load_partition(out_dir: Path, ep_id: int) -> Optional[Dict]:
    p = out_dir / "partition" / str(ep_id) / "partition.json"
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
    if not rewrite_sub.get("keep", True):
        audit["rewrite"] = "ambiguous"
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
        description="Stage 2: fold rewriter + partition results into the filter pipeline",
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--from_yaml", default=None,
                    help="Selection YAML carrying expname / run_name so this "
                         "stage knows which experiment's filter dir to read.  "
                         "Typically the prior stage's current.yaml.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # Apply experiment identity (expname / run_name) before resolving paths.
    resolve_selection(cfg, args.from_yaml)

    filt_dir = get_filter_dir(cfg)
    if not filt_dir.exists():
        raise SystemExit(f"No filters/ at {filt_dir} — run stage 1 first.")
    out_dir = filt_dir.parent
    split   = get_split(cfg)

    # Read survivors from the prior stage via current.yaml.
    current = filt_dir / "current.yaml"
    if not current.exists():
        raise SystemExit(f"No current.yaml at {current} — run stage 1 first.")
    prior_ids = set(int(x) for x in load_keep(current.resolve())
                                            .get("instruction_ids", []))
    if not prior_ids:
        raise SystemExit("Prior stage produced no surviving episodes.")

    # Reload the dataset, restricted to prior survivors.
    resolve_selection(cfg, str(current.resolve()))
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
        partition_ep = _load_partition(out_dir, ep.instruction_id)

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

        for sub_idx in range(len(ep.sub_paths)):
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
    keep_path = write_keep_yaml(
        filt_dir, STAGE_NUM, STAGE_NAME, split,
        instruction_ids=sorted(keep_sub_paths.keys()),
        sub_paths=keep_sub_paths,
        cfg=cfg,
    )
    drop_path = write_drop_yaml(
        filt_dir, STAGE_NUM, STAGE_NAME, split,
        dropped=dict(sorted(dropped.items(), key=lambda kv: int(kv[0]))),
    )
    save_audit(audit, filt_dir)
    current_path = update_current(filt_dir, keep_path)

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
    print()
    print("Outputs:")
    print(f"  {keep_path}")
    print(f"  {drop_path}")
    print(f"  {current_path}  ->  {keep_path.name}")
    print(f"  {filt_dir / 'audit.json'}")


if __name__ == "__main__":
    main()
