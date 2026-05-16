"""Consolidate per-sub-path info into a single dataset file.

Reads every survivor (scan, instruction_id, sub_idx) from
``survivor.yaml`` and assembles one record per sub-trajectory pulling
together:

  • text       — full instruction + per-sub split (landmark / spatial)
  • geometry   — sub_path_nodes / spatial_path / landmark_path / heading
                  / partition kind / direction_mismatch
  • target     — target_instance_ids + status + matched semantic category
                  + landmark_visible_at_partition
  • rescue     — landmark / category / grounding method (when applicable)
  • pointers   — last rollout frame, partition viz PNG, etc.

Inputs are written by earlier stages — this script does not call the
LLM, the simulator, or YOLO. It is a pure aggregation.

Output: ``results/{run}/dataset.json`` — top-level JSON list of records.

Usage
-----
  python src/check/consolidate_dataset.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --exp configs/selection/val_unseen/one_scene_partial.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import (
    discover_rewrite_suffix,
    get_run_dir,
    resolve_exp,
)
from src.dataset.landmark_rxr import episodes_from_config


def _load_partition(run_dir: Path, scan: str, ep_id: int) -> Optional[Dict]:
    p = run_dir / "partition" / scan / str(ep_id) / "partition.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _partition_for_sub(part_json: Dict, sub_idx: int) -> Optional[Dict]:
    for part in (part_json or {}).get("partitions", []):
        if int(part.get("sub_idx", -1)) == sub_idx:
            return part
    return None


def _load_rewrite(run_dir: Path, scan: str, ep_id: int, suffix: str) -> Optional[Dict]:
    p = (
        run_dir
        / "rewrite"
        / scan
        / str(ep_id)
        / f"sub_instructions_rewritten{suffix}.json"
    )
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f) or {}
    return data.get("episode")


def _rewrite_for_sub(rewrite_episode: Dict, sub_idx: int) -> Optional[Dict]:
    for sp in (rewrite_episode or {}).get("sub_paths", []):
        if int(sp.get("sub_idx", -1)) == sub_idx:
            return sp
    return None


def _load_target_db(run_dir: Path, scan: str) -> Optional[Dict]:
    p = run_dir / "target_instances" / scan / "target_instances.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _target_for_sub(target_db: Dict, ep_id: int, sub_idx: int) -> Optional[Dict]:
    return (
        (target_db or {})
        .get("target_instances", {})
        .get(str(ep_id), {})
        .get(str(sub_idx))
    )


def _build_record(
    ep,
    sub_idx: int,
    part_json: Optional[Dict],
    rewrite_episode: Optional[Dict],
    target_db: Optional[Dict],
) -> Dict[str, Any]:
    part        = _partition_for_sub(part_json or {}, sub_idx)
    rew         = _rewrite_for_sub(rewrite_episode or {}, sub_idx)
    target      = _target_for_sub(target_db or {}, ep.instruction_id, sub_idx)

    sub_pair = ep.sub_paths[sub_idx] if sub_idx < len(ep.sub_paths) else [None, None]
    sub_text = ep.sub_instructions[sub_idx] if sub_idx < len(ep.sub_instructions) else ""

    rec: Dict[str, Any] = {
        "scan":            ep.scan,
        "instruction_id":  ep.instruction_id,
        "path_id":         ep.path_id,
        "sub_idx":         sub_idx,
        "language":        ep.language,
        "instruction":     ep.instruction,
        "sub_instruction": sub_text,
        "sub_path":        list(sub_pair),
    }

    # Rewrite block.
    if rew:
        rec["landmark"]             = rew.get("landmark")
        rec["landmark_category"]    = rew.get("landmark_category")
        rec["landmark_instruction"] = rew.get("landmark_instruction")
        rec["spatial_instruction"]  = rew.get("spatial_instruction")
        components = rew.get("components") or []
        rec["components"] = [
            {
                "original_mention": c.get("original_mention"),
                "semantic_label":   c.get("semantic_label"),
                "description":      c.get("description"),
            }
            for c in components
        ]

    # Partition / geometry block.
    if part:
        rec["sub_path_nodes"]    = part.get("sub_path_nodes")
        rec["spatial_path"]      = part.get("spatial_path")
        rec["landmark_path"]     = part.get("landmark_path")
        rec["partition_kind"]    = part.get("kind")
        rec["instruction_kind"]  = part.get("instruction_kind")
        rec["direction_mismatch"] = part.get("direction_mismatch")

    # Target block.
    if target:
        rec["target_instance_ids"]        = target.get("target_instance_ids") or []
        rec["target_status"]              = target.get("status")
        rec["matched_semantic_category"]  = target.get("matched_category")
        rec["matched_semantic_categories"] = target.get("matched_categories")
        rec["landmark_visible_at_partition"] = (
            target.get("visibility_status") in ("unique", "ambiguous")
        )
        rec["visibility_status"]          = target.get("visibility_status")
        # Pointer to the partition-point viz PNG (when rendered by 08).
        if "partition_viz_path" in target:
            rec["partition_viz_path"] = target["partition_viz_path"]

    return rec


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Consolidate surviving sub-trajectory info into one dataset file.",
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", default=None,
                    help="Experiment handle (selection YAML path or expname). "
                         "Auto-merges survivor.yaml.")
    ap.add_argument("--out", default=None,
                    help="Output JSON path. Default: <run_dir>/dataset.json.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    resolve_exp(cfg, args.exp, apply_current=True)

    run_dir = get_run_dir(cfg)
    if not run_dir.exists():
        raise SystemExit(f"No run_dir at {run_dir}")

    episodes = episodes_from_config(cfg)
    if not episodes:
        raise SystemExit("No surviving episodes — run the filter pipeline first.")

    sub_paths_filter = cfg.get("selection", {}).get("sub_paths") or {}
    if not sub_paths_filter:
        print(
            "WARNING: survivor.yaml has no `sub_paths` field — every sub-path "
            "of each surviving episode will be consolidated. Run 03/04 first "
            "for a tighter set."
        )

    suffix = discover_rewrite_suffix(run_dir / "rewrite") or ""

    # Group survivors per (scan, ep_id) so we hit each partition.json /
    # rewrite JSON / target_instances.json once.
    per_scan_target_cache: Dict[str, Optional[Dict]] = {}

    records: List[Dict[str, Any]] = []
    missing: List[str] = []
    for ep in episodes:
        ep_subs = sub_paths_filter.get(int(ep.instruction_id))
        if ep_subs is None:
            ep_subs = list(range(len(ep.sub_paths)))
        else:
            ep_subs = sorted({int(s) for s in ep_subs})

        if ep.scan not in per_scan_target_cache:
            per_scan_target_cache[ep.scan] = _load_target_db(run_dir, ep.scan)
        target_db = per_scan_target_cache[ep.scan]

        part_json = _load_partition(run_dir, ep.scan, ep.instruction_id)
        rewrite_episode = _load_rewrite(run_dir, ep.scan, ep.instruction_id, suffix)

        for sub_idx in ep_subs:
            rec = _build_record(ep, sub_idx, part_json, rewrite_episode, target_db)
            records.append(rec)
            if part_json is None or _partition_for_sub(part_json, sub_idx) is None:
                missing.append(f"{ep.instruction_id}#{sub_idx}: no partition.json")

    out_path = Path(args.out).expanduser() if args.out else (run_dir / "dataset.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # Summary print.
    n = len(records)
    by_status = {}
    n_with_target = 0
    n_visible = 0
    n_rescued = 0
    scans = set()
    for r in records:
        scans.add(r["scan"])
        by_status[r.get("target_status") or "unknown"] = by_status.get(r.get("target_status") or "unknown", 0) + 1
        if r.get("target_instance_ids"):
            n_with_target += 1
        if r.get("landmark_visible_at_partition"):
            n_visible += 1
        if r.get("target_status") == "rescued" or r.get("rescue_instance_id") is not None:
            n_rescued += 1

    print()
    print(f"=== consolidate summary ===")
    print(f"  records             : {n}")
    print(f"  scans               : {len(scans)}  ({', '.join(sorted(scans))})")
    print(f"  with target_id      : {n_with_target}")
    print(f"  landmark visible    : {n_visible}")
    print(f"  rescue-touched      : {n_rescued}")
    print(f"  by target_status    :")
    for status, count in sorted(by_status.items(), key=lambda kv: -kv[1]):
        print(f"    {status:<25s} {count}")
    if missing:
        print(f"  ⚠ partition missing : {len(missing)} sub-paths")
        for m in missing[:5]:
            print(f"      {m}")
    print()
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
