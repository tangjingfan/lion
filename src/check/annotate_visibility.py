"""
LION-Bench — Landmark visibility annotation (semantic panorama).

For each surviving sub-path (from the latest ``filters/current.yaml``),
renders a 360° semantic panorama at the **partition point** with
Habitat-Lab's ``rgbds_agent`` (same sensor setup as rollout) and tags
whether the landmark category is visible — and how many distinct
instances are visible — via
:meth:`VisibilityChecker.check_landmark_visibility_semantic`.

This is **not a filter** — no sub-paths are dropped, ``current.yaml`` is
not advanced, and the audit is not modified.  The output is a single
classification JSON downstream tools (e.g. an uniqueness filter) can read.

Status values per sub-path:
  • ``visible``                    — ≥ 1 instance of the matched MP40 category
  • ``not_visible``                — category found but 0 instances in FOV
  • ``no_match``                   — landmark text doesn't map to any MP40 cat
  • ``partition_pos_unresolvable`` — couldn't compute partition pos
  • ``partition_json_missing``     — episode has no partition.json

Output (per-scan)
-----------------
  ``{base_dir}/{run_name}/landmark_visibility/{scan}/visibility.json``

Usage
-----
  python src/check/annotate_visibility.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --from_yaml configs/selection/exp.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import (
    get_filter_dir,
    get_run_dir,
    get_split,
    load_keep,
    resolve_selection,
)
from src.dataset.landmark_rxr import episodes_from_config
from src.env.connectivity import load_connectivity
from src.process.landmark_remap import lookup_mention_labels
from src.process.visibility import VisibilityChecker


def _resolve_partition_pos(
    partition_sub:  Dict,
    virtual_nodes:  Dict[str, List[float]],
    scan_db:        Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    """Reconstruct the partition point's 3-D position.

    ``partition.json`` strips ``partition_pos`` from each entry, but the
    partition node id is preserved at ``spatial_path[-1]``:
      • real MP3D node id     → look up in ``scan_db``
      • ``virt:...`` virtual id → 3-D pos comes from ``virtual_nodes``
    """
    spatial_path = partition_sub.get("spatial_path") or []
    if not spatial_path:
        return None
    boundary_id = spatial_path[-1]
    if isinstance(boundary_id, str) and boundary_id.startswith("virt:"):
        pos = virtual_nodes.get(boundary_id)
        return np.asarray(pos, dtype=np.float32) if pos is not None else None
    if boundary_id in scan_db:
        return np.asarray(scan_db[boundary_id], dtype=np.float32)
    return None


def _semantic_labels(
    rewrite_sub:      Dict,
    landmark_mapping: Dict,
    scan:             str,
) -> List[str]:
    """Resolve a sub-path's components to candidate MP40 labels.

    Primary source is the ``landmark_mapping`` lookup keyed by ``scan`` +
    mention.  ``lookup_mention_labels`` accepts both the new per-scan
    format ``{scan: {mention: [labels]}}`` and the legacy flat format
    ``{mention: [labels]}``.  Falls back to this component's own
    ``semantic_label`` when the mention is absent from the mapping.

    Returns a deduplicated, "unknown"-stripped list.
    """
    labels: List[str] = []

    def _add(label: str) -> None:
        s = (label or "").strip()
        if s and s.lower() not in ("unknown", "") and s not in labels:
            labels.append(s)

    for comp in rewrite_sub.get("components") or []:
        mention = (comp.get("original_mention") or "").strip().lower()
        mapped  = lookup_mention_labels(landmark_mapping, scan, mention)
        if mapped:
            for label in mapped:
                _add(label)
        else:
            _add(comp.get("semantic_label") or "")

    return labels


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Annotate landmark visibility (per sub-path) at the "
                    "partition point — classification only, no dropping",
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--from_yaml", default=None,
                    help="Selection / current.yaml carrying expname so the "
                         "tool reads / writes under the right experiment dir.")
    ap.add_argument("--min_pixel_count", type=int, default=50,
                    help="Min pixels per instance to count as visible (default 50)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    resolve_selection(cfg, args.from_yaml)

    filt_dir = get_filter_dir(cfg)
    if not filt_dir.exists():
        raise SystemExit(f"No filters/ at {filt_dir} — run filter pipeline first.")
    out_dir = filt_dir.parent
    split   = get_split(cfg)

    current = filt_dir / "current.yaml"
    if not current.exists():
        raise SystemExit(f"No current.yaml at {current}")

    prior_keep = load_keep(current.resolve())
    prior_subs = prior_keep.get("sub_paths")
    if not prior_subs:
        raise SystemExit(
            "Prior current.yaml has no `sub_paths` field — annotate_visibility "
            "expects sub-path-level survivors (run filter stages 1-3 first).",
        )

    # Locate per-scan rewrite JSONs.  Pick a single suffix variant
    # consistently across all scans (``_filtered`` first).
    rewrite_dir = out_dir / "rewrite"
    if not rewrite_dir.exists():
        raise SystemExit(f"No rewrite dir under {rewrite_dir}")
    scan_dirs = [d for d in sorted(rewrite_dir.iterdir()) if d.is_dir()]
    rewrite_by_scan: Dict[str, Dict[str, Dict]] = {}
    mapping_by_scan: Dict[str, Dict] = {}
    chosen_suffix: Optional[str] = None
    for suffix in ("_filtered", ""):
        if any((d / f"sub_instructions_rewritten{suffix}.json").exists()
               for d in scan_dirs):
            chosen_suffix = suffix
            break
    if chosen_suffix is None:
        raise SystemExit(f"No rewrite JSON under {rewrite_dir}/*/")
    for scan_dir in scan_dirs:
        scan = scan_dir.name
        rw = scan_dir / f"sub_instructions_rewritten{chosen_suffix}.json"
        lm = scan_dir / f"landmark_mapping{chosen_suffix}.json"
        if not rw.exists():
            continue
        with open(rw) as f:
            rewrite_by_scan[scan] = json.load(f).get("episodes", {})
        print(f"Loaded rewrite ({scan}): {rw}")
        if lm.exists():
            with open(lm) as f:
                mapping_by_scan[scan] = json.load(f) or {}
            print(f"Loaded landmark mapping ({scan}): {lm}  "
                  f"({len(mapping_by_scan[scan])} mentions)")
        else:
            mapping_by_scan[scan] = {}
            print(f"  no landmark_mapping{chosen_suffix}.json under {scan_dir}/")

    # Load episodes restricted to prior survivors.
    episodes  = episodes_from_config(cfg)
    needed_ids = {int(k) for k in prior_subs.keys()}
    episodes  = [ep for ep in episodes if ep.instruction_id in needed_ids]
    if not episodes:
        raise SystemExit("No episodes loaded.")

    # Group by scan so each scene loads once.
    by_scan: Dict[str, list] = defaultdict(list)
    for ep in episodes:
        by_scan[ep.scan].append(ep)
    needed_scans = sorted(by_scan.keys())

    db = load_connectivity(
        scenes_dir=cfg["scenes"]["scenes_dir"],
        scans=needed_scans,
        json_dir=cfg["dataset"].get("connectivity_json_dir"),
        pkl_path=cfg["dataset"].get("connectivity_pkl"),
    )

    # ── Walk sub-paths and classify (one annotation file per scan) ──
    out_root = get_run_dir(cfg) / "landmark_visibility"
    out_root.mkdir(parents=True, exist_ok=True)
    written_paths: List[Path] = []
    status_counts: Dict[str, int] = defaultdict(int)
    n_total = 0

    checker = VisibilityChecker(cfg["env"], cfg["scenes"]["scenes_dir"])
    try:
        for scan in needed_scans:
            scan_eps          = by_scan[scan]
            scan_db           = db.get(scan, {})
            scan_rewrite_eps  = rewrite_by_scan.get(scan, {})
            scan_landmark_map = mapping_by_scan.get(scan, {})
            print(f"\n[{scan}] loading scene  ({len(scan_eps)} episode(s)) …")
            checker.load_scene(scan_eps[0].scene_file)

            scan_annotations: Dict[str, Dict[str, Dict]] = {}

            for ep in scan_eps:
                ep_id     = ep.instruction_id
                ep_id_str = str(ep_id)
                sub_idxs  = prior_subs.get(ep_id) or prior_subs.get(ep_id_str) or []

                ep_anno: Dict[str, Dict] = {}

                part_path = (out_dir / "partition" / scan / ep_id_str
                             / "partition.json")
                if not part_path.exists():
                    for sub_idx in sub_idxs:
                        n_total += 1
                        ep_anno[str(int(sub_idx))] = {"status": "partition_json_missing"}
                        status_counts["partition_json_missing"] += 1
                    scan_annotations[ep_id_str] = ep_anno
                    continue

                with open(part_path) as f:
                    part_json = json.load(f)
                partition_subs = {
                    int(s["sub_idx"]): s for s in part_json.get("partitions", [])
                }
                virtual_nodes = part_json.get("virtual_nodes", {})

                rewrite_ep   = scan_rewrite_eps.get(ep_id_str) or {}
                rewrite_subs = {
                    int(s["sub_idx"]): s for s in rewrite_ep.get("sub_paths", [])
                }

                for sub_idx in sub_idxs:
                    sub_idx = int(sub_idx)
                    n_total += 1

                    rw         = rewrite_subs.get(sub_idx, {})
                    landmark   = rw.get("landmark", "")
                    sem_labels = _semantic_labels(rw, scan_landmark_map, scan)
                    part_sub   = partition_subs.get(sub_idx, {})
                    pos        = _resolve_partition_pos(part_sub, virtual_nodes, scan_db)

                    record: Dict = {
                        "landmark":        landmark,
                        "semantic_labels": sem_labels,
                    }

                    if pos is None:
                        record["status"] = "partition_pos_unresolvable"
                    else:
                        result = checker.check_landmark_visibility_semantic(
                            pos, landmark,
                            semantic_labels=sem_labels or None,
                            min_pixel_count=args.min_pixel_count,
                        )
                        record.update({
                            "n_instances":      result.get("n_instances", 0),
                            "matched_category": result.get("matched_category"),
                            "matched_categories": result.get("matched_categories", []),
                            "matched_by":       result.get("matched_by"),
                            "pixel_count":      result.get("pixel_count", 0),
                            "pixel_fraction":   result.get("pixel_fraction", 0.0),
                            "instances":        result.get("instances", []),
                        })
                        if result.get("visible"):
                            record["status"] = "visible"
                        elif result.get("matched_category") is None:
                            record["status"] = "no_match"
                        else:
                            record["status"] = "not_visible"

                    status_counts[record["status"]] += 1
                    ep_anno[str(sub_idx)] = record

                    print(f"  [{ep_id} sub {sub_idx:<2}] "
                          f"{record['status']:<28s}  "
                          f"landmark={landmark!r}  "
                          f"cat={record.get('matched_category')!r}  "
                          f"n={record.get('n_instances', 0)}")

                scan_annotations[ep_id_str] = ep_anno

            scan_dir = out_root / scan
            scan_dir.mkdir(parents=True, exist_ok=True)
            scan_out = scan_dir / "visibility.json"
            payload = {
                "split":           split,
                "scan":            scan,
                "expname":         cfg.get("output", {}).get("expname"),
                "run_name":        cfg.get("output", {}).get("run_name"),
                "min_pixel_count": args.min_pixel_count,
                "source_keep":     str(current.resolve()),
                "annotations":     scan_annotations,
            }
            with open(scan_out, "w") as f:
                json.dump(payload, f, indent=2)
            written_paths.append(scan_out)
            print(f"  → {scan_out}")
    finally:
        checker.close()

    print(f"\n=== Landmark Visibility (semantic) ===")
    print(f"  sub-paths annotated : {n_total}")
    print(f"  status breakdown:")
    for status, n in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        pct = (n / n_total) if n_total else 0.0
        print(f"    {status:<28s} {n:>4d}  ({pct:.1%})")
    print(f"\nOutputs ({len(written_paths)} scan file(s)):")
    for p in written_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
