"""
Select target semantic instance ids from visibility annotations.

For each visible landmark, choose a target instance automatically when it is
unambiguous enough:

  • one visible instance          -> view_unique
  • largest / second largest >= R -> view_dominant
  • otherwise                    -> ambiguous

The output is intended as the input for later same-instance robust visibility
checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import get_run_dir, resolve_selection
from src.check.annotate_visibility import _resolve_partition_pos
from src.check.query_scene_instance import (
    _draw_house_instance_viz,
    _render_mask_for_rollout_frame,
)
from src.env.connectivity import load_connectivity
from src.process.visibility import VisibilityChecker


def _iter_records(data: Dict[str, Any]) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    for ep_id, ep_records in (data.get("annotations") or {}).items():
        if not isinstance(ep_records, dict):
            continue
        for sub_idx, record in ep_records.items():
            if isinstance(record, dict):
                yield str(ep_id), str(sub_idx), record


def _select(record: Dict[str, Any], dominance_ratio: float) -> Dict[str, Any]:
    instances = list(record.get("instances") or [])
    instances.sort(key=lambda x: int(x.get("n_pixels") or 0), reverse=True)

    base = {
        "landmark": record.get("landmark"),
        "semantic_labels": record.get("semantic_labels", []),
        "matched_category": record.get("matched_category"),
        "matched_categories": record.get("matched_categories", []),
        "visibility_status": record.get("status"),
        "candidates": instances,
        "target_instance_ids": [],
        "dominance_ratio": None,
    }

    if record.get("status") != "visible":
        return {**base, "status": f"visibility:{record.get('status') or 'unknown'}"}
    if not instances:
        return {**base, "status": "no_visible_instance"}
    if len(instances) == 1:
        return {
            **base,
            "status": "view_unique",
            "target_instance_ids": [int(instances[0]["id"])],
        }

    top = int(instances[0].get("n_pixels") or 0)
    second = int(instances[1].get("n_pixels") or 0)
    ratio = float("inf") if second <= 0 and top > 0 else (top / second if second else 0.0)
    if ratio >= dominance_ratio:
        return {
            **base,
            "status": "view_dominant",
            "target_instance_ids": [int(instances[0]["id"])],
            "dominance_ratio": ratio,
        }

    return {
        **base,
        "status": "ambiguous",
        "dominance_ratio": ratio,
    }


def _resolve_visibility_paths(path: Path) -> List[Path]:
    if path.exists():
        return [path]
    parent = path.parent
    paths = sorted(parent.glob("*/visibility.json"))
    if paths:
        return paths
    raise SystemExit(f"Visibility JSON not found: {path}")


def _infer_run_dir_from_visibility_path(vis_path: Path) -> Path:
    """Infer results/{run_name} from either aggregate or per-scan visibility JSON."""
    if vis_path.parent.parent.name == "landmark_visibility":
        return vis_path.parent.parent.parent
    return vis_path.parent.parent


def _partition_json_path(run_dir: Path, scan: str, ep_id: str) -> Optional[Path]:
    candidates = [
        run_dir / "partition" / scan / ep_id / "partition.json",
        run_dir / "partition" / ep_id / "partition.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_partition_position(
    run_dir: Path,
    scan: str,
    ep_id: str,
    sub_idx: str,
    scan_db: Dict[str, np.ndarray],
) -> Tuple[Optional[np.ndarray], int, Optional[str]]:
    part_path = _partition_json_path(run_dir, scan, ep_id)
    if part_path is None:
        return None, 1, "partition_json_missing"

    with open(part_path) as f:
        part_json = json.load(f)

    partition_subs = {
        int(s["sub_idx"]): s for s in part_json.get("partitions", [])
        if "sub_idx" in s
    }
    part_sub = partition_subs.get(int(sub_idx), {})
    pos = _resolve_partition_pos(
        part_sub,
        part_json.get("virtual_nodes", {}),
        scan_db,
    )
    if pos is None:
        return None, max(len(partition_subs), 1), "partition_pos_unresolvable"
    return pos, max(len(partition_subs), 1), None


def _safe_instruction_id(ep_id: str) -> Any:
    try:
        return int(ep_id)
    except ValueError:
        return ep_id


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select target instance ids from landmark visibility JSON.",
    )
    ap.add_argument("--config", default="configs/rollout/rollout_landmark_rxr.yaml")
    ap.add_argument("--from_yaml", default=None,
                    help="Selection YAML carrying expname / run_name.")
    ap.add_argument("--visibility_json", default=None,
                    help="Direct path to landmark_visibility/visibility.json.")
    ap.add_argument("--out", default=None,
                    help="Output JSON path. Default: {run}/target_instances/target_instances.json")
    ap.add_argument("--dominance_ratio", type=float, default=3.0,
                    help="Pick largest instance when largest/second >= this ratio.")
    ap.add_argument("--save_viz", action="store_true", default=True,
                    help="Save a Habitat mask PNG for each selected target instance.")
    ap.add_argument("--no_save_viz", action="store_false", dest="save_viz",
                    help="Do not save per-target visualization PNGs.")
    ap.add_argument("--viz_mode", choices=("mask", "topdown"), default="mask",
                    help="Visualization type. Default: Habitat rollout-style mask.")
    ap.add_argument("--info_width", type=int, default=300,
                    help="Right info panel width for Habitat mask visualizations.")
    ap.add_argument("--print_ambiguous", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.visibility_json:
        vis_path = Path(args.visibility_json)
        run_dir = _infer_run_dir_from_visibility_path(vis_path)
        default_out = run_dir / "target_instances" / "target_instances.json"
    else:
        resolve_selection(cfg, args.from_yaml)
        run_dir = get_run_dir(cfg)
        vis_path = run_dir / "landmark_visibility" / "visibility.json"
        default_out = run_dir / "target_instances" / "target_instances.json"

    vis_paths = _resolve_visibility_paths(vis_path)

    selections: Dict[str, Dict[str, Any]] = {}
    counts: Counter = Counter()
    viz_count = 0
    viz_errors = 0
    scenes_dir = cfg.get("scenes", {}).get("scenes_dir", "")
    viz_root = (Path(args.out).parent if args.out else default_out.parent) / "viz"

    checker: Optional[VisibilityChecker] = None
    connectivity_by_scan: Dict[str, Dict[str, np.ndarray]] = {}
    if args.save_viz and args.viz_mode == "mask":
        checker = VisibilityChecker(cfg["env"], scenes_dir)

    try:
        for one_vis_path in vis_paths:
            with open(one_vis_path) as f:
                visibility = json.load(f)
            scan = visibility.get("scan")
            for ep_id, sub_idx, record in _iter_records(visibility):
                selected = _select(record, args.dominance_ratio)
                if scan:
                    selected["scan"] = scan
                if args.save_viz and scan and selected.get("target_instance_ids"):
                    target_id = int(selected["target_instance_ids"][0])
                    viz_path = (
                        viz_root
                        / scan
                        / str(ep_id)
                        / f"sub_{int(sub_idx):03d}_id_{target_id}.png"
                    )
                    try:
                        if args.viz_mode == "topdown":
                            _draw_house_instance_viz(
                                scenes_dir=scenes_dir,
                                scan=scan,
                                instance_id=target_id,
                                out_path=viz_path,
                            )
                            selected["viz_path"] = str(viz_path)
                        else:
                            if checker is None:
                                raise RuntimeError("Habitat checker was not initialized.")
                            if scan not in connectivity_by_scan:
                                db = load_connectivity(
                                    scenes_dir=scenes_dir,
                                    scans=[scan],
                                    json_dir=cfg["dataset"].get("connectivity_json_dir"),
                                    pkl_path=cfg["dataset"].get("connectivity_pkl"),
                                )
                                connectivity_by_scan[scan] = db.get(scan, {})
                            pos, sub_total, pos_error = _load_partition_position(
                                run_dir=run_dir,
                                scan=scan,
                                ep_id=str(ep_id),
                                sub_idx=str(sub_idx),
                                scan_db=connectivity_by_scan[scan],
                            )
                            if pos_error or pos is None:
                                raise RuntimeError(pos_error or "partition_pos_unresolvable")
                            rv = _render_mask_for_rollout_frame(
                                checker=checker,
                                scan=scan,
                                instance_id=target_id,
                                frame_record={
                                    "position": [float(x) for x in pos],
                                    "heading": 0.0,
                                    "instruction_id": _safe_instruction_id(str(ep_id)),
                                    "instruction": selected.get("landmark") or "",
                                    "landmark": selected.get("landmark") or "",
                                    "sub_idx": int(sub_idx),
                                    "sub_total": sub_total,
                                    "step": 0,
                                    "action": "SELECT_TARGET",
                                },
                                out_path=viz_path,
                                info_width=args.info_width,
                            )
                            selected["viz_path"] = rv["path"]
                            selected["viz_target_pixels"] = rv.get("target_visible_pixels")
                    except Exception as exc:
                        selected["viz_error"] = str(exc)
                        viz_errors += 1
                    else:
                        viz_count += 1
                selections.setdefault(ep_id, {})[sub_idx] = selected
                counts[selected["status"]] += 1
    finally:
        if checker is not None:
            checker.close()

    out_path = Path(args.out) if args.out else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_visibility": [str(p) for p in vis_paths],
        "dominance_ratio": args.dominance_ratio,
        "summary": dict(counts),
        "target_instances": selections,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    total = sum(counts.values())
    auto = counts["view_unique"] + counts["view_dominant"]
    print("=== Target Instance Selection ===")
    print(f"visibility_json : {vis_paths[0] if len(vis_paths) == 1 else f'{len(vis_paths)} files'}")
    print(f"output          : {out_path}")
    print(f"dominance_ratio : {args.dominance_ratio:.2f}")
    print(f"records         : {total}")
    auto_pct = (auto / total) if total else 0.0
    print(f"auto selected   : {auto}  ({auto_pct:.1%})")
    if args.save_viz:
        print(f"viz saved       : {viz_count}")
        if viz_errors:
            print(f"viz errors      : {viz_errors}")
    print("status breakdown:")
    for status, n in counts.most_common():
        pct = (n / total) if total else 0.0
        print(f"  {status:<20s} {n:>5d}  ({pct:.1%})")

    if args.print_ambiguous:
        print()
        print("ambiguous:")
        for ep_id, sub_map in selections.items():
            for sub_idx, selected in sub_map.items():
                if selected["status"] != "ambiguous":
                    continue
                print(
                    f"  ep={ep_id:<8s} sub={sub_idx:<3s} "
                    f"ratio={selected.get('dominance_ratio'):.2f} "
                    f"landmark={selected.get('landmark')!r}"
                )


if __name__ == "__main__":
    main()
