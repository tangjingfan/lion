"""
Select target semantic instance ids from enumerated candidate annotations.

For each visible landmark, choose a target instance:

  • one visible instance   -> view_unique
  • multiple instances     -> view_nearest  (closest to the sub-path end point)

When instance centers can't be resolved, falls back to the largest-pixel
instance with status ``view_nearest_fallback``.

The output is intended as the input for later same-instance robust visibility
checks.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import get_run_dir, resolve_selection
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


def _load_last_rollout_frames(
    run_dir: Path, scan: str,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Return the last rollout-viz frame record for each ``(ep_id, sub_idx)``."""
    index_path = run_dir / "rollout_viz" / scan / "frames.jsonl"
    if not index_path.exists():
        return {}

    last: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with open(index_path) as f:
        for line_no, line in enumerate(f, start=1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep_id = rec.get("instruction_id")
            sub_idx = rec.get("sub_idx")
            if ep_id is None or sub_idx is None:
                continue
            key = (str(ep_id), str(sub_idx))
            rec["_frame_index"] = line_no - 1
            rel_path = rec.get("rel_path")
            if rel_path:
                rec["_rollout_frame"] = str(run_dir / "rollout_viz" / scan / rel_path)
            prev = last.get(key)
            if prev is None or int(rec.get("step") or 0) >= int(prev.get("step") or 0):
                last[key] = rec
    return last


def _copy_last_rollout_frame(
    frame_record: Dict[str, Any],
    out_path: Path,
) -> Optional[str]:
    src = frame_record.get("_rollout_frame")
    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, out_path)
    return str(out_path)


def _copy_existing_png(src: Optional[str], out_path: Path) -> Optional[str]:
    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, out_path)
    return str(out_path)


def _record_candidates(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return candidate instances from either new or legacy annotation shape."""
    candidates = record.get("candidates")
    if candidates is None:
        candidates = record.get("instances")
    items = list(candidates or [])
    items.sort(key=lambda x: int(x.get("n_pixels") or 0), reverse=True)
    return items


def _candidate_viz_path(record: Dict[str, Any], target_id: int) -> Optional[str]:
    for cand in _record_candidates(record):
        try:
            iid = int(cand.get("id"))
        except (TypeError, ValueError):
            continue
        if iid == target_id and cand.get("viz_path"):
            return str(cand["viz_path"])
    return None


def _record_visibility_status(record: Dict[str, Any]) -> str:
    """Normalize new ``uniqueness`` and legacy ``status`` fields."""
    uniqueness = record.get("uniqueness")
    if uniqueness:
        return str(uniqueness)
    status = record.get("status")
    if status:
        return str(status)
    return "unknown"


def _select(
    record: Dict[str, Any],
    end_pos: Optional[np.ndarray] = None,
    instance_centers: Optional[Dict[int, np.ndarray]] = None,
    center_source: Optional[str] = None,
) -> Dict[str, Any]:
    instances = _record_candidates(record)
    visibility_status = _record_visibility_status(record)

    base = {
        "landmark": record.get("landmark"),
        "semantic_labels": record.get("semantic_labels", []),
        "matched_category": record.get("matched_category"),
        "matched_categories": record.get("matched_categories", []),
        "visibility_status": visibility_status,
        "candidates": instances,
        "target_instance_ids": [],
        "selection_distance": None,
        "candidate_distances": None,
        "center_source": center_source,
    }

    if visibility_status in ("visible", "unique", "ambiguous"):
        pass
    else:
        return {**base, "status": f"visibility:{visibility_status}"}
    if not instances:
        return {**base, "status": "no_visible_instance"}
    if visibility_status == "unique" or len(instances) == 1:
        return {
            **base,
            "status": "view_unique",
            "target_instance_ids": [int(instances[0]["id"])],
        }

    # Multiple instances — pick the one whose center is closest to the
    # sub-path end (last-step) position.
    distances: Dict[int, float] = {}
    if end_pos is not None and instance_centers:
        for inst in instances:
            iid = int(inst["id"])
            ctr = instance_centers.get(iid)
            if ctr is None:
                continue
            distances[iid] = float(np.linalg.norm(np.asarray(ctr) - end_pos))

    if distances:
        nearest_id = min(distances, key=distances.get)
        return {
            **base,
            "status": "view_nearest",
            "target_instance_ids": [nearest_id],
            "selection_distance": distances[nearest_id],
            "candidate_distances": {str(k): round(v, 4) for k, v in distances.items()},
        }

    # No instance positions available — fall back to largest-pixel.
    return {
        **base,
        "status": "view_nearest_fallback",
        "target_instance_ids": [int(instances[0]["id"])],
        "fallback_reason": (
            "subpath_end_pos_unresolvable"
            if end_pos is None else "instance_centers_unavailable"
        ),
    }


def _resolve_visibility_paths(path: Path) -> List[Path]:
    if path.exists():
        return [path]
    parent = path.parent
    paths = sorted(parent.glob("*/visibility.json"))
    if paths:
        return paths
    raise SystemExit(f"Visibility JSON not found: {path}")


def _resolve_target_instance_paths(path: Path) -> List[Path]:
    if path.name == "target_instances.json" and path.parent.name == "target_instances":
        paths = sorted(path.parent.glob("*/target_instances.json"))
        if paths:
            return paths
    if path.exists():
        if path.is_dir():
            paths = sorted(path.glob("*/target_instances.json"))
            if paths:
                return paths
        return [path]
    parent = path.parent
    paths = sorted(parent.glob("*/target_instances.json"))
    if paths:
        return paths
    raise SystemExit(f"Target instances JSON not found: {path}")


def _infer_run_dir_from_visibility_path(vis_path: Path) -> Path:
    """Infer results/{run_name} from either aggregate or per-scan visibility JSON."""
    if vis_path.parent.parent.name == "landmark_visibility":
        return vis_path.parent.parent.parent
    return vis_path.parent.parent


def _infer_run_dir_from_target_instances_path(path: Path) -> Path:
    """Infer results/{run_name} from target_instances JSON paths."""
    if path.parent.name == "target_instances":
        return path.parent.parent
    if path.parent.parent.name == "target_instances":
        return path.parent.parent.parent
    return path.parent.parent


def _partition_json_path(run_dir: Path, scan: str, ep_id: str) -> Optional[Path]:
    candidates = [
        run_dir / "partition" / scan / ep_id / "partition.json",
        run_dir / "partition" / ep_id / "partition.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _resolve_subpath_end_pos(
    partition_sub: Dict,
    virtual_nodes: Dict[str, List[float]],
    scan_db: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    """Resolve the 3-D position of the sub-path's last node (final step).

    Prefers ``sub_path_nodes[-1]`` (always a real MP3D node); falls back to
    ``landmark_path[-1]`` which may be a virtual ``virt:...`` id.
    """
    end_id = None
    sub_nodes = partition_sub.get("sub_path_nodes") or []
    if sub_nodes:
        end_id = sub_nodes[-1]
    else:
        landmark_path = partition_sub.get("landmark_path") or []
        if landmark_path:
            end_id = landmark_path[-1]
    if end_id is None:
        return None
    if isinstance(end_id, str) and end_id.startswith("virt:"):
        pos = virtual_nodes.get(end_id)
        return np.asarray(pos, dtype=np.float32) if pos is not None else None
    if end_id in scan_db:
        return np.asarray(scan_db[end_id], dtype=np.float32)
    return None


def _resolve_partition_pos(
    partition_sub: Dict,
    virtual_nodes: Dict[str, List[float]],
    scan_db: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    """Resolve the partition point: the boundary after the spatial segment."""
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


def _load_subpath_end_position(
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
    pos = _resolve_subpath_end_pos(
        part_sub,
        part_json.get("virtual_nodes", {}),
        scan_db,
    )
    if pos is None:
        return None, max(len(partition_subs), 1), "subpath_end_pos_unresolvable"
    return pos, max(len(partition_subs), 1), None


def _load_subpath_positions(
    run_dir: Path,
    scan: str,
    ep_id: str,
    sub_idx: str,
    scan_db: Dict[str, np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, Optional[str]]:
    part_path = _partition_json_path(run_dir, scan, ep_id)
    if part_path is None:
        return None, None, 1, "partition_json_missing"

    with open(part_path) as f:
        part_json = json.load(f)

    partition_subs = {
        int(s["sub_idx"]): s for s in part_json.get("partitions", [])
        if "sub_idx" in s
    }
    part_sub = partition_subs.get(int(sub_idx), {})
    virtual_nodes = part_json.get("virtual_nodes", {})
    end_pos = _resolve_subpath_end_pos(part_sub, virtual_nodes, scan_db)
    partition_pos = _resolve_partition_pos(part_sub, virtual_nodes, scan_db)
    err = None
    if end_pos is None:
        err = "subpath_end_pos_unresolvable"
    elif partition_pos is None:
        err = "partition_pos_unresolvable"
    return end_pos, partition_pos, max(len(partition_subs), 1), err


def _safe_instruction_id(ep_id: str) -> Any:
    try:
        return int(ep_id)
    except ValueError:
        return ep_id


def _instance_centers_from_scene(checker: VisibilityChecker) -> Dict[int, np.ndarray]:
    """Map raw semantic-sensor instance id → habitat-space AABB center.

    Reads from the currently loaded Habitat scene's semantic annotations.
    Instance id is the integer suffix of each object's ``obj.id`` (e.g.
    ``object_331`` → ``331``), matching the ids the semantic sensor emits.
    """
    sim = getattr(checker, "_sim", None)
    if sim is None:
        return {}
    try:
        scene = sim.semantic_annotations()
        objects = list(getattr(scene, "objects", []) or [])
    except Exception:
        return {}
    centers: Dict[int, np.ndarray] = {}
    for obj in objects:
        if obj is None:
            continue
        try:
            iid = int(str(obj.id).split("_")[-1])
        except (AttributeError, TypeError, ValueError):
            continue
        aabb = getattr(obj, "aabb", None)
        ctr = getattr(aabb, "center", None) if aabb is not None else None
        if ctr is None:
            continue
        try:
            centers[iid] = np.asarray(ctr, dtype=np.float32)
        except Exception:
            continue
    return centers


def _instance_centers_from_house(scenes_dir: str, scan: str) -> Dict[int, np.ndarray]:
    """Map MP3D instance id → object center parsed from the scan's ``.house``.

    The target instance ids emitted by the semantic sensor line up with MP3D
    object ids in the ``O`` rows, whose fields 4:7 are the object center.  This
    keeps nearest-instance selection working even when Habitat cannot be
    imported in the current environment.
    """
    house_path = Path(scenes_dir) / "mp3d" / scan / f"{scan}.house"
    if not house_path.exists():
        return {}

    centers: Dict[int, np.ndarray] = {}
    with open(house_path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] != "O" or len(parts) < 7:
                continue
            try:
                iid = int(parts[1])
                center = np.asarray([float(x) for x in parts[4:7]], dtype=np.float32)
            except ValueError:
                continue
            centers[iid] = center
    return centers


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select target instance ids from landmark visibility JSON.",
    )
    ap.add_argument("--config", default="configs/rollout/rollout_landmark_rxr.yaml")
    ap.add_argument("--from_yaml", default=None,
                    help="Selection YAML carrying expname / run_name.")
    ap.add_argument("--visibility_json", default=None,
                    help="Legacy direct path to landmark_visibility/visibility.json.")
    ap.add_argument("--target_instances_json", default=None,
                    help="Direct path to target_instances/{scan}/target_instances.json "
                         "or a target_instances directory.")
    ap.add_argument("--out", default=None,
                    help="Optional aggregate output JSON path. By default the "
                         "selection is written back into each per-scan "
                         "target_instances/{scan}/target_instances.json file.")
    ap.add_argument("--save_viz", action="store_true", default=True,
                    help="Save a Habitat mask PNG for each selected target instance.")
    ap.add_argument("--no_save_viz", action="store_false", dest="save_viz",
                    help="Do not save per-target visualization PNGs.")
    ap.add_argument("--viz_mode", choices=("mask", "topdown"), default="mask",
                    help="Visualization type. Default: Habitat rollout-style mask.")
    ap.add_argument("--info_width", type=int, default=300,
                    help="Right info panel width for Habitat mask visualizations.")
    ap.add_argument("--print_multi", action="store_true",
                    help="Print per-record details when >1 candidate instances.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.visibility_json and args.target_instances_json:
        raise SystemExit("Pass only one of --visibility_json or --target_instances_json.")

    if args.visibility_json:
        vis_path = Path(args.visibility_json)
        run_dir = _infer_run_dir_from_visibility_path(vis_path)
        source_paths = _resolve_visibility_paths(vis_path)
        source_kind = "visibility_json"
    elif args.target_instances_json:
        ti_path = Path(args.target_instances_json)
        run_dir = _infer_run_dir_from_target_instances_path(ti_path)
        source_paths = _resolve_target_instance_paths(ti_path)
        source_kind = "target_instances_json"
    else:
        resolve_selection(cfg, args.from_yaml)
        run_dir = get_run_dir(cfg)
        ti_path = run_dir / "target_instances" / "target_instances.json"
        source_paths = _resolve_target_instance_paths(ti_path)
        source_kind = "target_instances_json"

    selections: Dict[str, Dict[str, Any]] = {}
    counts: Counter = Counter()
    written_paths: List[Path] = []
    viz_count = 0
    viz_errors = 0
    viz_skipped = 0
    last_frame_viz_count = 0
    last_frame_instance_viz_count = 0
    partition_viz_count = 0
    last_frame_missing = 0
    scenes_dir = cfg.get("scenes", {}).get("scenes_dir", "")
    default_target_root = run_dir / "target_instances"
    viz_root = (
        Path(args.out).parent if args.out else default_target_root
    ) / "viz_last_frame_instance"
    partition_viz_root = (
        Path(args.out).parent if args.out else default_target_root
    ) / "viz_partition"
    last_frame_viz_root = (
        Path(args.out).parent if args.out else default_target_root
    ) / "viz_last_frame"

    # Selection needs instance centers for >1-candidate cases.  Prefer Habitat
    # AABB centers when available; fall back to MP3D .house object centers.
    checker: VisibilityChecker = VisibilityChecker(cfg["env"], scenes_dir)
    connectivity_by_scan: Dict[str, Dict[str, np.ndarray]] = {}

    def _ensure_scan_loaded(
        scan_name: str,
    ) -> Tuple[Dict[str, np.ndarray], Dict[int, np.ndarray], str]:
        if scan_name not in connectivity_by_scan:
            db = load_connectivity(
                scenes_dir=scenes_dir,
                scans=[scan_name],
                json_dir=cfg["dataset"].get("connectivity_json_dir"),
                pkl_path=cfg["dataset"].get("connectivity_pkl"),
            )
            connectivity_by_scan[scan_name] = db.get(scan_name, {})
        house_centers = _instance_centers_from_house(scenes_dir, scan_name)
        try:
            checker.load_scene(f"mp3d/{scan_name}/{scan_name}.glb")
        except Exception as exc:
            if house_centers:
                print(
                    f"  WARN: could not load Habitat scene for {scan_name}: {exc}. "
                    f"Using {len(house_centers)} .house instance centers."
                )
                return connectivity_by_scan[scan_name], house_centers, "house"
            print(
                f"  WARN: could not load Habitat scene for {scan_name}: {exc}. "
                "Multi-candidate selection will fall back to largest pixel count."
            )
            return connectivity_by_scan[scan_name], {}, "unavailable"
        scene_centers = _instance_centers_from_scene(checker)
        if scene_centers:
            return connectivity_by_scan[scan_name], scene_centers, "habitat"
        if house_centers:
            print(
                f"  WARN: Habitat scene for {scan_name} exposed no centers; "
                f"using {len(house_centers)} .house instance centers."
            )
            return connectivity_by_scan[scan_name], house_centers, "house"
        return connectivity_by_scan[scan_name], {}, "unavailable"

    try:
        for source_path in source_paths:
            with open(source_path) as f:
                annotations = json.load(f)
            scan = annotations.get("scan")
            source_selections: Dict[str, Dict[str, Any]] = {}
            source_counts: Counter = Counter()

            scan_db: Dict[str, np.ndarray] = {}
            instance_centers: Dict[int, np.ndarray] = {}
            center_source = "unavailable"
            last_frames: Dict[Tuple[str, str], Dict[str, Any]] = {}
            if scan:
                scan_db, instance_centers, center_source = _ensure_scan_loaded(scan)
                last_frames = _load_last_rollout_frames(run_dir, scan)
            habitat_loaded = center_source == "habitat"

            for ep_id, sub_idx, record in _iter_records(annotations):
                end_pos: Optional[np.ndarray] = None
                partition_pos: Optional[np.ndarray] = None
                sub_total = 1
                last_frame = last_frames.get((str(ep_id), str(sub_idx)))
                if scan and scan_db:
                    end_pos, partition_pos, sub_total, _ = _load_subpath_positions(
                        run_dir=run_dir,
                        scan=scan,
                        ep_id=str(ep_id),
                        sub_idx=str(sub_idx),
                        scan_db=scan_db,
                    )

                selected = _select(
                    record,
                    end_pos=end_pos,
                    instance_centers=instance_centers,
                    center_source=center_source,
                )
                if scan:
                    selected["scan"] = scan
                if last_frame:
                    selected["rollout_last_frame"] = last_frame.get("_rollout_frame")
                    selected["rollout_last_frame_index"] = last_frame.get("_frame_index")
                    selected["rollout_last_step"] = last_frame.get("step")
                    if args.save_viz and scan:
                        last_out = (
                            last_frame_viz_root
                            / scan
                            / str(ep_id)
                            / f"sub_{int(sub_idx):03d}_last.png"
                        )
                        copied = _copy_last_rollout_frame(last_frame, last_out)
                        if copied:
                            selected["rollout_last_viz_path"] = copied
                            last_frame_viz_count += 1
                elif scan:
                    last_frame_missing += 1
                if args.save_viz and scan and selected.get("target_instance_ids"):
                    target_id = int(selected["target_instance_ids"][0])
                    partition_viz_path = (
                        partition_viz_root
                        / scan
                        / str(ep_id)
                        / f"sub_{int(sub_idx):03d}_id_{target_id}.png"
                    )
                    copied_partition = _copy_existing_png(
                        _candidate_viz_path(record, target_id),
                        partition_viz_path,
                    )
                    if copied_partition:
                        selected["partition_viz_path"] = copied_partition
                        partition_viz_count += 1
                    elif habitat_loaded and partition_pos is not None:
                        try:
                            rv = _render_mask_for_rollout_frame(
                                checker=checker,
                                scan=scan,
                                instance_id=target_id,
                                frame_record={
                                    "position": [float(x) for x in partition_pos],
                                    "heading": 0.0,
                                    "instruction_id": _safe_instruction_id(str(ep_id)),
                                    "instruction": selected.get("landmark") or "",
                                    "landmark": selected.get("landmark") or "",
                                    "sub_idx": int(sub_idx),
                                    "sub_total": sub_total,
                                    "step": 0,
                                    "action": "SELECT_TARGET_PARTITION",
                                },
                                out_path=partition_viz_path,
                                info_width=args.info_width,
                            )
                            selected["partition_viz_path"] = rv["path"]
                            selected["partition_viz_target_pixels"] = rv.get(
                                "target_visible_pixels"
                            )
                            partition_viz_count += 1
                        except Exception as exc:
                            selected["partition_viz_error"] = str(exc)
                            viz_errors += 1
                    else:
                        selected["partition_viz_skipped"] = (
                            "candidate viz missing and Habitat scene unavailable"
                            if not habitat_loaded else "partition_pos_unresolvable"
                        )

                    viz_path = (
                        viz_root
                        / scan
                        / str(ep_id)
                        / f"sub_{int(sub_idx):03d}_id_{target_id}.png"
                    )
                    if args.viz_mode == "mask" and not habitat_loaded:
                        viz_skipped += 1
                        selected["viz_skipped"] = (
                            f"Habitat scene unavailable; center_source={center_source}"
                        )
                    else:
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
                                if end_pos is None:
                                    raise RuntimeError("subpath_end_pos_unresolvable")
                                render_frame = dict(last_frame or {})
                                if not render_frame:
                                    render_frame = {
                                        "position": [float(x) for x in end_pos],
                                        "heading": 0.0,
                                        "instruction_id": _safe_instruction_id(str(ep_id)),
                                        "instruction": selected.get("landmark") or "",
                                        "landmark": selected.get("landmark") or "",
                                        "sub_idx": int(sub_idx),
                                        "sub_total": sub_total,
                                        "step": 0,
                                        "action": "SELECT_TARGET",
                                    }
                                else:
                                    render_frame["landmark"] = selected.get("landmark") or ""
                                    render_frame["instruction"] = (
                                        render_frame.get("sub_instruction")
                                        or selected.get("landmark")
                                        or ""
                                    )
                                    render_frame["sub_total"] = int(
                                        render_frame.get("sub_total") or sub_total
                                    )
                                rv = _render_mask_for_rollout_frame(
                                    checker=checker,
                                    scan=scan,
                                    instance_id=target_id,
                                    frame_record=render_frame,
                                    out_path=viz_path,
                                    info_width=args.info_width,
                                )
                                selected["last_frame_instance_viz_path"] = rv["path"]
                                selected["last_frame_instance_target_pixels"] = rv.get(
                                    "target_visible_pixels"
                                )
                                selected["viz_path"] = rv["path"]
                        except Exception as exc:
                            selected["viz_error"] = str(exc)
                            viz_errors += 1
                        else:
                            viz_count += 1
                            last_frame_instance_viz_count += 1
                selections.setdefault(ep_id, {})[sub_idx] = selected
                source_selections.setdefault(ep_id, {})[sub_idx] = selected
                counts[selected["status"]] += 1
                source_counts[selected["status"]] += 1

            if args.out is None:
                annotations["selection_rule"] = (
                    "single -> view_unique; multi -> nearest-to-subpath-end"
                )
                annotations["selection_summary"] = dict(source_counts)
                annotations["target_instances"] = source_selections
                with open(source_path, "w") as f:
                    json.dump(annotations, f, indent=2)
                written_paths.append(source_path)
    finally:
        checker.close()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            f"source_{source_kind}": [str(p) for p in source_paths],
            "selection_rule": "single -> view_unique; multi -> nearest-to-subpath-end",
            "summary": dict(counts),
            "target_instances": selections,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        written_paths.append(out_path)

    total = sum(counts.values())
    auto = counts["view_unique"] + counts["view_nearest"] + counts["view_nearest_fallback"]
    print("=== Target Instance Selection ===")
    print(f"{source_kind:<16s}: {source_paths[0] if len(source_paths) == 1 else f'{len(source_paths)} files'}")
    print(f"output          : {written_paths[0] if len(written_paths) == 1 else f'{len(written_paths)} files'}")
    print(f"records         : {total}")
    auto_pct = (auto / total) if total else 0.0
    print(f"auto selected   : {auto}  ({auto_pct:.1%})")
    if args.save_viz:
        print(f"last-frame raw  : {last_frame_viz_count}")
        print(f"last-frame inst : {last_frame_instance_viz_count}")
        print(f"partition viz   : {partition_viz_count}")
        if last_frame_missing:
            print(f"last-frame miss : {last_frame_missing}")
        if viz_skipped:
            print(f"viz skipped     : {viz_skipped}")
        if viz_errors:
            print(f"viz errors      : {viz_errors}")
    print("status breakdown:")
    for status, n in counts.most_common():
        pct = (n / total) if total else 0.0
        print(f"  {status:<24s} {n:>5d}  ({pct:.1%})")

    if args.print_multi:
        print()
        print("multi-candidate sub-paths:")
        for ep_id, sub_map in selections.items():
            for sub_idx, selected in sub_map.items():
                if selected["status"] not in ("view_nearest", "view_nearest_fallback"):
                    continue
                dist = selected.get("selection_distance")
                dist_str = f"{dist:.3f}m" if dist is not None else "  —  "
                print(
                    f"  ep={ep_id:<8s} sub={sub_idx:<3s} "
                    f"status={selected['status']:<22s} "
                    f"dist={dist_str}  "
                    f"chosen={selected['target_instance_ids']}  "
                    f"all_dists={selected.get('candidate_distances')}  "
                    f"landmark={selected.get('landmark')!r}"
                )


if __name__ == "__main__":
    main()
