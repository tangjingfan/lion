"""Target-instance selection rule + its inputs (pipeline step 08).

The decision core of ``select_target_instances``:

  • one visible instance   -> view_unique
  • multiple instances     -> view_nearest  (closest to the sub-path end point)
  • centers unavailable    -> view_nearest_fallback (largest pixel count)

plus the readers it depends on: annotation-record accessors (tolerant of
the legacy schema), partition.json pose resolution, and the two
instance-center sources (Habitat AABB / .house rows).  CLI orchestration
and viz copying stay in ``src/check/select_target_instances.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.process.target_annotation import resolve_partition_pos


# ── Annotation-record accessors ──────────────────────────────────────────
def record_candidates(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return candidate instances from either new or legacy annotation shape."""
    candidates = record.get("candidates")
    if candidates is None:
        candidates = record.get("instances")
    items = list(candidates or [])
    items.sort(key=lambda x: int(x.get("n_pixels") or 0), reverse=True)
    return items


def candidate_viz_path(record: Dict[str, Any], target_id: int) -> Optional[str]:
    for cand in record_candidates(record):
        try:
            iid = int(cand.get("id"))
        except (TypeError, ValueError):
            continue
        if iid == target_id and cand.get("viz_path"):
            return str(cand["viz_path"])
    return None


def record_visibility_status(record: Dict[str, Any]) -> str:
    """Return the visibility tag for one record.

    Reads the new ``visibility`` field written by ``list_target_instances``.
    Falls back to a legacy ``uniqueness`` string (when the file was written
    by an older version where ``uniqueness`` conflated visibility) and
    finally to ``status`` for the very earliest format.
    """
    v = record.get("visibility")
    if v:
        return str(v)
    # Legacy: 3-state uniqueness used to encode visibility too.
    legacy_uniq = record.get("uniqueness")
    if isinstance(legacy_uniq, str) and legacy_uniq:
        if legacy_uniq in ("unique", "ambiguous"):
            return "visible"
        return legacy_uniq
    status = record.get("status")
    if status:
        return str(status)
    return "unknown"


def record_is_unique(record: Dict[str, Any]) -> Optional[bool]:
    """Return ``True`` / ``False`` when the record has a definitive
    uniqueness verdict, ``None`` when not visible / unknown.

    With the new schema ``uniqueness`` is a bool when visible. The
    legacy format stored the strings ``"unique"`` / ``"ambiguous"`` /
    ``"not_visible"`` here instead.
    """
    u = record.get("uniqueness")
    if isinstance(u, bool):
        return u
    if u == "unique":
        return True
    if u == "ambiguous":
        return False
    return None


# ── The selection rule ───────────────────────────────────────────────────
def select_target(
    record: Dict[str, Any],
    end_pos: Optional[np.ndarray] = None,
    instance_centers: Optional[Dict[int, np.ndarray]] = None,
    center_source: Optional[str] = None,
) -> Dict[str, Any]:
    instances = record_candidates(record)
    visibility = record_visibility_status(record)
    is_unique = record_is_unique(record)

    base = {
        "landmark": record.get("landmark"),
        "semantic_labels": record.get("semantic_labels", []),
        "matched_category": record.get("matched_category"),
        "matched_categories": record.get("matched_categories", []),
        # Carry both fields through so consolidate / inspection can show
        # the split picture without having to re-derive it.
        "visibility":       visibility,
        "uniqueness":       record.get("uniqueness"),
        "visibility_status": visibility,
        "candidates": instances,
        "target_instance_ids": [],
        "selection_distance": None,
        "candidate_distances": None,
        "center_source": center_source,
    }

    if visibility != "visible":
        return {**base, "status": f"visibility:{visibility}"}
    if not instances:
        return {**base, "status": "no_visible_instance"}
    if is_unique is True or len(instances) == 1:
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


# ── partition.json pose access ───────────────────────────────────────────
def partition_json_path(run_dir: Path, scan: str, ep_id: str) -> Optional[Path]:
    candidates = [
        run_dir / "partition" / scan / ep_id / "partition.json",
        run_dir / "partition" / ep_id / "partition.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_subpath_end_pos(
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


def load_subpath_positions(
    run_dir: Path,
    scan: str,
    ep_id: str,
    sub_idx: str,
    scan_db: Dict[str, np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, Optional[str]]:
    part_path = partition_json_path(run_dir, scan, ep_id)
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
    end_pos = resolve_subpath_end_pos(part_sub, virtual_nodes, scan_db)
    partition_pos = resolve_partition_pos(part_sub, virtual_nodes, scan_db)
    err = None
    if end_pos is None:
        err = "subpath_end_pos_unresolvable"
    elif partition_pos is None:
        err = "partition_pos_unresolvable"
    return end_pos, partition_pos, max(len(partition_subs), 1), err


# ── Instance-center sources ──────────────────────────────────────────────
def instance_centers_from_scene(checker) -> Dict[int, np.ndarray]:
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


def instance_centers_from_house(scenes_dir: str, scan: str) -> Dict[int, np.ndarray]:
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
