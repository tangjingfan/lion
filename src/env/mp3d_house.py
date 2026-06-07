"""MP3D ``.house`` file parsing helpers.

The ``.house`` sidecar of every MP3D scan describes the scene's semantic
annotation in plain text rows; the ones used here:

  * ``C`` rows define ``category_index -> mpcat40_name``.
  * ``O`` rows define ``instance_id, category_index, x, y, z`` in MP3D
    coordinates (x=right, y=forward, z=up).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.env.connectivity import _mp3d_to_habitat


def instance_meta_from_house(scenes_dir: str, scan: str) -> Dict[int, Dict]:
    """Map ``instance_id -> {category, center}`` from a scan's .house.

    Habitat's ``sim.semantic_annotations()`` is unreliable through the
    LION ``VisibilityChecker`` wrapper (returns empty objects list in our
    setup), so we read the same data straight from ``.house``.

    Centers are converted to Habitat coordinates via
    :func:`src.env.connectivity._mp3d_to_habitat` so they're comparable
    with node positions returned by ``load_connectivity``.
    """
    house_path = Path(scenes_dir) / "mp3d" / scan / f"{scan}.house"
    if not house_path.exists():
        return {}

    cat_index_to_mpcat40: Dict[int, str] = {}
    out: Dict[int, Dict] = {}
    with open(house_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "C" and len(parts) >= 6:
                try:
                    cat_index = int(parts[1])
                except ValueError:
                    continue
                name = parts[5].replace("#", " ").strip().lower()
                if name:
                    cat_index_to_mpcat40[cat_index] = name
            elif parts[0] == "O" and len(parts) >= 7:
                try:
                    iid = int(parts[1])
                    cat_index = int(parts[3])
                    x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
                except ValueError:
                    continue
                out[iid] = {
                    "_cat_index": cat_index,
                    "center":     _mp3d_to_habitat(x, y, z),
                }
    for iid, meta in out.items():
        meta["category"] = cat_index_to_mpcat40.get(meta.pop("_cat_index"), "")
    return out


def parse_house_categories(house_path: Path) -> Dict[int, Dict[str, Any]]:
    """Map ``category_index -> {raw_name, mpcat40_name, ...}`` from C rows."""
    categories: Dict[int, Dict[str, Any]] = {}
    with open(house_path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] != "C" or len(parts) < 6:
                continue
            try:
                cat_index = int(parts[1])
                mpcat40_index = int(parts[4])
            except ValueError:
                continue
            categories[cat_index] = {
                "category_index": cat_index,
                "raw_category_id": int(parts[2]) if parts[2].isdigit() else parts[2],
                "raw_name": parts[3].replace("#", " "),
                "mpcat40_index": mpcat40_index,
                "mpcat40_name": parts[5].replace("#", " "),
            }
    return categories


def parse_house_object(
    scenes_dir: str,
    scan: str,
    instance_id: int,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Look up one O-row instance by id; also return the 10 nearest ids.

    Returns ``(match_or_None, nearby)`` where ``nearby`` is sorted by
    instance-id distance (a debugging aid when the queried id is absent).
    """
    house_path = Path(scenes_dir) / "mp3d" / scan / f"{scan}.house"
    if not house_path.exists():
        return None, []

    categories = parse_house_categories(house_path)
    nearby: List[Dict[str, Any]] = []
    match: Optional[Dict[str, Any]] = None

    with open(house_path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] != "O" or len(parts) < 5:
                continue
            try:
                obj_id = int(parts[1])
                region_id = int(parts[2])
                cat_index = int(parts[3])
            except ValueError:
                continue

            cat = categories.get(cat_index, {})
            entry = {
                "instance_id": obj_id,
                "region_id": region_id,
                "category": {
                    "index": cat_index,
                    "raw_name": cat.get("raw_name"),
                    "mpcat40_name": cat.get("mpcat40_name"),
                    "mpcat40_index": cat.get("mpcat40_index"),
                },
                # In MP3D .house object rows, fields 4:7 are the object center.
                "center": [float(x) for x in parts[4:7]],
                "raw_fields": parts,
            }
            nearby.append({
                "instance_id": obj_id,
                "category": cat.get("mpcat40_name") or cat.get("raw_name"),
                "distance": abs(obj_id - instance_id),
            })
            if obj_id == instance_id:
                match = entry

    nearby.sort(key=lambda e: (e["distance"], e["instance_id"]))
    return match, nearby[:10]
