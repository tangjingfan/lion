"""MP3D ``.house`` file parsing helpers.

The ``.house`` sidecar of every MP3D scan describes the scene's semantic
annotation in plain text rows; the ones used here:

  * ``C`` rows define ``category_index -> mpcat40_name``.
  * ``O`` rows define ``instance_id, category_index, x, y, z`` in MP3D
    coordinates (x=right, y=forward, z=up).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

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
