"""Shared Habitat-Lab / Habitat-sim setup helpers.

Both :class:`src.env.habitat_env.HabitatEnv` and
:class:`src.process.visibility.VisibilityChecker` create the same
``rgbds_agent`` simulator from ``configs/habitat/rgbds_sim.yaml`` and
translate its raw semantic instance ids to MP40 categories the same way.
These helpers factor out that shared setup so the two stay in lock-step.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def ensure_habitat_lab_importable() -> None:
    """Add the vendored habitat-lab/ to sys.path if not already importable."""
    try:
        import habitat  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    habitat_lab_src = (
        Path(__file__).resolve().parents[2]
        / "external" / "habitat-lab" / "habitat-lab"
    )
    if habitat_lab_src.exists():
        sys.path.insert(0, str(habitat_lab_src))


def configure_sensor(
    sensor_cfg: Any,
    overrides: Dict[str, Any],
    sensor_height: float,
) -> None:
    """Patch a Habitat-Lab sensor config in place: position + size/fov fields."""
    sensor_cfg.position = overrides.get("position", [0.0, sensor_height, 0.0])
    for key in (
        "width",
        "height",
        "hfov",
        "orientation",
        "min_depth",
        "max_depth",
        "normalize_depth",
        "sensor_subtype",
    ):
        if key in overrides and hasattr(sensor_cfg, key):
            value = overrides[key]
            if key in {"width", "height", "hfov"}:
                value = int(round(value))
            setattr(sensor_cfg, key, value)


def build_semantic_tables(
    scene: Any,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, int]]:
    """Build instance-id → MP40 (category_id, category_name) lookup tables.

    ``scene`` is the object returned by ``sim.semantic_annotations()``.
    MP3D instance ids are not always dense, so the arrays are sized to
    ``max_id + 1`` with gaps left as -1 / "".

    Returns ``(id_map, name_map, name_to_cat_id)``:
      * ``id_map``  — int32 array, instance id → category id (-1 = unknown)
      * ``name_map``— object array, instance id → category name ("" = unknown)
      * ``name_to_cat_id`` — lower-cased category name → category id

    ``id_map`` / ``name_map`` are ``None`` (and ``name_to_cat_id`` empty) when
    *scene* has no usable annotations.
    """
    if scene is None or not getattr(scene, "objects", None):
        return None, None, {}

    pairs = []
    name_to_cat_id: Dict[str, int] = {}
    for obj in scene.objects:
        if obj is None:
            continue
        try:
            inst_id = int(obj.id.split("_")[-1])
        except (AttributeError, ValueError):
            continue
        cat = obj.category
        cat_id = cat.index()
        cat_name = cat.name()
        pairs.append((inst_id, cat_id, cat_name))
        if cat_name:
            name_to_cat_id[cat_name.lower()] = cat_id

    if not pairs:
        return None, None, {}

    max_id = max(p[0] for p in pairs)
    id_map = np.full(max_id + 1, -1, dtype=np.int32)
    name_map = np.full(max_id + 1, "", dtype=object)
    for inst_id, cat_id, cat_name in pairs:
        id_map[inst_id] = cat_id
        name_map[inst_id] = cat_name

    return id_map, name_map, name_to_cat_id
