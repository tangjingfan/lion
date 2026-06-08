"""Pure pixel/geometry helpers for the perturb-visibility robustness check.

Extracted from ``src/check/perturb_visibility.py`` so the decision logic can be
unit-tested without spinning up Habitat (see ``tests/test_visibility_analysis.py``).
Everything here operates on plain numpy arrays / dicts:

  * ``perturbed_positions``   — sample a circle of poses around a start node
  * ``check_targets_visible`` — count target-instance pixels in a semantic panorama
  * ``target_required_pixels``— per-target pixel thresholds (abs + relative)
  * ``category_for_instance`` / ``target_categories`` — instance id → MP40 category
  * ``check_same_category_instances`` — count visible non-target same-category instances
  * ``overlay_target``        — blend a red highlight over the target mask

The semantic name lookup is passed in as ``name_map`` (the per-scene
instance-id → category-name array produced by ``build_semantic_tables``),
so callers don't need a live ``VisibilityChecker``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np


def perturbed_positions(
    start_pos: np.ndarray, radius: float, n: int,
) -> List[Dict[str, Any]]:
    """``n`` points on a circle of radius ``radius`` in the X-Z plane.

    Habitat is Y-up, so the floor lies in X-Z; perturbations stay
    coplanar with the start node.  Angle 0° points toward +X.
    """
    out: List[Dict[str, Any]] = []
    for k in range(n):
        theta = 2.0 * math.pi * k / n
        dx = radius * math.cos(theta)
        dz = radius * math.sin(theta)
        pos = np.asarray(
            [start_pos[0] + dx, start_pos[1], start_pos[2] + dz],
            dtype=np.float32,
        )
        out.append({
            "k":          k,
            "angle_deg":  round(math.degrees(theta), 1),
            "raw_pos":    [float(pos[0]), float(pos[1]), float(pos[2])],
            "raw_pos_np": pos,
        })
    return out


def check_targets_visible(
    sem:          np.ndarray,
    target_ids:   List[int],
    min_pixel:    int,
    required_pixels: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    """Count pixels per target instance id in ``sem``.

    Returns ``{"hits": [...]}`` with one entry per target id whose pixel
    count meets the target-specific threshold, and aggregate counts.
    """
    if not target_ids:
        return {"hits": [], "n_visible_targets": 0, "any_visible": False,
                "max_pixels": 0, "target_mask": None, "required_pixels": {}}
    required_pixels = required_pixels or {}
    hits: List[Dict[str, Any]] = []
    max_pixels = 0
    target_mask = np.zeros_like(sem, dtype=bool)
    for tid in target_ids:
        threshold = int(required_pixels.get(int(tid), min_pixel))
        m = (sem == int(tid))
        n = int(m.sum())
        if n > max_pixels:
            max_pixels = n
        if n >= threshold:
            hits.append({
                "id": int(tid),
                "n_pixels": n,
                "required_pixels": threshold,
            })
            target_mask |= m
    return {
        "hits":              hits,
        "n_visible_targets": len(hits),
        "any_visible":       bool(hits),
        "max_pixels":        max_pixels,
        "target_mask":       target_mask,
        "required_pixels":   {str(k): int(v) for k, v in required_pixels.items()},
    }


def target_required_pixels(
    entry: Dict[str, Any],
    target_ids: List[int],
    min_pixel: int,
    original_fraction: float,
) -> Dict[int, int]:
    """Per-target threshold: max(min_pixel, fraction of original pixels)."""
    by_id = {
        int(c.get("id")): int(c.get("n_pixels") or 0)
        for c in (entry.get("candidates") or [])
        if c.get("id") is not None
    }
    out: Dict[int, int] = {}
    for tid in target_ids:
        original = by_id.get(int(tid), 0)
        rel = int(math.ceil(float(original) * original_fraction)) if original > 0 else 0
        out[int(tid)] = max(int(min_pixel), rel)
    return out


def category_for_instance(
    name_map: Optional[np.ndarray],
    instance_id: int,
) -> Optional[str]:
    """Look up an instance id's MP40 category name in ``name_map`` (or None)."""
    if name_map is None:
        return None
    if instance_id < 0 or instance_id >= len(name_map):
        return None
    name = str(name_map[int(instance_id)] or "").strip()
    return name or None


def target_categories(
    entry: Dict[str, Any],
    target_ids: List[int],
    name_map: Optional[np.ndarray],
) -> List[str]:
    """Resolve target instance ids to semantic category names.

    Prefers the per-candidate ``category`` recorded in ``entry``, falling
    back to the scene ``name_map`` and finally to the entry's matched
    category fields.  De-duplicated, case-insensitive, order-preserving.
    """
    cats: List[str] = []

    def add(cat: Optional[str]) -> None:
        s = (cat or "").strip()
        if s and s.lower() not in {c.lower() for c in cats}:
            cats.append(s)

    candidates = entry.get("candidates") or []
    for tid in target_ids:
        for cand in candidates:
            if int(cand.get("id") or -1) == int(tid):
                add(cand.get("category"))
                break
        add(category_for_instance(name_map, int(tid)))

    if not cats:
        add(entry.get("matched_category"))
        for cat in entry.get("matched_categories") or []:
            add(cat)
    return cats


def check_same_category_instances(
    sem: np.ndarray,
    target_ids: List[int],
    categories: List[str],
    name_map: Optional[np.ndarray],
    min_pixel: int,
) -> Dict[str, Any]:
    """Find visible non-target instances sharing the target category."""
    target_set = {int(x) for x in target_ids}
    cat_set = {(c or "").strip().lower() for c in categories if c}
    if sem is None or not cat_set:
        return {
            "hits": [],
            "n_other_same_category": 0,
            "category_unique": None,
            "same_category_mask": None,
        }

    hits: List[Dict[str, Any]] = []
    same_category_mask = np.zeros_like(sem, dtype=bool)
    instance_ids, counts = np.unique(sem, return_counts=True)
    for iid_raw, n_raw in zip(instance_ids, counts):
        iid = int(iid_raw)
        if iid in target_set:
            continue
        n = int(n_raw)
        if n < min_pixel:
            continue
        cat = category_for_instance(name_map, iid)
        if (cat or "").strip().lower() not in cat_set:
            continue
        hits.append({"id": iid, "category": cat, "n_pixels": n})
        same_category_mask |= (sem == iid)

    hits.sort(key=lambda x: int(x.get("n_pixels") or 0), reverse=True)
    return {
        "hits": hits,
        "n_other_same_category": len(hits),
        "category_unique": len(hits) == 0,
        "same_category_mask": same_category_mask,
    }


def overlay_target(rgb: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Blend a red highlight over pixels where ``mask`` is True."""
    out = rgb.copy()
    if mask is None or not mask.any():
        return out
    red = np.array([240, 70, 70], dtype=np.uint8)
    blend = (0.55 * out[mask].astype(np.float32)
             + 0.45 * red.astype(np.float32)).astype(np.uint8)
    out[mask] = blend
    return out
