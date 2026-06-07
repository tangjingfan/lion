"""Per-sub-path target annotation primitives (pipeline step 07).

The pure decision pieces of ``list_target_instances``:

  * resolving the partition (turn) pose from a partition.json record
  * resolving a sub-path's landmark mentions to MP40 candidate labels
  * the visibility / uniqueness classification

The semantic rendering itself lives in
:class:`src.process.visibility.VisibilityChecker`; the CLI orchestration
in ``src/check/list_target_instances.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.process.landmark_remap import lookup_mention_labels


def resolve_partition_pos(
    partition_sub:  Dict,
    virtual_nodes:  Dict[str, List[float]],
    scan_db:        Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    """Resolve the 3-D position of the partition (turn) point.

    Uses ``spatial_path[-1]`` — the boundary node between this sub-path
    and the next.  This is often a virtual ``virt:...`` id (resolved
    via ``virtual_nodes``) when the partition was inserted in the
    middle of an edge rather than at an MP3D node.
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


def semantic_labels_for_sub(
    rewrite_sub:      Dict,
    landmark_mapping: Dict,
    scan:             str,
) -> List[str]:
    """Resolve a sub-path's components to candidate MP40 labels.

    The per-scan ``landmark_mapping`` (refined in step 06) is the sole
    source of truth — no fallback to the rewriter's per-component
    ``semantic_label``. That fallback used to let coarse buckets
    (``appliances`` / ``furniture`` / ``objects`` / ``lighting``) sneak
    back in for fine-grained mentions like ``fridge`` / ``stove``, even
    after step 06 dropped them. If the mapping returns ``[]`` the
    landmark is intentionally treated as unmapped at this stage and the
    downstream visibility check records ``no_match``.

    Returns a deduplicated, ``"unknown"``-stripped list.
    """
    labels: List[str] = []

    def _add(label: str) -> None:
        s = (label or "").strip()
        if s and s.lower() not in ("unknown", "") and s not in labels:
            labels.append(s)

    for comp in rewrite_sub.get("components") or []:
        mention = (comp.get("original_mention") or "").strip().lower()
        for label in lookup_mention_labels(landmark_mapping, scan, mention):
            _add(label)

    return labels


def classify_visibility(
    pos:               Optional[np.ndarray],
    visibility_result: Optional[Dict[str, Any]],
) -> Tuple[str, Any]:
    """Split visibility from uniqueness.

    Returns ``(visibility, uniqueness)``:

    - ``visibility`` (str): one of ``"visible"``, ``"not_visible"``,
      ``"no_match"``, or ``"partition_pos_unresolvable"``. Anything other
      than ``"visible"`` means the landmark isn't usable at this pose.
    - ``uniqueness``: ``True`` (exactly 1 visible instance), ``False``
      (multiple visible instances), or the string ``"not_visible"`` when
      ``visibility != "visible"``. Using the same string as the visibility
      tag makes it obvious downstream that there's nothing to pick.
    """
    if pos is None:
        return "partition_pos_unresolvable", "not_visible"
    if visibility_result is None or not (visibility_result.get("matched_categories") or []):
        return "no_match", "not_visible"
    n = int(visibility_result.get("n_instances") or 0)
    if n == 0:
        return "not_visible", "not_visible"
    return "visible", (True if n == 1 else False)
