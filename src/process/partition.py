"""
Sub-path partition: split each sub-trajectory into a spatial portion
(accomplishing the turn / short forward) and a landmark portion
(walking to the landmark).

The partition point is a *continuous* position along the sub-path, reported
via a ceiling node index ``p`` and an interpolation factor ``alpha``:

    partition_pos = (1-alpha) * pos[p-1] + alpha * pos[p]

  • ``alpha == 1.0`` ⇒ partition lies exactly on MP3D node ``p``
  • ``alpha ∈ (0, 1)`` ⇒ partition is a *virtual* point on edge
    ``(p-1) → p``; its node id is encoded from the 3-D position as
    ``virt:{x:+.4f}_{y:+.4f}_{z:+.4f}`` (see :func:`_virt_id`).

The rule is driven by the *geometry* of the sub-path (not the instruction
text — rewritten instructions can be wrong about left/right/forward):
  • Find the first edge whose heading differs from the start heading by
    more than TURN_THRESH_DEG — classify as ``around`` if the magnitude
    also exceeds AROUND_THRESH_DEG, else ``left``/``right`` by sign.
  • If such a turn edge exists, advance forward from its destination
    until the accumulated distance reaches FORWARD_DISTANCE_M.
  • Otherwise treat the sub-path as pure forward and partition at the
    point whose cumulative distance from node 0 equals FORWARD_DISTANCE_M.

The ``spatial_instruction`` text is still parsed (``instruction_kind``)
and compared against the geometric ``kind`` (``direction_mismatch`` flag)
so dataset-level instruction errors surface in the output.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── Hyper-parameters (tunable; see configs/partition/partition.yaml) ──────
TURN_THRESH_DEG    = 45.0
AROUND_THRESH_DEG  = 120.0
FORWARD_DISTANCE_M = 1.0      # metres of forward motion defining spatial part


def heading_of_edge(p_from: np.ndarray, p_to: np.ndarray) -> float:
    """Clockwise heading from north (−Z) in radians, in Habitat frame."""
    dx = float(p_to[0] - p_from[0])
    dz = float(p_to[2] - p_from[2])
    return math.atan2(dx, -dz)


def signed_angle_diff(a: float, b: float) -> float:
    """Smallest signed difference a − b wrapped to [−π, π]."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


def _classify_spatial(spatial: str) -> str:
    """Parse the kind {around|left|right|forward} from the instruction text."""
    s = (spatial or "").lower()
    if "turn around" in s:
        return "around"
    if "turn left" in s:
        return "left"
    if "turn right" in s:
        return "right"
    if "go forward" in s:
        return "forward"
    return "forward"  # unknown → treat as forward


_KIND_TO_INSTRUCTION = {
    "forward": "Go forward.",
    "left":    "Turn left.",
    "right":   "Turn right.",
    "around":  "Turn around.",
}


def _classify_by_geometry(
    turn_deltas: Sequence[float],
    turn_thresh_deg: float,
    around_thresh_deg: float,
) -> Tuple[str, Optional[int]]:
    """Infer ``(kind, turn_k)`` from edge-heading deltas.

    Returns the first edge whose ``|Δ|`` exceeds ``turn_thresh_deg``, and
    the kind inferred from that edge's signed delta (positive ⇒ right,
    negative ⇒ left, ``|Δ| > around_thresh_deg`` ⇒ around).  If no such
    edge exists, returns ``("forward", None)``.
    """
    turn_rad   = math.radians(turn_thresh_deg)
    around_rad = math.radians(around_thresh_deg)
    for k, d in enumerate(turn_deltas):
        if abs(d) > turn_rad:
            if abs(d) > around_rad:
                return "around", k
            return ("right" if d > 0 else "left"), k
    return "forward", None


def _edge_lengths(positions: Sequence[np.ndarray]) -> List[float]:
    """Euclidean length of each consecutive edge (length K = len(positions)-1)."""
    return [
        float(np.linalg.norm(positions[k + 1] - positions[k]))
        for k in range(len(positions) - 1)
    ]


def _virt_id(pos: np.ndarray) -> str:
    """Encode a 3-D position as a virtual node id.  The ``virt:`` prefix
    and the ``_``/``:`` separators cannot appear in MP3D hex node ids, so
    these never collide with real nodes in ``scan_db``.
    """
    return "virt:{:+.4f}_{:+.4f}_{:+.4f}".format(
        float(pos[0]), float(pos[1]), float(pos[2]),
    )


def _forward_partition_point_from(
    start_idx: int,
    positions: Sequence[np.ndarray],
    edge_lengths: Sequence[float],
    distance_m: float,
) -> Tuple[int, float, np.ndarray]:
    """Walk ``distance_m`` forward from ``positions[start_idx]`` along the
    sub-path edges; return ``(partition_idx, alpha, pos)``:

      • ``partition_idx`` — smallest node index whose cumulative distance
        from ``start_idx`` reaches ``distance_m``.  Clamps to ``K`` when
        the remaining path is shorter than ``distance_m``.
      • ``alpha ∈ [0, 1]`` — fraction along edge ``(partition_idx-1) → partition_idx``.
        1.0 means partition sits exactly on node ``partition_idx``; values
        in (0, 1) mean partition is strictly between two nodes (virtual).
      • ``pos`` — interpolated 3-D position of the partition point.
    """
    K = len(edge_lengths)
    if start_idx >= K or distance_m <= 0.0:
        # No room to walk forward, or zero distance requested.
        return start_idx, 1.0, np.asarray(positions[start_idx], dtype=float).copy()

    acc = 0.0
    for k in range(start_idx, K):
        edge_len = edge_lengths[k]
        if acc + edge_len >= distance_m:
            remaining = distance_m - acc
            alpha = remaining / edge_len if edge_len > 0.0 else 1.0
            pos = positions[k] + alpha * (positions[k + 1] - positions[k])
            return k + 1, float(alpha), np.asarray(pos, dtype=float)
        acc += edge_len

    # Sub-path shorter than requested distance — clamp to last node.
    return K, 1.0, np.asarray(positions[K], dtype=float).copy()


def partition_subpath(
    positions:     Sequence[np.ndarray],
    start_heading: float,
    spatial:       str,
    turn_thresh_deg:    float = TURN_THRESH_DEG,
    around_thresh_deg:  float = AROUND_THRESH_DEG,
    forward_distance_m: float = FORWARD_DISTANCE_M,
) -> Dict[str, any]:
    """Return partition metadata for one sub-path.

    Returns
    -------
    dict with:
      partition_idx       int          — ceiling node index in [0, K]
      partition_alpha     float        — fraction along edge (p-1)→p, in [0, 1]
      partition_pos       list[float]  — interpolated 3-D position [x, y, z]
      turn_node_idx       int|None     — edge where the turn was detected
      kind                str          — geometric kind: left|right|around|forward
      instruction_kind    str          — kind parsed from the instruction text
      direction_mismatch  bool         — True when the two disagree
      geometric_spatial_instruction
                          str          — authoritative spatial text derived from
                                         geometry: "Go forward." / "Turn left."
                                         / "Turn right." / "Turn around."
      edge_headings       list[float]
      edge_lengths        list[float]  — Euclidean length (m) of each edge
      turn_deltas         list[float]  — signed diff of each edge heading vs start
      reason              str          — human-readable explanation
    """
    K = len(positions) - 1
    instruction_kind = _classify_spatial(spatial)

    if K <= 0:
        pos0 = np.asarray(positions[0], dtype=float).copy() if positions else np.zeros(3)
        return {
            "partition_idx":      0,
            "partition_alpha":    1.0,
            "partition_pos":      pos0.tolist(),
            "turn_node_idx":      None,
            "kind":               "forward",
            "instruction_kind":   instruction_kind,
            "direction_mismatch": False,  # no geometry to compare against
            "geometric_spatial_instruction": _KIND_TO_INSTRUCTION["forward"],
            "edge_headings":      [],
            "edge_lengths":       [],
            "turn_deltas":        [],
            "reason":             "degenerate sub-path (<2 nodes)",
        }

    edge_headings: List[float] = [
        heading_of_edge(positions[k], positions[k + 1]) for k in range(K)
    ]
    edge_lengths: List[float] = _edge_lengths(positions)
    turn_deltas: List[float] = [
        signed_angle_diff(h, start_heading) for h in edge_headings
    ]

    # Authoritative classification comes from geometry, not the instruction.
    kind, turn_k = _classify_by_geometry(
        turn_deltas, turn_thresh_deg, around_thresh_deg,
    )
    direction_mismatch = (kind != instruction_kind)

    if turn_k is None:
        # No turn in geometry → plain forward rule from node 0.
        start_idx = 0
        base_reason = f"no edge exceeded {turn_thresh_deg:.0f}°; " \
                      f"forward {forward_distance_m:.2f} m"
    else:
        # Walk forward from the turn edge's destination node.
        start_idx = turn_k + 1
        base_reason = (
            f"turn at edge {turn_k} "
            f"(Δ={math.degrees(turn_deltas[turn_k]):+.0f}° ⇒ {kind}); "
            f"+{forward_distance_m:.2f} m forward"
        )

    p, alpha, pos = _forward_partition_point_from(
        start_idx, positions, edge_lengths, forward_distance_m,
    )
    reason = f"{base_reason} → p={p}, α={alpha:.2f}"
    if direction_mismatch:
        reason += f"  ⚠ instruction_kind={instruction_kind}"

    turn_delta_deg = (
        float(math.degrees(turn_deltas[turn_k])) if turn_k is not None else None
    )

    return {
        "partition_idx":      p,
        "partition_alpha":    alpha,
        "partition_pos":      [float(pos[0]), float(pos[1]), float(pos[2])],
        "turn_node_idx":      turn_k,
        "turn_delta_deg":     turn_delta_deg,
        "kind":               kind,
        "instruction_kind":   instruction_kind,
        "direction_mismatch": direction_mismatch,
        "geometric_spatial_instruction": _KIND_TO_INSTRUCTION[kind],
        "edge_headings":      edge_headings,
        "edge_lengths":       edge_lengths,
        "turn_deltas":        turn_deltas,
        "reason":             reason,
    }


def partition_episode(
    episode,
    scan_db:   Dict[str, np.ndarray],
    rewritten: Optional[Dict] = None,
    turn_thresh_deg:    float = TURN_THRESH_DEG,
    around_thresh_deg:  float = AROUND_THRESH_DEG,
    forward_distance_m: float = FORWARD_DISTANCE_M,
) -> List[Dict]:
    """Run ``partition_subpath`` for every sub-path in one episode.

    Parameters
    ----------
    episode:
        A LandmarkRxREpisode.
    scan_db:
        ``{node_id → np.array([x, y, z])}`` for this episode's scan.
    rewritten:
        Optional parsed rewritten-episode dict (value under
        ``rewritten["episodes"][str(instruction_id)]``).  When given,
        spatial instructions come from the rewrite; otherwise all sub-paths
        are treated as "forward" (fallback).
    turn_thresh_deg, around_thresh_deg, forward_distance_m:
        Hyper-parameters forwarded to :func:`partition_subpath`.
    """
    results: List[Dict] = []

    rewrite_by_idx: Dict[int, Dict] = {}
    if rewritten is not None:
        for sub in rewritten.get("sub_paths", []):
            rewrite_by_idx[int(sub["sub_idx"])] = sub

    for i, sub_nodes in enumerate(episode.sub_paths):
        if len(sub_nodes) < 2:
            results.append({
                "sub_idx":  i,
                "error":    "sub-path has < 2 nodes",
            })
            continue

        try:
            positions = [scan_db[n] for n in sub_nodes]
        except KeyError as exc:
            results.append({
                "sub_idx": i,
                "error":   f"node not in DB: {exc}",
            })
            continue

        start_heading = (
            episode.headings[i]
            if episode.headings and i < len(episode.headings)
            else episode.heading
        )

        rw = rewrite_by_idx.get(i, {})
        spatial = rw.get("spatial_instruction", "Go forward.")

        out = partition_subpath(
            positions, start_heading, spatial,
            turn_thresh_deg=turn_thresh_deg,
            around_thresh_deg=around_thresh_deg,
            forward_distance_m=forward_distance_m,
        )

        # Build the two path segments; they share the partition point
        # (a real MP3D node when alpha==1.0, else a virtual "virt:..." node).
        p     = out["partition_idx"]
        alpha = out["partition_alpha"]
        if alpha >= 1.0 - 1e-9 or p == 0:
            out["partition_on_edge"] = None
            spatial_seg  = list(sub_nodes[: p + 1])
            landmark_seg = list(sub_nodes[p:])
        else:
            vid = _virt_id(np.asarray(out["partition_pos"]))
            out["partition_on_edge"] = [sub_nodes[p - 1], sub_nodes[p]]
            spatial_seg  = list(sub_nodes[:p]) + [vid]
            landmark_seg = [vid] + list(sub_nodes[p:])

        out.update({
            "sub_idx":             i,
            "sub_path_nodes":      list(sub_nodes),
            "spatial_path":        spatial_seg,
            "landmark_path":       landmark_seg,
            "start_heading":       start_heading,
            "spatial_instruction": spatial,
            "landmark":            rw.get("landmark", ""),
            "landmark_instruction": rw.get("landmark_instruction", ""),
        })
        results.append(out)

    return results
