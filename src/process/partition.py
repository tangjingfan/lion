"""
Sub-path partition: split each sub-trajectory into a spatial portion
(accomplishing the turn / short forward) and a landmark portion
(walking to the landmark).

The partition point is a node index ``p`` within the sub-path, such that:
    spatial_part  = nodes[0 : p+1]
    landmark_part = nodes[p   : K+1]
(they share the boundary node ``p``).

The rule is driven by the rewritten ``spatial_instruction``:
  • "Turn left" / "Turn right" — find the first edge whose heading differs
    from the start heading by more than TURN_THRESH_DEG, mark that as the
    turn node, then advance forward until the accumulated distance
    (starting from the turn edge's destination) reaches FORWARD_DISTANCE_M.
  • "Turn around"              — same but with AROUND_THRESH_DEG.
  • "Go forward"               — partition at the first node whose
    cumulative distance from node 0 reaches FORWARD_DISTANCE_M.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

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


def _edge_lengths(positions: Sequence[np.ndarray]) -> List[float]:
    """Euclidean length of each consecutive edge (length K = len(positions)-1)."""
    return [
        float(np.linalg.norm(positions[k + 1] - positions[k]))
        for k in range(len(positions) - 1)
    ]


def _forward_partition_from(
    start_idx: int,
    edge_lengths: Sequence[float],
    distance_m: float,
) -> int:
    """Smallest index ``p ≥ start_idx`` such that cumulative edge length
    from ``start_idx`` to ``p`` reaches ``distance_m``.  Clamps to K when
    the remaining sub-path is shorter than ``distance_m``.
    """
    K = len(edge_lengths)
    if start_idx >= K:
        return K
    acc = 0.0
    for k in range(start_idx, K):
        acc += edge_lengths[k]
        if acc >= distance_m:
            return k + 1
    return K


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
      partition_idx  int       — node index in [0, K]
      turn_node_idx  int|None  — where the turn was detected (None for "forward")
      kind           str       — "left" | "right" | "around" | "forward"
      edge_headings  list[float]
      edge_lengths   list[float] — Euclidean length (m) of each edge
      turn_deltas    list[float] — signed diff of each edge heading vs start
      reason         str        — human-readable explanation
    """
    K = len(positions) - 1
    kind = _classify_spatial(spatial)

    if K <= 0:
        return {
            "partition_idx": 0,
            "turn_node_idx": None,
            "kind":          kind,
            "edge_headings": [],
            "edge_lengths":  [],
            "turn_deltas":   [],
            "reason":        "degenerate sub-path (<2 nodes)",
        }

    edge_headings: List[float] = [
        heading_of_edge(positions[k], positions[k + 1]) for k in range(K)
    ]
    edge_lengths: List[float] = _edge_lengths(positions)
    turn_deltas: List[float] = [
        signed_angle_diff(h, start_heading) for h in edge_headings
    ]

    if kind == "forward":
        p = _forward_partition_from(0, edge_lengths, forward_distance_m)
        return {
            "partition_idx": p,
            "turn_node_idx": None,
            "kind":          kind,
            "edge_headings": edge_headings,
            "edge_lengths":  edge_lengths,
            "turn_deltas":   turn_deltas,
            "reason":        f"forward {forward_distance_m:.2f} m → p={p}",
        }

    thresh_rad = math.radians(
        around_thresh_deg if kind == "around" else turn_thresh_deg
    )
    turn_k: Optional[int] = None
    for k, d in enumerate(turn_deltas):
        if abs(d) > thresh_rad:
            turn_k = k
            break

    if turn_k is None:
        # No clear turn detected; fall back to walking forward from start.
        p = _forward_partition_from(0, edge_lengths, forward_distance_m)
        return {
            "partition_idx": p,
            "turn_node_idx": None,
            "kind":          kind,
            "edge_headings": edge_headings,
            "edge_lengths":  edge_lengths,
            "turn_deltas":   turn_deltas,
            "reason":        f"no edge exceeded {math.degrees(thresh_rad):.0f}°; "
                             f"fell back to forward {forward_distance_m:.2f} m → p={p}",
        }

    # Start accumulating forward distance from the turn edge's destination node.
    p = _forward_partition_from(turn_k + 1, edge_lengths, forward_distance_m)
    return {
        "partition_idx": p,
        "turn_node_idx": turn_k,
        "kind":          kind,
        "edge_headings": edge_headings,
        "edge_lengths":  edge_lengths,
        "turn_deltas":   turn_deltas,
        "reason":        f"turn at edge {turn_k} "
                         f"(Δ={math.degrees(turn_deltas[turn_k]):+.0f}°); "
                         f"+{forward_distance_m:.2f} m forward → p={p}",
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
        out.update({
            "sub_idx":             i,
            "sub_path_nodes":      list(sub_nodes),
            "start_heading":       start_heading,
            "spatial_instruction": spatial,
            "landmark":            rw.get("landmark", ""),
            "landmark_instruction": rw.get("landmark_instruction", ""),
        })
        results.append(out)

    return results
