"""
Sub-path partition: split each sub-trajectory into a spatial portion
(the first short stretch from the start) and a landmark portion (the
remainder, walking to the landmark).

The partition point is a *continuous* position along the sub-path, reported
via a ceiling node index ``p`` and an interpolation factor ``alpha``:

    partition_pos = (1-alpha) * pos[p-1] + alpha * pos[p]

  • ``alpha == 1.0`` ⇒ partition lies exactly on MP3D node ``p``
  • ``alpha ∈ (0, 1)`` ⇒ partition is a *virtual* point on edge
    ``(p-1) → p``; its node id is encoded from the 3-D position as
    ``virt:{x:+.4f}_{y:+.4f}_{z:+.4f}`` (see :func:`_virt_id`).

Placement rule
--------------
When rollout frame metadata is available, use the already segmented rollout
steps.  For a turn instruction, consume the initial turn actions, record how
many degrees the agent turned, then cut after ``FORWARD_DISTANCE_M`` metres of
forward motion.  For a forward instruction, cut after the same distance from
the start of the rollout segment.  If rollout metadata is unavailable, fall
back to walking ``FORWARD_DISTANCE_M`` metres along the reference sub-path.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── Hyper-parameters (tunable; see configs/partition/partition.yaml) ──────
TURN_THRESH_DEG    = 45.0
AROUND_THRESH_DEG  = 120.0
FORWARD_DISTANCE_M = 0.5      # metres of forward motion defining spatial part
MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3


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


def _classify_turn_delta(
    delta_deg: float,
    turn_thresh_deg: float,
    around_thresh_deg: float,
) -> str:
    if abs(delta_deg) >= around_thresh_deg:
        return "around"
    if abs(delta_deg) >= turn_thresh_deg:
        return "right" if delta_deg > 0 else "left"
    return "forward"


def _classify_rollout_turn_delta(delta_deg: float, around_thresh_deg: float) -> str:
    """Rollout actions are explicit turns; classify by sign, not min angle."""
    if abs(delta_deg) >= around_thresh_deg:
        return "around"
    return "right" if delta_deg > 0 else "left"


def _classify_reference_partition_kind(
    turn_deltas: Sequence[float],
    partition_idx: int,
    turn_thresh_deg: float,
    around_thresh_deg: float,
) -> Tuple[str, Optional[int]]:
    """Classify only the geometry up to the partition point.

    The full sub-path often bends later while walking to the landmark.  That
    later bend should not label the spatial partition itself.
    """
    if not turn_deltas:
        return "forward", None
    max_edge = max(0, min(partition_idx - 1, len(turn_deltas) - 1))
    best_k = max(range(max_edge + 1), key=lambda k: abs(turn_deltas[k]))
    best_deg = math.degrees(turn_deltas[best_k])
    return (
        _classify_turn_delta(best_deg, turn_thresh_deg, around_thresh_deg),
        best_k,
    )


def _edge_lengths(positions: Sequence[np.ndarray]) -> List[float]:
    """Euclidean length of each consecutive edge (length K = len(positions)-1)."""
    return [
        float(np.linalg.norm(positions[k + 1] - positions[k]))
        for k in range(len(positions) - 1)
    ]


def _as_pos(record: Dict) -> Optional[np.ndarray]:
    pos = record.get("position")
    if pos is None:
        return None
    return np.asarray(pos, dtype=float)


def _project_to_polyline(
    pos: np.ndarray,
    positions: Sequence[np.ndarray],
) -> Tuple[int, float]:
    """Return ``(ceil_node_idx, alpha)`` for the nearest reference edge."""
    K = len(positions) - 1
    if K <= 0:
        return 0, 1.0

    best_k = 0
    best_alpha = 0.0
    best_dist = float("inf")
    for k in range(K):
        a = np.asarray(positions[k], dtype=float)
        b = np.asarray(positions[k + 1], dtype=float)
        ab = b - a
        denom = float(np.dot(ab, ab))
        alpha = 0.0 if denom <= 0.0 else float(np.dot(pos - a, ab) / denom)
        alpha = max(0.0, min(1.0, alpha))
        proj = a + alpha * ab
        dist = float(np.linalg.norm(pos - proj))
        if dist < best_dist:
            best_dist = dist
            best_k = k
            best_alpha = alpha
    return best_k + 1, best_alpha


def _cut_after_distance(
    records: Sequence[Dict],
    start_i: int,
    distance_m: float,
) -> Tuple[np.ndarray, float, Optional[int], str]:
    """Interpolate the rollout position after ``distance_m`` travelled."""
    if not records:
        return np.zeros(3), 0.0, None, "no rollout frames"

    start_i = max(0, min(start_i, len(records) - 1))
    prev = _as_pos(records[start_i])
    if prev is None:
        prev = np.zeros(3)
    if distance_m <= 0.0:
        return prev.copy(), 0.0, records[start_i].get("step"), "zero threshold"

    walked = 0.0
    for i in range(start_i + 1, len(records)):
        cur = _as_pos(records[i])
        if cur is None:
            continue
        seg = float(np.linalg.norm(cur - prev))
        if walked + seg >= distance_m and seg > 0.0:
            remain = distance_m - walked
            alpha = remain / seg
            cut = prev + alpha * (cur - prev)
            return (
                np.asarray(cut, dtype=float),
                float(distance_m),
                records[i].get("step"),
                f"rollout forward {distance_m:.2f} m",
            )
        walked += seg
        prev = cur

    return (
        np.asarray(prev, dtype=float),
        float(walked),
        records[-1].get("step"),
        f"rollout segment shorter than {distance_m:.2f} m; clamped",
    )


def _partition_point_from_rollout(
    records: Sequence[Dict],
    instruction_kind: str,
    distance_m: float,
    turn_thresh_deg: float,
    around_thresh_deg: float,
) -> Optional[Dict]:
    """Choose a partition point using action-level rollout frames."""
    records = [r for r in records if _as_pos(r) is not None]
    if not records:
        return None

    start_pos = _as_pos(records[0])
    start_heading = records[0].get("heading")
    turn_deg = 0.0
    turn_steps = 0
    turn_threshold_i: Optional[int] = None
    prev_heading = start_heading

    for i in range(1, len(records)):
        action = records[i].get("action")
        heading = records[i].get("heading")
        if action in (TURN_LEFT, TURN_RIGHT):
            turn_steps += 1
            if prev_heading is not None and heading is not None:
                turn_deg += math.degrees(signed_angle_diff(float(heading), float(prev_heading)))
            prev_heading = heading
            if abs(turn_deg) >= turn_thresh_deg:
                turn_threshold_i = i
                break
            continue
        prev_heading = heading if heading is not None else prev_heading

    if turn_threshold_i is not None:
        walk_start_i = turn_threshold_i
        cut_pos, walked, cut_step, cut_reason = _cut_after_distance(
            records, walk_start_i, distance_m,
        )
        reason = (
            f"rollout cumulative turn {turn_deg:+.1f} deg crossed "
            f"{turn_thresh_deg:.1f} deg at step {records[walk_start_i].get('step')} "
            f"over {turn_steps} turn step(s), "
            f"then {cut_reason}"
        )
        rollout_kind = _classify_rollout_turn_delta(turn_deg, around_thresh_deg)
    else:
        cut_pos, walked, cut_step, cut_reason = _cut_after_distance(
            records, 0, distance_m,
        )
        reason = (
            f"rollout cumulative turn {turn_deg:+.1f} deg did not cross "
            f"{turn_thresh_deg:.1f} deg; {cut_reason}"
        )
        walk_start_i = 0
        rollout_kind = "forward"

    total_walk = 0.0
    prev = start_pos
    for rec in records[1:]:
        cur = _as_pos(rec)
        if cur is None:
            continue
        total_walk += float(np.linalg.norm(cur - prev))
        prev = cur

    return {
        "partition_pos": [float(cut_pos[0]), float(cut_pos[1]), float(cut_pos[2])],
        "partition_source": "rollout_frames",
        "partition_virtual": True,
        "rollout_turn_deg": float(turn_deg),
        "rollout_turn_steps": int(turn_steps),
        "rollout_turn_threshold_deg": float(turn_thresh_deg),
        "rollout_turn_threshold_step": (
            int(records[turn_threshold_i].get("step"))
            if turn_threshold_i is not None and records[turn_threshold_i].get("step") is not None
            else None
        ),
        "rollout_kind": rollout_kind,
        "rollout_walk_m": float(walked),
        "rollout_segment_walk_m": float(total_walk),
        "rollout_cut_step": int(cut_step) if cut_step is not None else None,
        "rollout_walk_start_step": records[walk_start_i].get("step"),
        "reason": reason,
    }


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
    rollout_frames: Optional[Sequence[Dict]] = None,
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

    rollout_partition = _partition_point_from_rollout(
        rollout_frames or [],
        instruction_kind,
        forward_distance_m,
        turn_thresh_deg,
        around_thresh_deg,
    )
    if rollout_partition is not None:
        pos = np.asarray(rollout_partition["partition_pos"], dtype=float)
        p, alpha = _project_to_polyline(pos, positions)
        base_reason = rollout_partition["reason"]
        partition_source = "rollout_frames"
        partition_virtual = True
        reference_kind, turn_k = _classify_reference_partition_kind(
            turn_deltas, p, turn_thresh_deg, around_thresh_deg,
        )
        kind = rollout_partition.get("rollout_kind") or reference_kind
    else:
        p, alpha, pos = _forward_partition_point_from(
            0, positions, edge_lengths, forward_distance_m,
        )
        reference_kind, turn_k = _classify_reference_partition_kind(
            turn_deltas, p, turn_thresh_deg, around_thresh_deg,
        )
        total_len = sum(edge_lengths)
        if forward_distance_m >= total_len:
            base_reason = (
                f"reference forward {forward_distance_m:.2f} m exceeds path length "
                f"{total_len:.2f} m; clamped to end"
            )
        else:
            base_reason = f"reference forward {forward_distance_m:.2f} m from start"
        partition_source = "reference_path"
        partition_virtual = False
        kind = reference_kind

    direction_mismatch = (kind != instruction_kind)

    reason = f"{base_reason} → p={p}, α={alpha:.2f}"
    if direction_mismatch:
        reason += (
            f"  instruction_kind={instruction_kind}, partition_kind={kind}"
        )

    turn_delta_deg = (
        float(math.degrees(turn_deltas[turn_k])) if turn_k is not None else None
    )

    out = {
        "partition_idx":      p,
        "partition_alpha":    alpha,
        "partition_pos":      [float(pos[0]), float(pos[1]), float(pos[2])],
        "partition_source":   partition_source,
        "partition_virtual":  partition_virtual,
        "forward_distance_m": float(forward_distance_m),
        "turn_node_idx":      turn_k,
        "turn_delta_deg":     turn_delta_deg,
        "kind":               kind,
        "reference_kind":     reference_kind,
        "instruction_kind":   instruction_kind,
        "direction_mismatch": direction_mismatch,
        "geometric_spatial_instruction": _KIND_TO_INSTRUCTION[kind],
        "edge_headings":      edge_headings,
        "edge_lengths":       edge_lengths,
        "turn_deltas":        turn_deltas,
        "reason":             reason,
    }
    if rollout_partition is not None:
        for key in (
            "rollout_turn_deg", "rollout_turn_steps",
            "rollout_turn_threshold_deg", "rollout_turn_threshold_step",
            "rollout_kind",
            "rollout_walk_m", "rollout_segment_walk_m", "rollout_cut_step",
            "rollout_walk_start_step",
        ):
            out[key] = rollout_partition.get(key)
    return out


def partition_episode(
    episode,
    scan_db:   Dict[str, np.ndarray],
    rewritten: Optional[Dict] = None,
    turn_thresh_deg:    float = TURN_THRESH_DEG,
    around_thresh_deg:  float = AROUND_THRESH_DEG,
    forward_distance_m: float = FORWARD_DISTANCE_M,
    rollout_frames_by_sub: Optional[Dict[int, Sequence[Dict]]] = None,
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
            rollout_frames=(rollout_frames_by_sub or {}).get(i),
        )

        # Build the two path segments; they share the partition point
        # (a real MP3D node when alpha==1.0, else a virtual "virt:..." node).
        p     = out["partition_idx"]
        alpha = out["partition_alpha"]
        use_virtual = bool(out.get("partition_virtual")) or (
            alpha < 1.0 - 1e-9 and p > 0
        )
        if not use_virtual:
            out["partition_on_edge"] = None
            spatial_seg  = list(sub_nodes[: p + 1])
            landmark_seg = list(sub_nodes[p:])
        else:
            vid = _virt_id(np.asarray(out["partition_pos"]))
            out["partition_on_edge"] = (
                [sub_nodes[p - 1], sub_nodes[p]] if 0 < p < len(sub_nodes) else None
            )
            keep_to = max(1, min(p, len(sub_nodes)))
            spatial_seg  = list(sub_nodes[:keep_to]) + [vid]
            landmark_seg = [vid] + list(sub_nodes[keep_to:])

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
