"""
VLN evaluation metrics for LION-Bench.

Metrics (matching CL_CoTNav / standard VLN literature):
  NE   — Navigation Error            (metres, lower is better)
  SR   — Success Rate                (%, higher is better)
  SPL  — Success weighted by Path Length
  NDTW — Normalized Dynamic Time Warping   (path fidelity)
  SDTW — Success weighted DTW
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
#  Per-episode metric computation
# ---------------------------------------------------------------------------

def navigation_error(agent_pos: np.ndarray, goal_pos: np.ndarray) -> float:
    """Euclidean distance from final agent position to goal (metres)."""
    return float(np.linalg.norm(np.array(agent_pos) - np.array(goal_pos)))


def success(ne: float, success_distance: float = 3.0) -> bool:
    return ne <= success_distance


def spl(
    succeeded: bool,
    agent_path_length: float,
    reference_path_length: float,
) -> float:
    """Success weighted by (normalized inverse) Path Length.

    SPL = success * shortest_path_length / max(agent_path_length, shortest_path_length)
    """
    if not succeeded:
        return 0.0
    denom = max(agent_path_length, reference_path_length)
    if denom == 0.0:
        return 1.0
    return reference_path_length / denom


def ndtw(
    agent_positions: List[np.ndarray],
    reference_positions: List[np.ndarray],
    success_distance: float = 3.0,
) -> float:
    """Normalized Dynamic Time Warping between agent and reference paths.

    Uses a simple O(n*m) DTW without FastDTW for correctness.
    Returns value in [0, 1] (higher is better).
    """
    agent = [np.array(p, dtype=np.float32) for p in agent_positions]
    ref = [np.array(p, dtype=np.float32) for p in reference_positions]

    n, m = len(agent), len(ref)
    dtw_grid = np.full((n + 1, m + 1), np.inf)
    dtw_grid[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(agent[i - 1] - ref[j - 1]))
            dtw_grid[i, j] = cost + min(
                dtw_grid[i - 1, j],
                dtw_grid[i, j - 1],
                dtw_grid[i - 1, j - 1],
            )

    dtw_score = dtw_grid[n, m]
    # Normalise by the number of steps and success_distance
    ndtw_score = math.exp(-dtw_score / (success_distance * max(n, m)))
    return float(ndtw_score)


def sdtw(ndtw_score: float, succeeded: bool) -> float:
    """Success-weighted DTW."""
    return ndtw_score if succeeded else 0.0


def path_length(positions: List[np.ndarray]) -> float:
    """Total Euclidean path length of an ordered list of positions."""
    total = 0.0
    for i in range(1, len(positions)):
        total += float(np.linalg.norm(
            np.array(positions[i]) - np.array(positions[i - 1])
        ))
    return total


# ---------------------------------------------------------------------------
#  Episode-level result accumulation
# ---------------------------------------------------------------------------

def compute_episode_metrics(
    agent_positions: List[np.ndarray],
    episode_reference_path: List[List[float]],
    goal_position: List[float],
    success_distance: float = 3.0,
) -> Dict[str, float]:
    """Compute all metrics for a single completed episode.

    Parameters
    ----------
    agent_positions:
        Trajectory of the agent's XYZ positions (one per step + final).
    episode_reference_path:
        The ground-truth reference path (list of [x,y,z]).
    goal_position:
        The goal node position [x,y,z].
    success_distance:
        Distance threshold in metres for counting as success.
    """
    ref_pos = [np.array(p, dtype=np.float32) for p in episode_reference_path]
    goal = np.array(goal_position, dtype=np.float32)

    ne = navigation_error(agent_positions[-1], goal)
    sr = float(success(ne, success_distance))

    agent_pl = path_length(agent_positions)
    ref_pl = path_length(ref_pos)
    spl_score = spl(bool(sr), agent_pl, ref_pl)

    ndtw_score = ndtw(agent_positions, ref_pos, success_distance)
    sdtw_score = sdtw(ndtw_score, bool(sr))

    return {
        "ne": ne,
        "sr": sr,
        "spl": spl_score,
        "ndtw": ndtw_score,
        "sdtw": sdtw_score,
        "agent_path_length": agent_pl,
        "reference_path_length": ref_pl,
    }


def aggregate_metrics(
    per_episode: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """Compute mean metrics over all completed episodes."""
    if not per_episode:
        return {}
    keys = list(next(iter(per_episode.values())).keys())
    agg: Dict[str, float] = {}
    for k in keys:
        vals = [ep[k] for ep in per_episode.values() if k in ep]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
    agg["num_episodes"] = float(len(per_episode))
    return agg
