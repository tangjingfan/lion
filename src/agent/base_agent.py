"""
Abstract base agent and built-in agents.

Built-in agents
---------------
  DummyAgent   — random actions, for pipeline smoke-testing
  OracleAgent  — follows the GT reference path using habitat-sim's
                 GreedyGeodesicFollower (upper-bound / GT trajectory)

To plug in a real agent:
  1. Subclass BaseAgent
  2. Implement reset() and step()
  3. Register in build_agent() below
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from src.dataset.landmark_rxr import LandmarkRxREpisode
from src.env.habitat_env import MOVE_FORWARD, STOP, TURN_LEFT, TURN_RIGHT


class BaseAgent(ABC):
    """Interface every LION-Bench agent must implement."""

    @abstractmethod
    def reset(self, episode: LandmarkRxREpisode) -> None:
        """Called once at the start of each episode before the first step."""

    @abstractmethod
    def step(self, obs: Dict[str, Any]) -> int:
        """Return a discrete action given the current observation.

        Parameters
        ----------
        obs : dict with keys "rgb", "semantic", "semantic_id",
            "semantic_name", "instruction", "position", "heading"

        Returns
        -------
        int — STOP(0), MOVE_FORWARD(1), TURN_LEFT(2), TURN_RIGHT(3)
        """


# ---------------------------------------------------------------------------
#  DummyAgent
# ---------------------------------------------------------------------------

class DummyAgent(BaseAgent):
    """Random-action agent — useful for verifying the rollout pipeline."""

    _ACTIONS = [MOVE_FORWARD, MOVE_FORWARD, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]

    def reset(self, episode: LandmarkRxREpisode) -> None:
        pass

    def step(self, obs: Dict[str, Any]) -> int:
        return random.choice(self._ACTIONS)


# ---------------------------------------------------------------------------
#  OracleAgent  (GT upper bound)
# ---------------------------------------------------------------------------

class OracleAgent(BaseAgent):
    """Follows the GT reference path using habitat-sim's GreedyGeodesicFollower.

    Advances through the navmesh-snapped GT waypoints in order.  For each
    step it asks the follower for the best action toward the current waypoint;
    once the agent is within goal_radius it moves on to the next waypoint and
    stops after the last one.

    set_env() must be called after every env.reset() (rollout.py does this
    automatically) because reconfigure() rebuilds the pathfinder per scene.

    Note: habitat-sim 0.3.1 has no ShortestPathFollower in Python bindings;
    GreedyGeodesicFollower is the equivalent (geodesic / navmesh shortest path).
    We use integer action keys so next_action_along() returns the same action
    IDs consumed by Habitat-Lab's simulator wrapper.
    """

    # GreedyGeodesicFollower returns the exact action keys we register.
    _KEY_TO_ACTION = {
        MOVE_FORWARD: MOVE_FORWARD,
        TURN_LEFT: TURN_LEFT,
        TURN_RIGHT: TURN_RIGHT,
    }

    def __init__(self) -> None:
        self._follower = None       # habitat_sim.nav.GreedyGeodesicFollower
        self._waypoints: List[List[float]] = []
        self._wp_idx: int = 0
        self._goal_radius: float = 0.5   # metres — advance to next waypoint

    def set_env(self, env) -> None:
        """Bind follower to the current scene's pathfinder. Call after each env.reset()."""
        import habitat_sim
        pathfinder = env.get_pathfinder()
        agent      = env._sim.get_agent(0)
        self._follower = habitat_sim.nav.GreedyGeodesicFollower(
            pathfinder=pathfinder,
            agent=agent,
            goal_radius=self._goal_radius,
            forward_key=MOVE_FORWARD,
            left_key=TURN_LEFT,
            right_key=TURN_RIGHT,
        )

    def reset(self, episode: LandmarkRxREpisode) -> None:
        # reference_path is at navmesh height (set by env.reset)
        self._waypoints = episode.reference_path[1:]  # skip start node
        self._wp_idx = 0
        if self._follower is not None:
            self._follower.reset()

    def step(self, obs: Dict[str, Any]) -> int:
        if self._follower is None or self._wp_idx >= len(self._waypoints):
            return STOP

        goal = np.array(self._waypoints[self._wp_idx], dtype=np.float32)
        cur  = obs["position"]

        # Advance waypoint when close enough
        if np.linalg.norm(cur - goal) < self._goal_radius:
            self._wp_idx += 1
            if self._wp_idx >= len(self._waypoints):
                return STOP
            goal = np.array(self._waypoints[self._wp_idx], dtype=np.float32)

        try:
            action_key = self._follower.next_action_along(goal)
        except Exception:
            return STOP

        if action_key is None:
            # Follower says we're at this waypoint — move to next
            self._wp_idx += 1
            return STOP if self._wp_idx >= len(self._waypoints) else MOVE_FORWARD

        return self._KEY_TO_ACTION.get(action_key, STOP)


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------

def build_agent(cfg: dict) -> BaseAgent:
    """Instantiate an agent from the "agent" config section."""
    agent_type = cfg.get("type", "dummy").lower()
    if agent_type == "dummy":
        return DummyAgent()
    if agent_type == "oracle":
        return OracleAgent()
    raise ValueError(
        f"Unknown agent type: {agent_type!r}. "
        "Implement a BaseAgent subclass and register it here."
    )
