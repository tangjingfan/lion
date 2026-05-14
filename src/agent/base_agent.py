"""
Abstract base agent and built-in agents.

Built-in agents
---------------
  DummyAgent   — random actions, for pipeline smoke-testing
  OracleAgent  — follows the GT reference path using habitat-lab's
                 ShortestPathFollower in a per-waypoint inner loop
                 (CL_CoTNav style; the rollout driver consumes
                 OracleAgent.follower directly and does not call step()).

To plug in a real agent:
  1. Subclass BaseAgent
  2. Implement reset() and step()
  3. Register in build_agent() below
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List

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
    """Owns a ShortestPathFollower for following the GT reference path.

    The rollout driver (``rollout._drive_oracle``) consumes
    ``self.follower`` and ``self.waypoints`` directly in a per-waypoint
    inner loop (CL_CoTNav pattern), so ``step()`` is never called and
    raises if used.

    set_env() must be called after every env.reset() because make_sim
    can rebuild the underlying simulator per scene.
    """

    def __init__(self, goal_radius: float = 0.5) -> None:
        self._goal_radius = float(goal_radius)
        self._follower = None
        self._waypoints: List[List[float]] = []

    def set_env(self, env) -> None:
        """Bind the follower to the current scene's simulator."""
        from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
        self._follower = ShortestPathFollower(
            sim=env._sim,
            goal_radius=self._goal_radius,
            return_one_hot=False,
            stop_on_error=True,
        )

    def reset(self, episode: LandmarkRxREpisode) -> None:
        # reference_path is at navmesh height (set by env.reset); skip the
        # start node since the agent is already placed there.
        self._waypoints = list(episode.reference_path[1:])

    def step(self, obs: Dict[str, Any]) -> int:
        raise NotImplementedError(
            "OracleAgent is driven by rollout._drive_oracle via the "
            "ShortestPathFollower; step() should not be called."
        )

    @property
    def follower(self):
        return self._follower

    @property
    def waypoints(self) -> List[List[float]]:
        return self._waypoints


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------

def build_agent(cfg: dict) -> BaseAgent:
    """Instantiate an agent from the top-level rollout config.

    Reads agent type from cfg["agent"] and OracleAgent's follower
    ``goal_radius`` from cfg["env"] so the parameter lives next to the
    rest of the simulator config.
    """
    agent_cfg = cfg.get("agent", {})
    env_cfg = cfg.get("env", {})
    agent_type = agent_cfg.get("type", "dummy").lower()
    if agent_type == "dummy":
        return DummyAgent()
    if agent_type == "oracle":
        return OracleAgent(goal_radius=env_cfg.get("goal_radius", 0.5))
    raise ValueError(
        f"Unknown agent type: {agent_type!r}. "
        "Implement a BaseAgent subclass and register it here."
    )
