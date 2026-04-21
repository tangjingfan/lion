"""
Habitat-Lab/Habitat-sim based continuous navigation environment for LION-Bench.

Uses Habitat-Lab's ``habitat.simulator.agents.rgbds_agent`` sensor setup.
Exposes a minimal VLN interface:
  - reset(episode)  →  obs
  - step(action)    →  obs, done, info
  - close()

Actions (int)
-------------
  0 = stop
  1 = move_forward
  2 = turn_left
  3 = turn_right

Observation dict
----------------
  "rgb"           : np.ndarray (H, W, 3) uint8    equirect panorama
  "semantic"      : np.ndarray (H, W)    int32    raw instance ids from Habitat-Sim
  "semantic_id"   : np.ndarray (H, W)    int32    MP40 category id per pixel (-1 = unknown)
  "semantic_name" : np.ndarray (H, W)    object   MP40 category name per pixel ("" = unknown)
  "instruction"   : str
  "position"      : np.ndarray (3,)      float32  current agent XYZ (Habitat frame)
  "heading"       : float                current agent heading in radians

Info dict (only meaningful after step / on done=True)
-----------------------------------------------------
  "steps"            : int
  "distance_to_goal" : float  (metres)
  "success"          : bool
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.dataset.landmark_rxr import LandmarkRxREpisode
from src.env.connectivity import (
    ConnectivityDB,
    heading_to_rotation,
    path_to_positions,
)
HABITAT_AGENT_NAME = "rgbds_agent"

# Action constants — exported so agent/rollout code can use them
STOP = 0
MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3

class HabitatEnv:
    """Thin wrapper around habitat_sim.Simulator for VLN evaluation."""

    def __init__(self, cfg: dict, connectivity_db: ConnectivityDB):
        """
        Parameters
        ----------
        cfg:
            The "env" section of rollout_landmark_rxr.yaml.
        connectivity_db:
            Pre-loaded connectivity graph (scan → node → position).
        """
        self._cfg = cfg
        self._db = connectivity_db
        self._scenes_dir = None  # set by rollout before first reset
        self._sim = None
        self._current_episode: Optional[LandmarkRxREpisode] = None
        self._step_count = 0
        # Instance-id → MP40 category id / name lookup tables, rebuilt per scene.
        # Habitat-Sim's semantic sensor emits instance ids; MP3D categories come
        # from sim.semantic_annotations().
        self._sem_id_map: Optional[np.ndarray] = None
        self._sem_name_map: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    #  Public lifecycle
    # ------------------------------------------------------------------

    def set_scenes_dir(self, scenes_dir: str) -> None:
        self._scenes_dir = scenes_dir.rstrip("/")

    def reset(self, episode: LandmarkRxREpisode) -> Dict[str, Any]:
        """Place the agent at the episode's start node and return first obs."""
        self._current_episode = episode
        self._step_count = 0

        # Build (or reconfigure) the simulator first — needed for pathfinder
        scene_file = f"{self._scenes_dir}/{episode.scene_file}"
        self._init_sim(scene_file)
        self._build_semantic_mapping()

        # Resolve node IDs → positions in Habitat frame.
        # JSON source: already at navmesh level (camera_y - height field).
        # House source: at camera height; _snap() will project to navmesh.
        path_positions = path_to_positions(self._db, episode.scan, episode.path)

        # Snap to navmesh — no-op for JSON-derived positions, projects down
        # for house-file positions.  Result is the agent-base (floor) height.
        nav_positions = [self._snap(p) for p in path_positions]

        episode.start_position = nav_positions[0].tolist()
        episode.goal_position = nav_positions[-1].tolist()
        episode.reference_path = [p.tolist() for p in nav_positions]

        # Place agent.  Pass the raw path position so snap_point inside
        # _place_agent can locate the correct navmesh island for multi-floor
        # buildings (camera height is closer to the right floor than navmesh).
        rotation_xyzw = heading_to_rotation(episode.heading)
        self._place_agent(path_positions[0], rotation_xyzw)

        return self._get_obs()

    def get_pathfinder(self):
        """Expose pathfinder so agents (e.g. OracleAgent) can use it."""
        return self._sim.pathfinder if self._sim else None

    def step(self, action: int) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        """Execute one discrete action.

        Returns
        -------
        obs, done, info
        """
        if action == STOP or self._sim is None:
            done = True
            obs = self._get_obs()
            info = self._build_info(done=True)
            return obs, done, info

        self._sim.step(action)
        self._step_count += 1

        max_steps = self._cfg.get("max_steps", 500)
        done = self._step_count >= max_steps
        obs = self._get_obs()
        info = self._build_info(done=done)
        return obs, done, info

    def close(self) -> None:
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _init_sim(self, scene_file: str) -> None:
        """Create the Habitat-Lab simulator for a scene."""
        _ensure_habitat_lab_importable()

        from habitat.config import read_write
        from habitat.config.default import get_config
        from habitat.sims import make_sim

        env_cfg = self._cfg
        config_path = env_cfg.get("habitat_config", "configs/habitat/rgbds_sim.yaml")
        cfg = get_config(config_path)
        sim_cfg = cfg.habitat.simulator

        with read_write(sim_cfg):
            sim_cfg.scene = scene_file
            sim_cfg.scene_dataset = env_cfg.get("scene_dataset", "default")
            sim_cfg.forward_step_size = env_cfg.get("forward_step_size", 0.25)
            sim_cfg.turn_angle = int(round(env_cfg.get("turn_angle", 15.0)))
            sim_cfg.default_agent_id = 0
            sim_cfg.agents_order = [HABITAT_AGENT_NAME]

            agent_cfg = sim_cfg.agents[HABITAT_AGENT_NAME]
            agent_cfg.height = env_cfg.get("height", 0.88)
            agent_cfg.radius = env_cfg.get("radius", 0.18)

            # Depth is not used — strip it from the rgbds_agent composition
            # so Habitat-Sim doesn't spin up a depth render target.
            if "depth_sensor" in agent_cfg.sim_sensors:
                del agent_cfg.sim_sensors.depth_sensor

            sensor_height = env_cfg.get("sensor_height", 1.5)
            pano_w = int(env_cfg.get("panorama_width", 1024))
            pano_h = pano_w // 2  # equirect: 360° horizontal, 180° vertical
            pano_overrides = {"width": pano_w, "height": pano_h}

            self._configure_sensor(
                agent_cfg.sim_sensors.rgb_sensor,
                pano_overrides,
                sensor_height,
            )
            self._configure_sensor(
                agent_cfg.sim_sensors.semantic_sensor,
                pano_overrides,
                sensor_height,
            )

        if self._sim is not None:
            self._sim.close()
        self._sim = make_sim(sim_cfg.type, config=sim_cfg)

    @staticmethod
    def _configure_sensor(
        sensor_cfg: Any,
        overrides: Dict[str, Any],
        sensor_height: float,
    ) -> None:
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

    def _place_agent(self, position: np.ndarray, rotation_xyzw: np.ndarray) -> None:
        """Place agent on the navmesh below the viewpoint camera position.

        house-file positions are camera locations (eye level, ~1.25 m above floor).
        We snap to the navmesh so that movement stays on the floor, and the
        sensor_height offset in _init_sim lifts the camera back to eye level.
        """
        import habitat_sim
        import quaternion as qt

        # Snap to nearest navmesh point — keeps agent on the floor for movement
        nav_pos = self._sim.pathfinder.snap_point(position)
        if not np.isfinite(nav_pos).all():
            nav_pos = position   # fallback if navmesh unavailable

        agent = self._sim.initialize_agent(0)
        state = habitat_sim.AgentState()
        state.position = np.array(nav_pos, dtype=np.float32)

        # habitat-sim quaternion convention: [w, x, y, z]
        x, y, z, w = rotation_xyzw
        state.rotation = qt.quaternion(w, x, y, z)

        agent.set_state(state)

    def _snap(self, position: np.ndarray) -> np.ndarray:
        """Snap a camera-height position to the nearest navmesh point."""
        nav = self._sim.pathfinder.snap_point(position)
        if np.isfinite(nav).all():
            return np.array(nav, dtype=np.float32)
        return np.array(position, dtype=np.float32)

    def _build_semantic_mapping(self) -> None:
        """Cache instance-id → (category_id, category_name) tables for this scene.

        MP3D instance ids are not always dense, so we size the table to
        ``max_id + 1`` and leave gaps as -1 / "".
        """
        self._sem_id_map = None
        self._sem_name_map = None
        if self._sim is None:
            return

        scene = self._sim.semantic_annotations()
        if scene is None or not getattr(scene, "objects", None):
            return

        pairs = []
        for obj in scene.objects:
            if obj is None:
                continue
            try:
                inst_id = int(obj.id.split("_")[-1])
            except (AttributeError, ValueError):
                continue
            cat = obj.category
            pairs.append((inst_id, cat.index(), cat.name()))

        if not pairs:
            return

        max_id = max(p[0] for p in pairs)
        id_map = np.full(max_id + 1, -1, dtype=np.int32)
        name_map = np.full(max_id + 1, "", dtype=object)
        for inst_id, cat_id, cat_name in pairs:
            id_map[inst_id] = cat_id
            name_map[inst_id] = cat_name

        self._sem_id_map = id_map
        self._sem_name_map = name_map

    def _map_semantic(
        self, semantic: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Translate instance ids to MP40 category ids / names via the cached table."""
        if self._sem_id_map is None or self._sem_name_map is None:
            empty_ids = np.full(semantic.shape, -1, dtype=np.int32)
            empty_names = np.full(semantic.shape, "", dtype=object)
            return empty_ids, empty_names

        idx = np.clip(semantic, 0, len(self._sem_id_map) - 1)
        return self._sem_id_map[idx], self._sem_name_map[idx]

    def _get_obs(self) -> Dict[str, Any]:
        if self._sim is None:
            return {}

        raw = self._sim.get_sensor_observations()
        rgb = raw["rgb"][..., :3]
        semantic = raw["semantic"]

        semantic_id, semantic_name = self._map_semantic(semantic)

        agent_state = self._sim.get_agent(0).get_state()
        pos = np.array(agent_state.position, dtype=np.float32)

        # Extract heading from quaternion  (rotation around Y axis)
        import quaternion as qt
        q = agent_state.rotation   # numpy quaternion [w, x, y, z]
        # heading = 2 * arctan2(q.y, q.w), negated for clockwise convention
        heading = -2.0 * math.atan2(q.y, q.w)

        instruction = (
            self._current_episode.instruction
            if self._current_episode is not None
            else ""
        )

        return {
            "rgb": rgb,
            "semantic": semantic,
            "semantic_id": semantic_id,
            "semantic_name": semantic_name,
            "instruction": instruction,
            "position": pos,
            "heading": heading,
        }

    def _build_info(self, *, done: bool) -> Dict[str, Any]:
        info: Dict[str, Any] = {"steps": self._step_count}

        if not done or self._current_episode is None:
            return info

        agent_state = self._sim.get_agent(0).get_state()
        cur_pos = np.array(agent_state.position, dtype=np.float32)
        goal_pos = np.array(self._current_episode.goal_position, dtype=np.float32)

        dist = float(np.linalg.norm(cur_pos - goal_pos))
        success_dist = self._cfg.get("success_distance", 3.0)

        info["distance_to_goal"] = dist
        info["success"] = dist <= success_dist
        return info


def _ensure_habitat_lab_importable() -> None:
    try:
        import habitat  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    habitat_lab_src = (
        Path(__file__).resolve().parents[2]
        / "external"
        / "habitat-lab"
        / "habitat-lab"
    )
    if habitat_lab_src.exists():
        sys.path.insert(0, str(habitat_lab_src))
