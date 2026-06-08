"""
Sub-path visibility checker for Landmark-RxR.

Uses Habitat-Lab's ``rgbds_agent`` (same setup as rollout) so visibility /
uniqueness checks render exactly what the rollout agent perceives — RGB and
semantic equirectangular panoramas at eye height.
``check_landmark_visibility_semantic`` counts distinct instance ids of
the landmark category in the semantic panorama — fast, and it matches
what the agent's semantic sensor would actually report.

API
---
    from src.process.visibility import VisibilityChecker

    checker = VisibilityChecker(env_cfg, scenes_dir)
    checker.load_scene(scene_file)
    vis  = checker.check_landmark_visibility_semantic(pos_end, "bath",
                                                       ["bath"])
    checker.close()
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from src.env.habitat_setup import (
    build_semantic_tables,
    configure_sensor,
    ensure_habitat_lab_importable,
)


class VisibilityChecker:
    """Habitat-Lab wrapper that mirrors rollout's ``rgbds_agent`` setup.

    The simulator is created the same way as :class:`HabitatEnv`: load the
    same ``rgbds_sim.yaml``, strip the depth sensor, configure RGB and
    semantic equirectangular sensors at eye height.

    Parameters
    ----------
    env_cfg:
        The ``env`` block of the rollout YAML (``cfg["env"]``).  Provides
        ``habitat_config``, ``scene_dataset``, ``height``/``radius``,
        ``sensor_height``, ``panorama_width``, ``forward_step_size``,
        ``turn_angle``.
    scenes_dir:
        Root of the scene_datasets directory (``cfg["scenes"]["scenes_dir"]``).
    """

    def __init__(
        self,
        env_cfg:    Dict[str, Any],
        scenes_dir: str,
    ) -> None:
        self._env_cfg     = env_cfg
        self._scenes_dir  = scenes_dir.rstrip("/")
        self._sensor_h    = float(env_cfg.get("sensor_height", 1.5))
        self._sim         = None
        self._scene_file  = None
        # Instance-id → MP40 category id / name lookup tables (rebuilt per scene).
        self._sem_id_map:    Optional[np.ndarray] = None
        self._sem_name_map:  Optional[np.ndarray] = None
        # name (lower) → cat_id index for fast pixel-mask lookup.
        self._name_to_cat_id: Dict[str, int] = {}

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------

    def load_scene(self, scene_file: str) -> None:
        """(Re-)initialise the Habitat-Lab simulator for a new scene.

        Mirrors :meth:`HabitatEnv._init_sim` step-for-step: load
        ``rgbds_sim.yaml``, set scene + dataset, strip depth, resize the
        rgb/semantic sensors to a 360° panorama at ``sensor_height``, and
        enable physics so cast_ray works.
        """
        full_path = f"{self._scenes_dir}/{scene_file}"
        if self._scene_file == full_path and self._sim is not None:
            return

        if self._sim is not None:
            self._sim.close()

        ensure_habitat_lab_importable()
        from habitat.config import read_write
        from habitat.config.default import get_config
        from habitat.sims import make_sim

        env_cfg = self._env_cfg
        config_path = env_cfg.get("habitat_config",
                                  "configs/habitat/rgbds_sim.yaml")
        cfg     = get_config(config_path)
        sim_cfg = cfg.habitat.simulator

        with read_write(sim_cfg):
            sim_cfg.scene             = full_path
            sim_cfg.scene_dataset     = env_cfg.get("scene_dataset", "default")
            sim_cfg.forward_step_size = env_cfg.get("forward_step_size", 0.25)
            sim_cfg.turn_angle        = int(round(env_cfg.get("turn_angle", 15.0)))
            sim_cfg.default_agent_id  = 0
            sim_cfg.agents_order      = ["rgbds_agent"]
            # Required so cast_ray returns hit results below.
            sim_cfg.enable_physics    = True

            agent_cfg = sim_cfg.agents["rgbds_agent"]
            agent_cfg.height = env_cfg.get("height", 0.88)
            agent_cfg.radius = env_cfg.get("radius", 0.18)

            pano_w = int(env_cfg.get("panorama_width", 1024))
            pano_h = pano_w // 2  # equirect: 2:1 aspect
            pano_overrides = {"width": pano_w, "height": pano_h}

            configure_sensor(agent_cfg.sim_sensors.rgb_sensor,
                             pano_overrides, self._sensor_h)
            configure_sensor(agent_cfg.sim_sensors.semantic_sensor,
                             pano_overrides, self._sensor_h)
            # Depth sensor (configured as equirectangular by
            # configs/habitat/rgbds_sim.yaml). Used by point-cloud /
            # voxel-based visibility scoring downstream — unproject
            # equirect depth + semantic to get per-instance point
            # clouds, then voxelise.
            if "depth_sensor" in agent_cfg.sim_sensors:
                configure_sensor(agent_cfg.sim_sensors.depth_sensor,
                                 pano_overrides, self._sensor_h)

        self._sim        = make_sim(sim_cfg.type, config=sim_cfg)
        self._scene_file = full_path
        self._build_semantic_mapping()

    def close(self) -> None:
        if self._sim is not None:
            self._sim.close()
            self._sim          = None
            self._scene_file   = None
            self._sem_id_map   = None
            self._sem_name_map = None
            self._name_to_cat_id = {}

    # ------------------------------------------------------------------
    #  Semantic mapping
    # ------------------------------------------------------------------

    def _build_semantic_mapping(self) -> None:
        """Cache instance-id → MP40 (category_id, category_name) tables.

        Translates Habitat-Sim's raw semantic sensor (instance ids) to MP40
        category names per pixel; also builds the name → cat_id lookup used by
        the semantic visibility check.
        """
        scene = self._sim.semantic_annotations() if self._sim is not None else None
        (
            self._sem_id_map,
            self._sem_name_map,
            self._name_to_cat_id,
        ) = build_semantic_tables(scene)

    # ------------------------------------------------------------------
    #  Semantic-image visibility (panorama-based, rgbds_agent native)
    # ------------------------------------------------------------------

    def check_landmark_visibility_semantic(
        self,
        pos:             np.ndarray,
        landmark:        str,
        semantic_labels: Optional[List[str]] = None,
        heading:         float = 0.0,
        min_pixel_count: int   = 50,
    ) -> Dict[str, Any]:
        """Render a 360° semantic panorama at *pos* and count distinct
        instance ids whose MP40 category matches the landmark.

        Faster and more aligned with what the rollout agent perceives than
        a per-instance raycast scheme: a single render replaces N raycasts.
        Heading mostly only changes which side of the
        panorama the landmark falls on, so any heading works for visibility
        / counting (default 0.0).

        Matching strategy:
          1. Match every label in *semantic_labels* and aggregate all matched
             MP40 categories.
          2. Substring match on the raw *landmark* phrase as a fallback when
             no semantic label matches.

        Instance ids whose pixel count is below ``min_pixel_count`` are
        treated as noise (e.g. a sliver visible through a doorway) and
        excluded from the visibility count.

        Returns
        -------
        dict with:
          visible           bool         — at least one instance survives
          n_instances       int          — distinct instances after filtering
          pixel_count       int          — total matching pixels
          pixel_fraction    float        — pixel_count / total
          matched_category  str | None  — first matched category, for legacy callers
          matched_categories list[str]  — all matched categories that were counted
          matched_by        "semantic_label" | "phrase_fallback" | None
          instances         list[dict]   — [{"id": int, "category": str,
                                             "n_pixels": int}, ...]
        """
        if self._sim is None:
            raise RuntimeError("Call load_scene() before check_landmark_visibility_semantic().")
        if self._sem_id_map is None or not self._name_to_cat_id:
            return {
                "visible":          False,
                "n_instances":      0,
                "pixel_count":      0,
                "pixel_fraction":   0.0,
                "matched_category": None,
                "matched_categories": [],
                "matched_by":       None,
                "instances":        [],
                "note":             "semantic mapping unavailable",
            }

        # ── 1. Pick target MP40 category ids ────────────────────────────
        target_cat_ids: List[int] = []
        matched_cats:   List[str] = []
        matched_by:     Optional[str] = None

        def _add_match(cat_name: str, cat_id: int) -> None:
            if cat_id in target_cat_ids:
                return
            target_cat_ids.append(cat_id)
            matched_cats.append(cat_name)

        for label in (semantic_labels or []):
            if not label:
                continue
            key = label.lower().strip()
            if key in ("unknown", ""):
                continue
            if key in self._name_to_cat_id:
                _add_match(key, self._name_to_cat_id[key])
                matched_by = "semantic_label"
                continue
            # Loose substring match against scene category names.
            for cat_name, cat_id in self._name_to_cat_id.items():
                if key in cat_name or cat_name in key:
                    _add_match(cat_name, cat_id)
                    matched_by = "semantic_label"
                    break

        if not target_cat_ids:
            lm_lower = landmark.lower().strip()
            for cat_name, cat_id in self._name_to_cat_id.items():
                if not cat_name:
                    continue
                if cat_name in lm_lower or lm_lower in cat_name:
                    _add_match(cat_name, cat_id)
                    matched_by = "phrase_fallback"
                    break

        if not target_cat_ids:
            return {
                "visible":          False,
                "n_instances":      0,
                "pixel_count":      0,
                "pixel_fraction":   0.0,
                "matched_category": None,
                "matched_categories": [],
                "matched_by":       None,
                "instances":        [],
                "note":             f"no MP40 category matches '{landmark}'",
            }

        # ── 2. Render + map instance ids → category ids per pixel ───────
        sem = self.render_semantic(pos, heading)             # (H, W) int32
        sem_clip = np.clip(sem, 0, len(self._sem_id_map) - 1)
        cat_image = self._sem_id_map[sem_clip]               # (H, W) cat ids

        target_mask = np.isin(cat_image, target_cat_ids)
        target_pixels = int(target_mask.sum())
        if target_pixels == 0:
            return {
                "visible":          False,
                "n_instances":      0,
                "pixel_count":      0,
                "pixel_fraction":   0.0,
                "matched_category": matched_cats[0] if matched_cats else None,
                "matched_categories": matched_cats,
                "matched_by":       matched_by,
                "instances":        [],
            }

        # ── 3. Count distinct instance ids; drop tiny ones as noise ─────
        instance_ids, counts = np.unique(sem[target_mask], return_counts=True)
        instances: List[Dict[str, Any]] = []
        kept_pixels = 0
        for iid, n in zip(instance_ids, counts):
            n = int(n)
            if n >= min_pixel_count:
                iid_int = int(iid)
                idx = min(max(iid_int, 0), len(self._sem_name_map) - 1)
                instances.append({
                    "id": iid_int,
                    "category": str(self._sem_name_map[idx] or ""),
                    "n_pixels": n,
                })
                kept_pixels += n

        return {
            "visible":          len(instances) > 0,
            "n_instances":      len(instances),
            "pixel_count":      kept_pixels,
            "pixel_fraction":   float(kept_pixels) / float(sem.size),
            "matched_category": matched_cats[0] if matched_cats else None,
            "matched_categories": matched_cats,
            "matched_by":       matched_by,
            "instances":        instances,
        }

    # ------------------------------------------------------------------
    #  Rendering helpers
    # ------------------------------------------------------------------

    def render_observation(self, pos: np.ndarray, heading: float) -> Dict[str, np.ndarray]:
        """Place the agent and capture the rgbds_agent sensors.

        Parameters
        ----------
        pos:
            Navmesh-level 3-D position (sensor_height is applied via the
            sensor spec, not by modifying pos).
        heading:
            Clockwise heading from north in radians (same convention as
            LandmarkRxR heading field).

        Returns
        -------
        dict with:
          ``rgb``      — (H, W, 3) uint8  equirectangular RGB panorama
          ``semantic`` — (H, W)    int32  raw Habitat-Sim instance ids
          ``depth``    — (H, W)    float32 radial distance from sensor
                                            origin to first hit per ray
                                            (equirectangular geometry)
          ``rgb_viz``  — alias for ``rgb`` (back-compat with viz.py callers)
        """
        if self._sim is None:
            raise RuntimeError("Call load_scene() before render_observation().")

        import habitat_sim
        import quaternion as qt

        agent = self._sim.get_agent(0)
        state = habitat_sim.AgentState()
        state.position = np.array(pos, dtype=np.float32)
        half = -heading / 2.0
        state.rotation = qt.quaternion(math.cos(half), 0.0, math.sin(half), 0.0)
        agent.set_state(state)

        raw = self._sim.get_sensor_observations()
        obs: Dict[str, np.ndarray] = {}
        if "rgb" in raw:
            rgb = raw["rgb"][..., :3].astype(np.uint8)
            obs["rgb"]     = rgb
            obs["rgb_viz"] = rgb            # alias for viz.py callers
        if "semantic" in raw:
            obs["semantic"] = np.asarray(raw["semantic"], dtype=np.int32)
        if "depth" in raw:
            d = np.asarray(raw["depth"], dtype=np.float32)
            if d.ndim == 3 and d.shape[-1] == 1:
                d = d[..., 0]
            obs["depth"] = d
        return obs

    def render_semantic(self, pos: np.ndarray, heading: float = 0.0) -> np.ndarray:
        """Capture the raw semantic equirectangular panorama at *pos*.

        Returns instance ids per pixel (int32).  Use ``self._sem_id_map`` /
        ``self._sem_name_map`` to translate to MP40 categories.
        """
        sem = self.render_observation(pos, heading).get("semantic")
        if sem is None:
            raise RuntimeError("Semantic sensor not in observation.")
        return sem

