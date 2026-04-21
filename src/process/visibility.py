"""
Sub-path visibility checker for Landmark-RxR.

For each sub-path (start_node → end_node), back-traces a ray from the
end-point's eye position toward the start-point and records whether any
obstacle blocks the line of sight.

Requires habitat-sim built with Bullet physics (--bullet).

Usage
-----
    python src/check/check_visibility.py \
        --config configs/rollout/rollout_landmark_rxr.yaml

API
---
    from src.process.visibility import VisibilityChecker

    checker = VisibilityChecker(scenes_dir, sensor_height=1.5,
                                rgb_cfg={"width": 256, "height": 256, "hfov": 90})
    checker.load_scene(scene_file)
    result = checker.check(pos_a, pos_b)          # navmesh-level positions
    checker.save_subpath_viz(pos_start, pos_end,  # render + compose PNG
                             result, out_path, sub_instruction="...")
    checker.close()
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.env.rgbd_agent import make_rgbd_agent_config

# MP3D .house file room-type code → human-readable name
_MP3D_ROOM_TYPES: Dict[str, str] = {
    "a": "bathroom",
    "b": "bedroom",
    "c": "closet",
    "d": "dining room",
    "e": "entryway",
    "f": "family room",
    "g": "garage",
    "h": "hallway",
    "i": "library",
    "j": "laundry room",
    "k": "kitchen",
    "l": "living room",
    "m": "meeting room",
    "n": "lounge",
    "o": "office",
    "p": "porch",
    "r": "rec room",
    "s": "stairs",
    "t": "toilet",
    "u": "utility room",
    "v": "tv room",
    "w": "gym",
    "x": "outdoor",
    "y": "balcony",
    "z": "other room",
    "B": "bar",
    "C": "classroom",
    "D": "dining booth",
    "S": "spa",
    "Z": "storage",
}


def parse_house_rooms(scenes_dir: str, scan: str) -> List[str]:
    """Return sorted list of unique room type names present in the scene.

    Parses R lines from the MP3D .house file and maps the single-letter
    room type code to a human-readable name via _MP3D_ROOM_TYPES.
    """
    house_path = Path(scenes_dir) / "mp3d" / scan / f"{scan}.house"
    if not house_path.exists():
        return []
    names: set = set()
    with open(house_path) as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] == "R" and len(parts) >= 6:
                code = parts[5]
                name = _MP3D_ROOM_TYPES.get(code)
                if name:
                    names.add(name)
    return sorted(names)


def match_room(landmark: str, room_list: List[str]) -> Optional[str]:
    """Return the best-matching room name from room_list for the landmark string.

    Tries substring matching (case-insensitive) in both directions.
    Returns None if no match found.
    """
    lm = landmark.lower()
    for room in room_list:
        r = room.lower()
        if r in lm or lm in r:
            return room
    return None


class VisibilityChecker:
    """Habitat-sim wrapper for eye-level raycasting and optional RGB rendering.

    Parameters
    ----------
    scenes_dir:
        Root of the scene_datasets directory (same as rollout config).
    sensor_height:
        Camera height above navmesh in metres.
    rgb_cfg:
        Optional dict with keys ``width``, ``height``, ``hfov`` (degrees).
        When provided the sim includes an RGB-D rendering agent so that
        render_observation(), render_rgb(), and save_subpath_viz() can be
        called.
    """

    def __init__(
        self,
        scenes_dir: str,
        sensor_height: float = 1.5,
        rgb_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._scenes_dir = scenes_dir.rstrip("/")
        self._sensor_h   = sensor_height
        self._rgb_cfg    = rgb_cfg
        self._sim        = None
        self._scene_file = None

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------

    def load_scene(self, scene_file: str) -> None:
        """(Re-)initialise the simulator for a new scene."""
        import habitat_sim

        full_path = f"{self._scenes_dir}/{scene_file}"
        if self._scene_file == full_path and self._sim is not None:
            return

        if self._sim is not None:
            self._sim.close()

        backend = habitat_sim.SimulatorConfiguration()
        backend.scene_id           = full_path
        backend.load_semantic_mesh = True
        backend.enable_physics     = True   # required for cast_ray

        if self._rgb_cfg:
            render_cfg = {
                "height": 0.88,
                "radius": 0.18,
                "sensor_height": self._sensor_h,
                "rgb": {
                    "width": self._rgb_cfg["width"],
                    "height": self._rgb_cfg["height"],
                    "hfov": self._rgb_cfg.get("hfov", 90),
                    "viz_width": self._rgb_cfg["width"],
                    "viz_height": self._rgb_cfg["height"],
                    "vfov": self._rgb_cfg["height"] / self._rgb_cfg["width"] * 360.0,
                },
                "depth": {
                    "width": self._rgb_cfg.get("depth_width", self._rgb_cfg["width"]),
                    "height": self._rgb_cfg.get("depth_height", self._rgb_cfg["height"]),
                    "hfov": self._rgb_cfg.get("depth_hfov", self._rgb_cfg.get("hfov", 90)),
                    "min_depth": self._rgb_cfg.get("min_depth"),
                    "max_depth": self._rgb_cfg.get("max_depth"),
                },
            }
            agent_cfg = make_rgbd_agent_config(
                habitat_sim,
                render_cfg,
                include_rgb=False,
                include_rgb_viz=True,
                include_depth=True,
                include_actions=False,
            )
        else:
            agent_cfg = habitat_sim.agent.AgentConfiguration()
            agent_cfg.sensor_specifications = []

        self._sim        = habitat_sim.Simulator(
            habitat_sim.Configuration(backend, [agent_cfg])
        )
        self._scene_file = full_path

    def close(self) -> None:
        if self._sim is not None:
            self._sim.close()
            self._sim        = None
            self._scene_file = None

    # ------------------------------------------------------------------
    #  Core visibility check
    # ------------------------------------------------------------------

    def check(
        self,
        pos_start: np.ndarray,
        pos_end:   np.ndarray,
    ) -> Dict[str, Any]:
        """Back-trace a ray from pos_end toward pos_start at eye level.

        Both positions should be at navmesh height; sensor_height is added
        internally to get eye-level positions.

        Returns
        -------
        dict with:
          visible         bool
          distance        float  — straight-line eye-to-eye distance (m)
          obstacle        None | dict
            hit_distance  float  — distance along ray to first obstacle
            hit_fraction  float  — hit_distance / distance
            hit_point     list   — 3-D world coordinate of hit
            object_id     int
            semantic_cat  str
        """
        if self._sim is None:
            raise RuntimeError("Call load_scene() before check().")

        import habitat_sim

        eye_start = np.array(pos_start, dtype=np.float32)
        eye_start[1] += self._sensor_h
        eye_end = np.array(pos_end, dtype=np.float32)
        eye_end[1] += self._sensor_h

        vec  = eye_start - eye_end
        dist = float(np.linalg.norm(vec))

        if dist < 1e-4:
            return {"visible": True, "distance": 0.0, "obstacle": None}

        ray    = habitat_sim.geo.Ray(origin=eye_end, direction=vec / dist)
        result = self._sim.cast_ray(ray, max_distance=dist)

        if not result.has_hits():
            return {"visible": True, "distance": dist, "obstacle": None}

        hit = result.hits[0]
        return {
            "visible": False,
            "distance": dist,
            "obstacle": {
                "hit_distance": float(hit.ray_distance),
                "hit_fraction": float(hit.ray_distance / dist),
                "hit_point":    list(hit.point),
                "object_id":    hit.object_id,
                "semantic_cat": self._semantic_cat(hit.object_id),
            },
        }

    # ------------------------------------------------------------------
    #  Landmark uniqueness check
    # ------------------------------------------------------------------

    def check_landmark_uniqueness(
        self,
        pos_end:         np.ndarray,
        landmark:        str,
        semantic_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """From the end position, count how many instances of the landmark
        category are visible to the agent.

        Matching strategy (in priority order):
          1. Exact category match against each label in *semantic_labels*
             (grounded MP3D labels from the LLM rewriter components).
          2. Substring match on the raw *landmark* phrase as a fallback.

        For each matching object a ray is cast from the eye position to the
        object's AABB centre; the object is considered visible if the ray
        reaches it without being blocked by a different object first.

        Parameters
        ----------
        pos_end:
            Navmesh-level 3-D position of the sub-path end node.
        landmark:
            Landmark noun phrase (used as fallback when semantic_labels fail).
        semantic_labels:
            Ordered list of grounded MP3D category labels to try first
            (e.g. ["stair", "door"]).  Each is tried in turn; the first
            that finds at least one scene object wins.

        Returns
        -------
        dict with:
          unique           bool | None  — True if exactly one instance visible;
                                         None when no matching objects found
          visible_count    int          — visible matching instances
          total_in_scene   int          — all matching instances in scene
          matched_category str | None   — the semantic category that was matched
          matched_by       str          — "semantic_label" | "phrase_fallback"
          visible_ids      list[int]    — object_ids of visible instances
        """
        if self._sim is None:
            raise RuntimeError("Call load_scene() before check_landmark_uniqueness().")

        eye = np.array(pos_end, dtype=np.float32)
        eye[1] += self._sensor_h

        scene_objects = [
            obj for obj in self._sim.semantic_scene.objects
            if obj is not None and obj.category is not None
            and obj.category.name()
        ]

        candidates:  List     = []
        matched_cat: Optional[str] = None
        matched_by:  str      = "phrase_fallback"

        # ── 1. Try each grounded semantic label (exact / near-exact) ────
        for label in (semantic_labels or []):
            if not label or label.lower() in ("unknown", ""):
                continue
            label_lower = label.lower().strip()
            hits = [
                obj for obj in scene_objects
                if label_lower == obj.category.name().lower().strip()
                or label_lower in obj.category.name().lower()
                or obj.category.name().lower() in label_lower
            ]
            if hits:
                candidates  = hits
                matched_cat = hits[0].category.name()
                matched_by  = "semantic_label"
                break

        # ── 2. Phrase substring fallback ─────────────────────────────────
        if not candidates:
            lm_lower = landmark.lower().strip()
            for obj in scene_objects:
                cat = obj.category.name().lower()
                if lm_lower in cat or cat in lm_lower:
                    candidates.append(obj)
                    matched_cat = obj.category.name()

        if not candidates:
            return {
                "unique":           None,
                "visible_count":    0,
                "total_in_scene":   0,
                "matched_category": None,
                "matched_by":       matched_by,
                "visible_ids":      [],
                "note": f"no objects found matching '{landmark}'",
            }

        # ── cast rays to AABB centre + 6 face centres per candidate ─────
        visible_ids: list = []
        for obj in candidates:
            if self._aabb_visible_from(eye, obj):
                visible_ids.append(obj.id)

        n_visible = len(visible_ids)
        return {
            "unique":           n_visible == 1,
            "visible_count":    n_visible,
            "total_in_scene":   len(candidates),
            "matched_category": matched_cat,
            "matched_by":       matched_by,
            "visible_ids":      visible_ids,
        }

    # ------------------------------------------------------------------
    #  Episode-level helper
    # ------------------------------------------------------------------

    def check_episode(
        self,
        episode,
        db: Dict,
    ) -> List[Dict[str, Any]]:
        """Check visibility for every sub_path in one episode."""
        scan_db = db.get(episode.scan, {})
        results = []

        for sub_path in episode.sub_paths:
            if len(sub_path) < 2:
                continue
            start_node = sub_path[0]
            end_node   = sub_path[-1]

            if start_node not in scan_db or end_node not in scan_db:
                results.append({
                    "start_node": start_node,
                    "end_node":   end_node,
                    "error":      "node not found in connectivity DB",
                })
                continue

            pos_start = scan_db[start_node]
            pos_end   = scan_db[end_node]

            vis = self.check(pos_start, pos_end)
            results.append({
                "start_node": start_node,
                "end_node":   end_node,
                "start_pos":  pos_start.tolist(),
                "end_pos":    pos_end.tolist(),
                **vis,
            })

        return results

    # ------------------------------------------------------------------
    #  Rendering helpers
    # ------------------------------------------------------------------

    def render_observation(self, pos: np.ndarray, heading: float) -> Dict[str, np.ndarray]:
        """Place the RGB-D rendering agent and capture configured sensors.

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
        dict with ``rgb_viz`` and ``depth`` arrays.
        """
        if self._sim is None:
            raise RuntimeError("Call load_scene() before render_observation().")
        if not self._rgb_cfg:
            raise RuntimeError("Pass rgb_cfg to VisibilityChecker to enable rendering.")

        import habitat_sim
        import quaternion as qt

        agent = self._sim.initialize_agent(0)
        state = habitat_sim.AgentState()
        state.position = np.array(pos, dtype=np.float32)
        half = -heading / 2.0
        state.rotation = qt.quaternion(math.cos(half), 0.0, math.sin(half), 0.0)
        agent.set_state(state)

        raw = self._sim.get_sensor_observations()
        obs: Dict[str, np.ndarray] = {}
        if "rgb_viz" in raw:
            obs["rgb_viz"] = raw["rgb_viz"][..., :3].astype(np.uint8)
        if "rgb" in raw:
            obs["rgb"] = raw["rgb"][..., :3].astype(np.uint8)
        if "depth" in raw:
            obs["depth"] = raw["depth"].astype(np.float32)
        return obs

    def render_rgb(self, pos: np.ndarray, heading: float) -> np.ndarray:
        """Place the RGB-D rendering agent and capture the RGB panorama."""
        obs = self.render_observation(pos, heading)
        rgb = obs.get("rgb_viz")
        if rgb is None:
            rgb = obs.get("rgb")
        if rgb is None:
            raise RuntimeError("RGB rendering sensor is not configured.")
        return rgb

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------

    def _aabb_visible_from(self, eye: np.ndarray, obj) -> bool:
        """Return True if any probe ray from *eye* reaches *obj*.

        Probes: AABB centre + 6 face centres (one per axis direction).
        The face centres are shrunk 10 % inward so they stay inside the
        object surface rather than sitting exactly on the boundary.

        Two hit conditions are accepted:
          1. the first raycast hit uses the same object id
          2. the first hit point lands inside this object's AABB, with a
             small margin to tolerate thin wall-mounted objects such as
             mirrors or panels whose collision ids may differ
        """
        import habitat_sim

        c = np.array(obj.aabb.center, dtype=np.float32)
        h = np.array(obj.aabb.sizes,  dtype=np.float32) * 0.4  # 80 % of half-extent
        half = np.array(obj.aabb.sizes, dtype=np.float32) * 0.5
        margin = np.maximum(0.05, 0.25 * half)
        aabb_lo = c - half - margin
        aabb_hi = c + half + margin

        probes = [
            c,
            c + np.array([h[0],    0,    0], dtype=np.float32),
            c - np.array([h[0],    0,    0], dtype=np.float32),
            c + np.array([   0, h[1],    0], dtype=np.float32),
            c - np.array([   0, h[1],    0], dtype=np.float32),
            c + np.array([   0,    0, h[2]], dtype=np.float32),
            c - np.array([   0,    0, h[2]], dtype=np.float32),
        ]

        for pt in probes:
            vec  = pt - eye
            dist = float(np.linalg.norm(vec))
            if dist < 0.05:          # agent is essentially inside the object
                return True
            ray    = habitat_sim.geo.Ray(origin=eye, direction=vec / dist)
            result = self._sim.cast_ray(ray, max_distance=dist + 0.5)
            if not result.has_hits():
                return True                       # unobstructed
            hit = result.hits[0]
            if hit.object_id == obj.id:
                return True                       # first hit is the target
            hp = np.array(hit.point, dtype=np.float32)
            if np.all(hp >= aabb_lo) and np.all(hp <= aabb_hi):
                return True                       # collision landed on target region
        return False

    def _semantic_cat(self, object_id: int) -> str:
        try:
            obj = self._sim.semantic_scene.objects[object_id]
            if obj is not None and obj.category is not None:
                return obj.category.name()
        except Exception:
            pass
        return "unknown"


from src.viz import save_obs_strip, save_subpath_viz  # noqa: F401  re-exported for callers


# ---------------------------------------------------------------------------
#  Pipeline helpers  (called by scripts — return all_results for JSON saving)
# ---------------------------------------------------------------------------

def run_visibility_check(
    episodes,
    db: Dict,
    checker: "VisibilityChecker",
    viz_root: Optional[Any] = None,
    rgb_cfg: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """Run the visibility check pipeline over all episodes.

    Groups episodes by scan (one sim load per scan).  Optionally renders
    3-panel PNGs when *rgb_cfg* and *viz_root* are provided.

    Parameters
    ----------
    episodes:   list of LandmarkRxREpisode
    db:         connectivity DB from load_connectivity()
    checker:    a VisibilityChecker instance (scene not yet loaded)
    viz_root:   Path — parent directory for PNG output; None = no rendering
    rgb_cfg:    dict with width/height/hfov; None = no rendering

    Returns
    -------
    dict keyed by str(instruction_id), sorted numerically.
    Each value: {"scan", "language", "sub_paths": [...]}.
    """
    from collections import defaultdict
    from pathlib import Path as _Path

    by_scan: Dict[str, list] = defaultdict(list)
    for ep in episodes:
        by_scan[ep.scan].append(ep)

    all_results: Dict[str, Dict] = {}

    for scan, scan_eps in by_scan.items():
        print(f"\n[{scan}]  loading scene …")
        checker.load_scene(scan_eps[0].scene_file)

        for ep in scan_eps:
            scan_db     = db.get(ep.scan, {})
            sub_results = []
            n_sub       = len(ep.sub_paths)

            for idx, sub_path in enumerate(ep.sub_paths):
                if len(sub_path) < 2:
                    continue

                start_node = sub_path[0]
                end_node   = sub_path[-1]

                if start_node not in scan_db or end_node not in scan_db:
                    sub_results.append({
                        "start_node": start_node,
                        "end_node":   end_node,
                        "error":      "node not found in connectivity DB",
                    })
                    continue

                pos_start = scan_db[start_node]
                pos_end   = scan_db[end_node]
                sub_instr = (ep.sub_instructions[idx]
                             if idx < len(ep.sub_instructions) else "")

                result = checker.check(pos_start, pos_end)
                sub_results.append({
                    "start_node":      start_node,
                    "end_node":        end_node,
                    "start_pos":       pos_start.tolist(),
                    "end_pos":         pos_end.tolist(),
                    "sub_instruction": sub_instr,
                    **result,
                })

                if result["visible"]:
                    print(f"  [{idx}] VISIBLE  dist={result['distance']:.2f}m"
                          f"  instr_id={ep.instruction_id}")
                else:
                    obs = result["obstacle"]
                    print(f"  [{idx}] BLOCKED  dist={result['distance']:.2f}m"
                          f"  hit={obs['hit_distance']:.2f}m"
                          f" ({obs['hit_fraction']*100:.0f}%)"
                          f"  cat={obs['semantic_cat']}"
                          f"  instr_id={ep.instruction_id}")

                if rgb_cfg is not None and viz_root is not None:
                    viz_path = (_Path(viz_root) / str(ep.instruction_id)
                                / f"sub_{idx:02d}.png")
                    save_subpath_viz(
                        checker=checker,
                        pos_start=pos_start, pos_end=pos_end,
                        result=result, out_path=viz_path,
                        sub_idx=idx, sub_total=n_sub,
                        sub_instruction=sub_instr,
                        episode_id=ep.instruction_id, scan=ep.scan,
                    )

            all_results[str(ep.instruction_id)] = {
                "scan":      ep.scan,
                "language":  ep.language,
                "sub_paths": sub_results,
            }

    return dict(sorted(all_results.items(), key=lambda kv: int(kv[0])))


def run_landmark_uniqueness_check(
    episodes,
    db: Dict,
    rewritten: Dict,
    checker: "VisibilityChecker",
    landmark_mapping: Optional[Dict[str, List[str]]] = None,
    obs_dir: Optional[Any] = None,
    img_w: int = 320,
    img_h: int = 240,
) -> Dict[str, Dict]:
    """Run the landmark uniqueness check pipeline over all episodes.

    Parameters
    ----------
    episodes:         list of LandmarkRxREpisode
    db:               connectivity DB from load_connectivity()
    rewritten:        parsed sub_instructions_rewritten.json (dict with "episodes" key)
    checker:          a VisibilityChecker instance (scene not yet loaded)
    landmark_mapping: parsed landmark_mapping.json — original_mention → [semantic_labels].
                      Used as fallback when a component's semantic_label is "unknown".
    obs_dir:          Path — if given, render a 4-view observation strip per sub-path
    img_w, img_h:     per-view render resolution when obs_dir is set

    Returns
    -------
    dict keyed by str(instruction_id), sorted numerically.
    Each value: {"scan", "sub_paths": [...]}.
    """
    from collections import defaultdict
    from pathlib import Path as _Path

    by_scan: Dict[str, list] = defaultdict(list)
    for ep in episodes:
        by_scan[ep.scan].append(ep)

    all_results: Dict[str, Dict] = {}

    for scan, scan_eps in by_scan.items():
        print(f"\n[{scan}]  loading scene …")
        checker.load_scene(scan_eps[0].scene_file)
        scan_db    = db.get(scan, {})
        room_list  = parse_house_rooms(checker._scenes_dir, scan)
        print(f"  rooms: {room_list}")

        for ep in scan_eps:
            ep_rewritten     = rewritten["episodes"][str(ep.instruction_id)]
            sub_path_results = []
            n_sub            = len(ep_rewritten["sub_paths"])

            for sub_entry in ep_rewritten["sub_paths"]:
                sub_idx  = sub_entry["sub_idx"]
                landmark = sub_entry.get("landmark", "")

                if sub_entry.get("landmark_category") == "spatial":
                    sub_path_results.append({
                        "sub_idx": sub_idx, "landmark": landmark,
                        "skipped": "spatial landmark ignored",
                    })
                    continue

                if sub_entry.get("landmark_category") == "room":
                    matched_room = match_room(landmark, room_list)
                    sub_path_results.append({
                        "sub_idx":      sub_idx,
                        "landmark":     landmark,
                        "category":     "room",
                        "matched_room": matched_room,
                        "room_list":    room_list,
                    })
                    continue

                if sub_idx >= len(ep.sub_paths):
                    sub_path_results.append({
                        "sub_idx": sub_idx, "landmark": landmark,
                        "error": "sub_idx out of range",
                    })
                    continue

                end_node = ep.sub_paths[sub_idx][-1]
                if end_node not in scan_db:
                    sub_path_results.append({
                        "sub_idx": sub_idx, "landmark": landmark,
                        "error": f"end node '{end_node}' not in connectivity DB",
                    })
                    continue

                if not landmark:
                    sub_path_results.append({
                        "sub_idx": sub_idx, "landmark": landmark,
                        "error": "empty landmark",
                    })
                    continue

                pos_end = scan_db[end_node]

                # Build ordered semantic label list from components
                semantic_labels: List[str] = []
                for comp in sub_entry.get("components", []):
                    label = comp.get("semantic_label", "").strip()
                    mention = comp.get("original_mention", "").strip().lower()
                    if label and label.lower() not in ("unknown", ""):
                        if label not in semantic_labels:
                            semantic_labels.append(label)
                    elif mention and landmark_mapping:
                        # fallback: look up mention in cross-episode mapping
                        for mapped in landmark_mapping.get(mention, []):
                            if mapped.lower() not in ("unknown", "") \
                                    and mapped not in semantic_labels:
                                semantic_labels.append(mapped)

                result = checker.check_landmark_uniqueness(
                    pos_end, landmark, semantic_labels=semantic_labels or None
                )
                sub_path_results.append({"sub_idx": sub_idx, "landmark": landmark,
                                         "semantic_labels_used": semantic_labels,
                                         **result})

                u   = result["unique"]
                cat = result["matched_category"] or "—"
                by  = result.get("matched_by", "")
                if u is None:
                    tag = "NOT_FOUND "
                elif result["visible_count"] == 0:
                    tag = "NOT_VISIBLE"
                elif u:
                    tag = "UNIQUE     "
                else:
                    tag = "AMBIGUOUS  "
                print(f"  [{sub_idx}] {tag}  "
                      f"vis={result['visible_count']}/{result['total_in_scene']}  "
                      f"cat={cat!r}  by={by}  landmark={landmark!r}"
                      f"  ep={ep.instruction_id}")

                if obs_dir is not None:
                    heading = (ep.headings[sub_idx]
                               if ep.headings and sub_idx < len(ep.headings)
                               else ep.heading)
                    obs_path = (_Path(obs_dir) / str(ep.instruction_id)
                                / f"sub_{sub_idx:02d}.png")
                    save_obs_strip(
                        checker=checker,
                        pos_end=pos_end,
                        heading=heading,
                        episode_id=ep.instruction_id,
                        scan=ep.scan,
                        sub_idx=sub_idx,
                        sub_total=n_sub,
                        landmark=landmark,
                        instruction=sub_entry.get("landmark_instruction", ""),
                        original=sub_entry.get("original", ""),
                        out_path=obs_path,
                        result=result,
                        img_w=img_w,
                        img_h=img_h,
                    )

            all_results[str(ep.instruction_id)] = {
                "scan":      ep.scan,
                "sub_paths": sub_path_results,
            }

    return dict(sorted(all_results.items(), key=lambda kv: int(kv[0])))
