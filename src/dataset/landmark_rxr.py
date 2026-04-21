"""
Load and filter Landmark-RxR episodes.

Episode fields (from LandmarkRxR_*.json):
  instruction_id  int   — unique per instruction text (primary key)
  path_id         int   — trajectory ID; multiple instructions can share one path
  scan            str   — MP3D scene ID  (e.g. "X7HyMhZNoso")
  heading         float — agent starting heading in radians (azimuth, clockwise from north)
  path            list  — ordered list of MP3D viewpoint node IDs
  instruction     str   — full natural-language instruction
  language        str   — e.g. "en-US", "en-IN"
  sub_paths       list  — path broken into per-landmark segments (list of [start,end] node pairs)
  sub_instructions list — instruction for each sub-path segment
  headings        list  — heading at each sub-path waypoint
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class LandmarkRxREpisode:
    instruction_id: int
    path_id: int
    scan: str
    heading: float          # radians, clockwise from north
    path: List[str]         # ordered MP3D viewpoint node IDs
    instruction: str
    language: str
    sub_paths: List[List[str]]
    sub_instructions: List[str]
    headings: List[float]   # heading at each sub-path waypoint

    # Filled in by the env from the connectivity graph
    start_position: Optional[List[float]] = field(default=None, repr=False)
    start_rotation: Optional[List[float]] = field(default=None, repr=False)
    reference_path: Optional[List[List[float]]] = field(default=None, repr=False)
    goal_position: Optional[List[float]] = field(default=None, repr=False)

    @property
    def scene_file(self) -> str:
        """Relative path used by Habitat-sim to locate the scene GLB."""
        return f"mp3d/{self.scan}/{self.scan}.glb"

    @property
    def path_key(self) -> str:
        """Unique string key for this episode (mirrors CL_CoTNav convention)."""
        return str(self.instruction_id)


def load_episodes(
    data_path: str,
    *,
    include_scans: Optional[List[str]] = None,
    exclude_scans: Optional[List[str]] = None,
    instruction_ids: Optional[List[int]] = None,
    languages: Optional[List[str]] = None,
    max_episodes: Optional[int] = None,
) -> List[LandmarkRxREpisode]:
    """Load and filter Landmark-RxR episodes from a JSON split file.

    Parameters
    ----------
    data_path:
        Full path to LandmarkRxR_{split}.json.
    include_scans:
        Whitelist of scan IDs.  None / [] = accept all scans.
    exclude_scans:
        Blacklist of scan IDs applied after include_scans filter.
    instruction_ids:
        Explicit list of instruction_ids to keep.  None / [] = keep all.
    languages:
        Language whitelist (e.g. ["en-US"]).  None / [] = keep all.
    max_episodes:
        Hard cap on total returned episodes.  None = no cap.

    Returns
    -------
    List of LandmarkRxREpisode in the order they appear in the JSON.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    with open(path) as f:
        raw = json.load(f)

    # Normalise filter inputs
    include_scans = set(include_scans) if include_scans else None
    exclude_scans = set(exclude_scans) if exclude_scans else set()
    instr_id_set = set(instruction_ids) if instruction_ids else None
    lang_set = set(languages) if languages else None

    episodes: List[LandmarkRxREpisode] = []
    for rec in raw:
        scan = rec["scan"]

        if include_scans is not None and scan not in include_scans:
            continue
        if scan in exclude_scans:
            continue

        instr_id = int(rec["instruction_id"])
        if instr_id_set is not None and instr_id not in instr_id_set:
            continue

        lang = rec.get("language", "")
        if lang_set is not None and lang not in lang_set:
            continue

        episodes.append(
            LandmarkRxREpisode(
                instruction_id=instr_id,
                path_id=int(rec["path_id"]),
                scan=scan,
                heading=float(rec["heading"]),
                path=rec["path"],
                instruction=rec["instruction"],
                language=lang,
                sub_paths=rec.get("sub_paths", []),
                sub_instructions=rec.get("sub_instructions", []),
                headings=rec.get("headings", []),
            )
        )

        if max_episodes is not None and len(episodes) >= max_episodes:
            break

    return episodes


def episodes_from_config(cfg: dict) -> List[LandmarkRxREpisode]:
    """Convenience wrapper that reads filter parameters from a config dict."""
    dataset_cfg = cfg["dataset"]
    scene_cfg = cfg.get("scenes", {})
    sel_cfg = cfg.get("selection", {})

    # Honour from_yaml: load instruction_ids from a replay file
    instruction_ids = sel_cfg.get("instruction_ids") or []
    from_yaml = sel_cfg.get("from_yaml")
    if from_yaml:
        import yaml
        with open(from_yaml) as f:
            replay = yaml.safe_load(f)
        instruction_ids = replay.get("instruction_ids", instruction_ids)

    return load_episodes(
        data_path=dataset_cfg["data_path"],
        include_scans=scene_cfg.get("include") or None,
        exclude_scans=scene_cfg.get("exclude") or [],
        instruction_ids=instruction_ids or None,
        languages=sel_cfg.get("languages") or None,
        max_episodes=sel_cfg.get("max_episodes"),
    )
