"""
Load and query the MP3D connectivity graph.

Source formats supported:
  1. MP3D .house files  (already in scene_datasets/ — no extra download needed)
  2. Directory of {scan}_connectivity.json files  (R2R standard, optional)
  3. Pre-built pickle connectivity_graphs.pkl      (CL_CoTNav format, optional)

Coordinate systems
------------------
MP3D .house / connectivity JSON positions are in MP3D frame:
    x = right,  y = forward (into building),  z = UP

Habitat-sim frame:
    x = right,  y = UP,  z = backward

Transformation → habitat_pos = [mp3d_x, mp3d_z, -mp3d_y]

Verified: .house positions match connectivity JSON positions within ~3 cm.
All nodes in .house are marked "included" in the connectivity JSON.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# {scan_id: {node_id: np.array([x, y, z])}} in Habitat frame
ConnectivityDB = Dict[str, Dict[str, np.ndarray]]


def load_connectivity(
    scenes_dir: str,
    scans: Optional[List[str]] = None,
    *,
    json_dir: Optional[str] = None,
    pkl_path: Optional[str] = None,
) -> ConnectivityDB:
    """Load MP3D connectivity for the requested scans.

    Strategy
    --------
    1. If ``pkl_path`` is given, load everything from that pickle.
    2. Otherwise, for each scan:
         a. If ``json_dir`` is given and ``{scan}_connectivity.json`` exists
            there, load from that file (positions at navmesh level, accurate).
         b. Otherwise fall back to parsing ``{scan}.house`` (positions at
            camera height; will be snapped to navmesh by the environment).

    Parameters
    ----------
    scenes_dir:
        Root of the scene_datasets directory.  Always required for .house
        fallback, e.g. ~/project/VLN_dataset/data/scene_datasets/.
    scans:
        Scan IDs to load.  None = all .house files found under scenes_dir.
    json_dir:
        Optional directory of {scan}_connectivity.json files.
    pkl_path:
        Optional path to a connectivity_graphs.pkl (highest priority).
    """
    if pkl_path:
        return _load_from_pickle(Path(pkl_path), scans)

    mp3d_dir = Path(scenes_dir) / "mp3d"
    if scans is None:
        scans = [p.name for p in mp3d_dir.iterdir() if p.is_dir()]

    db: ConnectivityDB = {}
    json_path = Path(json_dir) if json_dir else None

    for scan in scans:
        json_file = json_path / f"{scan}_connectivity.json" if json_path else None
        if json_file and json_file.exists():
            with open(json_file) as f:
                db[scan] = _parse_json_nodes(json.load(f))
        else:
            house_file = mp3d_dir / scan / f"{scan}.house"
            if not house_file.exists():
                raise FileNotFoundError(
                    f"No connectivity JSON and no .house file for scan {scan!r}.\n"
                    f"  Looked for: {json_file}\n"
                    f"  Looked for: {house_file}"
                )
            db[scan] = _parse_house_file(house_file)

    return db


def get_position(db: ConnectivityDB, scan: str, node_id: str) -> np.ndarray:
    """Return the Habitat-frame 3-D position of a node."""
    return db[scan][node_id]


def path_to_positions(
    db: ConnectivityDB, scan: str, node_ids: List[str]
) -> List[np.ndarray]:
    """Convert a list of node IDs to Habitat-frame 3-D positions."""
    return [get_position(db, scan, nid) for nid in node_ids]


def heading_to_rotation(heading: float) -> np.ndarray:
    """Convert a Landmark-RxR heading (radians, clockwise from north) to a
    quaternion [x, y, z, w] for habitat-sim.

    In Habitat-sim the agent faces -Z at heading=0 (north).
    Positive heading rotates clockwise viewed from above.
    """
    half = -heading / 2.0
    return np.array([0.0, np.sin(half), 0.0, np.cos(half)], dtype=np.float32)


# ---------------------------------------------------------------------------
# MP3D coordinate transform (shared by all loaders)
# ---------------------------------------------------------------------------

def _mp3d_to_habitat(x: float, y: float, z: float) -> np.ndarray:
    """MP3D (x=right, y=forward, z=up) → Habitat (x=right, y=up, z=back)."""
    return np.array([x, z, -y], dtype=np.float32)


# ---------------------------------------------------------------------------
# Loader: .house files
# ---------------------------------------------------------------------------

def _parse_house_file(house_file: Path) -> Dict[str, np.ndarray]:
    """Extract {node_id → habitat_pos} from a single .house file.

    Positions are at camera height (eye level, ~1.5 m above floor).
    The environment's _snap() will project them to the navmesh.

    .house P-line format:
        P  {node_id}  {idx}  {region}  {?}  {x}  {y}  {z}  ...
    Columns 5-7 are the position in MP3D frame.
    """
    nodes: Dict[str, np.ndarray] = {}
    with open(house_file) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] != "P":
                continue
            node_id = parts[1]
            x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
            nodes[node_id] = _mp3d_to_habitat(x, y, z)
    return nodes


def _parse_json_nodes(nodes: list) -> Dict[str, np.ndarray]:
    """Parse connectivity JSON nodes.

    Each node's ``pose`` stores the camera-to-world transform (MP3D frame).
    The ``height`` field is the camera height *above the navmesh* for that
    viewpoint.  Subtracting it converts the camera position to a navmesh-level
    position, which is what habitat-sim's pathfinder and metrics expect.
    """
    node_map: Dict[str, np.ndarray] = {}
    for node in nodes:
        if not node.get("included", True):
            continue
        pose = np.array(node["pose"], dtype=np.float64).reshape(4, 4)
        mp3d = pose[:3, 3]                          # camera pos in MP3D frame
        cam_hab = _mp3d_to_habitat(mp3d[0], mp3d[1], mp3d[2])

        # Subtract camera height above navmesh (MP3D z = Habitat y = UP).
        # This converts the eye-level camera position to the navmesh surface,
        # consistent with the agent's base position during navigation.
        cam_hab[1] -= node.get("height", 1.5)

        node_map[node["image_id"]] = cam_hab
    return node_map


# ---------------------------------------------------------------------------
# Loader: CL_CoTNav pickle  (optional alternative)
# ---------------------------------------------------------------------------

def _load_from_pickle(
    pkl_path: Path, scans: Optional[List[str]]
) -> ConnectivityDB:
    """Load from connectivity_graphs.pkl — positions already in Habitat frame."""
    with open(pkl_path, "rb") as f:
        conn_graphs = pickle.load(f)

    keys = scans if scans is not None else list(conn_graphs.keys())
    db: ConnectivityDB = {}
    for scan_id in keys:
        g = conn_graphs[scan_id]
        db[scan_id] = {
            node: np.array(g.nodes[node]["position"], dtype=np.float32)
            for node in g.nodes
        }
    return db
