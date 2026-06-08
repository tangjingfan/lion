"""Equirectangular panorama geometry: unprojection + per-instance voxels.

Shared by the landmark-synthesis rescue (step 11) and any tool that
needs "which instances are visible from this pose, and how much of
them" answered in a distance-independent way.  Voxel counts measure the
spatial extent of the visible surface, decoupled from how close the
agent happens to be — cleaner than pixel counts, which conflate size
with proximity.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

# Voxel size for downsampling the unprojected point cloud (metres in
# world coordinates). 0.1 m = 10 cm cells — small enough to capture
# the shape of furniture and people, large enough that nearby points
# collapse into the same cell so a 1-m-wide fridge surface contributes
# tens of voxels rather than thousands of unique pixels.
VOXEL_SIZE_M = 0.10


def heading_from_to(pos_from: np.ndarray, pos_to: np.ndarray) -> float:
    """Clockwise heading from north (−Z) in radians, to face pos_to from pos_from.

    Habitat frame: +X is east and −Z is north, so heading 0 looks toward −Z
    and increases clockwise (toward +X).
    """
    dx = float(pos_to[0] - pos_from[0])
    dz = float(pos_to[2] - pos_from[2])
    return math.atan2(dx, -dz)


def unproject_equirect(
    depth:     np.ndarray,
    semantic:  np.ndarray,
    pos:       np.ndarray,
    heading:   float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject an equirectangular depth + semantic panorama into a
    flat 3-D point cloud in **world** coordinates.

    Habitat's equirectangular sensor returns radial distance to first
    hit along each pixel's ray. For pixel ``(u, v)`` on a ``(H, W)``
    panorama:

      θ = 2π · u / W − π        ∈ [-π, π]   (yaw, +x = right of agent)
      φ = π/2 − π · v / H       ∈ [-π/2, π/2] (pitch, +y = up)

    The sensor frame at heading=0 has the agent facing −z, so:

      x_local =  d · cos(φ) · sin(θ)
      y_local =  d · sin(φ)
      z_local = -d · cos(φ) · cos(θ)

    A heading rotation around y maps local → world before adding ``pos``.
    Returns ``(points (N, 3) float32, instance_ids (N,) int32)`` with
    invalid pixels (non-finite depth, zero, or unannotated) filtered
    out so downstream consumers can voxelise directly.
    """
    H, W  = depth.shape[:2]
    u     = np.arange(W, dtype=np.float32).reshape(1, W)
    v     = np.arange(H, dtype=np.float32).reshape(H, 1)
    theta = 2.0 * np.pi * (u / W) - np.pi
    phi   = 0.5 * np.pi - np.pi * (v / H)
    cos_phi = np.cos(phi)
    d       = depth.astype(np.float32, copy=False)

    x_local = d * cos_phi * np.sin(theta)
    y_local = d * np.sin(phi)
    z_local = -d * cos_phi * np.cos(theta)

    valid = np.isfinite(d) & (d > 1e-3) & (d < 1e3) & (semantic >= 0)
    pts_local = np.stack([x_local[valid], y_local[valid], z_local[valid]], axis=1)
    iids      = semantic[valid].astype(np.int32, copy=False)
    if pts_local.size == 0:
        return pts_local.astype(np.float32), iids

    cos_h, sin_h = float(np.cos(heading)), float(np.sin(heading))
    rot = np.array([[ cos_h, 0.0, sin_h],
                    [   0.0, 1.0,   0.0],
                    [-sin_h, 0.0, cos_h]], dtype=np.float32)
    pts_world = pts_local @ rot.T + np.asarray(pos, dtype=np.float32)
    return pts_world.astype(np.float32, copy=False), iids


def voxelise_per_instance(
    points:    np.ndarray,
    iids:      np.ndarray,
    voxel_size: float,
) -> Dict[int, int]:
    """Return ``{instance_id: unique_voxel_count}``.

    Each 3-D point is binned into a voxel by ``floor(p / voxel_size)``;
    for each instance, count how many distinct cells its points occupy.
    Empty inputs return ``{}``.
    """
    if points.size == 0:
        return {}
    vox = np.floor(points / float(voxel_size)).astype(np.int64)
    # Combine instance id + (vx, vy, vz) into a single key so np.unique
    # gives us (iid, voxel) pairs in one shot.
    key = np.empty((points.shape[0], 4), dtype=np.int64)
    key[:, 0]  = iids.astype(np.int64, copy=False)
    key[:, 1:] = vox
    uniq = np.unique(key, axis=0)
    out: Dict[int, int] = {}
    for row in uniq:
        i = int(row[0])
        out[i] = out.get(i, 0) + 1
    return out


def visible_instance_voxels(
    depth:    np.ndarray,
    semantic: np.ndarray,
    pos:      np.ndarray,
    heading:  float,
    voxel_size: float = VOXEL_SIZE_M,
) -> Dict[int, int]:
    """Render-time helper: depth + semantic panorama → ``{iid: voxels}``.

    Pipeline: unproject equirect depth into world-frame points, drop
    invalid pixels, voxel-downsample at ``voxel_size`` metres, count
    distinct voxels per instance id.
    """
    points, iids = unproject_equirect(depth, semantic, pos, heading)
    return voxelise_per_instance(points, iids, voxel_size)
