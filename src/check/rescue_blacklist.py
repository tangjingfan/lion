"""Find replacement landmarks for sub-paths dropped by 03_blacklist_landmark.

The blacklist filter drops a sub-path when its instruction-derived
landmark is too generic to ground ("wall", "door", "room", "doorway",
...). This rescue does NOT recover the original landmark — it picks a
**different** referrable landmark visible at the sub-path's end pose,
producing a synthesized sub-instruction. Records flow through the rest
of the pipeline marked ``landmark_source = "synthesized"`` so downstream
consumers can choose to use or skip them.

A good replacement landmark must:

  1. Be visible at the sub-path **end pose** (we want the thing the
     agent walked towards).
  2. Be **progressively approached** along the landmark half — i.e.
     ``distance(end_pos, instance_center) < distance(partition_pos, instance_center)``.
  3. Have a concrete MPCAT40 category (not ``wall`` / ``door`` /
     ``window`` / ``floor`` / ``ceiling`` / etc.).
  4. Be referrable — preferentially a category that has **exactly one**
     instance visible at the end pose, so saying "go to the X" is
     unambiguous.

Output: ``target_instances/{scan}/blacklist_rescue.json`` — one
record per rescued sub-path. The consolidate step (11) reads this and
emits synthesized rows into ``dataset.json`` alongside the original
ones.

Usage
-----
  python src/check/rescue_blacklist.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --exp configs/selection/val_unseen/one_scene_partial.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import get_filter_dir, get_run_dir, resolve_exp
from src.check.query_scene_instance import _render_mask_for_rollout_frame
from src.dataset.landmark_rxr import episodes_from_config
from src.env.connectivity import _mp3d_to_habitat, load_connectivity
from src.process.visibility import VisibilityChecker


# Category names that are unreferrable on their own — we never pick
# these as a replacement landmark, even when they dominate the FOV.
# "railing" was previously included but user feedback flagged it as
# a valid navigation landmark ("walk to the railing"), so it stays out.
LANDMARK_BLACKLIST_CATS = {
    "wall", "floor", "ceiling", "door", "window", "doorway",
    "blinds", "curtain", "misc", "void",
}

# Coarse buckets that we'd prefer NOT to pick because they're not
# "concrete enough" on their own. We tolerate them only when nothing
# better is approached.
COARSE_BUCKETS = {
    "appliances", "objects", "furniture", "lighting",
}

# A landmark must get at least this much closer between partition and
# end pose to count as "progressively approached" (metres).
APPROACH_THRESHOLD_M = 0.5

# An instance must have at least this many visible pixels at the end
# pose to be considered a viable replacement landmark. 1000 px in a
# 1024×512 panorama (~0.19% of FOV) is a small but visually obvious
# object; 200 (the previous default) admitted near-invisible specks.
MIN_VISIBLE_PIXELS = 1000


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _blacklist_drops(filt_dir: Path) -> List[Tuple[int, int, Dict]]:
    """Return ``[(ep_id, sub_idx, drop_record), ...]`` from
    ``02_blacklist_dropped.yaml``."""
    payload = _load_yaml(filt_dir / "02_blacklist_dropped.yaml")
    out: List[Tuple[int, int, Dict]] = []
    for ep_id_str, info in (payload.get("dropped") or {}).items():
        for sub_idx_str, rec in ((info or {}).get("subs") or {}).items():
            try:
                out.append((int(ep_id_str), int(sub_idx_str), rec or {}))
            except ValueError:
                continue
    return out


def _instance_meta_from_house(scenes_dir: str, scan: str) -> Dict[int, Dict]:
    """Map ``instance_id -> {category, center_habitat}`` from a scan's .house.

    Habitat's ``sim.semantic_annotations()`` is unreliable through the
    LION ``VisibilityChecker`` wrapper (returns empty objects list in our
    setup), so we read the same data straight from ``.house``:

      * ``C`` rows define ``category_index -> mpcat40_name``.
      * ``O`` rows define ``instance_id, category_index, x, y, z`` in MP3D
        coordinates (x=right, y=forward, z=up).

    Centers are converted to Habitat coordinates via
    :func:`src.env.connectivity._mp3d_to_habitat` so they're comparable
    with node positions returned by ``load_connectivity``.
    """
    house_path = Path(scenes_dir) / "mp3d" / scan / f"{scan}.house"
    if not house_path.exists():
        return {}

    cat_index_to_mpcat40: Dict[int, str] = {}
    out: Dict[int, Dict] = {}
    with open(house_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "C" and len(parts) >= 6:
                try:
                    cat_index = int(parts[1])
                except ValueError:
                    continue
                name = parts[5].replace("#", " ").strip().lower()
                if name:
                    cat_index_to_mpcat40[cat_index] = name
            elif parts[0] == "O" and len(parts) >= 7:
                try:
                    iid = int(parts[1])
                    cat_index = int(parts[3])
                    x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
                except ValueError:
                    continue
                out[iid] = {
                    "_cat_index": cat_index,
                    "center":     _mp3d_to_habitat(x, y, z),
                }
    for iid, meta in out.items():
        meta["category"] = cat_index_to_mpcat40.get(meta.pop("_cat_index"), "")
    return out


def _resolve_node_pos(
    node_id: str, virtual_nodes: Dict[str, List[float]], scan_db: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    if isinstance(node_id, str) and node_id.startswith("virt:"):
        pos = virtual_nodes.get(node_id)
        return np.asarray(pos, dtype=np.float32) if pos is not None else None
    if node_id in scan_db:
        return np.asarray(scan_db[node_id], dtype=np.float32)
    return None


def _visible_instance_pixels(sem_array: np.ndarray) -> Dict[int, int]:
    """Return ``{instance_id: pixel_count}`` for a rendered semantic
    panorama (raw instance-id buffer)."""
    flat = sem_array.ravel()
    unique, counts = np.unique(flat, return_counts=True)
    return {int(i): int(c) for i, c in zip(unique, counts) if int(i) >= 0}


def _pick_replacement_landmark(
    visible_pixels:    Dict[int, int],
    inst_meta:         Dict[int, Dict],
    partition_pos:     np.ndarray,
    end_pos:           np.ndarray,
    scene_category_counts: Counter,
) -> Optional[Dict]:
    """Choose a single replacement landmark, or None if no candidate fits.

    Selection priority:
      1. category not in LANDMARK_BLACKLIST_CATS
      2. progressively approached: ``dist(end) < dist(partition) - APPROACH_THRESHOLD_M``
      3. visible >= MIN_VISIBLE_PIXELS at the end pose
      4. prefer category with exactly 1 visible instance in this FOV
         (unambiguous "the X" reference)
      5. among ties, prefer non-coarse-bucket
      6. final tiebreak: closest to end_pos
    """
    candidates: List[Dict] = []
    cat_visible_count: Counter = Counter()

    for iid, pix in visible_pixels.items():
        if iid not in inst_meta or pix < MIN_VISIBLE_PIXELS:
            continue
        cat = inst_meta[iid]["category"]
        if not cat or cat in LANDMARK_BLACKLIST_CATS:
            continue
        center = inst_meta[iid]["center"]
        dist_p = float(np.linalg.norm(partition_pos - center))
        dist_e = float(np.linalg.norm(end_pos - center))
        approached = (dist_p - dist_e) >= APPROACH_THRESHOLD_M
        if not approached:
            continue
        cat_visible_count[cat] += 1
        candidates.append({
            "instance_id": iid,
            "category":    cat,
            "pixel_count": pix,
            "dist_partition_m": dist_p,
            "dist_end_m":       dist_e,
            "approach_m":       dist_p - dist_e,
        })

    if not candidates:
        return None

    # Tier 1: category has exactly one visible instance in this FOV.
    unique_view = [c for c in candidates if cat_visible_count[c["category"]] == 1]
    pool = unique_view or candidates
    # Tier 2: prefer non-coarse buckets.
    fine = [c for c in pool if c["category"] not in COARSE_BUCKETS]
    pool = fine or pool

    # Tier 3: balance proximity-to-end against visual prominence.
    # ``score = dist_end_m / sqrt(pixel_count)`` — interprets as
    # "distance per unit visual size", so a small-but-close instance
    # (e.g. a tiny stair at 3 m) doesn't beat a huge-but-farther one
    # (e.g. a railing filling the FOV at 5 m). Lower = better.
    for c in pool:
        c["score"] = c["dist_end_m"] / max(1.0, float(c["pixel_count"]) ** 0.5)
    pool.sort(key=lambda c: c["score"])
    chosen = pool[0]
    chosen["scene_instance_count"]    = int(scene_category_counts.get(chosen["category"], 0))
    chosen["fov_instance_count"]      = int(cat_visible_count[chosen["category"]])
    chosen["unique_in_fov"]           = chosen["fov_instance_count"] == 1
    chosen["unique_in_scene"]         = chosen["scene_instance_count"] == 1
    return chosen


def _synth_sub_instruction(spatial: str, landmark_category: str) -> str:
    spatial = (spatial or "").strip().rstrip(".")
    article = "an" if landmark_category[:1] in "aeiou" else "a"
    if spatial:
        return f"{spatial}. Walk to {article} {landmark_category}."
    return f"Walk to {article} {landmark_category}."


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Find replacement landmarks for blacklist-dropped sub-paths.",
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", default=None,
                    help="Experiment handle (selection YAML path or expname).")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # apply_current=False — we WANT to consider blacklist-dropped sub-paths
    # which are not in survivor.yaml. We'll iterate the audit / drop yaml.
    resolve_exp(cfg, args.exp, apply_current=False)

    filt_dir = get_filter_dir(cfg)
    if not filt_dir.exists():
        raise SystemExit(f"No filters/ at {filt_dir}; run pipeline first.")

    run_dir   = get_run_dir(cfg)
    drops     = _blacklist_drops(filt_dir)
    if not drops:
        print("No blacklist drops to rescue.")
        return

    # Pull every episode from the dataset so we can look up sub_paths /
    # text per (ep_id, sub_idx). We do NOT filter — we want every episode
    # the blacklist drops mention.
    cfg.setdefault("selection", {})["instruction_ids"] = sorted({d[0] for d in drops})
    cfg["selection"]["sub_paths"] = {}    # unrestricted
    episodes = {int(ep.instruction_id): ep for ep in episodes_from_config(cfg)}

    # Load connectivity for node-position lookup.
    needed_scans = sorted({ep.scan for ep in episodes.values()})
    scan_db_all  = load_connectivity(
        scenes_dir=cfg["scenes"]["scenes_dir"],
        scans=needed_scans,
        json_dir=cfg["dataset"].get("connectivity_json_dir"),
        pkl_path=cfg["dataset"].get("connectivity_pkl"),
    )

    # Group drops by scan so we load each scene once.
    by_scan: Dict[str, List[Tuple[int, int, Dict]]] = defaultdict(list)
    for ep_id, sub_idx, rec in drops:
        ep = episodes.get(ep_id)
        if ep is None:
            continue
        by_scan[ep.scan].append((ep_id, sub_idx, rec))

    checker = VisibilityChecker(cfg["env"], cfg["scenes"]["scenes_dir"])
    rescued_count        = 0
    skipped_no_partition = 0
    skipped_no_pos       = 0
    skipped_no_candidate = 0
    output_paths: List[Path] = []

    try:
        for scan, scan_drops in sorted(by_scan.items()):
            checker.load_scene(f"mp3d/{scan}/{scan}.glb")
            inst_meta = _instance_meta_from_house(cfg["scenes"]["scenes_dir"], scan)
            scene_cat_counts: Counter = Counter(
                m["category"] for m in inst_meta.values() if m["category"]
            )
            scan_db = scan_db_all.get(scan) or {}

            scan_rescues: Dict[str, Dict[str, Dict]] = {}
            for ep_id, sub_idx, drop_rec in scan_drops:
                ep = episodes[ep_id]
                part_path = run_dir / "partition" / scan / str(ep_id) / "partition.json"
                if not part_path.exists():
                    skipped_no_partition += 1
                    continue
                with open(part_path) as f:
                    part_json = json.load(f)
                virtual_nodes = part_json.get("virtual_nodes") or {}
                part_sub = next(
                    (p for p in part_json.get("partitions") or []
                     if int(p.get("sub_idx", -1)) == sub_idx),
                    None,
                )
                if part_sub is None:
                    skipped_no_partition += 1
                    continue

                spatial_path  = part_sub.get("spatial_path") or []
                landmark_path = part_sub.get("landmark_path") or []
                if not spatial_path or not landmark_path:
                    skipped_no_pos += 1
                    continue
                partition_pos = _resolve_node_pos(spatial_path[-1], virtual_nodes, scan_db)
                end_pos       = _resolve_node_pos(landmark_path[-1], virtual_nodes, scan_db)
                if partition_pos is None or end_pos is None:
                    skipped_no_pos += 1
                    continue

                obs = checker.render_observation(end_pos.astype(np.float32), 0.0)
                sem = obs.get("semantic")
                if sem is None:
                    skipped_no_pos += 1
                    continue

                visible_pixels = _visible_instance_pixels(sem)
                pick = _pick_replacement_landmark(
                    visible_pixels=visible_pixels,
                    inst_meta=inst_meta,
                    partition_pos=partition_pos,
                    end_pos=end_pos,
                    scene_category_counts=scene_cat_counts,
                )
                if pick is None:
                    skipped_no_candidate += 1
                    continue

                spatial_instr = (part_sub.get("spatial_instruction") or "").strip()
                new_landmark  = pick["category"]
                new_sub_instr = _synth_sub_instruction(spatial_instr, new_landmark)

                # Render two viz frames for the new landmark:
                #   - partition pose: matches what 07/08 do for `original`
                #     records; the mask may be EMPTY for synthesized
                #     records because the chosen instance is selected for
                #     end-pose visibility, not partition-pose visibility.
                #   - end pose: by construction the chosen instance IS
                #     visible here, so the mask is always populated.
                viz_dir = (
                    run_dir / "target_instances" / "viz_blacklist_rescue"
                    / scan / str(ep_id)
                )
                viz_paths: Dict[str, Optional[str]] = {
                    "partition_viz_path": None,
                    "end_viz_path":       None,
                }
                for pose_key, pos, suffix, action in (
                    ("partition_viz_path", partition_pos, "partition", "BLACKLIST_RESCUE_P"),
                    ("end_viz_path",       end_pos,       "end",       "BLACKLIST_RESCUE_E"),
                ):
                    try:
                        rv = _render_mask_for_rollout_frame(
                            checker=checker,
                            scan=scan,
                            instance_id=int(pick["instance_id"]),
                            frame_record={
                                "position":       [float(x) for x in pos],
                                "heading":        0.0,
                                "instruction_id": ep_id,
                                "instruction":    new_sub_instr,
                                "landmark":       new_landmark,
                                "sub_idx":        sub_idx,
                                "sub_total":      len(ep.sub_paths),
                                "step":           0,
                                "action":         action,
                            },
                            out_path=viz_dir / f"sub_{sub_idx:03d}_{suffix}.png",
                            info_width=300,
                        )
                        viz_paths[pose_key] = rv["path"]
                    except Exception as exc:
                        print(
                            f"  WARN [{scan} ep={ep_id} sub={sub_idx}] {suffix} "
                            f"viz render failed: {exc}"
                        )

                scan_rescues.setdefault(str(ep_id), {})[str(sub_idx)] = {
                    "original_landmark":    drop_rec.get("landmark"),
                    "original_reason":      drop_rec.get("reason"),
                    "new_landmark":         new_landmark,
                    "new_instance_id":      pick["instance_id"],
                    "new_mpcat40":          pick["category"],
                    "new_sub_instruction":  new_sub_instr,
                    "spatial_instruction":  spatial_instr,
                    "landmark_instruction": f"Walk to a {new_landmark}.",
                    "pixel_count_at_end":   pick["pixel_count"],
                    "dist_partition_m":     round(pick["dist_partition_m"], 3),
                    "dist_end_m":           round(pick["dist_end_m"], 3),
                    "approach_m":           round(pick["approach_m"], 3),
                    "unique_in_fov":        pick["unique_in_fov"],
                    "unique_in_scene":      pick["unique_in_scene"],
                    "fov_instance_count":   pick["fov_instance_count"],
                    "scene_instance_count": pick["scene_instance_count"],
                    "partition_viz_path":   viz_paths["partition_viz_path"],
                    "end_viz_path":         viz_paths["end_viz_path"],
                }
                rescued_count += 1

            if scan_rescues:
                out_path = run_dir / "target_instances" / scan / "blacklist_rescue.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump({
                        "scan":       scan,
                        "approach_threshold_m": APPROACH_THRESHOLD_M,
                        "min_visible_pixels":   MIN_VISIBLE_PIXELS,
                        "blacklist_categories": sorted(LANDMARK_BLACKLIST_CATS),
                        "coarse_buckets":       sorted(COARSE_BUCKETS),
                        "rescues":    scan_rescues,
                    }, f, indent=2)
                output_paths.append(out_path)
    finally:
        checker.close()

    print()
    print(f"=== blacklist rescue summary ===")
    print(f"  total blacklist drops      : {len(drops)}")
    print(f"  rescued                    : {rescued_count}")
    print(f"  skipped (no partition.json): {skipped_no_partition}")
    print(f"  skipped (no node position) : {skipped_no_pos}")
    print(f"  skipped (no fit candidate) : {skipped_no_candidate}")
    print()
    for p in output_paths:
        print(f"  → {p}")


if __name__ == "__main__":
    main()
