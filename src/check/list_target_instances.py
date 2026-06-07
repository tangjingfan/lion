"""
LION-Bench — Enumerate candidate target instances at the partition point.

For each surviving sub-path (from the latest ``survivor.yaml``):
  1. Resolve the landmark text → matched MP40 category labels (via the
     per-scan ``landmark_mapping[_filtered].json`` from refine step 3b).
  2. Render a 360° semantic panorama at the **partition point** (the
     turn node between this sub-path and the next; usually a virtual
     ``virt:...`` node from ``partition.json``) and list every visible
     MP40 instance whose category matches the landmark.
  3. Tag each (ep, sub) with two fields, **visibility first, then
     uniqueness**:
       • ``visibility`` (str):
           - ``"visible"``                   — at least one matched instance
                                               visible from the partition pose
           - ``"not_visible"``               — category matched, zero visible
           - ``"no_match"``                  — landmark text doesn't map to any
                                               MP40 category in the scene
           - ``"partition_pos_unresolvable"`` — couldn't resolve partition pose
       • ``uniqueness``:
           - ``True``                        — exactly one visible candidate
                                               (only meaningful when visible)
           - ``False``                       — multiple visible candidates
                                               (only meaningful when visible)
           - ``"not_visible"``               — mirrors the visibility tag so
                                               downstream code never has to
                                               check visibility separately to
                                               know there's nothing to pick

This step intentionally counts instances **from the agent's vantage
point at the turn**, not the whole-scene total — because that count is
the one that determines whether downstream selection has a unique
target without disambiguation.

When ``--save_viz`` is on (default), a per-candidate rollout-style mask
PNG is also written at the same partition pose.

Pipeline prerequisites
----------------------
- ``survivor.yaml`` after stage 3 (partition) — sub-path-level
  survivors with ``partition.json`` written.
- ``rewrite/{scan}/landmark_mapping[_filtered].json`` — preferably the
  refined version from ``refine_landmark_mapping``.

Output (per scan)
-----------------
  ``{run_dir}/target_instances/{scan}/target_instances.json``

Viz (per candidate, when enabled)
---------------------------------
  ``{run_dir}/target_instances/viz/{scan}/{ep}/sub_{NNN}_cand_{IID}.png``

Usage
-----
  python src/check/list_target_instances.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --exp configs/selection/exp.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import (
    active_subs,
    append_sub_event,
    discover_rewrite_suffix,
    ensure_episode,
    finalize_audit,
    get_filter_dir,
    get_run_dir,
    get_split,
    get_survivor_path,
    iter_rewrite_files,
    load_audit,
    register_stage,
    resolve_exp,
    save_audit,
    strip_stage_events,
)
from src.instance_viz import render_mask_for_rollout_frame
from src.dataset.landmark_rxr import episodes_from_config
from src.env.connectivity import load_connectivity
from src.process.target_annotation import (
    classify_visibility,
    resolve_partition_pos,
    semantic_labels_for_sub,
)
from src.process.visibility import VisibilityChecker
from src.viz import _semantic_to_rgb


STAGE_NAME = "visibility"


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render a 360° panorama at every surviving sub-path's "
                    "partition point, list the visible instances of the "
                    "matched category, and tag uniqueness.",
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", default=None,
                    help="Experiment handle (selection YAML path or expname). "
                         "Auto-merges survivor.yaml so the survivor "
                         "sub_paths are restored even when passing the "
                         "original selection yaml.")
    ap.add_argument("--min_pixel_count", type=int, default=2000,
                    help="Min pixels per instance to count as visible. "
                         "Default 2000 — matches step 11's MIN_VISIBLE_PIXELS, "
                         "so 'visible' means the same thing everywhere in the "
                         "pipeline. 2000 px in a 1024x512 panorama is the "
                         "minimum visually recognisable region; smaller "
                         "regions tend to be easy to miss.")
    ap.add_argument("--save_viz", action="store_true", default=False,
                    help="Render a rollout-style mask PNG per candidate at "
                         "the partition pose. Default off — pass this flag "
                         "to opt in.")
    ap.add_argument("--no_save_viz", action="store_false", dest="save_viz",
                    help="(Deprecated; viz is off by default now.)")
    ap.add_argument("--info_width", type=int, default=300,
                    help="Right info panel width for viz panoramas.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    resolve_exp(cfg, args.exp, apply_current=True)

    out_dir = get_run_dir(cfg)
    filt_dir = get_filter_dir(cfg)
    split   = get_split(cfg)

    audit = load_audit(filt_dir, split)
    register_stage(audit, STAGE_NAME, min_pixel_count=args.min_pixel_count)
    strip_stage_events(audit, STAGE_NAME)

    survivor = get_survivor_path(cfg)
    if not survivor.exists():
        raise SystemExit(f"No survivor.yaml at {survivor} — run filter pipeline first.")

    # resolve_exp already merged survivor.yaml into cfg.selection.
    # Process only sub-paths that are still active (un-labeled by earlier
    # stages). Labeled drops stay in survivor.sub_paths for inspection /
    # rescue tooling, but we don't want to spend visibility-check compute
    # on them here — rescue paths (step 11 + step 09/10) handle them.
    prior_subs = active_subs(cfg)
    if not prior_subs:
        raise SystemExit(
            "survivor.yaml has no active sub-paths — run scripts 02→03→04 first "
            "(rewrite → blacklist → partition), and check that any aren't all "
            "labeled by an earlier filter.",
        )

    # Locate per-episode rewrite JSONs.
    rewrite_dir = out_dir / "rewrite"
    if not rewrite_dir.exists():
        raise SystemExit(f"No rewrite dir under {rewrite_dir}")
    chosen_suffix: Optional[str] = discover_rewrite_suffix(rewrite_dir)
    if chosen_suffix is None:
        raise SystemExit(f"No rewrite JSON under {rewrite_dir}/*/*/")

    rewrite_by_scan: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for scan, ep_id, p in iter_rewrite_files(rewrite_dir, chosen_suffix):
        with open(p) as f:
            rewrite_by_scan[scan][ep_id] = json.load(f)["episode"]
        print(f"Loaded rewrite ({scan}/{ep_id}): {p}")
    rewrite_by_scan = dict(rewrite_by_scan)

    mapping_by_scan: Dict[str, Dict] = {}
    scan_dirs = [d for d in sorted(rewrite_dir.iterdir()) if d.is_dir()]
    for scan_dir in scan_dirs:
        scan = scan_dir.name
        lm = scan_dir / f"landmark_mapping{chosen_suffix}.json"
        if lm.exists():
            with open(lm) as f:
                mapping_by_scan[scan] = json.load(f) or {}
            print(f"Loaded landmark mapping ({scan}): {lm}  "
                  f"({len(mapping_by_scan[scan])} mentions)")
        else:
            mapping_by_scan[scan] = {}
            print(f"  no landmark_mapping{chosen_suffix}.json under {scan_dir}/")

    # Load episodes restricted to prior survivors.
    episodes  = episodes_from_config(cfg)
    needed_ids = {int(k) for k in prior_subs.keys()}
    episodes  = [ep for ep in episodes if ep.instruction_id in needed_ids]
    if not episodes:
        raise SystemExit("No episodes loaded.")

    # Group by scan so each scene loads once.
    by_scan: Dict[str, list] = defaultdict(list)
    for ep in episodes:
        by_scan[ep.scan].append(ep)
    needed_scans = sorted(by_scan.keys())

    db = load_connectivity(
        scenes_dir=cfg["scenes"]["scenes_dir"],
        scans=needed_scans,
        json_dir=cfg["dataset"].get("connectivity_json_dir"),
        pkl_path=cfg["dataset"].get("connectivity_pkl"),
    )

    run_dir  = get_run_dir(cfg)
    out_root = run_dir / "target_instances"
    out_root.mkdir(parents=True, exist_ok=True)
    viz_root          = out_root / "viz"
    partition_obs_root = out_root / "partition_obs"
    partition_obs_count = 0
    written_paths: List[Path] = []
    uniqueness_counts: Dict[str, int] = defaultdict(int)
    viz_count = 0
    viz_errors = 0
    n_total = 0
    n_total_candidates = 0

    checker = VisibilityChecker(cfg["env"], cfg["scenes"]["scenes_dir"])
    try:
        for scan in needed_scans:
            scan_eps          = by_scan[scan]
            scan_db           = db.get(scan, {})
            scan_rewrite_eps  = rewrite_by_scan.get(scan, {})
            scan_landmark_map = mapping_by_scan.get(scan, {})
            print(f"\n[{scan}] loading scene  ({len(scan_eps)} episode(s)) …")
            checker.load_scene(scan_eps[0].scene_file)

            scan_annotations: Dict[str, Dict[str, Dict]] = {}

            for ep in scan_eps:
                ep_id     = ep.instruction_id
                ep_id_str = str(ep_id)
                sub_idxs  = prior_subs.get(ep_id) or prior_subs.get(ep_id_str) or []
                ep_audit  = ensure_episode(audit, ep)

                ep_anno: Dict[str, Dict] = {}

                part_path = (out_dir / "partition" / scan / ep_id_str
                             / "partition.json")
                partition_subs: Dict[int, Dict] = {}
                virtual_nodes: Dict[str, List[float]] = {}
                if part_path.exists():
                    with open(part_path) as f:
                        part_json = json.load(f)
                    partition_subs = {
                        int(s["sub_idx"]): s for s in part_json.get("partitions", [])
                    }
                    virtual_nodes = part_json.get("virtual_nodes", {})

                rewrite_ep   = scan_rewrite_eps.get(ep_id_str) or {}
                rewrite_subs = {
                    int(s["sub_idx"]): s for s in rewrite_ep.get("sub_paths", [])
                }
                sub_total = max(len(partition_subs), len(sub_idxs), 1)

                for sub_idx in sub_idxs:
                    sub_idx_int = int(sub_idx)
                    n_total += 1

                    rw         = rewrite_subs.get(sub_idx_int, {})
                    landmark   = rw.get("landmark", "")
                    sem_labels = semantic_labels_for_sub(rw, scan_landmark_map, scan)
                    part_sub   = partition_subs.get(sub_idx_int, {})
                    pos        = resolve_partition_pos(part_sub, virtual_nodes, scan_db)

                    visibility_result: Optional[Dict[str, Any]] = None
                    candidates: List[Dict[str, Any]] = []
                    matched_categories: List[str] = []
                    matched_category: Optional[str] = None

                    if pos is not None:
                        visibility_result = checker.check_landmark_visibility_semantic(
                            pos, landmark,
                            semantic_labels=sem_labels or None,
                            min_pixel_count=args.min_pixel_count,
                        )
                        candidates = list(visibility_result.get("instances") or [])
                        matched_categories = list(
                            visibility_result.get("matched_categories") or []
                        )
                        matched_category = visibility_result.get("matched_category")

                    visibility, uniqueness = classify_visibility(pos, visibility_result)
                    # For the breakdown counter, fold the bool form back
                    # into a stable string key.
                    counter_key = (
                        visibility if visibility != "visible"
                        else ("unique" if uniqueness is True else "ambiguous")
                    )
                    uniqueness_counts[counter_key] += 1
                    n_total_candidates += len(candidates)

                    append_sub_event(
                        ep_audit, sub_idx_int, stage=STAGE_NAME, action="labeled",
                        visibility=visibility,
                        uniqueness=uniqueness,
                        n_candidates=len(candidates),
                        matched_category=matched_category,
                        landmark=landmark,
                    )

                    # Save the clean partition-pose RGB + semantic panoramas
                    # once per (ep, sub) — independent of candidates, so
                    # not_visible sub-paths get a record too.
                    if args.save_viz and pos is not None:
                        try:
                            obs_dir = partition_obs_root / scan / ep_id_str
                            obs_dir.mkdir(parents=True, exist_ok=True)
                            checker.load_scene(f"mp3d/{scan}/{scan}.glb")
                            obs = checker.render_observation(
                                np.asarray(pos, dtype=np.float32), 0.0,
                            )
                            rgb = obs.get("rgb")
                            sem = obs.get("semantic")
                            if rgb is not None:
                                Image.fromarray(rgb).save(
                                    obs_dir / f"sub_{sub_idx_int:03d}_rgb.png"
                                )
                            if sem is not None:
                                Image.fromarray(_semantic_to_rgb(sem)).save(
                                    obs_dir / f"sub_{sub_idx_int:03d}_semantic.png"
                                )
                            partition_obs_count += 1
                        except Exception as exc:
                            print(
                                f"  WARN [{scan} ep={ep_id_str} sub={sub_idx_int}] "
                                f"partition obs render failed: {exc}"
                            )

                    record: Dict[str, Any] = {
                        "landmark":           landmark,
                        "semantic_labels":    sem_labels,
                        "matched_category":   matched_category,
                        "matched_categories": matched_categories,
                        "matched_by":         (visibility_result or {}).get("matched_by"),
                        "pixel_count":        int((visibility_result or {}).get("pixel_count") or 0),
                        "pixel_fraction":     float((visibility_result or {}).get("pixel_fraction") or 0.0),
                        "candidates":         candidates,
                        # Split fields: visibility decides whether the
                        # landmark is reachable from this pose at all;
                        # uniqueness is a bool only when visible, else
                        # mirrors the visibility tag.
                        "visibility":         visibility,
                        "uniqueness":         uniqueness,
                    }

                    # Per-candidate mask viz, rendered at the same partition pose.
                    if args.save_viz and candidates and pos is not None:
                        for cand in candidates:
                            cand_id = int(cand["id"])
                            viz_path = (
                                viz_root / scan / ep_id_str
                                / f"sub_{sub_idx_int:03d}_cand_{cand_id}.png"
                            )
                            try:
                                rv = render_mask_for_rollout_frame(
                                    checker=checker,
                                    scan=scan,
                                    instance_id=cand_id,
                                    frame_record={
                                        "position": [float(x) for x in pos],
                                        "heading": 0.0,
                                        "instruction_id": ep_id,
                                        "instruction": landmark or "",
                                        "landmark": landmark or "",
                                        "sub_idx": sub_idx_int,
                                        "sub_total": sub_total,
                                        "step": 0,
                                        "action": "ENUM_CANDIDATE",
                                    },
                                    out_path=viz_path,
                                    info_width=args.info_width,
                                )
                                cand["viz_path"] = rv["path"]
                                cand["viz_visible_pixels"] = rv.get(
                                    "target_visible_pixels"
                                )
                                viz_count += 1
                            except Exception as exc:
                                cand["viz_error"] = str(exc)
                                viz_errors += 1

                    ep_anno[str(sub_idx_int)] = record

                    cand_summary = ",".join(str(c["id"]) for c in candidates) or "—"
                    if visibility == "visible":
                        verdict = f"visible/unique={uniqueness}"
                    else:
                        verdict = visibility
                    print(f"  [{ep_id} sub {sub_idx_int:<2}] "
                          f"{verdict:<26s}  "
                          f"landmark={landmark!r}  "
                          f"cats={matched_categories or '—'}  "
                          f"ids=[{cand_summary}]")

                scan_annotations[ep_id_str] = ep_anno

            scan_dir = out_root / scan
            scan_dir.mkdir(parents=True, exist_ok=True)
            scan_out = scan_dir / "target_instances.json"
            payload = {
                "split":           split,
                "scan":            scan,
                "expname":         cfg.get("output", {}).get("expname"),
                "run_name":        cfg.get("output", {}).get("run_name"),
                "min_pixel_count": args.min_pixel_count,
                "source_keep":     str(survivor.resolve()),
                "annotations":     scan_annotations,
            }
            with open(scan_out, "w") as f:
                json.dump(payload, f, indent=2)
            written_paths.append(scan_out)
            print(f"  → {scan_out}")
    finally:
        checker.close()

    finalize_audit(audit)
    save_audit(audit, filt_dir)

    print(f"\n=== Target Instance Enumeration (partition-point FOV) ===")
    print(f"  sub-paths annotated : {n_total}")
    print(f"  total candidates    : {n_total_candidates}")
    print(f"  uniqueness breakdown:")
    for verdict, n in sorted(uniqueness_counts.items(), key=lambda kv: -kv[1]):
        pct = (n / n_total) if n_total else 0.0
        print(f"    {verdict:<28s} {n:>4d}  ({pct:.1%})")
    if args.save_viz:
        print(f"  viz saved           : {viz_count}")
        print(f"  partition obs saved : {partition_obs_count}  (rgb + semantic per sub-path)")
        if viz_errors:
            print(f"  viz errors          : {viz_errors}")
    print(f"\nOutputs ({len(written_paths)} scan file(s)):")
    for p in written_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
