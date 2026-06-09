"""Synthesize a replacement landmark when the original can't be grounded.

CLI + per-scan orchestration for pipeline step 11
(``scripts/11_rescue_blacklist.sh``).  The selection algorithm, source
collection, referrability table, and VLM naming live in
:mod:`src.process.synthesis`; equirect unprojection + voxel counting in
:mod:`src.env.geometry`; ``.house`` instance metadata in
:mod:`src.env.mp3d_house`.

Three upstream conditions feed this rescue (all treated uniformly):

  • ``origin = "blacklist"`` — step 02 dropped the sub because its
    instruction-derived landmark is too generic to ground (``wall``,
    ``door``, ``room``, ``doorway``, ...). Read from
    ``filters/02_blacklist_dropped.yaml``.
  • ``origin = "detection_failure"`` — step 09 YOLO-World (and the
    optional VLM fallback) couldn't locate the original landmark in the
    panorama. Read from the lifecycle audit's ``detection`` /
    ``rescue_failed`` events.
  • ``origin = "visibility_not_visible"`` — step 07 matched the
    original landmark text to a fine MPCat40 category but found no
    instance of that category visible at the partition pose (e.g.
    ``bath`` → ``bathtub`` matched, but no bathtub in view). Read
    from the lifecycle audit's ``visibility`` events with
    ``visibility == "not_visible"``.  Subs later grounded by the
    step-09 detection rescue are skipped — their original record is
    usable as-is.

In all cases the original landmark is not recoverable, so we pick a
**different** referrable landmark visible at the partition pose and
emit a synthesized sub-instruction. Records flow through the rest of
the pipeline marked ``synthesized = True`` with ``synthesized_from.origin``
recording which upstream branch fed them, so downstream consumers can
filter as they like.

Per-(scan, instance_id) VLM responses are cached in
``target_instances/<scan>/vlm_instance_labels.json`` so repeated runs
don't re-pay the VLM cost.

Output: ``target_instances/{scan}/blacklist_rescue.json`` — one record
per synthesized sub-path. The ``origin`` field on each record
distinguishes the upstream sources. The consolidate step (12) reads
this file and emits synthesized rows into ``dataset.json`` alongside
the original ones.

Usage
-----
  python src/check/rescue_blacklist.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --exp configs/selection/val_unseen/one_scene_partial.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline.audit import (
    append_sub_event,
    finalize_audit,
    load_audit,
    register_stage,
    save_audit,
    strip_stage_events,
)
from src.pipeline.config import (
    get_filter_dir,
    get_run_dir,
    get_split,
    resolve_exp,
)

STAGE_NAME = "rescue_blacklist"

from src.dataset.landmark_rxr import episodes_from_config
from src.instance_viz import render_mask_for_rollout_frame
from src.env.connectivity import load_connectivity
from src.env.geometry import VOXEL_SIZE_M, visible_instance_voxels
from src.env.mp3d_house import instance_meta_from_house
from src.process.rewriter import make_client
from src.process.synthesis import (
    MIN_VISIBILITY_RATIO,
    REFERRABLE_TABLE_PATH,
    SIZE_SATURATION_VOXELS,
    blacklist_drops,
    detection_failures,
    load_referrability_table,
    load_vlm_cache,
    merge_drops,
    pick_replacement_landmark,
    resolve_node_pos,
    save_vlm_cache,
    synth_sub_instruction,
    visibility_not_visible_failures,
)
from src.process.visibility import VisibilityChecker


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Synthesize a replacement landmark for sub-paths whose original "
            "landmark can't be grounded — either dropped by 02_blacklist or "
            "failed by 09_detection."
        ),
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", default=None,
                    help="Experiment handle (selection YAML path or expname).")
    ap.add_argument("--save_viz", action="store_true", default=False,
                    help="Render a rollout-style mask PNG at the partition "
                         "and end poses for each rescued landmark. Default "
                         "off — pass this flag to opt in.")
    ap.add_argument("--no_vlm_refine", action="store_true", default=False,
                    help="Skip the VLM step that names collective-bucket "
                         "instances (appliances / furniture / objects / "
                         "lighting / ...). With this flag, collective "
                         "candidates are skipped and only `fine`-tier "
                         "categories from the referrability table can win.")
    ap.add_argument("--vlm_model", default="gemini-2.5-flash",
                    help="VLM model used to name collective instances "
                         "(default gemini-2.5-flash).")
    ap.add_argument("--vlm_api_key", default=None,
                    help="API key for the VLM; defaults to GEMINI_API_KEY env.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # apply_current=False — we WANT to consider blacklist-dropped sub-paths
    # which are not in survivor.yaml. We'll iterate the audit / drop yaml.
    resolve_exp(cfg, args.exp, apply_current=False)

    filt_dir = get_filter_dir(cfg)
    if not filt_dir.exists():
        raise SystemExit(f"No filters/ at {filt_dir}; run pipeline first.")

    referrability = load_referrability_table()
    n_too_generic = sum(1 for v in referrability.values() if v == "too_generic")
    n_collective  = sum(1 for v in referrability.values() if v == "collective")
    n_fine        = sum(1 for v in referrability.values() if v == "fine")
    print(
        f"Referrability table: {len(referrability)} categories "
        f"({n_too_generic} too_generic, {n_collective} collective, {n_fine} fine)"
    )

    vlm_client = None
    vlm_model  = args.vlm_model
    if args.no_vlm_refine:
        print("VLM refinement: DISABLED (--no_vlm_refine)")
    elif n_collective == 0:
        print("VLM refinement: skipped (no collective categories in table)")
    else:
        api_key = args.vlm_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEY")
        if not api_key:
            print(
                "VLM refinement: DISABLED (no GEMINI_API_KEY); collective "
                "candidates will fall through to next-best fine candidates."
            )
        else:
            try:
                vlm_client = make_client(api_key=api_key)
                print(f"VLM refinement: enabled (model={vlm_model})")
            except Exception as exc:
                print(f"VLM refinement: DISABLED (client init failed: {exc})")
                vlm_client = None
    vlm_cache_by_scan: Dict[str, Dict[str, Dict]] = {}

    split = get_split(cfg)
    audit = load_audit(filt_dir, split)
    register_stage(audit, STAGE_NAME)
    strip_stage_events(audit, STAGE_NAME)

    run_dir           = get_run_dir(cfg)
    bl_drops          = blacklist_drops(filt_dir)
    detection_fails   = detection_failures(audit)
    visibility_misses = visibility_not_visible_failures(audit)
    drops             = merge_drops(bl_drops, detection_fails, visibility_misses)
    n_bl  = len(bl_drops)
    n_det = len(detection_fails)
    n_vis = len(visibility_misses)
    print(
        f"Synthesis candidates: {len(drops)}  "
        f"({n_bl} blacklist + {n_det} detection-fail + {n_vis} visibility:not_visible)"
    )
    if not drops:
        print("No candidates to rescue.")
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
            inst_meta = instance_meta_from_house(cfg["scenes"]["scenes_dir"], scan)
            scan_db = scan_db_all.get(scan) or {}

            scan_rescues: Dict[str, Dict[str, Dict]] = {}
            for ep_id, sub_idx, drop_rec in scan_drops:
                ep = episodes[ep_id]
                ep_audit = audit["episodes"].setdefault(str(ep_id), {
                    "scan": scan, "events": [], "sub_paths": {},
                })

                def _failed(reason: str) -> None:
                    append_sub_event(
                        ep_audit, sub_idx, stage=STAGE_NAME,
                        action="rescue_failed", reason=reason,
                        origin=drop_rec.get("origin") or "blacklist",
                        original_landmark=drop_rec.get("landmark"),
                    )

                part_path = run_dir / "partition" / scan / str(ep_id) / "partition.json"
                if not part_path.exists():
                    skipped_no_partition += 1
                    _failed("no_partition")
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
                    _failed("no_partition")
                    continue

                spatial_path  = part_sub.get("spatial_path") or []
                landmark_path = part_sub.get("landmark_path") or []
                if not spatial_path or not landmark_path:
                    skipped_no_pos += 1
                    _failed("no_pos")
                    continue
                partition_pos = resolve_node_pos(spatial_path[-1], virtual_nodes, scan_db)
                end_pos       = resolve_node_pos(landmark_path[-1], virtual_nodes, scan_db)
                if partition_pos is None or end_pos is None:
                    skipped_no_pos += 1
                    _failed("no_pos")
                    continue

                # Render at BOTH partition and end pose: selection is
                # driven by the partition/end visibility *ratio* (must be
                # at least MIN_VISIBILITY_RATIO) plus max distance
                # change toward the instance, so we need both
                # depth-unprojected voxel sets up-front.
                try:
                    p_obs = checker.render_observation(
                        partition_pos.astype(np.float32), 0.0,
                    )
                    e_obs = checker.render_observation(
                        end_pos.astype(np.float32), 0.0,
                    )
                except Exception as exc:
                    print(
                        f"  WARN [{scan} ep={ep_id} sub={sub_idx}] render "
                        f"failed: {exc}"
                    )
                    skipped_no_pos += 1
                    _failed("no_render")
                    continue

                p_sem = p_obs.get("semantic")
                e_sem = e_obs.get("semantic")
                p_depth = p_obs.get("depth")
                e_depth = e_obs.get("depth")
                if (p_sem is None or e_sem is None
                        or p_depth is None or e_depth is None):
                    skipped_no_pos += 1
                    _failed("no_render")
                    continue

                partition_voxels = visible_instance_voxels(
                    p_depth, p_sem, partition_pos, heading=0.0,
                )
                end_voxels = visible_instance_voxels(
                    e_depth, e_sem, end_pos, heading=0.0,
                )

                print(f"[{scan} ep={ep_id} sub={sub_idx}] rescue selection "
                      f"(orig_landmark={drop_rec.get('landmark')!r}):")
                vlm_viz_dir = (
                    run_dir / "viz_blacklist_rescue" / scan / str(ep_id)
                )
                pick = pick_replacement_landmark(
                    partition_voxels=partition_voxels,
                    end_voxels=end_voxels,
                    inst_meta=inst_meta,
                    partition_pos=partition_pos,
                    end_pos=end_pos,
                    referrability=referrability,
                    scan=scan,
                    ep_id=ep_id,
                    sub_idx=sub_idx,
                    sub_total=len(ep.sub_paths),
                    checker=checker,
                    vlm_client=vlm_client,
                    vlm_model=vlm_model,
                    vlm_cache=vlm_cache_by_scan.setdefault(scan, load_vlm_cache(
                        run_dir / "target_instances" / scan,
                    )),
                    viz_dir=vlm_viz_dir,
                    render_mask_fn=render_mask_for_rollout_frame,
                )
                if pick is None:
                    skipped_no_candidate += 1
                    _failed("no_fit_candidate")
                    continue

                spatial_instr = (part_sub.get("spatial_instruction") or "").strip()
                new_landmark  = pick["new_landmark"]
                new_sub_instr = synth_sub_instruction(spatial_instr, new_landmark)

                if args.save_viz:
                    # partition + end poses for the chosen replacement
                    # landmark. PNGs land under
                    # viz_blacklist_rescue/<scan>/<ep>/ (sibling of
                    # partition_obs/ and detection/, out of target_instances/
                    # which now holds only data). Paths are not recorded
                    # in the JSON to keep records compact — locate them
                    # by (ep, sub) on disk.
                    viz_dir = (
                        run_dir / "viz_blacklist_rescue"
                        / scan / str(ep_id)
                    )
                    for pos, suffix, action in (
                        (partition_pos, "partition", "BLACKLIST_RESCUE_P"),
                        (end_pos,       "end",       "BLACKLIST_RESCUE_E"),
                    ):
                        try:
                            render_mask_for_rollout_frame(
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
                        except Exception as exc:
                            print(
                                f"  WARN [{scan} ep={ep_id} sub={sub_idx}] {suffix} "
                                f"viz render failed: {exc}"
                            )

                origin = drop_rec.get("origin") or "blacklist"
                scan_rescues.setdefault(str(ep_id), {})[str(sub_idx)] = {
                    "origin":               origin,
                    "original_landmark":    drop_rec.get("landmark"),
                    "original_reason":      drop_rec.get("reason"),
                    "new_landmark":         new_landmark,
                    "new_landmark_source":  pick["new_landmark_source"],
                    "vlm":                  pick.get("vlm"),
                    "new_instance_id":      pick["instance_id"],
                    "new_mpcat40":          pick["category"],
                    "new_sub_instruction":  new_sub_instr,
                    "spatial_instruction":  spatial_instr,
                    "landmark_instruction": f"Walk to a {new_landmark}.",
                    # Same split-schema as step 07/08 records:
                    # visibility is always "visible" by construction
                    # (selection requires ratio ≥ MIN_VISIBILITY_RATIO,
                    # which implies non-zero partition pixels);
                    # uniqueness reflects whether the chosen category
                    # is unique in the partition FOV.
                    "visibility":           "visible",
                    "uniqueness":           bool(pick["unique_in_fov"]),
                }
                append_sub_event(
                    ep_audit, sub_idx, stage=STAGE_NAME, action="synthesized",
                    origin=origin,
                    new_landmark=new_landmark,
                    instance_id=int(pick["instance_id"]),
                    original_landmark=drop_rec.get("landmark"),
                    unique_in_fov=bool(pick["unique_in_fov"]),
                )
                rescued_count += 1

            if scan_rescues:
                out_path = run_dir / "target_instances" / scan / "blacklist_rescue.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump({
                        "scan":       scan,
                        "min_visibility_ratio":   MIN_VISIBILITY_RATIO,
                        "voxel_size_m":           VOXEL_SIZE_M,
                        "size_saturation_voxels": SIZE_SATURATION_VOXELS,
                        "referrable_table":       str(REFERRABLE_TABLE_PATH),
                        "vlm_model":              vlm_model if vlm_client else None,
                        "rescues":    scan_rescues,
                    }, f, indent=2)
                output_paths.append(out_path)

            # Persist per-instance VLM labels for this scan (idempotent).
            scan_dir = run_dir / "target_instances" / scan
            save_vlm_cache(scan_dir, scan, vlm_cache_by_scan.get(scan) or {})
    finally:
        checker.close()

    finalize_audit(audit)
    save_audit(audit, filt_dir)

    # Per-origin breakdown of the rescue verdict, so it's clear where
    # the rescued landmarks (and the misses) came from.
    by_origin: Dict[str, Dict[str, int]] = {}
    for ep_id_str, ep in (audit.get("episodes") or {}).items():
        for sub_idx_str, sp in (ep.get("sub_paths") or {}).items():
            for e in sp.get("events") or []:
                if e.get("stage") != STAGE_NAME:
                    continue
                origin = e.get("origin") or "blacklist"
                slot = by_origin.setdefault(origin, {"synthesized": 0, "rescue_failed": 0})
                if e.get("action") == "synthesized":
                    slot["synthesized"] += 1
                elif e.get("action") == "rescue_failed":
                    slot["rescue_failed"] += 1

    print()
    print(f"=== landmark synthesis summary ===")
    print(f"  candidates total           : {len(drops)}  "
          f"({n_bl} blacklist + {n_det} detection-fail + {n_vis} visibility:not_visible)")
    print(f"  synthesized                : {rescued_count}")
    print(f"  skipped (no partition.json): {skipped_no_partition}")
    print(f"  skipped (no node position) : {skipped_no_pos}")
    print(f"  skipped (no fit candidate) : {skipped_no_candidate}")
    if by_origin:
        print(f"  by origin:")
        for origin, counts in sorted(by_origin.items()):
            print(f"    {origin:<20s} synthesized={counts['synthesized']}  "
                  f"rescue_failed={counts['rescue_failed']}")
    print()
    for p in output_paths:
        print(f"  → {p}")


if __name__ == "__main__":
    main()
