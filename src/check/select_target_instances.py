"""
Select target semantic instance ids from enumerated candidate annotations.

For each visible landmark, choose a target instance:

  • one visible instance   -> view_unique
  • multiple instances     -> view_nearest  (closest to the sub-path end point)

When instance centers can't be resolved, falls back to the largest-pixel
instance with status ``view_nearest_fallback``.

The output is intended as the input for later same-instance robust visibility
checks.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import (
    append_sub_event,
    finalize_audit,
    get_filter_dir,
    get_run_dir,
    get_split,
    load_audit,
    register_stage,
    resolve_exp,
    save_audit,
    strip_stage_events,
)
from src.instance_viz import (
    draw_house_instance_viz,
    render_mask_for_rollout_frame,
)
from src.env.connectivity import load_connectivity
from src.process.target_selection import (
    candidate_viz_path,
    instance_centers_from_house,
    instance_centers_from_scene,
    load_subpath_positions,
    select_target,
)
from src.process.visibility import VisibilityChecker


def _iter_records(data: Dict[str, Any]) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    for ep_id, ep_records in (data.get("annotations") or {}).items():
        if not isinstance(ep_records, dict):
            continue
        for sub_idx, record in ep_records.items():
            if isinstance(record, dict):
                yield str(ep_id), str(sub_idx), record


def _load_last_rollout_frames(
    run_dir: Path, scan: str,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Return the last rollout-viz frame record for each ``(ep_id, sub_idx)``."""
    index_path = run_dir / "rollout_viz" / scan / "frames.jsonl"
    if not index_path.exists():
        return {}

    last: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with open(index_path) as f:
        for line_no, line in enumerate(f, start=1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep_id = rec.get("instruction_id")
            sub_idx = rec.get("sub_idx")
            if ep_id is None or sub_idx is None:
                continue
            key = (str(ep_id), str(sub_idx))
            rec["_frame_index"] = line_no - 1
            rel_path = rec.get("rel_path")
            if rel_path:
                rec["_rollout_frame"] = str(run_dir / "rollout_viz" / scan / rel_path)
            prev = last.get(key)
            if prev is None or int(rec.get("step") or 0) >= int(prev.get("step") or 0):
                last[key] = rec
    return last


def _copy_last_rollout_frame(
    frame_record: Dict[str, Any],
    out_path: Path,
) -> Optional[str]:
    src = frame_record.get("_rollout_frame")
    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, out_path)
    return str(out_path)


def _copy_existing_png(src: Optional[str], out_path: Path) -> Optional[str]:
    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, out_path)
    return str(out_path)


def _resolve_visibility_paths(path: Path) -> List[Path]:
    if path.exists():
        return [path]
    parent = path.parent
    paths = sorted(parent.glob("*/visibility.json"))
    if paths:
        return paths
    raise SystemExit(f"Visibility JSON not found: {path}")


def _resolve_target_instance_paths(path: Path) -> List[Path]:
    if path.name == "target_instances.json" and path.parent.name == "target_instances":
        paths = sorted(path.parent.glob("*/target_instances.json"))
        if paths:
            return paths
    if path.exists():
        if path.is_dir():
            paths = sorted(path.glob("*/target_instances.json"))
            if paths:
                return paths
        return [path]
    parent = path.parent
    paths = sorted(parent.glob("*/target_instances.json"))
    if paths:
        return paths
    raise SystemExit(f"Target instances JSON not found: {path}")


def _infer_run_dir_from_visibility_path(vis_path: Path) -> Path:
    """Infer results/{run_name} from either aggregate or per-scan visibility JSON."""
    if vis_path.parent.parent.name == "landmark_visibility":
        return vis_path.parent.parent.parent
    return vis_path.parent.parent


def _infer_run_dir_from_target_instances_path(path: Path) -> Path:
    """Infer results/{run_name} from target_instances JSON paths."""
    if path.parent.name == "target_instances":
        return path.parent.parent
    if path.parent.parent.name == "target_instances":
        return path.parent.parent.parent
    return path.parent.parent


def _safe_instruction_id(ep_id: str) -> Any:
    try:
        return int(ep_id)
    except ValueError:
        return ep_id


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select target instance ids from landmark visibility JSON.",
    )
    ap.add_argument("--config", default="configs/rollout/rollout_landmark_rxr.yaml")
    ap.add_argument("--exp", default=None,
                    help="Experiment handle (selection YAML path or expname). "
                         "Auto-merges survivor.yaml so the survivor "
                         "sub_paths are restored.")
    ap.add_argument("--visibility_json", default=None,
                    help="Legacy direct path to landmark_visibility/visibility.json.")
    ap.add_argument("--target_instances_json", default=None,
                    help="Direct path to target_instances/{scan}/target_instances.json "
                         "or a target_instances directory.")
    ap.add_argument("--out", default=None,
                    help="Optional aggregate output JSON path. By default the "
                         "selection is written back into each per-scan "
                         "target_instances/{scan}/target_instances.json file.")
    ap.add_argument("--save_viz", action="store_true", default=False,
                    help="Save a Habitat mask PNG for each selected target "
                         "instance. Default off — pass this flag to opt in.")
    ap.add_argument("--no_save_viz", action="store_false", dest="save_viz",
                    help="(Deprecated; viz is off by default now.)")
    ap.add_argument("--viz_mode", choices=("mask", "topdown"), default="mask",
                    help="Visualization type. Default: Habitat rollout-style mask.")
    ap.add_argument("--info_width", type=int, default=300,
                    help="Right info panel width for Habitat mask visualizations.")
    ap.add_argument("--print_multi", action="store_true",
                    help="Print per-record details when >1 candidate instances.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.visibility_json and args.target_instances_json:
        raise SystemExit("Pass only one of --visibility_json or --target_instances_json.")

    if args.visibility_json:
        vis_path = Path(args.visibility_json)
        run_dir = _infer_run_dir_from_visibility_path(vis_path)
        source_paths = _resolve_visibility_paths(vis_path)
        source_kind = "visibility_json"
    elif args.target_instances_json:
        ti_path = Path(args.target_instances_json)
        run_dir = _infer_run_dir_from_target_instances_path(ti_path)
        source_paths = _resolve_target_instance_paths(ti_path)
        source_kind = "target_instances_json"
    else:
        resolve_exp(cfg, args.exp, apply_current=True)
        run_dir = get_run_dir(cfg)
        ti_path = run_dir / "target_instances" / "target_instances.json"
        source_paths = _resolve_target_instance_paths(ti_path)
        source_kind = "target_instances_json"

    # Audit log (only when running through the standard pipeline path —
    # legacy --visibility_json / --target_instances_json invocations
    # write to whichever run_dir they infer from those paths).
    filt_dir = run_dir / "filters"
    filt_dir.mkdir(parents=True, exist_ok=True)
    split = get_split(cfg) if cfg else "unknown"
    audit = load_audit(filt_dir, split)
    register_stage(audit, "select")
    strip_stage_events(audit, "select")

    selections: Dict[str, Dict[str, Any]] = {}
    counts: Counter = Counter()
    written_paths: List[Path] = []
    viz_count = 0
    viz_errors = 0
    viz_skipped = 0
    last_frame_viz_count = 0
    last_frame_instance_viz_count = 0
    partition_viz_count = 0
    last_frame_missing = 0
    scenes_dir = cfg.get("scenes", {}).get("scenes_dir", "")
    default_target_root = run_dir / "target_instances"
    viz_root = (
        Path(args.out).parent if args.out else default_target_root
    ) / "viz_last_frame_instance"
    partition_viz_root = (
        Path(args.out).parent if args.out else default_target_root
    ) / "viz_partition"
    last_frame_viz_root = (
        Path(args.out).parent if args.out else default_target_root
    ) / "viz_last_frame"

    # Selection needs instance centers for >1-candidate cases.  Prefer Habitat
    # AABB centers when available; fall back to MP3D .house object centers.
    checker: VisibilityChecker = VisibilityChecker(cfg["env"], scenes_dir)
    connectivity_by_scan: Dict[str, Dict[str, np.ndarray]] = {}

    def _ensure_scan_loaded(
        scan_name: str,
    ) -> Tuple[Dict[str, np.ndarray], Dict[int, np.ndarray], str]:
        if scan_name not in connectivity_by_scan:
            db = load_connectivity(
                scenes_dir=scenes_dir,
                scans=[scan_name],
                json_dir=cfg["dataset"].get("connectivity_json_dir"),
                pkl_path=cfg["dataset"].get("connectivity_pkl"),
            )
            connectivity_by_scan[scan_name] = db.get(scan_name, {})
        house_centers = instance_centers_from_house(scenes_dir, scan_name)
        try:
            checker.load_scene(f"mp3d/{scan_name}/{scan_name}.glb")
        except Exception as exc:
            if house_centers:
                print(
                    f"  WARN: could not load Habitat scene for {scan_name}: {exc}. "
                    f"Using {len(house_centers)} .house instance centers."
                )
                return connectivity_by_scan[scan_name], house_centers, "house"
            print(
                f"  WARN: could not load Habitat scene for {scan_name}: {exc}. "
                "Multi-candidate selection will fall back to largest pixel count."
            )
            return connectivity_by_scan[scan_name], {}, "unavailable"
        scene_centers = instance_centers_from_scene(checker)
        if scene_centers:
            return connectivity_by_scan[scan_name], scene_centers, "habitat"
        if house_centers:
            print(
                f"  WARN: Habitat scene for {scan_name} exposed no centers; "
                f"using {len(house_centers)} .house instance centers."
            )
            return connectivity_by_scan[scan_name], house_centers, "house"
        return connectivity_by_scan[scan_name], {}, "unavailable"

    try:
        for source_path in source_paths:
            with open(source_path) as f:
                annotations = json.load(f)
            scan = annotations.get("scan")
            source_selections: Dict[str, Dict[str, Any]] = {}
            source_counts: Counter = Counter()

            scan_db: Dict[str, np.ndarray] = {}
            instance_centers: Dict[int, np.ndarray] = {}
            center_source = "unavailable"
            last_frames: Dict[Tuple[str, str], Dict[str, Any]] = {}
            if scan:
                scan_db, instance_centers, center_source = _ensure_scan_loaded(scan)
                last_frames = _load_last_rollout_frames(run_dir, scan)
            habitat_loaded = center_source == "habitat"

            for ep_id, sub_idx, record in _iter_records(annotations):
                end_pos: Optional[np.ndarray] = None
                partition_pos: Optional[np.ndarray] = None
                sub_total = 1
                last_frame = last_frames.get((str(ep_id), str(sub_idx)))
                if scan and scan_db:
                    end_pos, partition_pos, sub_total, _ = load_subpath_positions(
                        run_dir=run_dir,
                        scan=scan,
                        ep_id=str(ep_id),
                        sub_idx=str(sub_idx),
                        scan_db=scan_db,
                    )

                selected = select_target(
                    record,
                    end_pos=end_pos,
                    instance_centers=instance_centers,
                    center_source=center_source,
                )
                if scan:
                    selected["scan"] = scan
                if last_frame:
                    selected["rollout_last_frame"] = last_frame.get("_rollout_frame")
                    selected["rollout_last_frame_index"] = last_frame.get("_frame_index")
                    selected["rollout_last_step"] = last_frame.get("step")
                    if args.save_viz and scan:
                        last_out = (
                            last_frame_viz_root
                            / scan
                            / str(ep_id)
                            / f"sub_{int(sub_idx):03d}_last.png"
                        )
                        copied = _copy_last_rollout_frame(last_frame, last_out)
                        if copied:
                            selected["rollout_last_viz_path"] = copied
                            last_frame_viz_count += 1
                elif scan:
                    last_frame_missing += 1
                if args.save_viz and scan and selected.get("target_instance_ids"):
                    target_id = int(selected["target_instance_ids"][0])
                    partition_viz_path = (
                        partition_viz_root
                        / scan
                        / str(ep_id)
                        / f"sub_{int(sub_idx):03d}_id_{target_id}.png"
                    )
                    copied_partition = _copy_existing_png(
                        candidate_viz_path(record, target_id),
                        partition_viz_path,
                    )
                    if copied_partition:
                        selected["partition_viz_path"] = copied_partition
                        partition_viz_count += 1
                    elif habitat_loaded and partition_pos is not None:
                        try:
                            rv = render_mask_for_rollout_frame(
                                checker=checker,
                                scan=scan,
                                instance_id=target_id,
                                frame_record={
                                    "position": [float(x) for x in partition_pos],
                                    "heading": 0.0,
                                    "instruction_id": _safe_instruction_id(str(ep_id)),
                                    "instruction": selected.get("landmark") or "",
                                    "landmark": selected.get("landmark") or "",
                                    "sub_idx": int(sub_idx),
                                    "sub_total": sub_total,
                                    "step": 0,
                                    "action": "SELECT_TARGET_PARTITION",
                                },
                                out_path=partition_viz_path,
                                info_width=args.info_width,
                            )
                            selected["partition_viz_path"] = rv["path"]
                            selected["partition_viz_target_pixels"] = rv.get(
                                "target_visible_pixels"
                            )
                            partition_viz_count += 1
                        except Exception as exc:
                            selected["partition_viz_error"] = str(exc)
                            viz_errors += 1
                    else:
                        selected["partition_viz_skipped"] = (
                            "candidate viz missing and Habitat scene unavailable"
                            if not habitat_loaded else "partition_pos_unresolvable"
                        )

                    viz_path = (
                        viz_root
                        / scan
                        / str(ep_id)
                        / f"sub_{int(sub_idx):03d}_id_{target_id}.png"
                    )
                    if args.viz_mode == "mask" and not habitat_loaded:
                        viz_skipped += 1
                        selected["viz_skipped"] = (
                            f"Habitat scene unavailable; center_source={center_source}"
                        )
                    else:
                        try:
                            if args.viz_mode == "topdown":
                                draw_house_instance_viz(
                                    scenes_dir=scenes_dir,
                                    scan=scan,
                                    instance_id=target_id,
                                    out_path=viz_path,
                                )
                                selected["viz_path"] = str(viz_path)
                            else:
                                if end_pos is None:
                                    raise RuntimeError("subpath_end_pos_unresolvable")
                                render_frame = dict(last_frame or {})
                                if not render_frame:
                                    render_frame = {
                                        "position": [float(x) for x in end_pos],
                                        "heading": 0.0,
                                        "instruction_id": _safe_instruction_id(str(ep_id)),
                                        "instruction": selected.get("landmark") or "",
                                        "landmark": selected.get("landmark") or "",
                                        "sub_idx": int(sub_idx),
                                        "sub_total": sub_total,
                                        "step": 0,
                                        "action": "SELECT_TARGET",
                                    }
                                else:
                                    render_frame["landmark"] = selected.get("landmark") or ""
                                    render_frame["instruction"] = (
                                        render_frame.get("sub_instruction")
                                        or selected.get("landmark")
                                        or ""
                                    )
                                    render_frame["sub_total"] = int(
                                        render_frame.get("sub_total") or sub_total
                                    )
                                rv = render_mask_for_rollout_frame(
                                    checker=checker,
                                    scan=scan,
                                    instance_id=target_id,
                                    frame_record=render_frame,
                                    out_path=viz_path,
                                    info_width=args.info_width,
                                )
                                selected["last_frame_instance_viz_path"] = rv["path"]
                                selected["last_frame_instance_target_pixels"] = rv.get(
                                    "target_visible_pixels"
                                )
                                selected["viz_path"] = rv["path"]
                        except Exception as exc:
                            selected["viz_error"] = str(exc)
                            viz_errors += 1
                        else:
                            viz_count += 1
                            last_frame_instance_viz_count += 1
                # Merge selection result into the per-(ep, sub) annotations
                # record in place. Single source of truth — no separate
                # `target_instances` section so the file stays flat.
                record.update({
                    "target_instance_ids": selected.get("target_instance_ids", []),
                    "instance_id": (
                        int(selected["target_instance_ids"][0])
                        if selected.get("target_instance_ids") else None
                    ),
                    "status":              selected.get("status"),
                    "selection_distance":  selected.get("selection_distance"),
                    "candidate_distances": selected.get("candidate_distances"),
                    "center_source":       selected.get("center_source"),
                    "rescued":             record.get("rescued", False),
                })
                if selected.get("fallback_reason"):
                    record["fallback_reason"] = selected["fallback_reason"]
                if selected.get("viz_error"):
                    record["viz_error"] = selected["viz_error"]
                selections.setdefault(ep_id, {})[sub_idx] = selected
                source_selections.setdefault(ep_id, {})[sub_idx] = selected
                counts[selected["status"]] += 1
                source_counts[selected["status"]] += 1

                # Audit cell. Episode metadata is filled lazily — we
                # don't have a LandmarkRxREpisode here, only ids.
                ep_audit = audit["episodes"].setdefault(str(ep_id), {
                    "scan": scan, "events": [], "sub_paths": {},
                })
                tids = selected.get("target_instance_ids") or []
                append_sub_event(
                    ep_audit, sub_idx, stage="select", action="selected",
                    status=selected.get("status"),
                    target_instance_ids=tids,
                    instance_id=(int(tids[0]) if tids else None),
                    selection_distance=selected.get("selection_distance"),
                )

            if args.out is None:
                annotations["selection_rule"] = (
                    "single -> view_unique; multi -> nearest-to-subpath-end"
                )
                annotations["selection_summary"] = dict(source_counts)
                with open(source_path, "w") as f:
                    json.dump(annotations, f, indent=2)
                written_paths.append(source_path)
    finally:
        checker.close()

    finalize_audit(audit)
    save_audit(audit, filt_dir)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            f"source_{source_kind}": [str(p) for p in source_paths],
            "selection_rule": "single -> view_unique; multi -> nearest-to-subpath-end",
            "summary": dict(counts),
            "target_instances": selections,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        written_paths.append(out_path)

    total = sum(counts.values())
    auto = counts["view_unique"] + counts["view_nearest"] + counts["view_nearest_fallback"]
    print("=== Target Instance Selection ===")
    print(f"{source_kind:<16s}: {source_paths[0] if len(source_paths) == 1 else f'{len(source_paths)} files'}")
    print(f"output          : {written_paths[0] if len(written_paths) == 1 else f'{len(written_paths)} files'}")
    print(f"records         : {total}")
    auto_pct = (auto / total) if total else 0.0
    print(f"auto selected   : {auto}  ({auto_pct:.1%})")
    if args.save_viz:
        print(f"last-frame raw  : {last_frame_viz_count}")
        print(f"last-frame inst : {last_frame_instance_viz_count}")
        print(f"partition viz   : {partition_viz_count}")
        if last_frame_missing:
            print(f"last-frame miss : {last_frame_missing}")
        if viz_skipped:
            print(f"viz skipped     : {viz_skipped}")
        if viz_errors:
            print(f"viz errors      : {viz_errors}")
    print("status breakdown:")
    for status, n in counts.most_common():
        pct = (n / total) if total else 0.0
        print(f"  {status:<24s} {n:>5d}  ({pct:.1%})")

    if args.print_multi:
        print()
        print("multi-candidate sub-paths:")
        for ep_id, sub_map in selections.items():
            for sub_idx, selected in sub_map.items():
                if selected["status"] not in ("view_nearest", "view_nearest_fallback"):
                    continue
                dist = selected.get("selection_distance")
                dist_str = f"{dist:.3f}m" if dist is not None else "  —  "
                print(
                    f"  ep={ep_id:<8s} sub={sub_idx:<3s} "
                    f"status={selected['status']:<22s} "
                    f"dist={dist_str}  "
                    f"chosen={selected['target_instance_ids']}  "
                    f"all_dists={selected.get('candidate_distances')}  "
                    f"landmark={selected.get('landmark')!r}"
                )


if __name__ == "__main__":
    main()
