"""LION-Bench — Robustness check: target visibility under small position perturbations.

For each surviving sub-path with a chosen target instance, stand at the
**start of the sub-trajectory** and render a 360° semantic panorama at
``N`` positions arranged on a circle of radius ``R`` around the start
node.  Report whether each target instance id is visible from each
perturbed position — answering "is the target robustly visible from a
small neighbourhood of the start point, or only from one lucky pose?".

Inputs (per scan)
-----------------
  • ``{run_dir}/target_instances/{scan}/target_instances.json`` (or the
    legacy single ``{run_dir}/target_instances/target_instances.json``)
    — chosen target instance ids per (ep_id, sub_idx).
  • ``{run_dir}/filters/current.yaml`` — sub-path-level survivors.
  • Connectivity DB — for the start node's 3-D position.

Output (per scan)
-----------------
  ``{run_dir}/perturb_visibility/{scan}/summary.json`` indexing one
  ``visibility.json`` per sub-trajectory.

  When ``--save_viz`` is on (default), one PNG per perturbation is also
  written next to that sub-trajectory's JSON under
  ``{run_dir}/perturb_visibility/{scan}/{instruction_id}/sub_{NNN}/k_{KK}.png``
  — equirectangular RGB panorama with a red overlay highlighting the
  target-instance pixels and a one-line caption (k / angle / VIS / px /
  snap).

Usage
-----
  python src/check/perturb_visibility.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --from_yaml results/{run}/filters/current.yaml \\
      --radius 0.5 --n 8
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import (
    get_filter_dir,
    get_run_dir,
    get_split,
    load_keep,
    resolve_selection,
)
from src.dataset.landmark_rxr import episodes_from_config
from src.env.connectivity import load_connectivity
from src.process.visibility import VisibilityChecker
from src.process.visibility_analysis import (
    check_same_category_instances,
    check_targets_visible,
    overlay_target,
    perturbed_positions,
    target_categories as resolve_target_categories,
    target_required_pixels,
)


def _load_target_instances(
    run_dir: Path, scan: str,
) -> Dict[str, Dict[str, Dict]]:
    """Return ``{ep_id: {sub_idx: entry}}`` for one scan.

    Tries the per-scan layout first; falls back to the legacy single
    file and filters by each entry's ``scan`` field.
    """
    per_scan = run_dir / "target_instances" / scan / "target_instances.json"
    if per_scan.exists():
        with open(per_scan) as f:
            payload = json.load(f) or {}
        return payload.get("target_instances") or payload

    legacy = run_dir / "target_instances" / "target_instances.json"
    if not legacy.exists():
        return {}
    with open(legacy) as f:
        payload = json.load(f) or {}
    src = payload.get("target_instances") or {}
    out: Dict[str, Dict[str, Dict]] = {}
    for ep_id, sub_map in src.items():
        for sub_idx, entry in (sub_map or {}).items():
            if entry.get("scan") != scan:
                continue
            out.setdefault(ep_id, {})[sub_idx] = entry
    return out


def _maybe_snap(
    sim, pos: np.ndarray, snap: bool,
) -> Dict[str, Any]:
    """Snap ``pos`` to the nearest navmesh point when requested.

    Returns a dict with ``pos`` (the position to render at), ``snapped``
    (True iff the navmesh moved the point), and ``snap_dx`` (Euclidean
    distance the point moved during snapping).
    """
    if not snap or sim is None or sim.pathfinder is None:
        return {"pos": pos, "snapped": False, "snap_dx": 0.0}
    try:
        snapped = np.asarray(sim.pathfinder.snap_point(pos), dtype=np.float32)
    except Exception:
        return {"pos": pos, "snapped": False, "snap_dx": 0.0}
    if not np.all(np.isfinite(snapped)):
        return {"pos": pos, "snapped": False, "snap_dx": 0.0}
    dx = float(np.linalg.norm(snapped - pos))
    return {
        "pos":     snapped,
        "snapped": dx > 1e-4,
        "snap_dx": dx,
    }


def _save_perturbation_png(
    rgb_overlay: np.ndarray,
    caption:     str,
    out_path:    Path,
    panorama_w:  int = 768,
    visible:     bool = False,
) -> None:
    """Save one perturbation panorama with a thin caption bar on top.

    The panorama is downsampled to ``panorama_w`` while preserving its
    native equirectangular aspect ratio.  A 26 px caption strip above
    the image carries the ``k / angle / VIS-or-not / px / snap`` summary;
    the text colour reflects ``visible``.
    """
    from PIL import Image, ImageDraw, ImageFont

    pano = Image.fromarray(rgb_overlay)
    h = max(1, int(round(pano.height * panorama_w / max(1, pano.width))))
    pano = pano.resize((panorama_w, h), Image.BILINEAR)

    label_h = 26
    canvas = Image.new("RGB", (panorama_w, label_h + h), color=(20, 20, 25))
    draw   = ImageDraw.Draw(canvas)
    font   = ImageFont.load_default()
    col    = (140, 220, 140) if visible else (220, 130, 130)
    draw.text((6, 6), caption, fill=col, font=font)
    canvas.paste(pano, (0, label_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Robustness check: see target instance from N positions "
                    "on a small circle around each sub-trajectory's start.",
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--from_yaml", default=None,
                    help="Selection / current.yaml carrying expname.")
    ap.add_argument("--radius", type=float, default=0.5,
                    help="Perturbation circle radius in metres (default 0.5).")
    ap.add_argument("--n", type=int, default=8,
                    help="Number of perturbed positions on the circle (default 8).")
    ap.add_argument("--min_pixel_count", type=int, default=50,
                    help="Min pixels per target id to count as visible (default 50).")
    ap.add_argument("--min_original_pixel_fraction", type=float, default=0.5,
                    help="Also require target pixels in each perturbation to "
                         "be at least this fraction of the original selected "
                         "view's pixel count (default 0.5).")
    ap.add_argument("--no_snap", action="store_true",
                    help="Render at the raw perturbed position; do NOT snap to navmesh.")
    ap.add_argument("--save_viz", action="store_true", default=True,
                    help="Save one PNG per perturbation: the equirectangular "
                         "RGB panorama with target mask overlay (default on).")
    ap.add_argument("--no_save_viz", action="store_false", dest="save_viz",
                    help="Skip per-perturbation PNGs.")
    ap.add_argument("--panorama_width", type=int, default=768,
                    help="Output PNG width in px (default 768).")
    args = ap.parse_args()

    if args.n <= 0:
        raise SystemExit(f"--n must be positive (got {args.n})")
    if args.radius <= 0:
        raise SystemExit(f"--radius must be positive (got {args.radius})")
    if args.min_original_pixel_fraction < 0:
        raise SystemExit("--min_original_pixel_fraction must be non-negative")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    resolve_selection(cfg, args.from_yaml)

    filt_dir = get_filter_dir(cfg)
    if not filt_dir.exists():
        raise SystemExit(f"No filters/ at {filt_dir}")
    current = filt_dir / "current.yaml"
    if not current.exists():
        raise SystemExit(f"No current.yaml at {current}")
    prior_keep = load_keep(current.resolve())
    prior_subs = prior_keep.get("sub_paths") or {}
    if not prior_subs:
        raise SystemExit("current.yaml has no sub_paths field — run filter stages first.")

    run_dir = get_run_dir(cfg)
    split   = get_split(cfg)

    # Episodes restricted to prior survivors.
    needed_ids = {int(k) for k in prior_subs.keys()}
    episodes  = [ep for ep in episodes_from_config(cfg)
                 if ep.instruction_id in needed_ids]
    if not episodes:
        raise SystemExit("No episodes loaded.")

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

    out_root = run_dir / "perturb_visibility"
    out_root.mkdir(parents=True, exist_ok=True)

    snap_to_navmesh = not args.no_snap
    written_paths:  List[Path] = []
    summary_paths:  List[Path] = []
    grand_total     = 0
    grand_n_targets = 0
    grand_n_perturb = 0
    grand_n_visible = 0
    grand_all_vis   = 0
    grand_n_cat_unique = 0
    grand_all_cat_unique = 0
    grand_viz_saved = 0
    grand_viz_err   = 0

    checker = VisibilityChecker(cfg["env"], cfg["scenes"]["scenes_dir"])
    try:
        for scan in needed_scans:
            scan_eps  = by_scan[scan]
            scan_db   = db.get(scan, {})
            target_db = _load_target_instances(run_dir, scan)
            if not target_db:
                print(f"\n[{scan}] no target_instances.json — skipping")
                continue

            print(f"\n[{scan}] loading scene  ({len(scan_eps)} episode(s)) …")
            checker.load_scene(scan_eps[0].scene_file)
            sim = checker._sim  # noqa: SLF001 - we need pathfinder for snapping
            sem_name_map = getattr(checker, "_sem_name_map", None)

            scan_dir = out_root / scan
            scan_dir.mkdir(parents=True, exist_ok=True)
            scan_index: List[Dict[str, Any]] = []
            scan_total = 0
            scan_n_perturb = 0
            scan_n_visible = 0
            scan_all_vis = 0
            scan_n_cat_unique = 0
            scan_all_cat_unique = 0

            for ep in scan_eps:
                ep_id     = ep.instruction_id
                ep_id_str = str(ep_id)
                sub_idxs  = (prior_subs.get(ep_id)
                             or prior_subs.get(ep_id_str)
                             or [])
                ep_targets = target_db.get(ep_id_str) or {}
                if not ep_targets:
                    continue

                for sub_idx in sub_idxs:
                    sub_idx     = int(sub_idx)
                    sub_idx_str = str(sub_idx)
                    entry       = ep_targets.get(sub_idx_str)
                    if entry is None:
                        continue
                    target_ids = [int(x) for x in (entry.get("target_instance_ids") or [])]
                    sub_dir = scan_dir / ep_id_str / f"sub_{sub_idx:03d}"
                    sub_json = sub_dir / "visibility.json"

                    def write_sub(record: Dict[str, Any]) -> None:
                        payload = {
                            "split":           split,
                            "scan":            scan,
                            "instruction_id":  ep_id,
                            "sub_idx":         sub_idx,
                            "expname":         cfg.get("output", {}).get("expname"),
                            "run_name":        cfg.get("output", {}).get("run_name"),
                            "radius_m":        args.radius,
                            "n_perturbations": args.n,
                            "min_pixel_count": args.min_pixel_count,
                            "min_original_pixel_fraction": args.min_original_pixel_fraction,
                            "snap_to_navmesh": snap_to_navmesh,
                            "source_keep":     str(current.resolve()),
                            **record,
                        }
                        _write_json(sub_json, payload)
                        written_paths.append(sub_json)
                        scan_index.append({
                            "instruction_id": ep_id,
                            "sub_idx": sub_idx,
                            "status": payload.get("status"),
                            "landmark": payload.get("landmark"),
                            "target_instance_ids": payload.get("target_instance_ids", []),
                            "target_categories": payload.get("target_categories", []),
                            "n_visible": payload.get("n_visible"),
                            "all_visible": payload.get("all_visible"),
                            "n_category_unique": payload.get("n_category_unique"),
                            "all_category_unique": payload.get("all_category_unique"),
                            "path": _rel(sub_json, scan_dir),
                        })

                    if sub_idx >= len(ep.sub_paths):
                        write_sub({
                            "status": "sub_idx_out_of_range",
                            "target_instance_ids": target_ids,
                        })
                        continue

                    sub_nodes = ep.sub_paths[sub_idx]
                    if not sub_nodes:
                        write_sub({
                            "status": "empty_sub_path",
                            "target_instance_ids": target_ids,
                        })
                        continue
                    start_node = sub_nodes[0]
                    if start_node not in scan_db:
                        write_sub({
                            "status": "start_node_not_in_db",
                            "start_node": start_node,
                            "target_instance_ids": target_ids,
                        })
                        continue
                    start_pos = np.asarray(scan_db[start_node], dtype=np.float32)

                    if not target_ids:
                        write_sub({
                            "status":              "no_target",
                            "landmark":            entry.get("landmark"),
                            "start_node":          start_node,
                            "start_pos":           [float(x) for x in start_pos],
                            "target_instance_ids": [],
                        })
                        continue

                    target_categories = resolve_target_categories(
                        entry, target_ids, sem_name_map,
                    )
                    required_pixels = target_required_pixels(
                        entry,
                        target_ids,
                        args.min_pixel_count,
                        args.min_original_pixel_fraction,
                    )
                    perturbations: List[Dict[str, Any]] = []
                    n_visible = 0
                    n_category_unique = 0
                    sub_viz_dir = sub_dir
                    for p in perturbed_positions(start_pos, args.radius, args.n):
                        snap = _maybe_snap(sim, p["raw_pos_np"], snap_to_navmesh)
                        obs  = checker.render_observation(snap["pos"], heading=0.0)
                        sem  = obs.get("semantic")
                        rgb  = obs.get("rgb")
                        if sem is None:
                            vis = {"hits": [], "n_visible_targets": 0,
                                   "any_visible": False, "max_pixels": 0,
                                   "target_mask": None,
                                   "required_pixels": {str(k): int(v) for k, v in required_pixels.items()}}
                        else:
                            vis = check_targets_visible(
                                sem,
                                target_ids,
                                args.min_pixel_count,
                                required_pixels=required_pixels,
                            )
                        same_cat = check_same_category_instances(
                            sem,
                            target_ids,
                            target_categories,
                            sem_name_map,
                            args.min_pixel_count,
                        ) if sem is not None else {
                            "hits": [],
                            "n_other_same_category": 0,
                            "category_unique": None,
                            "same_category_mask": None,
                        }
                        if vis["any_visible"]:
                            n_visible += 1
                        category_unique = (
                            bool(vis["any_visible"])
                            and same_cat["n_other_same_category"] == 0
                        )
                        if category_unique:
                            n_category_unique += 1

                        record: Dict[str, Any] = {
                            "k":         p["k"],
                            "angle_deg": p["angle_deg"],
                            "raw_pos":   p["raw_pos"],
                            "rendered_pos": [float(snap["pos"][0]),
                                             float(snap["pos"][1]),
                                             float(snap["pos"][2])],
                            "snapped":   snap["snapped"],
                            "snap_dx":   round(snap["snap_dx"], 4),
                            "any_visible":     vis["any_visible"],
                            "n_visible_ids":   vis["n_visible_targets"],
                            "max_pixels":      vis["max_pixels"],
                            "required_pixels": vis["required_pixels"],
                            "hits":            vis["hits"],
                            "target_categories": target_categories,
                            "n_other_same_category": same_cat["n_other_same_category"],
                            "category_unique": category_unique,
                            "same_category_hits": same_cat["hits"],
                        }

                        if args.save_viz and rgb is not None:
                            viz_path = sub_viz_dir / f"k_{int(p['k']):02d}.png"
                            try:
                                rgb_overlay = overlay_target(
                                    rgb, vis.get("target_mask"),
                                )
                                verdict = "VIS" if vis["any_visible"] else "—  "
                                if category_unique:
                                    uniq = "UNIQ"
                                elif not vis["any_visible"]:
                                    uniq = "NO_TARGET"
                                else:
                                    uniq = f"SAME={same_cat['n_other_same_category']}"
                                caption = (
                                    f"ep={ep_id}  sub={sub_idx}  "
                                    f"k={int(p['k']):<2d}  "
                                    f"ang={p['angle_deg']:>5.1f}°  "
                                    f"{verdict}  "
                                    f"px={vis['max_pixels']}/{max(required_pixels.values() or [args.min_pixel_count])}  "
                                    f"{uniq}  "
                                    f"snap={'Y' if snap['snapped'] else 'N'} "
                                    f"({snap['snap_dx']:.2f}m)  "
                                    f"r={args.radius}m"
                                )
                                _save_perturbation_png(
                                    rgb_overlay=rgb_overlay,
                                    caption=caption,
                                    out_path=viz_path,
                                    panorama_w=args.panorama_width,
                                    visible=vis["any_visible"],
                                )
                                record["viz_path"] = str(viz_path)
                                grand_viz_saved += 1
                            except Exception as exc:
                                record["viz_error"] = str(exc)
                                grand_viz_err += 1

                        perturbations.append(record)

                    grand_total     += 1
                    grand_n_targets += len(target_ids)
                    grand_n_perturb += args.n
                    grand_n_visible += n_visible
                    grand_n_cat_unique += n_category_unique
                    scan_total += 1
                    scan_n_perturb += args.n
                    scan_n_visible += n_visible
                    scan_n_cat_unique += n_category_unique
                    if n_visible == args.n:
                        grand_all_vis += 1
                        scan_all_vis += 1
                    if n_category_unique == args.n:
                        grand_all_cat_unique += 1
                        scan_all_cat_unique += 1

                    sub_record: Dict[str, Any] = {
                        "status":              "ok",
                        "landmark":            entry.get("landmark"),
                        "start_node":          start_node,
                        "start_pos":           [float(x) for x in start_pos],
                        "target_instance_ids": target_ids,
                        "target_categories":    target_categories,
                        "required_pixels":      {str(k): int(v) for k, v in required_pixels.items()},
                        "min_original_pixel_fraction": args.min_original_pixel_fraction,
                        "n_perturbations":     args.n,
                        "n_visible":           n_visible,
                        "all_visible":         n_visible == args.n,
                        "n_category_unique":    n_category_unique,
                        "all_category_unique":  n_category_unique == args.n,
                        "perturbations":       perturbations,
                    }
                    if args.save_viz:
                        sub_record["viz_dir"] = str(sub_viz_dir)

                    write_sub(sub_record)

                    print(f"  [{ep_id} sub {sub_idx:<2}] "
                          f"{n_visible}/{args.n} visible  "
                          f"{n_category_unique}/{args.n} cat-unique  "
                          f"target_ids={target_ids}  "
                          f"req_px={required_pixels}  "
                          f"cat={target_categories}  "
                          f"landmark={entry.get('landmark')!r}")

            scan_out = scan_dir / "summary.json"
            payload = {
                "split":           split,
                "scan":            scan,
                "expname":         cfg.get("output", {}).get("expname"),
                "run_name":        cfg.get("output", {}).get("run_name"),
                "radius_m":        args.radius,
                "n_perturbations": args.n,
                "min_pixel_count": args.min_pixel_count,
                "min_original_pixel_fraction": args.min_original_pixel_fraction,
                "snap_to_navmesh": snap_to_navmesh,
                "source_keep":     str(current.resolve()),
                "summary": {
                    "sub_paths_checked": scan_total,
                    "total_perturbations": scan_n_perturb,
                    "perturbations_seeing": scan_n_visible,
                    "fully_robust": scan_all_vis,
                    "category_unique_views": scan_n_cat_unique,
                    "fully_category_unique": scan_all_cat_unique,
                },
                "sub_trajectories": scan_index,
            }
            _write_json(scan_out, payload)
            summary_paths.append(scan_out)
            legacy_pointer = scan_dir / "visibility.json"
            _write_json(legacy_pointer, {
                "deprecated": True,
                "message": (
                    "Perturb visibility is now stored per sub-trajectory. "
                    "Read summary.json, then follow sub_trajectories[*].path."
                ),
                "summary": "summary.json",
            })
            print(f"  → {scan_out}")
    finally:
        checker.close()

    pct_perturb = (grand_n_visible / grand_n_perturb) if grand_n_perturb else 0.0
    pct_all_vis = (grand_all_vis   / grand_total)     if grand_total     else 0.0
    pct_cat_unique = (
        grand_n_cat_unique / grand_n_perturb
        if grand_n_perturb else 0.0
    )
    pct_all_cat_unique = (
        grand_all_cat_unique / grand_total
        if grand_total else 0.0
    )
    print(f"\n=== Perturb Visibility ===")
    print(f"  sub-paths checked    : {grand_total}")
    print(f"  total perturbations  : {grand_n_perturb}")
    print(f"  perturbations seeing : {grand_n_visible}  ({pct_perturb:.1%})")
    print(f"  fully robust (all {args.n}/{args.n}): "
          f"{grand_all_vis}  ({pct_all_vis:.1%})")
    print(f"  category-unique views: {grand_n_cat_unique}  ({pct_cat_unique:.1%})")
    print(f"  fully category-unique: {grand_all_cat_unique}  ({pct_all_cat_unique:.1%})")
    if args.save_viz:
        print(f"  composite PNGs saved : {grand_viz_saved}"
              + (f"  (errors: {grand_viz_err})" if grand_viz_err else ""))
    print(f"\nSummary outputs ({len(summary_paths)} scan file(s)):")
    for p in summary_paths:
        print(f"  {p}")
    print(f"Sub-trajectory JSONs: {len(written_paths)}")
    for p in written_paths[:20]:
        print(f"  {p}")
    if len(written_paths) > 20:
        print(f"  ... {len(written_paths) - 20} more")


if __name__ == "__main__":
    main()
