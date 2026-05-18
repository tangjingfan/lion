"""Inspection viz — render a panorama at every sub-path's partition pose.

Walks ``filters/audit.json`` and renders, for every ``(ep, sub_idx)`` that
survived the cross_floor stage, a 360° RGB + depth + semantic panorama at
the partition-end pose (the point where the agent finishes the spatial
segment and must look for the landmark). The PNGs land under

    {run_dir}/target_instances/viz_inspection/{scan}/{ep_id}/sub_{idx}__{status}.png

so every potentially rescuable sub-path has its inspection frame in one
place, side-by-side with its sibling sub-paths from the same episode.

The filename status suffix encodes what the pipeline did with this sub:

  view_unique / view_nearest / view_nearest_fallback   target picked OK
  rescued                                              YOLO/VLM rescue hit
  synthesized                                          blacklist replaced
  visibility_not_visible                               vis check failed
  blacklist_<reason>                                   dropped at stage 02
  partition_<reason>                                   dropped at stage 04
  cross_floor_drop                                     skipped — no pose

A coloured title bar on each PNG repeats the status so a directory listing
of thumbnails is enough to triage what to rescue / discard / re-run.

This step reads-only — it touches no other pipeline state.

Usage
-----
  python src/check/inspection_viz.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --exp configs/selection/val_unseen/one_scene_partial.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import (
    get_filter_dir,
    get_run_dir,
    resolve_exp,
)
from src.dataset.landmark_rxr import episodes_from_config
from src.env.connectivity import load_connectivity
from src.process.visibility import VisibilityChecker
from src.viz import _compose


# ── status classification ────────────────────────────────────────────────


def _classify_status(
    ep_audit:       Dict[str, Any],
    sub_audit:      Dict[str, Any],
    target_for_sub: Optional[Dict[str, Any]],
    rescued:        bool,
) -> str:
    """Return a single status slug for one (ep, sub).

    Resolution order matches the pipeline's: a blacklist drop happens before
    a visibility check ever runs, so it takes precedence over any later
    status. Cross-floor drops are caller-filtered but handled defensively.
    """
    if rescued:
        return "synthesized"

    ep_stages = (ep_audit or {}).get("stages") or {}
    if (ep_stages.get("cross_floor") or {}).get("status") == "drop":
        return "cross_floor_drop"

    sub_stages = (sub_audit or {}).get("stages") or {}

    bl = sub_stages.get("blacklist") or {}
    if bl.get("status") == "drop":
        return f"blacklist_{bl.get('reason') or 'drop'}"

    pt = sub_stages.get("partition") or {}
    if pt.get("status") == "drop":
        rw = pt.get("rewrite")
        if rw and rw != "ok":
            tag = f"rewrite_{rw}"
        else:
            tag = f"partition_{pt.get('partition') or 'drop'}"
        return f"partition_{tag}"

    if target_for_sub:
        status = target_for_sub.get("status") or ""
        if status:
            return status
    return "no_target_data"


def _slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_")
    return s[:80] or "unknown"


def _bar_color_for_status(status: str) -> Tuple[int, int, int]:
    if status.startswith("blacklist_") or status.startswith("partition_"):
        return (130, 50, 60)
    if "not_visible" in status:
        return (140, 110, 40)
    if status in ("rescued", "synthesized"):
        return (50, 110, 70)
    if status.startswith("view_"):
        return (60, 90, 130)
    return (60, 60, 70)


# ── geometry helpers (mirrors rescue_blacklist._resolve_node_pos) ────────


def _resolve_node_pos(
    node_id:       Any,
    virtual_nodes: Dict[str, List[float]],
    scan_db:       Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    if isinstance(node_id, str) and node_id.startswith("virt:"):
        pos = virtual_nodes.get(node_id)
        return np.asarray(pos, dtype=np.float32) if pos is not None else None
    if node_id in scan_db:
        return np.asarray(scan_db[node_id], dtype=np.float32)
    return None


def _partition_end_pos(
    part_sub:      Dict[str, Any],
    virtual_nodes: Dict[str, List[float]],
    scan_db:       Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    """Pose at which the agent stops the spatial segment — i.e. the last
    node of ``spatial_path``. This is where a human inspector wants to
    "look around" to see if the landmark is visible."""
    spatial = part_sub.get("spatial_path") or []
    if not spatial:
        return None
    return _resolve_node_pos(spatial[-1], virtual_nodes, scan_db)


# ── render ───────────────────────────────────────────────────────────────


def _render_inspection_frame(
    checker:      VisibilityChecker,
    pos:          np.ndarray,
    ep:           Any,
    sub_idx:      int,
    sub_total:    int,
    status_label: str,
    landmark:     str,
    out_path:     Path,
    info_width:   int,
) -> None:
    from PIL import Image, ImageDraw

    obs = checker.render_observation(pos, 0.0)
    sem = obs.get("semantic")
    if sem is not None and checker._sem_id_map is not None:  # noqa: SLF001
        sem_clip = np.clip(sem, 0, len(checker._sem_id_map) - 1)  # noqa: SLF001
        obs["semantic_id"]   = checker._sem_id_map[sem_clip]      # noqa: SLF001
        obs["semantic_name"] = checker._sem_name_map[sem_clip]    # noqa: SLF001

    episode_stub = SimpleNamespace(
        scan             = ep.scan,
        instruction_id   = ep.instruction_id,
        instruction      = ep.instruction,
        sub_paths        = [None] * max(sub_total, 1),
        sub_instructions = list(getattr(ep, "sub_instructions", []) or []),
    )
    canvas = _compose(
        obs                   = obs,
        episode               = episode_stub,
        step                  = 0,
        action                = None,
        info_w                = info_width,
        sub_idx               = sub_idx,
        sub_total             = sub_total,
        mark_semantic_numbers = False,
    )

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    bar_h = 24
    draw.rectangle([(0, 0), (img.width, bar_h)],
                   fill=_bar_color_for_status(status_label))
    title = f"[{status_label}]"
    if landmark:
        title = f"{title}  landmark={landmark!r}"
    max_chars = max(20, (img.width - 10) // 7)
    if len(title) > max_chars:
        title = title[: max_chars - 3] + "..."
    draw.text((6, 7), title, fill=(245, 245, 245))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


# ── driver ───────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render an inspection panorama at every sub-path's "
                    "partition pose (one PNG per ep, sub).",
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", default=None,
                    help="Experiment handle (selection YAML path or expname). "
                         "survivor.yaml is NOT auto-merged — we walk audit.json "
                         "so blacklist/partition-dropped subs are included.")
    ap.add_argument("--info_width", type=int, default=600)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # apply_current=False: we want audit.json to drive iteration, not the
    # post-pipeline narrowed survivor set.
    resolve_exp(cfg, args.exp, apply_current=False)

    run_dir  = get_run_dir(cfg)
    filt_dir = get_filter_dir(cfg)
    audit_path = filt_dir / "audit.json"
    if not audit_path.exists():
        raise SystemExit(f"No audit.json at {audit_path}. Run the filter pipeline first.")
    with open(audit_path) as f:
        audit = json.load(f) or {}

    alive_ep_ids: List[int] = []
    for ep_id_str, ep_audit in (audit.get("episodes") or {}).items():
        cf = (ep_audit.get("stages") or {}).get("cross_floor") or {}
        if cf.get("status") == "drop":
            continue
        try:
            alive_ep_ids.append(int(ep_id_str))
        except ValueError:
            continue
    if not alive_ep_ids:
        raise SystemExit("No alive episodes after cross_floor stage.")

    # Side-load episodes (bypass survivor narrowing): give the loader the
    # full alive set with unrestricted sub_paths.
    cfg.setdefault("selection", {})["instruction_ids"] = sorted(alive_ep_ids)
    cfg["selection"]["sub_paths"] = {}
    episodes = {int(ep.instruction_id): ep for ep in episodes_from_config(cfg)}
    if not episodes:
        raise SystemExit("episodes_from_config returned nothing for the alive set.")

    needed_scans = sorted({ep.scan for ep in episodes.values()})
    scan_db_all = load_connectivity(
        scenes_dir = cfg["scenes"]["scenes_dir"],
        scans      = needed_scans,
        json_dir   = cfg["dataset"].get("connectivity_json_dir"),
        pkl_path   = cfg["dataset"].get("connectivity_pkl"),
    )

    # Pre-load per-scan target_instances + blacklist_rescue for status lookup.
    target_db_by_scan: Dict[str, Dict] = {}
    rescue_db_by_scan: Dict[str, Dict] = {}
    for scan in needed_scans:
        ti_path = run_dir / "target_instances" / scan / "target_instances.json"
        if ti_path.exists():
            with open(ti_path) as f:
                target_db_by_scan[scan] = json.load(f) or {}
        br_path = run_dir / "target_instances" / scan / "blacklist_rescue.json"
        if br_path.exists():
            with open(br_path) as f:
                rescue_db_by_scan[scan] = json.load(f) or {}

    # Group (ep, sub) by scan so each scene gets loaded once.
    by_scan: Dict[str, List[Tuple[int, int]]] = {}
    for ep_id, ep in episodes.items():
        ep_audit = audit["episodes"].get(str(ep_id)) or {}
        sub_audits = ep_audit.get("sub_paths") or {}
        if not sub_audits:
            # Episode survived cross_floor but no sub-path was ever
            # audited (e.g. only stage 01 has run so far). Fall back to
            # iterating the raw sub_paths range.
            for sub_idx in range(len(ep.sub_paths)):
                by_scan.setdefault(ep.scan, []).append((ep_id, sub_idx))
        else:
            for sub_idx_str in sub_audits:
                try:
                    sub_idx = int(sub_idx_str)
                except ValueError:
                    continue
                by_scan.setdefault(ep.scan, []).append((ep_id, sub_idx))

    out_root = run_dir / "target_instances" / "viz_inspection"
    out_root.mkdir(parents=True, exist_ok=True)

    checker = VisibilityChecker(cfg["env"], cfg["scenes"]["scenes_dir"])

    n_rendered            = 0
    n_skip_no_partition   = 0
    n_skip_no_pos         = 0
    n_skip_render_error   = 0
    status_counts: Dict[str, int] = {}

    try:
        for scan, items in sorted(by_scan.items()):
            checker.load_scene(f"mp3d/{scan}/{scan}.glb")
            scan_db   = scan_db_all.get(scan) or {}
            target_db = target_db_by_scan.get(scan) or {}
            rescue_db = rescue_db_by_scan.get(scan) or {}

            # Cache partition.json per ep within this scan.
            part_cache: Dict[int, Optional[Dict]] = {}

            for ep_id, sub_idx in sorted(set(items)):
                ep = episodes[ep_id]
                ep_audit  = audit["episodes"].get(str(ep_id)) or {}
                sub_audit = (ep_audit.get("sub_paths") or {}).get(str(sub_idx)) or {}

                if ep_id not in part_cache:
                    p = run_dir / "partition" / scan / str(ep_id) / "partition.json"
                    if p.exists():
                        with open(p) as f:
                            part_cache[ep_id] = json.load(f)
                    else:
                        part_cache[ep_id] = None
                part_json = part_cache[ep_id]
                if part_json is None:
                    n_skip_no_partition += 1
                    continue

                part_subs = {
                    int(p["sub_idx"]): p
                    for p in (part_json.get("partitions") or [])
                    if "sub_idx" in p
                }
                part_sub = part_subs.get(int(sub_idx))
                if part_sub is None:
                    n_skip_no_partition += 1
                    continue

                pos = _partition_end_pos(
                    part_sub,
                    part_json.get("virtual_nodes") or {},
                    scan_db,
                )
                if pos is None:
                    n_skip_no_pos += 1
                    continue

                target_for_sub = (
                    (target_db.get("target_instances") or {})
                    .get(str(ep_id), {})
                    .get(str(sub_idx))
                )
                rescue_for_sub = (
                    (rescue_db.get("rescues") or {})
                    .get(str(ep_id), {})
                    .get(str(sub_idx))
                )
                status = _classify_status(
                    ep_audit, sub_audit, target_for_sub,
                    rescued=bool(rescue_for_sub),
                )
                slug = _slug(status)

                landmark = ""
                if rescue_for_sub:
                    landmark = rescue_for_sub.get("new_landmark") or ""
                elif target_for_sub and target_for_sub.get("landmark"):
                    landmark = target_for_sub.get("landmark") or ""
                elif part_sub.get("landmark"):
                    landmark = part_sub.get("landmark") or ""

                sub_total = max(len(ep.sub_paths), 1)
                out_path = (
                    out_root
                    / scan
                    / str(ep_id)
                    / f"sub_{int(sub_idx):03d}__{slug}.png"
                )
                try:
                    _render_inspection_frame(
                        checker, pos, ep, sub_idx, sub_total,
                        status, landmark, out_path,
                        info_width=args.info_width,
                    )
                except Exception as exc:
                    n_skip_render_error += 1
                    print(f"  [render-error] ep={ep_id} sub={sub_idx}: {exc}")
                    continue

                n_rendered += 1
                status_counts[status] = status_counts.get(status, 0) + 1
    finally:
        checker.close()

    print()
    print("=== inspection viz summary ===")
    print(f"  rendered             : {n_rendered}")
    print(f"  skip (no partition)  : {n_skip_no_partition}")
    print(f"  skip (no pose)       : {n_skip_no_pos}")
    if n_skip_render_error:
        print(f"  skip (render error)  : {n_skip_render_error}")
    if status_counts:
        print(f"  by status:")
        for st, n in sorted(status_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {st:<35s} {n}")
    print()
    print(f"  → {out_root}")


if __name__ == "__main__":
    main()
