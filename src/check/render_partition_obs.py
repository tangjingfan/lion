"""Render an observation panorama at every sub-path's partition point.

Runs after ``04 partition``: walks each ``partition/{scan}/{ep_id}/partition.json``
and, for every ``(ep, sub_idx)`` present in ``survivor.yaml`` (which now
includes both active and labeled-drop subs past cross_floor), renders a
360° RGB + semantic panorama at the partition pose — the boundary point
where the spatial segment ends and the landmark segment begins
(``spatial_path[-1]``).

Output: one PNG per ``(ep, sub_idx)`` at

    {run_dir}/partition_obs/{scan}/{ep_id}/sub_{idx:03d}.png

(sibling of ``{run_dir}/partition/`` — the data the obs are rendered
from — so the inspection artifacts don't crowd the target_instances/
tree.)

Each PNG stacks the RGB panorama on top of the colored semantic panorama
plus a one-line title. Inspection-only — no downstream stage consumes
these.

Usage
-----
  python src/check/render_partition_obs.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --exp configs/selection/val_unseen/one_scene_partial.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import (
    get_run_dir,
    resolve_exp,
)
from src.dataset.landmark_rxr import episodes_from_config
from src.env.connectivity import load_connectivity
from src.process.visibility import VisibilityChecker
from src.viz import _semantic_to_rgb


# ── geometry helpers (mirror rescue_blacklist / inspection_viz) ──────────


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


def _partition_pos(
    part_sub:      Dict[str, Any],
    virtual_nodes: Dict[str, List[float]],
    scan_db:       Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    """Position at which the agent finishes the spatial segment of this
    sub-path — i.e. the last node of ``spatial_path``."""
    spatial = part_sub.get("spatial_path") or []
    if not spatial:
        return None
    return _resolve_node_pos(spatial[-1], virtual_nodes, scan_db)


# ── render ───────────────────────────────────────────────────────────────


_BG       = (30, 30, 50)
_FG       = (220, 220, 240)
_ACCENT   = (250, 225, 120)
_DIM      = (160, 160, 180)
_INFO_W   = 320
_FONT_PX  = 13


def _load_fonts() -> Tuple[Any, Any]:
    from PIL import ImageFont
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", _FONT_PX)
        font_bold = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", _FONT_PX)
        return font, font_bold
    except Exception:
        fallback = ImageFont.load_default()
        return fallback, fallback


def _draw_info_panel(
    info_w:  int,
    info_h:  int,
    lines:   List[Tuple[str, Tuple[int, int, int], bool]],
) -> Any:
    """Render a right-side info panel of (text, color, bold) lines."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (info_w, info_h), _BG)
    draw = ImageDraw.Draw(img)
    font, font_bold = _load_fonts()
    y = 12
    line_h = _FONT_PX + 7
    for text, color, bold in lines:
        if y + line_h > info_h:
            break
        draw.text((12, y), text, fill=color, font=(font_bold if bold else font))
        y += line_h
    return img


def _wrap_value(label: str, value: str, max_chars: int) -> List[str]:
    """Wrap a long value across continuation lines."""
    import textwrap
    prefix = f"{label:<12} : "
    body = textwrap.wrap(value, width=max(10, max_chars - len(prefix)))
    if not body:
        return [prefix.rstrip()]
    return [prefix + body[0]] + [" " * len(prefix) + ln for ln in body[1:]]


def _render_one(
    checker:  VisibilityChecker,
    pos:      np.ndarray,
    scan:     str,
    ep_id:    Any,
    sub_idx:  int,
    sub_total: int,
    landmark: str,
    out_path: Path,
) -> bool:
    from PIL import Image

    obs = checker.render_observation(pos, 0.0)
    rgb = obs.get("rgb")
    sem = obs.get("semantic")
    if rgb is None:
        return False

    h, w = rgb.shape[:2]
    if sem is not None:
        sem_rgb = _semantic_to_rgb(sem)
        # Defensive resize in case the semantic resolution differs.
        if sem_rgb.shape[:2] != rgb.shape[:2]:
            sem_rgb = np.asarray(
                Image.fromarray(sem_rgb).resize((w, h), Image.NEAREST)
            )
        stacked = np.zeros((h * 2, w, 3), dtype=np.uint8)
        stacked[:h] = rgb
        stacked[h:] = sem_rgb
    else:
        stacked = rgb

    left_img = Image.fromarray(stacked)

    # Right info panel — fields requested: scan, instruction_id, sub_idx,
    # landmark to find. Wider than the title bar used to be, so long
    # landmark phrases wrap cleanly.
    max_chars = max(20, (_INFO_W - 24) // 7)
    lines: List[Tuple[str, Tuple[int, int, int], bool]] = [
        (f"{'scan':<12} : {scan}", _DIM, False),
        (f"{'instr_id':<12} : {ep_id}", _ACCENT, True),
        (f"{'sub_idx':<12} : {sub_idx:03d} / {sub_total:03d}", _ACCENT, True),
        ("", _FG, False),
        ("LANDMARK:", _FG, True),
    ]
    if landmark:
        import textwrap
        for ln in textwrap.wrap(landmark, width=max_chars) or [landmark]:
            lines.append((ln, _ACCENT, False))
    else:
        lines.append(("(none)", _DIM, False))
    lines.append(("", _FG, False))
    lines.append(("RGB        (top)", _DIM, False))
    lines.append(("SEMANTIC (bottom)", _DIM, False))

    panel = _draw_info_panel(_INFO_W, left_img.height, lines)
    out_img = Image.new("RGB", (left_img.width + _INFO_W, left_img.height), _BG)
    out_img.paste(left_img, (0, 0))
    out_img.paste(panel, (left_img.width, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)
    return True


# ── driver ───────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render a partition-pose observation panorama for every "
                    "(ep, sub_idx) in survivor.yaml.",
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", default=None,
                    help="Experiment handle (selection YAML path or expname). "
                         "survivor.yaml is auto-merged.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    resolve_exp(cfg, args.exp, apply_current=True)

    run_dir  = get_run_dir(cfg)
    episodes = episodes_from_config(cfg)
    if not episodes:
        raise SystemExit("No episodes in survivor.yaml — run cross_floor first.")

    # sub_paths in survivor.yaml = every alive (ep, sub) past cross_floor
    # (active + labeled). Falls back to the episode's full range when the
    # earlier pipeline didn't emit a sub_paths key.
    sub_paths_filter = cfg.get("selection", {}).get("sub_paths") or {}

    needed_scans = sorted({ep.scan for ep in episodes})
    scan_db_all = load_connectivity(
        scenes_dir = cfg["scenes"]["scenes_dir"],
        scans      = needed_scans,
        json_dir   = cfg["dataset"].get("connectivity_json_dir"),
        pkl_path   = cfg["dataset"].get("connectivity_pkl"),
    )

    by_scan: Dict[str, List[Tuple[Any, List[int]]]] = {}
    for ep in episodes:
        subs_raw = sub_paths_filter.get(int(ep.instruction_id))
        if subs_raw is None:
            subs = list(range(len(ep.sub_paths)))
        else:
            subs = sorted({int(s) for s in subs_raw})
        if subs:
            by_scan.setdefault(ep.scan, []).append((ep, subs))

    out_root = run_dir / "partition_obs"
    out_root.mkdir(parents=True, exist_ok=True)

    checker = VisibilityChecker(cfg["env"], cfg["scenes"]["scenes_dir"])

    n_rendered          = 0
    n_skip_no_partition = 0
    n_skip_no_pos       = 0
    n_skip_render_err   = 0

    try:
        for scan, items in sorted(by_scan.items()):
            checker.load_scene(f"mp3d/{scan}/{scan}.glb")
            scan_db = scan_db_all.get(scan) or {}

            for ep, subs in items:
                part_path = (
                    run_dir / "partition" / scan / str(ep.instruction_id)
                    / "partition.json"
                )
                if not part_path.exists():
                    n_skip_no_partition += len(subs)
                    continue
                with open(part_path) as f:
                    part_json = json.load(f)
                part_subs_map = {
                    int(p["sub_idx"]): p
                    for p in (part_json.get("partitions") or [])
                    if "sub_idx" in p
                }
                virtual_nodes = part_json.get("virtual_nodes") or {}

                for sub_idx in subs:
                    part_sub = part_subs_map.get(int(sub_idx))
                    if part_sub is None:
                        n_skip_no_partition += 1
                        continue
                    pos = _partition_pos(part_sub, virtual_nodes, scan_db)
                    if pos is None:
                        n_skip_no_pos += 1
                        continue
                    out_path = (
                        out_root / scan / str(ep.instruction_id)
                        / f"sub_{int(sub_idx):03d}.png"
                    )
                    try:
                        ok = _render_one(
                            checker, pos,
                            scan      = scan,
                            ep_id     = ep.instruction_id,
                            sub_idx   = int(sub_idx),
                            sub_total = max(len(ep.sub_paths), 1),
                            landmark  = (part_sub.get("landmark") or ""),
                            out_path  = out_path,
                        )
                    except Exception as exc:
                        n_skip_render_err += 1
                        print(f"  [render-error] ep={ep.instruction_id} "
                              f"sub={sub_idx}: {exc}")
                        continue
                    if ok:
                        n_rendered += 1
                    else:
                        n_skip_render_err += 1
    finally:
        checker.close()

    print()
    print("=== partition_obs ===")
    print(f"  rendered            : {n_rendered}")
    if n_skip_no_partition:
        print(f"  skip (no partition) : {n_skip_no_partition}")
    if n_skip_no_pos:
        print(f"  skip (no pose)      : {n_skip_no_pos}")
    if n_skip_render_err:
        print(f"  skip (render error) : {n_skip_render_err}")
    print(f"  → {out_root}")


if __name__ == "__main__":
    main()
