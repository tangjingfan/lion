"""Instance-highlight rendering shared across pipeline stages.

Renders the rollout-style RGB + semantic panorama with a target-instance
mask strip appended — the visual used by step 07 (candidate viz), step 08
(selection viz), step 11 (the image the VLM names), and the
``query_scene_instance`` debug CLI.  Composes on top of :mod:`src.viz`
(the rollout frame composer).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from src.env.connectivity import load_connectivity
from src.env.mp3d_house import parse_house_categories
from src.viz import _compose, heading_toward


def nearest_viewpoint(
    cfg: Dict[str, Any],
    scenes_dir: str,
    scan: str,
    target_pos: Any,
) -> Tuple[str, Any]:
    db = load_connectivity(
        scenes_dir=scenes_dir,
        scans=[scan],
        json_dir=cfg["dataset"].get("connectivity_json_dir"),
        pkl_path=cfg["dataset"].get("connectivity_pkl"),
    )
    scan_db = db.get(scan, {})
    if not scan_db:
        raise RuntimeError(f"No connectivity nodes loaded for scan {scan}")
    target = target_pos
    best_id = min(
        scan_db,
        key=lambda node_id: float(
            (scan_db[node_id][0] - target[0]) ** 2
            + (scan_db[node_id][1] - target[1]) ** 2
            + (scan_db[node_id][2] - target[2]) ** 2
        ),
    )
    return best_id, scan_db[best_id]


def target_mask_rgb(mask) -> Any:
    import numpy as np

    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.where(mask, 255, 20).astype(np.uint8)
    rgb[..., 1] = np.where(mask, 60, 20).astype(np.uint8)
    rgb[..., 2] = np.where(mask, 60, 30).astype(np.uint8)
    return rgb


def render_rollout_style_instance_viz(
    checker,
    cfg: Dict[str, Any],
    scenes_dir: str,
    scan: str,
    instance_id: int,
    instance_center_habitat: Any,
    out_path: Path,
    info_width: int,
) -> Dict[str, Any]:
    from PIL import Image, ImageDraw
    import numpy as np

    target_pos = instance_center_habitat
    viewpoint_id, pos = nearest_viewpoint(cfg, scenes_dir, scan, target_pos)
    heading = heading_toward(pos, target_pos)

    checker.load_scene(f"mp3d/{scan}/{scan}.glb")
    obs = checker.render_observation(pos, heading)
    sem = obs.get("semantic")
    if sem is not None and checker._sem_id_map is not None:  # noqa: SLF001
        sem_clip = np.clip(sem, 0, len(checker._sem_id_map) - 1)  # noqa: SLF001
        obs["semantic_id"] = checker._sem_id_map[sem_clip]  # noqa: SLF001
        obs["semantic_name"] = checker._sem_name_map[sem_clip]  # noqa: SLF001

    canvas = _compose(
        obs=obs,
        episode=None,
        step=0,
        action=None,
        info_w=info_width,
        mark_semantic_numbers=False,
    )

    if sem is not None:
        target_mask = (sem == int(instance_id))
        mask_img = Image.fromarray(target_mask_rgb(target_mask))
        mask_img = mask_img.resize((obs["rgb"].shape[1], obs["rgb"].shape[0]), Image.NEAREST)
        draw = ImageDraw.Draw(mask_img)
        draw.rectangle([(0, 0), (mask_img.width, 20)], fill=(30, 30, 50))
        draw.text((5, 4), f"TARGET INSTANCE {instance_id}", fill=(100, 200, 255))

        base = Image.fromarray(canvas)
        out = Image.new("RGB", (base.width, base.height + mask_img.height), color=(25, 25, 35))
        out.paste(base, (0, 0))
        out.paste(mask_img, (0, base.height))
    else:
        out = Image.fromarray(canvas)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return {
        "path": str(out_path),
        "viewpoint_id": viewpoint_id,
        "viewpoint_pos": [float(x) for x in pos],
        "heading": float(heading),
        "target_visible_pixels": int((sem == int(instance_id)).sum()) if sem is not None else None,
    }


def render_mask_for_rollout_frame(
    checker,
    scan: str,
    instance_id: int,
    frame_record: Dict[str, Any],
    out_path: Path,
    info_width: int,
) -> Dict[str, Any]:
    from PIL import Image, ImageDraw
    import numpy as np

    pos = frame_record.get("position")
    heading = frame_record.get("heading")
    if pos is None or heading is None:
        raise ValueError("Frame record has no position/heading.")

    checker.load_scene(f"mp3d/{scan}/{scan}.glb")
    obs = checker.render_observation(np.asarray(pos, dtype=np.float32), float(heading))
    sem = obs.get("semantic")
    if sem is not None and checker._sem_id_map is not None:  # noqa: SLF001
        sem_clip = np.clip(sem, 0, len(checker._sem_id_map) - 1)  # noqa: SLF001
        obs["semantic_id"] = checker._sem_id_map[sem_clip]  # noqa: SLF001
        obs["semantic_name"] = checker._sem_name_map[sem_clip]  # noqa: SLF001

    episode = SimpleNamespace(
        scan=scan,
        instruction_id=frame_record.get("instruction_id", "unknown"),
        instruction=frame_record.get("instruction", ""),
        sub_paths=[None] * int(frame_record.get("sub_total") or 1),
        sub_instructions=[],
    )
    canvas = _compose(
        obs=obs,
        episode=episode,
        step=int(frame_record.get("step") or 0),
        action=frame_record.get("action"),
        info_w=info_width,
        sub_idx=int(frame_record.get("sub_idx") or 0),
        sub_total=int(frame_record.get("sub_total") or 1),
        mark_semantic_numbers=False,
    )

    if sem is not None:
        target_mask = (sem == int(instance_id))
        mask_img = Image.fromarray(target_mask_rgb(target_mask))
        mask_img = mask_img.resize((obs["rgb"].shape[1], obs["rgb"].shape[0]), Image.NEAREST)
        draw = ImageDraw.Draw(mask_img)
        draw.rectangle([(0, 0), (mask_img.width, 20)], fill=(30, 30, 50))
        landmark = str(frame_record.get("landmark") or frame_record.get("instruction") or "").strip()
        title = f"TARGET INSTANCE {instance_id}"
        if landmark:
            title = f"{title} | {landmark}"
        max_chars = max(20, (mask_img.width - 10) // 7)
        if len(title) > max_chars:
            title = title[: max_chars - 3] + "..."
        draw.text((5, 4), title, fill=(100, 200, 255))

        base = Image.fromarray(canvas)
        out = Image.new("RGB", (base.width, base.height + mask_img.height), color=(25, 25, 35))
        out.paste(base, (0, 0))
        out.paste(mask_img, (0, base.height))
    else:
        out = Image.fromarray(canvas)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return {
        "path": str(out_path),
        "source_frame": frame_record.get("_rollout_frame"),
        "frame_index": frame_record.get("_frame_index"),
        "position": [float(x) for x in pos],
        "heading": float(heading),
        "target_visible_pixels": int((sem == int(instance_id)).sum()) if sem is not None else None,
    }


def draw_house_instance_viz(
    scenes_dir: str,
    scan: str,
    instance_id: int,
    out_path: Path,
) -> Path:
    """Draw a simple top-down object-center map from the MP3D .house file."""
    import matplotlib.pyplot as plt

    house_path = Path(scenes_dir) / "mp3d" / scan / f"{scan}.house"
    if not house_path.exists():
        raise FileNotFoundError(f"House file not found: {house_path}")

    categories = parse_house_categories(house_path)
    objects: List[Dict[str, Any]] = []
    target = None

    with open(house_path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] != "O" or len(parts) < 7:
                continue
            try:
                obj_id = int(parts[1])
                region_id = int(parts[2])
                cat_index = int(parts[3])
                center = [float(x) for x in parts[4:7]]
            except ValueError:
                continue
            cat = categories.get(cat_index, {})
            entry = {
                "instance_id": obj_id,
                "region_id": region_id,
                "cat_index": cat_index,
                "category": cat.get("mpcat40_name") or cat.get("raw_name") or "unknown",
                "raw_category": cat.get("raw_name") or "unknown",
                "center": center,
            }
            objects.append(entry)
            if obj_id == instance_id:
                target = entry

    if target is None:
        raise ValueError(f"Instance {instance_id} not found in {house_path}")

    target_cat = target["category"]
    all_x = [o["center"][0] for o in objects]
    all_y = [-o["center"][2] for o in objects]
    same = [o for o in objects if o["category"] == target_cat]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.scatter(all_x, all_y, s=9, c="#c8c8c8", alpha=0.35,
               linewidths=0, label="all objects", zorder=1)
    if same:
        ax.scatter(
            [o["center"][0] for o in same],
            [-o["center"][2] for o in same],
            s=24, c="#2f80ed", alpha=0.75,
            edgecolors="white", linewidths=0.3,
            label=f"same category: {target_cat}", zorder=2,
        )

    tx, ty = target["center"][0], -target["center"][2]
    ax.scatter([tx], [ty], s=260, marker="*", c="#e03131",
               edgecolors="black", linewidths=0.8,
               label=f"target id={instance_id}", zorder=5)
    ax.text(
        tx, ty,
        f"  id={instance_id}\n  {target_cat}\n  raw={target['raw_category']}",
        fontsize=8, color="#111", va="center", ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#e03131", lw=0.8),
        zorder=6,
    )

    ax.set_aspect("equal")
    ax.set_title(f"{scan} instance {instance_id}", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("-z")
    ax.grid(True, lw=0.3, alpha=0.35)
    ax.legend(loc="best", fontsize=8)

    pad = max(1.0, 0.08 * max(max(all_x) - min(all_x), max(all_y) - min(all_y)))
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
