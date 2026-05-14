"""
Query one Habitat / MP3D semantic instance in a scene.

The semantic sensor emits raw instance ids.  In MP3D/Habitat annotations these
ids usually correspond to the integer suffix of ``obj.id`` (for example
``object_331`` -> semantic instance id ``331``), and MP3D's ``.house`` file also
stores object rows keyed by the same id.  This tool can query Habitat
annotations when available and falls back to parsing ``.house`` directly.

Usage
-----
  python src/check/query_scene_instance.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --scan X7HyMhZNoso \\
      --instance_id 331 \\
      --viz_out results/instance_331.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.env.connectivity import _mp3d_to_habitat, load_connectivity
from src.process.visibility import VisibilityChecker
from src.viz import _compose, heading_toward


def _instance_id(obj) -> Optional[int]:
    try:
        return int(str(obj.id).split("_")[-1])
    except (AttributeError, TypeError, ValueError):
        return None


def _vec(value) -> Optional[List[float]]:
    if value is None:
        return None
    try:
        return [float(x) for x in value]
    except TypeError:
        return None


def _object_payload(obj, annotation_index: int) -> Dict[str, Any]:
    cat = getattr(obj, "category", None)
    aabb = getattr(obj, "aabb", None)
    return {
        "instance_id": _instance_id(obj),
        "annotation_index": annotation_index,
        "object_id": getattr(obj, "id", None),
        "category": {
            "name": cat.name() if cat is not None else None,
            "index": int(cat.index()) if cat is not None else None,
        },
        "aabb": {
            "center": _vec(getattr(aabb, "center", None)),
            "sizes": _vec(getattr(aabb, "sizes", None)),
        },
    }


def _parse_house_categories(house_path: Path) -> Dict[int, Dict[str, Any]]:
    categories: Dict[int, Dict[str, Any]] = {}
    with open(house_path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] != "C" or len(parts) < 6:
                continue
            try:
                cat_index = int(parts[1])
                mpcat40_index = int(parts[4])
            except ValueError:
                continue
            categories[cat_index] = {
                "category_index": cat_index,
                "raw_category_id": int(parts[2]) if parts[2].isdigit() else parts[2],
                "raw_name": parts[3].replace("#", " "),
                "mpcat40_index": mpcat40_index,
                "mpcat40_name": parts[5].replace("#", " "),
            }
    return categories


def _parse_house_object(
    scenes_dir: str,
    scan: str,
    instance_id: int,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    house_path = Path(scenes_dir) / "mp3d" / scan / f"{scan}.house"
    if not house_path.exists():
        return None, []

    categories = _parse_house_categories(house_path)
    nearby: List[Dict[str, Any]] = []
    match: Optional[Dict[str, Any]] = None

    with open(house_path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] != "O" or len(parts) < 5:
                continue
            try:
                obj_id = int(parts[1])
                region_id = int(parts[2])
                cat_index = int(parts[3])
            except ValueError:
                continue

            cat = categories.get(cat_index, {})
            entry = {
                "instance_id": obj_id,
                "region_id": region_id,
                "category": {
                    "index": cat_index,
                    "raw_name": cat.get("raw_name"),
                    "mpcat40_name": cat.get("mpcat40_name"),
                    "mpcat40_index": cat.get("mpcat40_index"),
                },
                # In MP3D .house object rows, fields 4:7 are the object center.
                "center": [float(x) for x in parts[4:7]],
                "raw_fields": parts,
            }
            nearby.append({
                "instance_id": obj_id,
                "category": cat.get("mpcat40_name") or cat.get("raw_name"),
                "distance": abs(obj_id - instance_id),
            })
            if obj_id == instance_id:
                match = entry

    nearby.sort(key=lambda e: (e["distance"], e["instance_id"]))
    return match, nearby[:10]


def _draw_house_instance_viz(
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

    categories = _parse_house_categories(house_path)
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


def _nearest_viewpoint(
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


def _target_mask_rgb(mask) -> Any:
    import numpy as np

    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.where(mask, 255, 20).astype(np.uint8)
    rgb[..., 1] = np.where(mask, 60, 20).astype(np.uint8)
    rgb[..., 2] = np.where(mask, 60, 30).astype(np.uint8)
    return rgb


def _render_rollout_style_instance_viz(
    checker: VisibilityChecker,
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
    viewpoint_id, pos = _nearest_viewpoint(cfg, scenes_dir, scan, target_pos)
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
        mask_img = Image.fromarray(_target_mask_rgb(target_mask))
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


def _find_frame_index(rollout_frame: Path, frame_index: Optional[str]) -> Tuple[Path, Path]:
    frame_path = rollout_frame.expanduser().resolve()
    if frame_index:
        index_path = Path(frame_index).expanduser().resolve()
        try:
            viz_dir = index_path.parent
            rel_path = frame_path.relative_to(viz_dir)
        except ValueError:
            rel_path = Path(frame_path.name)
        return index_path, rel_path

    for parent in frame_path.parents:
        candidate = parent / "frames.jsonl"
        if candidate.exists():
            return candidate, frame_path.relative_to(parent)
    raise FileNotFoundError(
        f"Could not find frames.jsonl above rollout frame: {rollout_frame}. "
        "Re-run rollout after this update, or pass --frame_index explicitly."
    )


def _load_rollout_frame_record(
    rollout_frame: str,
    frame_index: Optional[str],
) -> Dict[str, Any]:
    index_path, rel_path = _find_frame_index(Path(rollout_frame), frame_index)
    rel_str = rel_path.as_posix()
    with open(index_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("rel_path") == rel_str:
                record["_frame_index"] = str(index_path)
                record["_rollout_frame"] = str(Path(rollout_frame).expanduser().resolve())
                return record
    raise KeyError(f"No frame record for {rel_str} in {index_path}")


def _render_mask_for_rollout_frame(
    checker: VisibilityChecker,
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
        mask_img = Image.fromarray(_target_mask_rgb(target_mask))
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


def _nearby_ids(objects, target_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    entries = []
    for idx, obj in enumerate(objects):
        iid = _instance_id(obj)
        if iid is None:
            continue
        cat = getattr(obj, "category", None)
        entries.append({
            "instance_id": iid,
            "annotation_index": idx,
            "category": cat.name() if cat is not None else None,
            "distance": abs(iid - target_id),
        })
    entries.sort(key=lambda e: (e["distance"], e["instance_id"]))
    return entries[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Query a semantic instance id in one MP3D/Habitat scene.",
    )
    ap.add_argument("--config", required=True,
                    help="Rollout YAML (provides env / scenes_dir).")
    ap.add_argument("--scan", default=None,
                    help="Scan id, e.g. X7HyMhZNoso.")
    ap.add_argument("--instance_id", type=int, required=True,
                    help="Raw semantic instance id from the semantic sensor.")
    ap.add_argument("--json", action="store_true",
                    help="Print machine-readable JSON only.")
    ap.add_argument("--source", choices=("auto", "habitat", "house"),
                    default="auto",
                    help="Where to query. auto tries Habitat first, then .house.")
    ap.add_argument("--viz_out", default=None,
                    help="Optional PNG path for a top-down visualization of "
                         "the queried instance in the scene.")
    ap.add_argument("--rollout_viz_out", default=None,
                    help="Optional rollout-style RGB + semantic PNG rendered "
                         "from the nearest connectivity viewpoint.")
    ap.add_argument("--rollout_frame", default=None,
                    help="Existing rollout_viz PNG to re-render from the same "
                         "position/heading and append a target-instance mask. "
                         "Requires rollout_viz/frames.jsonl.")
    ap.add_argument("--frame_index", default=None,
                    help="Optional frames.jsonl path for --rollout_frame.")
    ap.add_argument("--info_width", type=int, default=300,
                    help="Info panel width for --rollout_viz_out.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    frame_record = None
    if args.rollout_frame:
        try:
            frame_record = _load_rollout_frame_record(args.rollout_frame, args.frame_index)
        except Exception as exc:
            raise SystemExit(f"Cannot load rollout frame metadata: {exc}") from exc
        if args.scan is None:
            args.scan = frame_record.get("scan")
    if args.scan is None:
        raise SystemExit("Please pass --scan, or pass --rollout_frame with a frame index that contains scan.")

    scenes_dir = cfg["scenes"]["scenes_dir"]
    scene_file = f"mp3d/{args.scan}/{args.scan}.glb"
    payload: Dict[str, Any] = {
        "scan": args.scan,
        "scene_file": scene_file,
        "instance_id": args.instance_id,
        "found": False,
        "object": None,
        "source": None,
    }

    habitat_error = None
    checker = None
    if args.source in ("auto", "habitat"):
        checker = VisibilityChecker(cfg["env"], scenes_dir)
    try:
        if checker is not None:
            try:
                checker.load_scene(scene_file)
            except Exception as exc:
                habitat_error = str(exc)
                if args.source == "habitat":
                    raise
            else:
                sim = checker._sim  # noqa: SLF001 - debug/query tool
                scene = sim.semantic_annotations()
                objects = list(getattr(scene, "objects", []) or [])

                match = None
                for idx, obj in enumerate(objects):
                    if obj is not None and _instance_id(obj) == args.instance_id:
                        match = _object_payload(obj, idx)
                        break

                payload.update({
                    "found": match is not None,
                    "object": match,
                    "source": "habitat",
                })
                if match is None:
                    payload["nearby_instances"] = _nearby_ids(objects, args.instance_id)

        if args.source == "house" or (args.source == "auto" and not payload["found"]):
            match, nearby = _parse_house_object(
                scenes_dir, args.scan, args.instance_id
            )
            payload.update({
                "found": match is not None,
                "object": match,
                "source": "house",
            })
            if match is None:
                payload["nearby_instances"] = nearby
            if habitat_error:
                payload["habitat_error"] = habitat_error

        if args.viz_out:
            viz_path = _draw_house_instance_viz(
                scenes_dir, args.scan, args.instance_id, Path(args.viz_out)
            )
            payload["viz_path"] = str(viz_path)

        if args.rollout_viz_out:
            if checker is None:
                checker = VisibilityChecker(cfg["env"], scenes_dir)

            if frame_record is not None:
                try:
                    rollout_viz = _render_mask_for_rollout_frame(
                        checker=checker,
                        scan=args.scan,
                        instance_id=args.instance_id,
                        frame_record=frame_record,
                        out_path=Path(args.rollout_viz_out),
                        info_width=args.info_width,
                    )
                except Exception as exc:
                    raise SystemExit(
                        "Cannot render rollout-frame mask because Habitat "
                        f"could not be loaded/rendered: {exc}"
                    ) from exc
                payload["rollout_viz"] = rollout_viz
            elif not payload["found"] or not payload["object"]:
                raise SystemExit("Cannot render rollout viz: instance was not found.")
            else:
                center = payload["object"].get("center")
                center_frame = "mp3d"
                if center is None and payload["source"] == "habitat":
                    center = payload["object"].get("aabb", {}).get("center")
                    center_frame = "habitat"
                if center is None:
                    raise SystemExit("Cannot render rollout viz: no object center available.")
                if center_frame == "mp3d":
                    center_habitat = _mp3d_to_habitat(
                        float(center[0]), float(center[1]), float(center[2])
                    )
                else:
                    center_habitat = center
                try:
                    rollout_viz = _render_rollout_style_instance_viz(
                        checker=checker,
                        cfg=cfg,
                        scenes_dir=scenes_dir,
                        scan=args.scan,
                        instance_id=args.instance_id,
                        instance_center_habitat=center_habitat,
                        out_path=Path(args.rollout_viz_out),
                        info_width=args.info_width,
                    )
                except Exception as exc:
                    raise SystemExit(
                        "Cannot render rollout-style viz because Habitat could "
                        f"not be loaded/rendered: {exc}"
                    ) from exc
                payload["rollout_viz"] = rollout_viz

        if args.json:
            print(json.dumps(payload, indent=2))
            return

        print(f"scan        : {args.scan}")
        print(f"scene_file  : {scene_file}")
        print(f"instance_id : {args.instance_id}")
        print(f"source      : {payload['source']}")
        if payload.get("viz_path"):
            print(f"viz         : {payload['viz_path']}")
        if payload.get("rollout_viz"):
            rv = payload["rollout_viz"]
            print(f"rollout_viz : {rv['path']}")
            if "viewpoint_id" in rv:
                print(f"viewpoint   : {rv['viewpoint_id']}")
            if "source_frame" in rv:
                print(f"source_frame: {rv['source_frame']}")
            print(f"target_px   : {rv['target_visible_pixels']}")
        if match is None:
            print("found       : no")
            if habitat_error:
                print(f"habitat_err : {habitat_error}")
            nearby = payload.get("nearby_instances", [])
            if nearby:
                print("\nNearby annotated instance ids:")
                for item in nearby:
                    print(
                        f"  {item['instance_id']:>5}  "
                        f"category={item.get('category')}"
                    )
            return

        obj_payload = payload["object"]
        cat = obj_payload["category"]
        print("found       : yes")
        if payload["source"] == "habitat":
            aabb = obj_payload["aabb"]
            print(f"object_id   : {obj_payload['object_id']}")
            print(f"anno_index  : {obj_payload['annotation_index']}")
            print(f"category    : {cat['name']}")
            print(f"cat_id      : {cat['index']}")
            print(f"aabb_center : {aabb['center']}")
            print(f"aabb_sizes  : {aabb['sizes']}")
        else:
            print(f"region_id   : {obj_payload['region_id']}")
            print(f"category    : {cat['mpcat40_name']}")
            print(f"raw_category: {cat['raw_name']}")
            print(f"cat_id      : {cat['mpcat40_index']}")
            print(f"house_cat_id: {cat['index']}")
            print(f"center      : {obj_payload['center']}")
    finally:
        if checker is not None:
            checker.close()


if __name__ == "__main__":
    main()
