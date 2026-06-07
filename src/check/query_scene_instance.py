"""
Query one Habitat / MP3D semantic instance in a scene.

The semantic sensor emits raw instance ids.  In MP3D/Habitat annotations these
ids usually correspond to the integer suffix of ``obj.id`` (for example
``object_331`` -> semantic instance id ``331``), and MP3D's ``.house`` file also
stores object rows keyed by the same id.  This tool can query Habitat
annotations when available and falls back to parsing ``.house`` directly
(via :mod:`src.env.mp3d_house`); the renderings live in
:mod:`src.instance_viz`.

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
from typing import Any, Dict, List, Optional, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.env.connectivity import _mp3d_to_habitat
from src.env.mp3d_house import parse_house_object
from src.instance_viz import (
    draw_house_instance_viz,
    render_mask_for_rollout_frame,
    render_rollout_style_instance_viz,
)
from src.process.visibility import VisibilityChecker


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
            match, nearby = parse_house_object(
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
            viz_path = draw_house_instance_viz(
                scenes_dir, args.scan, args.instance_id, Path(args.viz_out)
            )
            payload["viz_path"] = str(viz_path)

        if args.rollout_viz_out:
            if checker is None:
                checker = VisibilityChecker(cfg["env"], scenes_dir)

            if frame_record is not None:
                try:
                    rollout_viz = render_mask_for_rollout_frame(
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
                    rollout_viz = render_rollout_style_instance_viz(
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
