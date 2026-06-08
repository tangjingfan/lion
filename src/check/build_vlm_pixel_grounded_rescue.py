"""Pixel-grounded semantic rescue for coarse-target failures.

CLI + per-record orchestration + debug viz for pipeline step 09
(``scripts/09_detection.sh``).  The detector wrapper, VLM grounding
fallback, source collection, and instance recovery live in
:mod:`src.process.detection`.

Primary path: run YOLO-World (open-vocabulary detector) on the re-rendered
rollout RGB panorama, prompted with the landmark phrase. The top-scored
detection's bbox is intersected with the Habitat semantic panorama, and
the bbox-majority MP3D instance is taken as the recovered target.

Fallback path (opt-in via ``--enable_vlm_fallback``): when YOLO-World
returns nothing above threshold, ask a VLM for the landmark's pixel and
bounding box and run the same instance-recovery step.

Steps:

  1. Read the survivor YAML and ``target_instances/{scan}/target_instances.json``.
  2. For each survivor whose selected target is grounded only to a coarse
     semantic bucket (or has no match at all), find the rollout frame pose
     for that (scan, episode, sub_idx).
  3. Re-render a clean RGB panorama and raw semantic panorama at the same pose.
  4. Run YOLO-World on the panorama to localise the landmark.
  5. Take the bbox-majority instance id from the semantic panorama.
  6. Write ``target_instances/{scan}/semantic_rescue_categories.json``.

This can rescue examples with ``target_instance_ids: []`` because the
instance is recovered from a detected box rather than from the previous
selector.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from PIL import Image, ImageDraw

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
    get_survivor_path,
    resolve_exp,
)

STAGE_NAME = "detection"

from src.process.coarse_labels import DEFAULT_COARSE_LABELS, _norm
from src.process.detection import (
    YoloWorldDetector,
    build_user_text,
    build_yolo_prompts,
    call_grounding_vlm_json,
    category_match_score,
    collect_from_dropped,
    collect_from_survivors,
    instance_at_pixel,
    load_frames,
    normalize_grounding,
)
from src.process.rewriter import make_client
from src.process.visibility import VisibilityChecker


def _load_yaml(path: Path) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def _dump_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)


# ── Debug viz ────────────────────────────────────────────────────────────
def _draw_point(rgb_path: Path, x: int, y: int, out_path: Path) -> Path:
    img = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    r = 9
    draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0), width=3)
    draw.line((x - 14, y, x + 14, y), fill=(255, 0, 0), width=2)
    draw.line((x, y - 14, x, y + 14), fill=(255, 0, 0), width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def _draw_bbox(rgb_path: Path, bbox: List[int], out_path: Path) -> Path:
    img = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    if bbox and len(bbox) == 4 and bbox[0] >= 0:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 0), width=4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def _draw_instance_mask(
    rgb_path: Path,
    sem: np.ndarray,
    instance_id: int,
    vlm_x: int,
    vlm_y: int,
    out_path: Path,
) -> Path:
    """Overlay the recovered instance's pixels in green for visual review."""
    img = Image.open(rgb_path).convert("RGBA")
    mask = (sem == int(instance_id))
    if mask.any():
        overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        overlay[mask] = (0, 220, 0, 110)
        overlay_img = Image.fromarray(overlay, mode="RGBA")
        img = Image.alpha_composite(img, overlay_img)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    r = 9
    draw.ellipse((vlm_x - r, vlm_y - r, vlm_x + r, vlm_y + r), outline=(255, 0, 0), width=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def _load_existing_payload(path: Path, scan: str, model: str, source: Path, append: bool = False) -> Dict:
    payload = {
        "scan": scan,
        "source_dropped_yaml": str(source),
        "model": model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "instances": {},
        "skipped": [],
    }
    if append and path.exists():
        existing = _load_json(path)
        if isinstance(existing.get("instances"), dict):
            payload["instances"].update(existing["instances"])
        if isinstance(existing.get("skipped"), list):
            payload["skipped"] = existing["skipped"]
    return payload


def _make_contact_sheet(summary_path: Path, out_path: Path) -> Optional[Path]:
    if not summary_path.exists():
        return None
    data = _load_json(summary_path)
    records = data.get("records") or []
    if not records:
        return None

    thumb_w = 320
    label_w = 460
    row_h = 150
    pad = 10
    cols = 4
    canvas_w = label_w + cols * thumb_w + (cols + 2) * pad
    canvas_h = max(1, len(records)) * (row_h + pad) + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    def paste_thumb(path_str: Optional[str], x: int, y: int) -> None:
        if not path_str:
            draw.rectangle((x, y, x + thumb_w, y + row_h), outline=(180, 180, 180))
            return
        path = Path(path_str)
        if not path.exists():
            draw.rectangle((x, y, x + thumb_w, y + row_h), outline=(180, 180, 180))
            draw.text((x + 8, y + 8), "missing", fill=(120, 0, 0))
            return
        img = Image.open(path).convert("RGB")
        img.thumbnail((thumb_w, row_h), Image.LANCZOS)
        bg = Image.new("RGB", (thumb_w, row_h), (20, 20, 25))
        bg.paste(img, ((thumb_w - img.width) // 2, (row_h - img.height) // 2))
        canvas.paste(bg, (x, y))

    for i, rec in enumerate(records):
        y = pad + i * (row_h + pad)
        status = rec.get("status")
        g = rec.get("vlm") or {}
        pix = rec.get("pixel_instance") or {}
        source = rec.get("source") or "yolo_world"
        sem_cat = rec.get("semantic_category")
        det_cat = g.get("category")
        score = g.get("score")
        cat_mismatch = (
            status == "rescued"
            and det_cat
            and sem_cat
            and category_match_score(str(det_cat), str(sem_cat)) < 0.5
        )
        score_str = f"{score:.2f}" if isinstance(score, (int, float)) else g.get("confidence")
        lines = [
            f"ep={rec.get('episode_id')} sub={rec.get('sub_idx')}  {status}  [{source}]",
            f"landmark: {rec.get('landmark')}",
            f"det: {det_cat} score={score_str}",
            f"xy=({g.get('x')},{g.get('y')}) inst={rec.get('instance_id')}",
            f"sem: {sem_cat}{'  [MISMATCH]' if cat_mismatch else ''}",
            f"recover: {pix.get('correction')}",
        ]
        x0 = pad
        draw.rectangle((x0, y, x0 + label_w, y + row_h), fill=(255, 255, 255))
        for j, line in enumerate(lines):
            color = (180, 0, 0) if (cat_mismatch and j == 4) else (0, 0, 0)
            draw.text((x0 + 8, y + 8 + j * 22), str(line), fill=color)

        x = label_w + 2 * pad
        paste_thumb(rec.get("rgb_path"), x, y)
        x += thumb_w + pad
        paste_thumb(rec.get("bbox_path"), x, y)
        x += thumb_w + pad
        paste_thumb(rec.get("debug_point_path"), x, y)
        x += thumb_w + pad
        paste_thumb(rec.get("mask_path"), x, y)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run YOLO-World on rollout panoramas and recover MP3D instance ids",
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--rewrite_config", default="configs/rewrite/rewrite_subinstructions.yaml")
    ap.add_argument("--exp", default=None,
                    help="Experiment handle (selection YAML path or expname). "
                         "Auto-merges survivor.yaml; the rescue reads "
                         "from current.yaml unless --dropped_yaml is given.")
    ap.add_argument("--dropped_yaml", default=None,
                    help="Legacy mode: collect examples from an existing stage-4 dropped YAML.")
    ap.add_argument("--coarse_label", action="append", default=None,
                    help="Coarse labels to rescue. Defaults to semantic_granularity defaults.")
    # YOLO-World options.
    ap.add_argument("--yolo_model", default="yolov8l-worldv2.pt",
                    help="YOLO-World checkpoint. Ultralytics auto-downloads on first use.")
    ap.add_argument("--yolo_conf", type=float, default=0.10,
                    help="Min confidence kept from YOLO-World predictions.")
    ap.add_argument("--yolo_imgsz", type=int, default=1024,
                    help="Detector input size (longest side).")
    ap.add_argument("--yolo_max_det", type=int, default=15)
    ap.add_argument("--yolo_device", default=None,
                    help="Override torch device (e.g. cuda:0, cpu). Defaults to ultralytics auto.")
    # VLM fallback (off by default).
    ap.add_argument("--enable_vlm_fallback", action="store_true",
                    help="If YOLO-World finds nothing above --yolo_conf, fall back to a VLM call.")
    ap.add_argument("--api_key", default=None,
                    help="Only used when --enable_vlm_fallback is set.")
    ap.add_argument("--model", default=None,
                    help="VLM model name for the fallback path.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--save_viz", action="store_true", default=False,
                    help="Save debug PNGs (input RGB, point, bbox, mask, "
                         "contact sheet) for each rescue attempt. Default "
                         "off — only semantic_rescue_categories.json is "
                         "produced.")
    ap.add_argument("--append", action="store_true",
                    help="Append to an existing semantic_rescue_categories.json instead of replacing this run's entries.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    resolve_exp(cfg, args.exp, apply_current=True)

    rw_cfg = _load_yaml(Path(args.rewrite_config)) if Path(args.rewrite_config).exists() else {}
    vlm_model = args.model or rw_cfg.get("model", "gemini-3-flash-preview")
    temperature = float(rw_cfg.get("temperature", 0.1))
    max_tokens = min(int(rw_cfg.get("max_tokens", 2048)), 4096)
    max_retries = int(rw_cfg.get("max_retries", 3))
    retry_delay = float(rw_cfg.get("retry_delay", 2.0))

    run_dir = get_run_dir(cfg)
    filt_dir = get_filter_dir(cfg)
    filt_dir.mkdir(parents=True, exist_ok=True)
    split = get_split(cfg)
    audit = load_audit(filt_dir, split)
    register_stage(audit, STAGE_NAME, yolo_model=args.yolo_model, yolo_conf=args.yolo_conf)
    strip_stage_events(audit, STAGE_NAME)

    source_yaml = Path(args.dropped_yaml).expanduser() if args.dropped_yaml else (
        get_survivor_path(cfg)
    )
    if not source_yaml.exists():
        raise SystemExit(f"No source YAML at {source_yaml}")

    frames = load_frames(run_dir)
    if args.dropped_yaml:
        records = collect_from_dropped(source_yaml)
        source_kind = "stage4_dropped"
    else:
        coarse = {_norm(x) for x in (args.coarse_label or sorted(DEFAULT_COARSE_LABELS))}
        records = collect_from_survivors(source_yaml.resolve(), run_dir, coarse)
        source_kind = "stage3_survivors"
    records_with_pose = [
        rec for rec in records
        if (rec["scan"], rec["episode_id"], rec["sub_idx"]) in frames
    ]
    print("=== YOLO-World pixel-grounded semantic rescue ===")
    print(f"  source yaml      : {source_yaml}")
    print(f"  source kind      : {source_kind}")
    print(f"  records          : {len(records)}")
    print(f"  with pose        : {len(records_with_pose)}")
    print(f"  yolo model       : {args.yolo_model}")
    print(f"  yolo conf / imgsz: {args.yolo_conf} / {args.yolo_imgsz}")
    print(f"  vlm fallback     : {'on' if args.enable_vlm_fallback else 'off'}")
    if args.dry_run:
        by_scan = defaultdict(int)
        for rec in records_with_pose:
            by_scan[rec["scan"]] += 1
        for scan, n in sorted(by_scan.items()):
            print(f"  [{scan}] records={n}")
        return

    detector = YoloWorldDetector(args.yolo_model, device=args.yolo_device)

    client = None
    if args.enable_vlm_fallback:
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit(
                "--enable_vlm_fallback set but no API key. "
                "Provide --api_key or set GEMINI_API_KEY."
            )
        client = make_client(api_key)

    scenes_dir = cfg.get("scenes", {}).get("scenes_dir", "")
    checker = VisibilityChecker(cfg.get("env", {}), scenes_dir)
    current_scan: Optional[str] = None
    payloads: Dict[str, Dict] = {}
    summaries: Dict[str, List[Dict]] = defaultdict(list)
    processed = 0
    try:
        for rec in records_with_pose:
            if args.limit is not None and processed >= args.limit:
                break
            scan = rec["scan"]
            frame = frames[(scan, rec["episode_id"], rec["sub_idx"])]
            if current_scan != scan:
                checker.load_scene(f"mp3d/{scan}/{scan}.glb")
                current_scan = scan

            obs = checker.render_observation(
                np.asarray(frame["position"], dtype=np.float32),
                float(frame["heading"]),
            )
            rgb = obs.get("rgb")
            sem = obs.get("semantic")
            if rgb is None or sem is None:
                continue
            h, w = rgb.shape[:2]

            out_dir = run_dir / "detection" / scan / rec["episode_id"]
            rgb_path = out_dir / f"sub_{int(rec['sub_idx']):03d}_rgb.png"
            if args.save_viz:
                out_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(rgb_path)

            print(f"[{scan} ep={rec['episode_id']} sub={rec['sub_idx']}] {rec['landmark']!r}")

            prompts = build_yolo_prompts(rec["landmark"])
            detections: List[Dict] = []
            if prompts:
                detections = detector.detect(
                    rgb,
                    prompts,
                    conf=args.yolo_conf,
                    imgsz=args.yolo_imgsz,
                    max_det=args.yolo_max_det,
                )

            grounding: Optional[Dict] = None
            chosen_det: Optional[Dict] = None
            pixel_info: Dict = {"instance_id": None}
            raw_payload: Any = {"detections": detections, "prompts": prompts}
            source = "yolo_world"

            # Take the top-scored detection — instance_at_pixel now
            # always returns the bbox-majority instance regardless of
            # MP3D category, so there's nothing to iterate looking for.
            # The category override (instance.category := landmark)
            # happens downstream in the summary record.
            if detections:
                chosen_det = detections[0]
                bbox_i = [int(round(v)) for v in chosen_det["bbox"]]
                cx = (bbox_i[0] + bbox_i[2]) // 2
                cy = (bbox_i[1] + bbox_i[3]) // 2
                pixel_info = instance_at_pixel(
                    sem, cx, cy, checker, bbox=bbox_i,
                )

            if chosen_det is not None:
                bbox_i = [int(round(v)) for v in chosen_det["bbox"]]
                cx = (bbox_i[0] + bbox_i[2]) // 2
                cy = (bbox_i[1] + bbox_i[3]) // 2
                score = float(chosen_det["score"])
                grounding = {
                    "is_visible": True,
                    "x": cx,
                    "y": cy,
                    "bbox": bbox_i,
                    "category": chosen_det["class_name"] or rec["landmark"],
                    "confidence": "high" if score >= 0.5 else ("medium" if score >= 0.25 else "low"),
                    "rationale": f"YOLO-World detected '{chosen_det['class_name']}' at conf {score:.2f}",
                    "score": score,
                }

            # Optional VLM fallback when YOLO finds nothing usable.
            if grounding is None and args.enable_vlm_fallback and client is not None:
                image_paths = [rgb_path]
                rel_path = frame.get("rel_path")
                if rel_path:
                    rollout_frame = run_dir / "rollout_viz" / scan / rel_path
                    if rollout_frame.exists():
                        image_paths.append(rollout_frame)
                try:
                    raw = call_grounding_vlm_json(
                        client,
                        model=vlm_model,
                        text=build_user_text(rec, w, h),
                        image_paths=image_paths,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                        label=f"{scan}:{rec['episode_id']}:{rec['sub_idx']}",
                    )
                    g = normalize_grounding(raw, w, h)
                except Exception as exc:
                    print(f"  vlm fallback failed: {exc}")
                    g = None
                if g is not None and g.get("is_visible"):
                    grounding = g
                    raw_payload = {"detections": detections, "prompts": prompts, "vlm_raw": raw}
                    source = "vlm_fallback"
                    pixel_info = instance_at_pixel(
                        sem,
                        g["x"],
                        g["y"],
                        checker,
                        bbox=g.get("bbox"),
                    )

            summary_base = {
                **rec,
                "frame": {
                    "rel_path": frame.get("rel_path"),
                    "step": frame.get("step"),
                    "position": frame.get("position"),
                    "heading": frame.get("heading"),
                },
                "vlm": grounding,           # kept name for backwards-compat with contact sheet
                "raw_vlm": raw_payload,
                "source": source,
                "rgb_path": str(rgb_path),
            }
            ep_audit = audit["episodes"].setdefault(rec["episode_id"], {
                "scan": scan, "events": [], "sub_paths": {},
            })
            if grounding is None:
                print(f"  -> not visible (no detection above {args.yolo_conf})")
                summaries[scan].append({**summary_base, "status": "not_visible"})
                append_sub_event(
                    ep_audit, rec["sub_idx"], stage=STAGE_NAME,
                    action="rescue_failed", reason="no_detection",
                    landmark=rec.get("landmark"),
                )
                processed += 1
                continue

            instance_id = pixel_info.get("instance_id")
            if args.save_viz:
                debug_path = _draw_point(
                    rgb_path,
                    grounding["x"],
                    grounding["y"],
                    out_dir / f"sub_{int(rec['sub_idx']):03d}_point.png",
                )
                bbox_path = _draw_bbox(
                    rgb_path,
                    grounding.get("bbox") or [-1, -1, -1, -1],
                    out_dir / f"sub_{int(rec['sub_idx']):03d}_bbox.png",
                )
                mask_path = _draw_instance_mask(
                    rgb_path,
                    sem,
                    instance_id,
                    grounding["x"],
                    grounding["y"],
                    out_dir / f"sub_{int(rec['sub_idx']):03d}_mask.png",
                ) if instance_id is not None else None
            else:
                debug_path = None
                bbox_path  = None
                mask_path  = None
            if instance_id is None:
                print(f"  -> bbox {grounding.get('bbox')} no MP3D instance recovered")
                summaries[scan].append({
                    **summary_base,
                    "status": "no_instance_at_pixel",
                    "debug_point_path": str(debug_path),
                    "bbox_path": str(bbox_path),
                    "pixel_instance": pixel_info,
                })
                append_sub_event(
                    ep_audit, rec["sub_idx"], stage=STAGE_NAME,
                    action="rescue_failed", reason="no_instance_in_bbox",
                    landmark=rec.get("landmark"),
                    bbox=grounding.get("bbox"),
                )
                processed += 1
                continue

            out_path = run_dir / "target_instances" / scan / "semantic_rescue_categories.json"
            payload = payloads.get(scan)
            if payload is None:
                payload = _load_existing_payload(out_path, scan, args.yolo_model, source_yaml, append=args.append)
                payload["source_kind"] = source_kind
                payload["yolo_model"] = args.yolo_model
                if args.enable_vlm_fallback:
                    payload["vlm_fallback_model"] = vlm_model
                payloads[scan] = payload

            # Record the LANDMARK as the rescued category so the output uses
            # the same word that was passed to YOLO as the prompt — downstream
            # consumers don't have to map the detector's class names (e.g.
            # "cooktop") back to the instruction phrase (e.g. "stove").
            recorded_category = (rec.get("landmark") or grounding["category"] or "unknown").strip().lower()
            key = str(instance_id)
            entry = payload["instances"].setdefault(key, {
                "instance_id": int(instance_id),
                "category": recorded_category,
                "confidence": grounding["confidence"],
                "is_rescuable": recorded_category != "unknown",
                "semantic_category": pixel_info.get("semantic_category"),
                "grounding_method": source,
                "landmarks": [],
                "examples": [],
                "image_paths": [],
            })
            if grounding["confidence"] == "high" or entry.get("category") in ("", "unknown"):
                entry["category"] = recorded_category
                entry["confidence"] = grounding["confidence"]
                entry["is_rescuable"] = recorded_category != "unknown"
                entry["grounding_method"] = source
            if rec["landmark"] and rec["landmark"] not in entry["landmarks"]:
                entry["landmarks"].append(rec["landmark"])
            entry["examples"].append({
                **rec,
                "frame": summary_base["frame"],
                "grounding": grounding,
                "source": source,
                "raw": raw_payload,
                "pixel_instance": pixel_info,
                "debug_point_path": str(debug_path),
                "bbox_path": str(bbox_path),
                "mask_path": str(mask_path) if mask_path else None,
            })
            summaries[scan].append({
                **summary_base,
                "status": "rescued",
                "instance_id": instance_id,
                "semantic_category": pixel_info.get("semantic_category"),
                "debug_point_path": str(debug_path),
                "bbox_path": str(bbox_path),
                "mask_path": str(mask_path) if mask_path else None,
                "pixel_instance": pixel_info,
            })
            append_sub_event(
                ep_audit, rec["sub_idx"], stage=STAGE_NAME,
                action="rescued", method=source,
                instance_id=int(instance_id),
                category=recorded_category,
                landmark=rec.get("landmark"),
            )
            extra_paths = [str(rgb_path), str(debug_path), str(bbox_path)]
            if mask_path:
                extra_paths.append(str(mask_path))
            for path in extra_paths:
                if path not in entry["image_paths"]:
                    entry["image_paths"].append(path)
            print(
                f"  -> instance={instance_id} semantic={pixel_info.get('semantic_category')} "
                f"category={entry['category']} conf={entry['confidence']}"
            )
            processed += 1
    finally:
        checker.close()

    finalize_audit(audit)
    save_audit(audit, filt_dir)

    for scan, payload in sorted(payloads.items()):
        out_path = run_dir / "target_instances" / scan / "semantic_rescue_categories.json"
        _dump_json(out_path, payload)
        print(f"wrote {out_path}")
    for scan, items in sorted(summaries.items()):
        summary_path = (
            run_dir / "detection" / scan / "summary.json"
        )
        _dump_json(summary_path, {
            "scan": scan,
            "source_yaml": str(source_yaml),
            "yolo_model": args.yolo_model,
            "yolo_conf": args.yolo_conf,
            "vlm_fallback_model": vlm_model if args.enable_vlm_fallback else None,
            "records": items,
        })
        print(f"wrote {summary_path}")
        if args.save_viz:
            sheet_path = summary_path.with_suffix(".png")
            made = _make_contact_sheet(summary_path, sheet_path)
            if made:
                print(f"wrote {made}")
    print(f"Processed VLM records: {processed}")


if __name__ == "__main__":
    main()
