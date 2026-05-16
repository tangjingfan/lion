"""Pixel-grounded semantic rescue for coarse-target failures.

Primary path: run YOLO-World (open-vocabulary detector) on the re-rendered
rollout RGB panorama, prompted with the landmark phrase + high-confidence
synonyms. Each detection is queried against the Habitat semantic panorama
to recover an MP3D instance id, preferring instances whose MPCat40 name is
a coarse bucket containing the prompted fine category (e.g. ``appliances``
for a ``stove`` prompt). The highest-confidence detection that hits a
category-matched instance wins.

Fallback path (opt-in via ``--enable_vlm_fallback``): when YOLO-World
returns nothing above threshold, ask a VLM for the landmark's pixel and
bounding box and run the same instance-recovery step.

Steps:

  1. Read the stage-3 survivor YAML and ``target_instances/{scan}/target_instances.json``.
  2. For each survivor whose selected target is grounded only to a coarse
     semantic bucket, find the rollout frame pose for that (scan, episode,
     sub_idx).
  3. Re-render a clean RGB panorama and raw semantic panorama at the same pose.
  4. Run YOLO-World on the panorama to localise the landmark.
  5. Query the raw semantic panorama inside each detection to recover the
     instance id via category-aware matching.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import get_filter_dir, get_run_dir, get_survivor_path, load_keep, resolve_exp
from src.check.build_semantic_rescue_categories import (
    _dump_json,
    _extract_json_object,
    _image_data_url,
    _load_json,
    _load_yaml,
)
from src.check.filter_semantic_granularity import (
    DEFAULT_COARSE_LABELS,
    _is_only_coarse_label,
    _label_candidates,
    _norm,
)
from src.process.rewriter import make_client
from src.process.visibility import VisibilityChecker


GROUNDING_SYSTEM = """\
You are grounding indoor navigation landmarks in a 360-degree equirectangular
RGB panorama.

Return ONLY a JSON object:
{
  "is_visible": true/false,
  "x": integer pixel coordinate,
  "y": integer pixel coordinate,
  "bbox": [x1, y1, x2, y2],
  "category": "short lowercase noun phrase or unknown",
  "confidence": "high|medium|low",
  "rationale": "one short sentence"
}

Rules:
- The FIRST image is the clean panorama. Return coordinates in this first image,
  with origin at the top-left (x increases right, y increases down).
- Additional images, when present, are context only — do not return their
  coordinates.
- The bounding box must TIGHTLY enclose only the landmark object — not the
  surrounding wall, counter, cabinet, or floor. If the object touches another
  object (e.g. a stove next to a cabinet, or a lamp on a table), shrink the
  box so it covers ONLY the landmark surface.
- The (x, y) point MUST land on the visible surface of the landmark itself,
  near the geometric center of the visible part of the object. It must not
  fall on an adjacent object such as a cabinet, counter, floor, wall, or
  ceiling.
- If the landmark is partly occluded, place (x, y) on the largest visible
  patch of the object.
- If the landmark is genuinely not visible anywhere in the panorama,
  set is_visible=false, x=-1, y=-1, bbox=[-1,-1,-1,-1], category="unknown".
- Prefer concrete categories such as "stove", "refrigerator", "dishwasher",
  "lamp", "blanket", "rug", "towel rack".
- Do not return a broad bucket such as "appliances", "objects", "furniture",
  or "lighting" as the category.
"""


# MP3D fine category names sometimes differ from how a VLM phrases the same
# object. Keep this list short and only include high-confidence pairs — wrong
# synonyms will silently misroute the rescue.
_CATEGORY_SYNONYMS: Dict[str, set] = {
    "stove": {"oven", "cooktop", "range", "stovetop", "stove top"},
    "oven": {"stove", "range", "cooktop"},
    "refrigerator": {"fridge"},
    "fridge": {"refrigerator"},
    "couch": {"sofa"},
    "sofa": {"couch"},
    "rug": {"carpet", "mat"},
    "carpet": {"rug"},
    "lamp": {"light", "lighting", "light fixture", "sconce"},
    "tv": {"television", "monitor", "screen"},
    "television": {"tv", "monitor"},
    "trash can": {"bin", "garbage can", "wastebin", "wastebasket"},
    "blanket": {"sheet", "comforter", "throw"},
    "towel rack": {"towel"},
    "sink": {"basin"},
    "toilet": {"commode"},
    "microwave": {"microwave oven"},
    "dishwasher": {"dish washer"},
    "curtain": {"drape", "drapes", "curtains"},
}


# Habitat's MP3D semantic sensor returns MPCat40 category names, which group
# many fine objects into a single coarse bucket — e.g. a stove, refrigerator,
# and dishwasher all share the "appliances" label at the pixel level. The
# whole point of this rescue is to add finer categories on top of those
# buckets, so when the VLM proposes "stove" we need to recognise an
# "appliances" instance in the bbox as the correct target.
_COARSE_TO_FINE: Dict[str, set] = {
    "appliances": {
        "stove", "refrigerator", "fridge", "oven", "dishwasher", "microwave",
        "washer", "dryer", "washing machine", "stovetop", "cooktop", "range",
        "coffee maker", "coffee machine", "toaster", "kettle", "range hood",
        "vent hood", "freezer", "wine cooler",
    },
    "lighting": {
        "lamp", "light", "light fixture", "sconce", "chandelier", "pendant",
        "ceiling light", "floor lamp", "table lamp", "wall light", "spotlight",
    },
    "tv_monitor": {"tv", "television", "monitor", "screen", "display"},
    "furniture": {
        "shelf", "shelves", "wardrobe", "bookshelf", "bookcase", "bench",
        "ottoman", "rack", "stand", "tv stand", "nightstand", "side table",
        "coffee table", "end table", "console", "armoire",
    },
    "objects": {
        "rug", "carpet", "blanket", "pillow", "vase", "clock", "trash can",
        "wastebasket", "bin", "basket", "book", "books", "toy", "frame",
        "speaker", "fan", "candle", "decoration", "plant pot", "tray",
        "bottle", "bowl", "plate", "cup", "magazine", "newspaper",
    },
    "cushion": {"pillow", "throw pillow"},
    "curtain": {"drape", "drapes", "curtains", "blinds"},
    "blinds": {"shutters", "shade", "shades"},
    "counter": {"countertop", "kitchen counter", "island"},
    "chest_of_drawers": {"dresser", "drawers", "chest"},
    "sink": {"basin", "washbasin"},
    "shower": {"shower stall", "shower door"},
    "bathtub": {"tub", "jacuzzi"},
    "towel": {"towel rack", "towel bar"},
    "clothes": {"clothing", "jacket", "shirt", "robe", "coat"},
    "shelving": {"shelf", "shelves", "rack"},
    "misc": set(),
}


def _category_match_score(vlm_cat: str, sem_name: str) -> float:
    """Heuristic similarity between a VLM category and an MP3D semantic name.

    Returns 0 for no match, 1.0 for exact, 0.85 for coarse-bucket containment
    (e.g. ``stove`` matches ``appliances``), 0.9 for synonyms, 0.5-0.75 for
    token overlap.
    """
    a = (vlm_cat or "").strip().lower()
    b = (sem_name or "").strip().lower()
    if not a or not b or a == "unknown" or b == "unknown":
        return 0.0
    if a == b:
        return 1.0
    # MPCat40 bucket containment is the workhorse: the semantic map labels
    # individual stoves as "appliances", so this is what fires on real data.
    if a in _COARSE_TO_FINE.get(b, set()):
        return 0.85
    if b in _COARSE_TO_FINE.get(a, set()):
        return 0.85
    if b in _CATEGORY_SYNONYMS.get(a, set()) or a in _CATEGORY_SYNONYMS.get(b, set()):
        return 0.9
    # Whole-word containment (e.g. "kitchen stove" vs "stove").
    a_toks = a.split()
    b_toks = b.split()
    if set(a_toks) <= set(b_toks) or set(b_toks) <= set(a_toks):
        return 0.75
    if set(a_toks) & set(b_toks):
        return 0.5
    return 0.0


def _build_yolo_prompts(landmark: str) -> List[str]:
    """Landmark phrase + high-confidence synonyms for open-vocab detection.

    YOLO-World takes a list of class strings; we keep the landmark itself
    first so the highest-confidence match for our exact phrase still ranks
    where it would have. Synonyms boost recall when the VLM-style canonical
    name (``refrigerator``) differs from the dataset phrasing (``fridge``).
    """
    base = (landmark or "").strip().lower()
    if not base:
        return []
    prompts = [base]
    for syn in sorted(_CATEGORY_SYNONYMS.get(base, set())):
        if syn and syn not in prompts:
            prompts.append(syn)
    return prompts


class YoloWorldDetector:
    """Thin wrapper around ultralytics YOLOWorld with class-prompt caching."""

    def __init__(self, model_path: str, device: Optional[str] = None) -> None:
        try:
            from ultralytics import YOLOWorld
        except ImportError as e:
            raise SystemExit(
                "ultralytics is required for YOLO-World grounding. Install:\n"
                "  pip install ultralytics\n"
                f"(original error: {e})"
            )
        self.model = YOLOWorld(model_path)
        if device:
            self.model.to(device)
        self._cached_classes: Optional[Tuple[str, ...]] = None

    def set_classes(self, classes: List[str]) -> None:
        key = tuple(classes)
        if self._cached_classes == key:
            return
        # ultralytics' YOLOWorld.set_classes re-tokenises via a CLIP text
        # encoder that allocates tokens on CPU; once the detector has moved
        # to CUDA (e.g. after the first predict()), this raises a device
        # mismatch. Bouncing the whole model through CPU for the encode and
        # back is cheap (~ms) and avoids the bug.
        target_device = None
        try:
            target_device = next(self.model.model.parameters()).device
        except Exception:
            pass
        moved = False
        if target_device is not None and str(target_device).startswith("cuda"):
            self.model.to("cpu")
            moved = True
        self.model.set_classes(list(classes))
        if moved:
            self.model.to(target_device)
        self._cached_classes = key

    def detect(
        self,
        image: np.ndarray,
        classes: List[str],
        conf: float = 0.05,
        imgsz: int = 1024,
        max_det: int = 20,
    ) -> List[Dict]:
        """Return detections sorted by descending score."""
        if not classes:
            return []
        self.set_classes(classes)
        # YOLO-World expects BGR or RGB; ultralytics auto-handles ndarray as
        # RGB when channels==3 and dtype==uint8, matching our render output.
        results = self.model.predict(
            image,
            conf=conf,
            imgsz=imgsz,
            max_det=max_det,
            verbose=False,
        )
        out: List[Dict] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.detach().cpu().numpy()
        scores = r.boxes.conf.detach().cpu().numpy()
        cls_ids = r.boxes.cls.detach().cpu().numpy().astype(int)
        names = r.names if isinstance(r.names, dict) else dict(enumerate(r.names))
        for i in range(len(xyxy)):
            ci = int(cls_ids[i])
            class_name = names.get(ci) if isinstance(names, dict) else None
            if not class_name and 0 <= ci < len(classes):
                class_name = classes[ci]
            x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
            out.append({
                "bbox": [x1, y1, x2, y2],
                "score": float(scores[i]),
                "class_idx": ci,
                "class_name": class_name or "",
            })
        out.sort(key=lambda d: -d["score"])
        return out


def _call_grounding_vlm_json(
    client,
    *,
    model: str,
    text: str,
    image_paths: List[Path],
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_delay: float,
    label: str,
) -> Dict:
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for image_path in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": _image_data_url(image_path)},
        })

    last_raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": GROUNDING_SYSTEM},
                        {"role": "user", "content": content},
                    ],
                )
            except TypeError:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": GROUNDING_SYSTEM},
                        {"role": "user", "content": content},
                    ],
                )
            last_raw = (resp.choices[0].message.content or "").strip()
            return _extract_json_object(last_raw)
        except Exception as exc:
            print(f"  [attempt {attempt}/{max_retries}] {label} error: {exc}")
            if attempt < max_retries:
                time.sleep(retry_delay)
    print(f"  [DEBUG] {label} final raw response:")
    print("  " + last_raw[:1000].replace("\n", "\n  "))
    raise RuntimeError(f"VLM grounding failed after {max_retries} retries ({label})")


def _load_frames(run_dir: Path) -> Dict[Tuple[str, str, str], Dict]:
    """Return last rollout frame metadata for each (scan, ep, sub_idx)."""
    out: Dict[Tuple[str, str, str], Dict] = {}
    root = run_dir / "rollout_viz"
    if not root.exists():
        return out
    for scan_dir in sorted(root.iterdir()):
        if not scan_dir.is_dir():
            continue
        frames_path = scan_dir / "frames.jsonl"
        if not frames_path.exists():
            continue
        with open(frames_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = (
                    str(rec.get("scan") or scan_dir.name),
                    str(rec.get("instruction_id")),
                    str(rec.get("sub_idx")),
                )
                prev = out.get(key)
                if prev is None or int(rec.get("step", -1)) >= int(prev.get("step", -1)):
                    out[key] = rec
    return out


def _load_target_db(run_dir: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    root = run_dir / "target_instances"
    if not root.exists():
        return out
    for scan_dir in sorted(root.iterdir()):
        path = scan_dir / "target_instances.json"
        if scan_dir.is_dir() and path.exists():
            out[scan_dir.name] = _load_json(path)
    return out


def _target_record(target_db: Dict[str, Dict], scan: str, ep_id: str, sub_idx: str) -> Dict:
    data = target_db.get(scan) or {}
    for section in ("target_instances", "annotations"):
        ep_entry = (data.get(section) or {}).get(str(ep_id)) or {}
        if not isinstance(ep_entry, dict):
            continue
        rec = ep_entry.get(str(sub_idx)) or {}
        if isinstance(rec, dict) and rec:
            if section == "annotations":
                rec = dict(rec)
                rec.setdefault("target_instance_ids", [])
            return rec
    return {}


def _collect_from_survivors(
    survivor_yaml: Path,
    run_dir: Path,
    coarse: set[str],
) -> List[Dict]:
    data = load_keep(survivor_yaml)
    target_db = _load_target_db(run_dir)
    if not target_db or not any(
        (db.get("target_instances") or db.get("annotations") or {})
        for db in target_db.values()
    ):
        raise SystemExit(
            "No target instance annotations found. Pixel-grounded rescue "
            "runs after list_target_instances and before the final stage-4 "
            "filter. Run:\n"
            f"  bash scripts/07_list_potential_instances.sh --exp {survivor_yaml}\n"
            "then rerun this rescue script."
        )
    records: List[Dict] = []
    for ep_id, subs in (data.get("sub_paths") or {}).items():
        for sub_idx in subs:
            # Scan lives in the target DB; search the per-scan files.  The
            # survivor YAML intentionally stays scan-agnostic.
            for scan in sorted(target_db):
                rec = _target_record(target_db, scan, str(ep_id), str(sub_idx))
                if not rec:
                    continue
                labels = _label_candidates(rec)
                is_bad, coarse_label = _is_only_coarse_label(
                    rec.get("landmark") or "", labels, coarse
                )
                if not is_bad:
                    continue
                records.append({
                    "scan": str(scan),
                    "episode_id": str(ep_id),
                    "sub_idx": str(sub_idx),
                    "landmark": rec.get("landmark") or "",
                    "coarse_label": coarse_label,
                    "semantic_labels": labels,
                    "previous_target_instance_ids": rec.get("target_instance_ids") or [],
                })
                break
    return records


def _collect_from_dropped(dropped_yaml: Path) -> List[Dict]:
    data = _load_yaml(dropped_yaml)
    records: List[Dict] = []
    for ep_id, ep_drop in (data.get("dropped") or {}).items():
        scan = ep_drop.get("scan")
        if not scan:
            continue
        for sub_idx, sub_drop in (ep_drop.get("subs") or {}).items():
            if sub_drop.get("reason") != "coarse_semantic_label":
                continue
            records.append({
                "scan": str(scan),
                "episode_id": str(ep_id),
                "sub_idx": str(sub_idx),
                "landmark": sub_drop.get("landmark") or "",
                "coarse_label": sub_drop.get("coarse_label"),
                "semantic_labels": sub_drop.get("semantic_labels") or [],
                "previous_target_instance_ids": sub_drop.get("target_instance_ids") or [],
            })
    return records


def _build_user_text(rec: Dict, width: int, height: int) -> str:
    return (
        f"IMAGE_SIZE: width={width}, height={height}\n"
        f"SCAN: {rec['scan']}\n"
        f"EPISODE_ID: {rec['episode_id']}\n"
        f"SUB_IDX: {rec['sub_idx']}\n"
        f"LANDMARK: {rec['landmark']}\n"
        f"COARSE_SEMANTIC_LABEL: {rec.get('coarse_label')}\n"
        f"PREVIOUS_TARGET_INSTANCE_IDS: "
        f"{json.dumps(rec.get('previous_target_instance_ids', []), ensure_ascii=False)}\n\n"
        "Find the landmark in this panorama and return one pixel coordinate "
        "on the object, plus the most specific visual category you can infer."
    )


def _normalize_grounding(raw: Dict, width: int, height: int) -> Dict:
    category = str(raw.get("category") or "unknown").strip().lower()
    confidence = str(raw.get("confidence") or "low").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    try:
        x = int(round(float(raw.get("x", -1))))
        y = int(round(float(raw.get("y", -1))))
    except Exception:
        x, y = -1, -1
    is_visible = bool(raw.get("is_visible"))
    if x < 0 or y < 0 or x >= width or y >= height:
        is_visible = False
        x, y = -1, -1
    if category in {"appliances", "objects", "furniture", "lighting"}:
        category = "unknown"
    bbox = raw.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        bbox = [x - 30, y - 30, x + 30, y + 30] if x >= 0 and y >= 0 else [-1, -1, -1, -1]
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except Exception:
        x1, y1, x2, y2 = -1, -1, -1, -1
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if is_visible:
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            is_visible = False
    else:
        x1, y1, x2, y2 = -1, -1, -1, -1
    return {
        "is_visible": is_visible,
        "x": x,
        "y": y,
        "bbox": [x1, y1, x2, y2],
        "category": category if is_visible and category else "unknown",
        "confidence": confidence,
        "rationale": str(raw.get("rationale") or "").strip(),
    }


def _instance_at_pixel(
    sem: np.ndarray,
    x: int,
    y: int,
    radius: int,
    checker: VisibilityChecker,
    preferred_categories: Optional[List[str]] = None,
    search_radius: int = 140,
    bbox: Optional[List[int]] = None,
) -> Dict:
    h, w = sem.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return {"instance_id": None, "semantic_category": None, "sample_counts": {}}

    sem_name_map = getattr(checker, "_sem_name_map", None)

    def category_for(instance_id: int) -> Optional[str]:
        if sem_name_map is not None and 0 <= instance_id < len(sem_name_map):
            return str(sem_name_map[instance_id] or "") or None
        return None

    target_cats = [c for c in (preferred_categories or []) if c and c != "unknown"]

    has_bbox = bool(bbox) and len(bbox) == 4 and int(bbox[0]) >= 0
    if has_bbox:
        bx1, by1, bx2, by2 = [int(v) for v in bbox]
    else:
        bx1, by1, bx2, by2 = x - 30, y - 30, x + 30, y + 30
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)

    def region_for(margin_x: int, margin_y: int) -> Tuple[np.ndarray, Tuple[int, int, int, int], Counter]:
        rx0 = max(0, bx1 - margin_x)
        rx1 = min(w, bx2 + margin_x + 1)
        ry0 = max(0, by1 - margin_y)
        ry1 = min(h, by2 + margin_y + 1)
        crop = sem[ry0:ry1, rx0:rx1]
        cnt: Counter = Counter()
        if crop.size:
            vals = crop.reshape(-1)
            vals = vals[vals >= 0]
            if vals.size:
                cnt = Counter(int(v) for v in vals.tolist())
        return crop, (rx0, ry0, rx1, ry1), cnt

    # Two search shells: tight (bbox+8) and wide (bbox * ~3 or +search_radius).
    # Tight gives the precise grounding; wide rescues stove/lamp cases where
    # the VLM box was slightly off and the appliances/lighting instance sits
    # next to it rather than inside.
    wide_margin_x = max(8, min(int(search_radius), max(bw, 160)))
    wide_margin_y = max(8, min(int(search_radius), max(bh, 120)))
    shells = [
        ("tight", 8, 8),
        ("wide", wide_margin_x, wide_margin_y),
    ]

    def nearest_pixel(instance_id: int, region: np.ndarray, rx0: int, ry0: int) -> Tuple[int, int]:
        mask_y, mask_x = np.where(region == int(instance_id))
        if not len(mask_x):
            return int(x), int(y)
        mx = mask_x + rx0
        my = mask_y + ry0
        dist2 = (mx - x) ** 2 + (my - y) ** 2
        j = int(np.argmin(dist2))
        return int(mx[j]), int(my[j])

    # 1) Prefer an instance whose MP3D semantic name matches the VLM category.
    #    The MP3D semantic sensor uses MPCat40 names, so the actual hit is
    #    almost always via the coarse-bucket containment branch (stove →
    #    appliances). We search tight first, then a wider shell.
    for shell_name, mx_margin, my_margin in shells:
        region, (rx0, ry0, rx1, ry1), region_counts = region_for(mx_margin, my_margin)
        best_id: Optional[int] = None
        best_score = 0.0
        best_pix = 0
        best_target = ""
        for inst_id, cnt in region_counts.items():
            if cnt < 4:
                continue
            sem_name = category_for(inst_id) or ""
            for target in target_cats:
                s = _category_match_score(target, sem_name)
                if s <= 0.0:
                    continue
                # Prefer exact / coarse-bucket matches; among those, prefer
                # the larger and more centered instance.
                mask_y, mask_x = np.where(region == int(inst_id))
                if not len(mask_x):
                    continue
                cx = float(mask_x.mean() + rx0)
                cy = float(mask_y.mean() + ry0)
                dist = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
                # Normalise distance against shell diagonal so the size bonus
                # is comparable across shells.
                shell_diag = ((rx1 - rx0) ** 2 + (ry1 - ry0) ** 2) ** 0.5 or 1.0
                proximity = max(0.0, 1.0 - dist / shell_diag)
                scored = s + 0.10 * proximity + 0.05 * min(cnt, 5000) / 5000.0
                if scored > best_score or (scored == best_score and cnt > best_pix):
                    best_id, best_score, best_pix, best_target = inst_id, scored, cnt, target

        if best_id is not None:
            qx, qy = nearest_pixel(best_id, region, rx0, ry0)
            return {
                "instance_id": int(best_id),
                "semantic_category": category_for(best_id),
                "sample_counts": {str(k): int(v) for k, v in region_counts.most_common(10)},
                "sample_pixel_count": int(best_pix),
                "vlm_pixel": {"x": int(x), "y": int(y)},
                "query_pixel": {"x": qx, "y": qy},
                "bbox": [int(rx0), int(ry0), int(rx1 - 1), int(ry1 - 1)],
                "correction": f"category_match_{shell_name}",
                "match_score": float(best_score),
                "match_category": best_target,
            }

    # No category match at any shell. For the bbox-majority fallback below
    # use the tight region.
    region, (rx0, ry0, rx1, ry1), region_counts = region_for(8, 8)

    # 2) Small radius at the VLM pixel.
    px0, px1 = max(0, x - radius), min(w, x + radius + 1)
    py0, py1 = max(0, y - radius), min(h, y + radius + 1)
    pix_vals = sem[py0:py1, px0:px1].reshape(-1)
    pix_vals = pix_vals[pix_vals >= 0]
    if pix_vals.size:
        pix_counts = Counter(int(v) for v in pix_vals.tolist())
        instance_id, count = pix_counts.most_common(1)[0]
        return {
            "instance_id": int(instance_id),
            "semantic_category": category_for(instance_id),
            "sample_counts": {str(k): int(v) for k, v in pix_counts.most_common(10)},
            "sample_pixel_count": int(count),
            "vlm_pixel": {"x": int(x), "y": int(y)},
            "query_pixel": {"x": int(x), "y": int(y)},
            "correction": "vlm_point",
        }

    # 3) Majority within the broader region.
    if region_counts:
        instance_id, count = region_counts.most_common(1)[0]
        qx, qy = nearest_pixel(instance_id, region, rx0, ry0)
        return {
            "instance_id": int(instance_id),
            "semantic_category": category_for(instance_id),
            "sample_counts": {str(k): int(v) for k, v in region_counts.most_common(10)},
            "sample_pixel_count": int(count),
            "vlm_pixel": {"x": int(x), "y": int(y)},
            "query_pixel": {"x": qx, "y": qy},
            "bbox": [int(rx0), int(ry0), int(rx1 - 1), int(ry1 - 1)],
            "correction": "bbox_majority",
        }

    return {"instance_id": None, "semantic_category": None, "sample_counts": {}}


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
            and _category_match_score(str(det_cat), str(sem_cat)) < 0.5
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
    # Instance recovery options.
    ap.add_argument("--sample_radius", type=int, default=5)
    ap.add_argument("--search_radius", type=int, default=140)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--append", action="store_true",
                    help="Append to an existing semantic_rescue_categories.json instead of replacing this run's entries.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    resolve_exp(cfg, args.exp, apply_current=True)

    rw_cfg = _load_yaml(Path(args.rewrite_config)) if Path(args.rewrite_config).exists() else {}
    vlm_model = args.model or rw_cfg.get("model", "gemini-2.5-flash")
    temperature = float(rw_cfg.get("temperature", 0.1))
    max_tokens = min(int(rw_cfg.get("max_tokens", 2048)), 4096)
    max_retries = int(rw_cfg.get("max_retries", 3))
    retry_delay = float(rw_cfg.get("retry_delay", 2.0))

    run_dir = get_run_dir(cfg)
    filt_dir = get_filter_dir(cfg)
    source_yaml = Path(args.dropped_yaml).expanduser() if args.dropped_yaml else (
        get_survivor_path(cfg)
    )
    if not source_yaml.exists():
        raise SystemExit(f"No source YAML at {source_yaml}")

    frames = _load_frames(run_dir)
    if args.dropped_yaml:
        records = _collect_from_dropped(source_yaml)
        source_kind = "stage4_dropped"
    else:
        coarse = {_norm(x) for x in (args.coarse_label or sorted(DEFAULT_COARSE_LABELS))}
        records = _collect_from_survivors(source_yaml.resolve(), run_dir, coarse)
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

            out_dir = run_dir / "target_instances" / scan / "vlm_pixel_grounding" / rec["episode_id"]
            rgb_path = out_dir / f"sub_{int(rec['sub_idx']):03d}_rgb.png"
            out_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(rgb_path)

            print(f"[{scan} ep={rec['episode_id']} sub={rec['sub_idx']}] {rec['landmark']!r}")

            prompts = _build_yolo_prompts(rec["landmark"])
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

            # Walk detections by descending score. Stop at the first one whose
            # bbox contains a category-matched MP3D instance. If none does,
            # keep the highest-confidence detection so downstream code still
            # has something to visualise.
            for det in detections:
                bbox_i = [int(round(v)) for v in det["bbox"]]
                cx = (bbox_i[0] + bbox_i[2]) // 2
                cy = (bbox_i[1] + bbox_i[3]) // 2
                preferred = [det["class_name"], rec.get("landmark") or ""]
                preferred = [c for c in preferred if c]
                info = _instance_at_pixel(
                    sem,
                    cx,
                    cy,
                    max(0, args.sample_radius),
                    checker,
                    preferred_categories=preferred,
                    search_radius=max(0, args.search_radius),
                    bbox=bbox_i,
                )
                if chosen_det is None:
                    chosen_det = det
                    pixel_info = info
                if str(info.get("correction", "")).startswith("category_match"):
                    chosen_det = det
                    pixel_info = info
                    break

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
                    raw = _call_grounding_vlm_json(
                        client,
                        model=vlm_model,
                        text=_build_user_text(rec, w, h),
                        image_paths=image_paths,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                        label=f"{scan}:{rec['episode_id']}:{rec['sub_idx']}",
                    )
                    g = _normalize_grounding(raw, w, h)
                except Exception as exc:
                    print(f"  vlm fallback failed: {exc}")
                    g = None
                if g is not None and g.get("is_visible"):
                    grounding = g
                    raw_payload = {"detections": detections, "prompts": prompts, "vlm_raw": raw}
                    source = "vlm_fallback"
                    preferred = [c for c in (g.get("category"), rec.get("landmark")) if c]
                    pixel_info = _instance_at_pixel(
                        sem,
                        g["x"],
                        g["y"],
                        max(0, args.sample_radius),
                        checker,
                        preferred_categories=preferred,
                        search_radius=max(0, args.search_radius),
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
            if grounding is None:
                print(f"  -> not visible (no detection above {args.yolo_conf})")
                summaries[scan].append({**summary_base, "status": "not_visible"})
                processed += 1
                continue

            instance_id = pixel_info.get("instance_id")
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
            if instance_id is None:
                print(f"  -> bbox {grounding.get('bbox')} no MP3D instance recovered")
                summaries[scan].append({
                    **summary_base,
                    "status": "no_instance_at_pixel",
                    "debug_point_path": str(debug_path),
                    "bbox_path": str(bbox_path),
                    "pixel_instance": pixel_info,
                })
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

            key = str(instance_id)
            entry = payload["instances"].setdefault(key, {
                "instance_id": int(instance_id),
                "category": grounding["category"],
                "confidence": grounding["confidence"],
                "is_rescuable": grounding["category"] != "unknown",
                "semantic_category": pixel_info.get("semantic_category"),
                "grounding_method": source,
                "landmarks": [],
                "examples": [],
                "image_paths": [],
            })
            if grounding["confidence"] == "high" or entry.get("category") in ("", "unknown"):
                entry["category"] = grounding["category"]
                entry["confidence"] = grounding["confidence"]
                entry["is_rescuable"] = grounding["category"] != "unknown"
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

    for scan, payload in sorted(payloads.items()):
        out_path = run_dir / "target_instances" / scan / "semantic_rescue_categories.json"
        _dump_json(out_path, payload)
        print(f"wrote {out_path}")
    for scan, items in sorted(summaries.items()):
        summary_path = (
            run_dir / "target_instances" / scan
            / "vlm_pixel_grounding" / "vlm_pixel_grounding_summary.json"
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
        sheet_path = summary_path.with_suffix(".png")
        made = _make_contact_sheet(summary_path, sheet_path)
        if made:
            print(f"wrote {made}")
    print(f"Processed VLM records: {processed}")


if __name__ == "__main__":
    main()
