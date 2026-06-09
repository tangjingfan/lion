"""Open-vocabulary detection rescue: YOLO-World grounding + instance recovery.

Domain logic for pipeline step 09 (``scripts/09_detection.sh``).  The
CLI, per-record orchestration, and debug-viz rendering stay in
``src/check/build_vlm_pixel_grounded_rescue.py``; this module owns:

  * source collection — which (scan, ep, sub) records are grounded only
    to a coarse semantic bucket (or not at all) and need the detector
  * the YOLO-World wrapper (class-prompt caching, device workaround)
  * the VLM grounding fallback (prompt, response normalization)
  * bbox-majority MP3D instance recovery from the semantic panorama
  * the coarse↔fine category-match heuristic
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.pipeline.survivor import load_keep
from src.process.coarse_labels import (
    _is_only_coarse_label,
    _label_candidates,
)
from src.process.vlm_common import extract_json_object_strict, image_data_url

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


def _load_yaml(path: Path) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def category_match_score(vlm_cat: str, sem_name: str) -> float:
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
    # Whole-word containment (e.g. "kitchen stove" vs "stove").
    a_toks = a.split()
    b_toks = b.split()
    if set(a_toks) <= set(b_toks) or set(b_toks) <= set(a_toks):
        return 0.75
    if set(a_toks) & set(b_toks):
        return 0.5
    return 0.0


def build_yolo_prompts(landmark: str) -> List[str]:
    """Return the single open-vocab YOLO-World prompt for this landmark.

    YOLO-World encodes class strings with CLIP, so semantically close
    phrases ("fridge" / "refrigerator", "couch" / "sofa") already sit
    near each other in the joint text/vision embedding space — passing
    a hand-curated synonym list as extra classes was redundant noise.
    Kept as a function (returning a list) so callers don't need to
    branch; future expansion can plug in here.
    """
    base = (landmark or "").strip().lower()
    return [base] if base else []


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


def call_grounding_vlm_json(
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
            "image_url": {"url": image_data_url(image_path)},
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
            return extract_json_object_strict(last_raw)
        except Exception as exc:
            print(f"  [attempt {attempt}/{max_retries}] {label} error: {exc}")
            if attempt < max_retries:
                time.sleep(retry_delay * (2 ** (attempt - 1)))
    print(f"  [DEBUG] {label} final raw response:")
    print("  " + last_raw[:1000].replace("\n", "\n  "))
    raise RuntimeError(f"VLM grounding failed after {max_retries} retries ({label})")


# ── Stage-input loading ──────────────────────────────────────────────────
def load_frames(run_dir: Path) -> Dict[Tuple[str, str, str], Dict]:
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


def load_target_db(run_dir: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    root = run_dir / "target_instances"
    if not root.exists():
        return out
    for scan_dir in sorted(root.iterdir()):
        path = scan_dir / "target_instances.json"
        if scan_dir.is_dir() and path.exists():
            out[scan_dir.name] = _load_json(path)
    return out


def target_record(target_db: Dict[str, Dict], scan: str, ep_id: str, sub_idx: str) -> Dict:
    """Read the per-(ep, sub) record. ``annotations`` is the primary
    section (after the step 07/08/10 merge); the legacy
    ``target_instances`` section is checked as a fallback for older runs."""
    data = target_db.get(scan) or {}
    for section in ("annotations", "target_instances"):
        ep_entry = (data.get(section) or {}).get(str(ep_id)) or {}
        if not isinstance(ep_entry, dict):
            continue
        rec = ep_entry.get(str(sub_idx)) or {}
        if isinstance(rec, dict) and rec:
            rec = dict(rec)
            rec.setdefault("target_instance_ids", [])
            return rec
    return {}


# ── Source collection ────────────────────────────────────────────────────
def collect_from_survivors(
    survivor_yaml: Path,
    run_dir: Path,
    coarse: set,
) -> List[Dict]:
    data = load_keep(survivor_yaml)
    target_db = load_target_db(run_dir)
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
                rec = target_record(target_db, scan, str(ep_id), str(sub_idx))
                if not rec:
                    continue
                landmark = (rec.get("landmark") or "").strip()
                if not landmark:
                    continue
                labels = _label_candidates(rec)
                is_coarse_only, coarse_label = _is_only_coarse_label(
                    landmark, labels, coarse
                )
                # Three cases reach YOLO:
                #   - no_match: refined mapping returned [], scene has no
                #     fine-category label for this landmark. YOLO may still
                #     find the object visually and land on a coarse-bucket
                #     MP3D instance (e.g. fridge → instance labeled
                #     "appliances").
                #   - coarse-only: every match is a coarse bucket
                #     (appliances / furniture / objects / lighting) and the
                #     landmark is fine. YOLO disambiguates the bucket.
                #   - else (has a real fine match already): skip.
                if labels and not is_coarse_only:
                    continue
                records.append({
                    "scan": str(scan),
                    "episode_id": str(ep_id),
                    "sub_idx": str(sub_idx),
                    "landmark": landmark,
                    "coarse_label": coarse_label or None,
                    "semantic_labels": labels,
                    "previous_target_instance_ids": rec.get("target_instance_ids") or [],
                    # Partition (see-then-go) pose from step 07 — detection
                    # grounds the landmark here, not at the rollout end pose.
                    "partition_pos": rec.get("partition_pos"),
                })
                break
    return records


def collect_from_dropped(dropped_yaml: Path) -> List[Dict]:
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


# ── Grounding-response handling ──────────────────────────────────────────
def build_user_text(rec: Dict, width: int, height: int) -> str:
    coarse = rec.get("coarse_label") or "(none — landmark has no scene category match)"
    return (
        f"IMAGE_SIZE: width={width}, height={height}\n"
        f"SCAN: {rec['scan']}\n"
        f"EPISODE_ID: {rec['episode_id']}\n"
        f"SUB_IDX: {rec['sub_idx']}\n"
        f"LANDMARK: {rec['landmark']}\n"
        f"COARSE_SEMANTIC_LABEL: {coarse}\n"
        f"PREVIOUS_TARGET_INSTANCE_IDS: "
        f"{json.dumps(rec.get('previous_target_instance_ids', []), ensure_ascii=False)}\n\n"
        "Find the landmark in this panorama and return one pixel coordinate "
        "on the object, plus the most specific visual category you can infer."
    )


def normalize_grounding(raw: Dict, width: int, height: int) -> Dict:
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


def instance_at_pixel(
    sem: np.ndarray,
    x: int,
    y: int,
    checker,
    bbox: Optional[List[int]] = None,
) -> Dict:
    """Return the dominant MP3D instance inside the YOLO bbox.

    Simplified policy: trust the detection. Take the bbox-majority
    instance id and let the caller override its category to the
    landmark name. No category-match verification, no fallback tiers
    — if YOLO drew the wrong bbox, the override is wrong, and that's
    a problem for the detection layer, not for instance lookup.
    """
    h, w = sem.shape[:2]
    sem_name_map = getattr(checker, "_sem_name_map", None)

    def category_for(instance_id: int) -> Optional[str]:
        if sem_name_map is not None and 0 <= instance_id < len(sem_name_map):
            return str(sem_name_map[instance_id] or "") or None
        return None

    has_bbox = bool(bbox) and len(bbox) == 4 and int(bbox[0]) >= 0
    if has_bbox:
        bx1, by1, bx2, by2 = [int(v) for v in bbox]
    else:
        # Legacy point-only call site: build a small box around (x, y)
        # so the rest of the function still works.
        bx1, by1, bx2, by2 = x - 30, y - 30, x + 30, y + 30
    bx1 = max(0, bx1)
    by1 = max(0, by1)
    bx2 = min(w, bx2 + 1)
    by2 = min(h, by2 + 1)

    crop = sem[by1:by2, bx1:bx2]
    if not crop.size:
        return {"instance_id": None, "semantic_category": None, "sample_counts": {}}
    vals = crop.reshape(-1)
    vals = vals[vals >= 0]
    if not vals.size:
        return {"instance_id": None, "semantic_category": None, "sample_counts": {}}
    counts = Counter(int(v) for v in vals.tolist())
    instance_id, count = counts.most_common(1)[0]
    return {
        "instance_id":        int(instance_id),
        "semantic_category":  category_for(instance_id),
        "sample_counts":      {str(k): int(v) for k, v in counts.most_common(10)},
        "sample_pixel_count": int(count),
        "vlm_pixel":          {"x": int(x), "y": int(y)},
        "bbox":               [int(bx1), int(by1), int(bx2 - 1), int(by2 - 1)],
        "correction":         "bbox_majority",
    }
