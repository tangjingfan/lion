"""Build a per-scan VLM rescue dictionary for coarse semantic targets.

Input is usually ``filters/04_semantic_granularity_dropped.yaml``.  For each
drop caused by ``coarse_semantic_label`` that still has a selected instance id,
the script sends the saved target visualization(s) to a vision-language model
and asks for a finer category.  Output is written per scan:

    target_instances/{scan}/semantic_rescue_categories.json

The dictionary is intentionally separate from the deterministic filter output:
downstream code can opt in by querying ``instances[str(instance_id)]``.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import get_filter_dir, get_run_dir, resolve_selection
from src.process.rewriter import make_client


SYSTEM_PROMPT = """\
You are auditing Matterport3D semantic instance annotations for indoor navigation.

You will see one or more visualization images for the SAME highlighted target
instance. The dataset semantic label is a coarse bucket such as "appliances",
"objects", "furniture", or "lighting". Your job is to decide whether the
highlighted instance can be assigned a more specific visual category.

Return ONLY a JSON object:
{
  "is_rescuable": true/false,
  "category": "short lowercase noun phrase or unknown",
  "confidence": "high|medium|low",
  "rationale": "one short sentence"
}

Rules:
- Use the highlighted target instance, not the whole scene.
- Prefer concrete object categories such as "stove", "refrigerator",
  "dishwasher", "lamp", "blanket", "rug", "towel rack".
- If the highlight is missing, too ambiguous, or cannot be visually separated
  from surrounding objects, set is_rescuable=false and category="unknown".
- Do not return the coarse bucket itself as the category.
"""


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


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _extract_json_object(raw: str) -> Dict:
    raw = (raw or "").strip()
    candidates = [raw]
    if raw.startswith("```"):
        body = raw[3:]
        if body.startswith("json"):
            body = body[4:]
        if body.endswith("```"):
            body = body[:-3]
        candidates.append(body.strip())
    first = raw.find("{")
    last = raw.rfind("}")
    if first >= 0 and last > first:
        candidates.append(raw[first:last + 1])
    last_err: Exception = ValueError("empty response")
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except Exception as exc:
            last_err = exc
    raise ValueError(f"could not parse JSON: {last_err}")


def _call_vlm_json(
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
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                )
            except TypeError:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
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
    raise RuntimeError(f"VLM failed after {max_retries} retries ({label})")


def _target_db_for_scan(run_dir: Path, scan: str) -> Dict:
    path = run_dir / "target_instances" / scan / "target_instances.json"
    if not path.exists():
        return {}
    return _load_json(path)


def _lookup_target_record(run_dir: Path, scan: str, ep_id: str, sub_idx: str) -> Dict:
    db = _target_db_for_scan(run_dir, scan)
    targets = db.get("target_instances") or {}
    ep_entry = targets.get(str(ep_id)) or {}
    if not isinstance(ep_entry, dict):
        return {}
    rec = ep_entry.get(str(sub_idx)) or {}
    return rec if isinstance(rec, dict) else {}


def _selected_candidate_viz(rec: Dict, instance_id: int) -> Optional[Path]:
    for cand in rec.get("candidates") or []:
        if cand.get("id") == instance_id and cand.get("viz_path"):
            path = Path(cand["viz_path"])
            if path.exists():
                return path
    return None


def _record_image_paths(rec: Dict, instance_id: int) -> List[Path]:
    paths: List[Path] = []
    for key in ("partition_viz_path", "rollout_last_viz_path"):
        val = rec.get(key)
        if val:
            path = Path(val)
            if path.exists():
                paths.append(path)
    cand_viz = _selected_candidate_viz(rec, instance_id)
    if cand_viz:
        paths.append(cand_viz)

    seen = set()
    unique: List[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _collect_instances(run_dir: Path, dropped_yaml: Path) -> Tuple[Dict[str, Dict[int, Dict]], List[Dict]]:
    data = _load_yaml(dropped_yaml)
    grouped: Dict[str, Dict[int, Dict]] = defaultdict(dict)
    skipped: List[Dict] = []

    for ep_id, ep_drop in (data.get("dropped") or {}).items():
        scan = ep_drop.get("scan")
        if not scan:
            continue
        for sub_idx, sub_drop in (ep_drop.get("subs") or {}).items():
            if sub_drop.get("reason") != "coarse_semantic_label":
                continue
            instance_ids = [int(x) for x in (sub_drop.get("target_instance_ids") or [])]
            if not instance_ids:
                skipped.append({
                    "scan": scan,
                    "episode_id": str(ep_id),
                    "sub_idx": str(sub_idx),
                    "landmark": sub_drop.get("landmark"),
                    "reason": "no_target_instance_id",
                })
                continue

            rec = _lookup_target_record(run_dir, scan, str(ep_id), str(sub_idx))
            for instance_id in instance_ids:
                entry = grouped[scan].setdefault(instance_id, {
                    "instance_id": instance_id,
                    "coarse_labels": set(),
                    "landmarks": set(),
                    "examples": [],
                    "image_paths": [],
                })
                if sub_drop.get("coarse_label"):
                    entry["coarse_labels"].add(str(sub_drop["coarse_label"]))
                if sub_drop.get("landmark"):
                    entry["landmarks"].add(str(sub_drop["landmark"]))
                image_paths = _record_image_paths(rec, instance_id)
                entry["image_paths"].extend(image_paths)
                entry["examples"].append({
                    "episode_id": str(ep_id),
                    "sub_idx": str(sub_idx),
                    "landmark": sub_drop.get("landmark"),
                    "coarse_label": sub_drop.get("coarse_label"),
                    "semantic_labels": sub_drop.get("semantic_labels") or [],
                    "image_paths": [str(p) for p in image_paths],
                })

    for scan_entries in grouped.values():
        for entry in scan_entries.values():
            entry["coarse_labels"] = sorted(entry["coarse_labels"])
            entry["landmarks"] = sorted(entry["landmarks"])
            unique_paths: List[Path] = []
            seen = set()
            for path in entry["image_paths"]:
                key = str(path)
                if path.exists() and key not in seen:
                    unique_paths.append(path)
                    seen.add(key)
            entry["image_paths"] = unique_paths

    return grouped, skipped


def _build_user_text(scan: str, entry: Dict) -> str:
    examples = [
        {
            "episode_id": ex.get("episode_id"),
            "sub_idx": ex.get("sub_idx"),
            "landmark": ex.get("landmark"),
            "coarse_label": ex.get("coarse_label"),
        }
        for ex in entry.get("examples", [])[:8]
    ]
    return (
        f"SCAN: {scan}\n"
        f"INSTANCE_ID: {entry['instance_id']}\n"
        f"COARSE_LABELS: {json.dumps(entry.get('coarse_labels', []), ensure_ascii=False)}\n"
        f"LANDMARK_MENTIONS: {json.dumps(entry.get('landmarks', []), ensure_ascii=False)}\n"
        f"EXAMPLES: {json.dumps(examples, ensure_ascii=False)}\n\n"
        "The images show this target instance highlighted. Decide whether a "
        "specific visual category can rescue this coarse semantic annotation."
    )


def _normalize_result(raw: Dict, coarse_labels: Iterable[str]) -> Dict:
    category = str(raw.get("category") or "unknown").strip().lower()
    confidence = str(raw.get("confidence") or "low").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    coarse = {str(x).strip().lower() for x in coarse_labels}
    is_rescuable = bool(raw.get("is_rescuable")) and category and category != "unknown"
    if category in coarse:
        is_rescuable = False
        category = "unknown"
    return {
        "is_rescuable": is_rescuable,
        "category": category if is_rescuable else "unknown",
        "confidence": confidence,
        "rationale": str(raw.get("rationale") or "").strip(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Use a VLM to build per-scan rescue categories for coarse semantic instances",
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--rewrite_config", default="configs/rewrite/rewrite_subinstructions.yaml")
    ap.add_argument("--from_yaml", default=None,
                    help="Selection/current YAML used to resolve the run dir.")
    ap.add_argument("--dropped_yaml", default=None,
                    help="Defaults to {run}/filters/04_semantic_granularity_dropped.yaml.")
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--model", default=None,
                    help="Vision-capable model. Defaults to rewrite_config model.")
    ap.add_argument("--max_images", type=int, default=3,
                    help="Maximum images to send per scan/instance.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Debug limit on number of instances to send to the VLM.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Collect instances and print counts without calling the VLM.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    resolve_selection(cfg, args.from_yaml)

    rw_cfg: Dict = {}
    rw_path = Path(args.rewrite_config)
    if rw_path.exists():
        rw_cfg = _load_yaml(rw_path)
    model = args.model or rw_cfg.get("model", "gemini-2.5-flash")
    temperature = float(rw_cfg.get("temperature", 0.1))
    max_tokens = min(int(rw_cfg.get("max_tokens", 2048)), 4096)
    max_retries = int(rw_cfg.get("max_retries", 3))
    retry_delay = float(rw_cfg.get("retry_delay", 2.0))

    run_dir = get_run_dir(cfg)
    filt_dir = get_filter_dir(cfg)
    dropped_yaml = Path(args.dropped_yaml).expanduser() if args.dropped_yaml else (
        filt_dir / "04_semantic_granularity_dropped.yaml"
    )
    if not dropped_yaml.exists():
        raise SystemExit(f"No dropped YAML at {dropped_yaml}")

    grouped, skipped = _collect_instances(run_dir, dropped_yaml)
    total_instances = sum(len(v) for v in grouped.values())
    total_with_images = sum(
        1 for scan_entries in grouped.values()
        for entry in scan_entries.values()
        if entry.get("image_paths")
    )
    print("=== semantic rescue category collection ===")
    print(f"  dropped yaml     : {dropped_yaml}")
    print(f"  scans            : {len(grouped)}")
    print(f"  instances        : {total_instances}")
    print(f"  with images      : {total_with_images}")
    print(f"  skipped no id    : {len(skipped)}")

    if args.dry_run:
        for scan, scan_entries in sorted(grouped.items()):
            print(f"  [{scan}] instances={len(scan_entries)}")
            for instance_id, entry in sorted(scan_entries.items())[:10]:
                print(
                    f"    {instance_id}: landmarks={entry['landmarks']} "
                    f"images={len(entry['image_paths'])}"
                )
        return

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: provide --api_key or set GEMINI_API_KEY env var.")
    client = make_client(api_key)

    processed = 0
    for scan, scan_entries in sorted(grouped.items()):
        out_path = run_dir / "target_instances" / scan / "semantic_rescue_categories.json"
        payload = {
            "scan": scan,
            "source_dropped_yaml": str(dropped_yaml),
            "model": model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "instances": {},
            "skipped": [x for x in skipped if x.get("scan") == scan],
        }

        if out_path.exists():
            existing = _load_json(out_path)
            if isinstance(existing.get("instances"), dict):
                payload["instances"].update(existing["instances"])
            if isinstance(existing.get("skipped"), list):
                payload["skipped"] = existing["skipped"]

        for instance_id, entry in sorted(scan_entries.items()):
            key = str(instance_id)
            if key in payload["instances"] and payload["instances"][key].get("category"):
                print(f"[{scan} #{instance_id}] already exists, skipping")
                continue
            if args.limit is not None and processed >= args.limit:
                break
            image_paths = list(entry.get("image_paths") or [])[:max(1, args.max_images)]
            if not image_paths:
                payload["instances"][key] = {
                    "instance_id": instance_id,
                    "is_rescuable": False,
                    "category": "unknown",
                    "confidence": "low",
                    "rationale": "No saved visualization image was available.",
                    "coarse_labels": entry.get("coarse_labels", []),
                    "landmarks": entry.get("landmarks", []),
                    "examples": entry.get("examples", []),
                    "image_paths": [],
                }
                continue

            print(f"[{scan} #{instance_id}] landmarks={entry['landmarks']} images={len(image_paths)}")
            raw = _call_vlm_json(
                client,
                model=model,
                text=_build_user_text(scan, entry),
                image_paths=image_paths,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                retry_delay=retry_delay,
                label=f"{scan}#{instance_id}",
            )
            result = _normalize_result(raw, entry.get("coarse_labels", []))
            payload["instances"][key] = {
                "instance_id": instance_id,
                **result,
                "coarse_labels": entry.get("coarse_labels", []),
                "landmarks": entry.get("landmarks", []),
                "examples": entry.get("examples", []),
                "image_paths": [str(p) for p in image_paths],
            }
            print(
                f"  -> {payload['instances'][key]['category']} "
                f"({payload['instances'][key]['confidence']}, "
                f"rescuable={payload['instances'][key]['is_rescuable']})"
            )
            processed += 1

        _dump_json(out_path, payload)
        print(f"  wrote {out_path}")

    print(f"\nProcessed VLM instances: {processed}")


if __name__ == "__main__":
    main()
