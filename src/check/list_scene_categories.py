"""
LION-Bench — List MP40 / MP3D semantic categories present in a scene.

Useful for debugging visibility/uniqueness mismatches: when a landmark gets
``no_match`` even though the rewriter mapped it to a sensible label, this
tool tells you what category names the scene's semantic mesh actually
exposes (e.g. "refrigerator" rather than "fridge", "stairs" rather than
"stair case").

It can also read a selection / filter YAML, infer which scenes are used by the
selected instructions, and dump the MP3D ``.house`` object vocabulary for those
scenes.  By default, writes JSON dumps under the current experiment's
``{output.base_dir}/{run_name}/scene_categories/`` folder; the console listing
is just a preview.

Usage
-----
  # One scan (writes under {output.base_dir}/{run_name}/scene_categories/)
  python src/check/list_scene_categories.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --scan X7HyMhZNoso

  # Multiple scans (pass --scan repeatedly)
  python src/check/list_scene_categories.py --config ... \\
      --scan X7HyMhZNoso --scan oLBMNvg9in8

  # Infer scans from a selection/filter YAML and dump object lists:
  python src/check/list_scene_categories.py --config ... \\
      --from_yaml configs/selection/val_unseen_example.yaml --objects_only

  # Filter to categories whose name matches a regex (case-insensitive):
  python src/check/list_scene_categories.py --config ... \\
      --scan X7HyMhZNoso --grep "fridge|refrig|stair"

  # Override output directory:
  python src/check/list_scene_categories.py --config ... --scan ... \\
      --out_dir results/my_scenes/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import apply_selection_yaml, get_run_dir, resolve_selection
from src.dataset.landmark_rxr import episodes_from_config
from src.process.rewriter import parse_house_objects
from src.process.visibility import VisibilityChecker


def _list_categories(
    checker:    VisibilityChecker,
    scene_file: str,
) -> Tuple[Counter, Dict[int, str]]:
    """Load the scene, return (instance counts per category, cat_id → name)."""
    checker.load_scene(scene_file)
    sim = checker._sim                    # noqa: SLF001 - debug tool
    scene = sim.semantic_annotations()

    counts:        Counter           = Counter()
    cat_id_to_name: Dict[int, str]   = {}

    if scene is None or not getattr(scene, "objects", None):
        return counts, cat_id_to_name

    for obj in scene.objects:
        if obj is None or obj.category is None:
            continue
        name = (obj.category.name() or "").strip()
        if not name:
            continue
        counts[name] += 1
        cat_id_to_name[int(obj.category.index())] = name
    return counts, cat_id_to_name


def _unique_scans_from_episodes(cfg: dict) -> List[str]:
    """Return sorted scan ids used by the currently materialized selection."""
    episodes = episodes_from_config(cfg)
    return sorted({ep.scan for ep in episodes})


def _merge_unique_scans(*scan_groups: Sequence[str]) -> List[str]:
    seen = set()
    scans: List[str] = []
    for group in scan_groups:
        for scan in group:
            if scan in seen:
                continue
            scans.append(scan)
            seen.add(scan)
    return scans


def main() -> None:
    ap = argparse.ArgumentParser(
        description="List MP40 categories present in a Habitat scene",
    )
    ap.add_argument("--config", required=True,
                    help="Rollout YAML (provides env / scenes_dir).")
    ap.add_argument("--scan", action="append", default=[],
                    help="Scan id (e.g. X7HyMhZNoso). Pass multiple times "
                         "for several scenes.")
    ap.add_argument("--selection", default=None,
                    help="Selection YAML to merge before inferring scans "
                         "from selected instructions.")
    ap.add_argument("--from_yaml", default=None,
                    help="Selection/filter/replay YAML to merge before "
                         "inferring scans from selected instructions.")
    ap.add_argument("--objects_only", action="store_true",
                    help="Only write MP3D .house object lists for inferred "
                         "or explicit scans; skip Habitat semantic categories.")
    ap.add_argument("--grep", default=None,
                    help="Case-insensitive regex; only print categories "
                         "whose name matches.")
    ap.add_argument("--sort_by_name", action="store_true",
                    help="Sort alphabetically (default: by instance count).")
    ap.add_argument("--out_dir", default=None,
                    help="Where to write per-scan JSONs.  Default: "
                         "{output.base_dir}/{run_name}/scene_categories, "
                         "where run_name is derived from expname.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.selection:
        apply_selection_yaml(cfg, args.selection)
    resolve_selection(cfg, args.from_yaml)

    pattern: Optional[re.Pattern] = (
        re.compile(args.grep, re.IGNORECASE) if args.grep else None
    )

    out_dir = Path(args.out_dir) if args.out_dir else get_run_dir(cfg) / "scene_categories"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes_dir = cfg["scenes"]["scenes_dir"]
    inferred_scans: List[str] = []
    if args.selection or args.from_yaml:
        inferred_scans = _unique_scans_from_episodes(cfg)
        if inferred_scans:
            print("Inferred scans from selected instructions:")
            for scan in inferred_scans:
                print(f"  {scan}")

    scans = _merge_unique_scans(args.scan, inferred_scans)
    if not scans:
        raise SystemExit(
            "No scans provided. Use --scan, --selection, or --from_yaml."
        )

    checker: Optional[VisibilityChecker] = None
    if not args.objects_only:
        checker = VisibilityChecker(cfg["env"], scenes_dir)
    try:
        object_lists: Dict[str, List[str]] = {}
        for scan in scans:
            scene_file = f"mp3d/{scan}/{scan}.glb"
            object_list = parse_house_objects(scenes_dir, scan)
            object_lists[scan] = object_list

            print(f"\n=== {scan} ===")
            print(f"  object list entries: {len(object_list)}")

            object_payload = {
                "scan": scan,
                "scene_file": scene_file,
                "object_list": object_list,
            }

            if args.objects_only:
                json_path = out_dir / f"{scan}_objects.json"
                with open(json_path, "w") as f:
                    json.dump(object_payload, f, indent=2)
                print(f"  objects json → {json_path}")
                continue

            assert checker is not None
            counts, cat_id_to_name = _list_categories(checker, scene_file)

            if not counts:
                print("  (no categories — scene has no semantic annotations)")
                payload = {
                    **object_payload,
                    "total_instances": 0,
                    "unique_categories": 0,
                    "categories": [],
                }
                json_path = out_dir / f"{scan}.json"
                with open(json_path, "w") as f:
                    json.dump(payload, f, indent=2)
                print(f"  json → {json_path}")
                continue

            n_obj = sum(counts.values())
            print(f"  total instances: {n_obj}")
            print(f"  unique categories: {len(counts)}")

            cat_id_by_name = {n: i for i, n in cat_id_to_name.items()}
            full_entries: List[Dict[str, Any]] = [
                {"name": n, "cat_id": cat_id_by_name.get(n, -1), "count": int(c)}
                for n, c in sorted(counts.items(), key=lambda nc: (-nc[1], nc[0]))
            ]

            # ── Always write the full list to JSON (no grep filter applied
            # here — JSON is the canonical artifact, grep only filters the
            # console preview).
            payload = {
                "scan":              scan,
                "scene_file":        scene_file,
                "object_list":       object_list,
                "total_instances":   n_obj,
                "unique_categories": len(counts),
                "categories":        full_entries,
            }
            json_path = out_dir / f"{scan}.json"
            with open(json_path, "w") as f:
                json.dump(payload, f, indent=2)

            # ── Console preview (optionally grep / sort) ─────────────
            preview = list(full_entries)
            if pattern is not None:
                preview = [e for e in preview if pattern.search(e["name"])]
            if args.sort_by_name:
                preview.sort(key=lambda e: e["name"].lower())

            if not preview:
                print(f"  (no categories match /{args.grep}/)")
            else:
                print(f"  {'count':>5}  {'cat_id':>6}  category")
                for e in preview:
                    print(f"  {e['count']:>5d}  {e['cat_id']:>6d}  {e['name']}")

            print(f"  json → {json_path}")

        if args.selection or args.from_yaml or len(scans) > 1:
            aggregate_path = out_dir / "object_lists.json"
            with open(aggregate_path, "w") as f:
                json.dump(object_lists, f, indent=2)
            print(f"\nObject lists → {aggregate_path}")
    finally:
        if checker is not None:
            checker.close()


if __name__ == "__main__":
    main()
