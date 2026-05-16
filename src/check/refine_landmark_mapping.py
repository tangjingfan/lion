"""LION-Bench — Refine landmark_mapping[_filtered].json with per-scene LLM remap.

Reads (per scan):
  • ``{run_dir}/rewrite/{scan}/{instruction_id}/sub_instructions_rewritten[_filtered].json``
    — source of every non-spatial mention used by that scene's episodes.
  • ``{run_dir}/scene_categories/{scan}/objects.json`` — instantiated MPCAT40
    object vocabulary for that scan (produced by
    ``scripts/05_get_object_list.sh --objects_only``).

For each scan, asks an LLM to map every mention to candidate labels
drawn ONLY from that scan's object list — refining the rewriter's own
mapping with a focused per-scene prompt.

Pipeline placement
------------------
This tool only needs filter stage 2 (rewrite) to have written the
per-episode rewrite JSONs and ``list_scene_categories`` to have written
``scene_categories/{scan}/objects.json``. It does **not** depend on
partition (filter stage 3); the ``--from_yaml`` argument is used only to
resolve the experiment's run directory, so either the original selection
YAML or any later ``filters/*.yaml`` works.

Output (overwrites the rewriter's per-scan mapping in place):
  ``{run_dir}/rewrite/{scan}/landmark_mapping[_filtered].json``  with
  shape ``{mention: [labels...]}`` (flat, scan-specific).

Usage
-----
  GEMINI_API_KEY=your_key \\
  python src/check/refine_landmark_mapping.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --from_yaml configs/selection/exp.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import (
    get_run_dir,
    load_rewrite_by_scan,
    resolve_selection,
)
from src.process.landmark_remap import remap_scan_mentions
from src.process.rewriter import make_client, parse_house_objects


def _load_object_list(
    scene_categories_dir: Path, scan: str, scenes_dir: str,
) -> List[str]:
    """Prefer the cached ``{scan}/objects.json``; fall back to live parse."""
    json_path = scene_categories_dir / scan / "objects.json"
    if json_path.exists():
        with open(json_path) as f:
            payload = json.load(f) or {}
        source = payload.get("object_list_source")
        if source in {"house_mpcat40_instances", "habitat_mpcat40_instances"}:
            return list(payload.get("object_list") or [])
        print(
            f"  WARN: {json_path} has legacy object_list source "
            "— reparsing MPCAT40 labels"
        )
        return parse_house_objects(scenes_dir, scan)
    print(f"  WARN: {json_path} missing — falling back to parse_house_objects")
    return parse_house_objects(scenes_dir, scan)


def _discover_per_scan_rewrites(
    rewrite_dir: Path,
) -> Tuple[Dict[str, Dict[str, Dict]], str]:
    """Return ``{scan: {ep_id: episode}}`` and the suffix in use.

    Tries ``_filtered`` first across all scans, then unfiltered.
    """
    if not rewrite_dir.exists():
        raise SystemExit(f"No rewrite dir: {rewrite_dir}")
    by_scan, suffix, _paths = load_rewrite_by_scan(rewrite_dir)
    if not by_scan:
        raise SystemExit(
            f"No sub_instructions_rewritten[_filtered].json under "
            f"{rewrite_dir}/*/*/"
        )
    return by_scan, suffix


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-scene LLM refinement of landmark_mapping[_filtered].json",
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--rewrite_config",
                    default="configs/rewrite/rewrite_subinstructions.yaml")
    ap.add_argument("--from_yaml", default=None)
    ap.add_argument("--api_key", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    resolve_selection(cfg, args.from_yaml)

    rw_cfg: Dict = {}
    rw_path = Path(args.rewrite_config)
    if rw_path.exists():
        with open(rw_path) as f:
            rw_cfg = yaml.safe_load(f) or {}
    model       = rw_cfg.get("model",        "gemini-2.0-flash")
    temperature = float(rw_cfg.get("temperature", 0.1))
    # The remap response is one big JSON object covering every mention
    # in the scene, so it benefits from a much larger output window than
    # the per-episode rewriter calls.
    max_tokens  = int(rw_cfg.get("max_tokens",     8192))
    max_retries = int(rw_cfg.get("max_retries",    3))
    retry_delay = float(rw_cfg.get("retry_delay",  2.0))

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: provide --api_key or set GEMINI_API_KEY env var.")
        sys.exit(1)

    run_dir       = get_run_dir(cfg)
    rewrite_dir   = run_dir / "rewrite"
    scene_cat_dir = run_dir / "scene_categories"

    rewrites, suffix = _discover_per_scan_rewrites(rewrite_dir)
    print(f"Discovered rewrites for {len(rewrites)} scan(s) "
          f"(suffix={suffix!r})")

    client     = make_client(api_key)
    scenes_dir = cfg.get("scenes", {}).get("scenes_dir", "")
    summary: Dict[str, Tuple[int, int]] = {}
    t0 = time.time()

    for scan, rewrite_episodes in sorted(rewrites.items()):
        # Collect mentions this scan actually uses.
        mentions_set: set = set()
        for ep in rewrite_episodes.values():
            for sub in ep.get("sub_paths", []):
                if sub.get("landmark_category") == "spatial":
                    continue
                for comp in sub.get("components", []):
                    m = (comp.get("original_mention") or "").strip().lower()
                    if m and m != "unknown":
                        mentions_set.add(m)
        mentions = sorted(mentions_set)
        if not mentions:
            print(f"\n[{scan}] no mentions — skipping")
            continue

        object_list = _load_object_list(scene_cat_dir, scan, scenes_dir)
        if not object_list:
            print(f"\n[{scan}] empty object list — emitting empty mapping")
            mapping: Dict[str, List[str]] = {m: [] for m in mentions}
        else:
            print(f"\n[{scan}] mentions={len(mentions)}  "
                  f"object_list={len(object_list)}")
            result = remap_scan_mentions(
                client=client, model=model,
                scan=scan, mentions=mentions, object_list=object_list,
                temperature=temperature, max_tokens=max_tokens,
                max_retries=max_retries, retry_delay=retry_delay,
            )
            mapping = {m: result.get(m, []) for m in mentions}
            n_match = sum(1 for v in mapping.values() if v)
            print(f"  matched: {n_match}/{len(mentions)}")

        out_path = rewrite_dir / scan / f"landmark_mapping{suffix}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False, sort_keys=True)
        print(f"  → {out_path}")
        summary[scan] = (
            sum(1 for v in mapping.values() if v),
            len(mapping),
        )

    n_total = sum(t for _, t in summary.values())
    n_match = sum(m for m, _ in summary.values())
    pct = (n_match / n_total) if n_total else 0.0
    print(f"\n=== refine_landmark_mapping ===")
    print(f"  scans            : {len(summary)}")
    print(f"  mentions (total) : {n_total}")
    print(f"  matched          : {n_match}  ({pct:.1%})")
    print(f"  time             : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
