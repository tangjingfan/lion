"""
LION-Bench — Landmark uniqueness check  (CLI entry point).

All pipeline logic lives in src/process/visibility.py :: run_landmark_uniqueness_check().

Usage
-----
  python src/check/check_landmark_uniqueness.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml

Output
------
  results/val_unseen/obs/landmark_uniqueness.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dataset.landmark_rxr import episodes_from_config
from src.env.connectivity import load_connectivity
from src.process.visibility import VisibilityChecker, run_landmark_uniqueness_check


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check landmark uniqueness from the sub-path end position"
    )
    p.add_argument("--config", required=True,
                   help="Rollout YAML config (scene / connectivity / dataset paths)")
    p.add_argument("--from_yaml", default=None,
                   help="Selection YAML to override config's selection.from_yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.from_yaml:
        cfg.setdefault("selection", {})["from_yaml"] = args.from_yaml

    out_cfg  = cfg.get("output", {})
    base_dir = Path(out_cfg.get("base_dir", "results")).expanduser()
    run_name = (out_cfg.get("run_name")
                or Path(cfg["dataset"]["data_path"]).stem.replace("LandmarkRxR_", ""))
    out_dir  = base_dir / run_name
    obs_root = out_dir / "obs"
    json_out = obs_root / "landmark_uniqueness.json"

    uniq_cfg       = cfg.get("uniqueness", {})
    render_obs     = uniq_cfg.get("render_obs", False)
    img_w          = uniq_cfg.get("img_width",  320)
    img_h          = uniq_cfg.get("img_height", 240)
    rewrite_dir    = out_dir / "rewrite"
    rewritten_path = Path(uniq_cfg["rewritten_path"]) if uniq_cfg.get("rewritten_path") \
                     else rewrite_dir / "sub_instructions_rewritten.json"
    # derive mapping path from rewritten path: same dir, same suffix (_filtered or not)
    suffix       = "_filtered" if rewritten_path.stem.endswith("_filtered") else ""
    mapping_path = rewritten_path.parent / f"landmark_mapping{suffix}.json"

    if not rewritten_path.exists():
        print(f"ERROR: rewritten JSON not found: {rewritten_path}")
        print("Run src/check/rewrite_subinstructions.py first.")
        sys.exit(1)

    with open(rewritten_path) as f:
        rewritten = json.load(f)

    landmark_mapping = {}
    if mapping_path.exists():
        with open(mapping_path) as f:
            landmark_mapping = json.load(f)
        print(f"Loaded landmark mapping: {len(landmark_mapping)} entries from {mapping_path}")
    else:
        print(f"No landmark_mapping.json found at {mapping_path}, skipping fallback.")

    episodes = episodes_from_config(cfg)
    if not episodes:
        print("No episodes matched. Exiting.")
        return

    rewritten_ids = set(rewritten["episodes"].keys())
    episodes = [ep for ep in episodes if str(ep.instruction_id) in rewritten_ids]
    if not episodes:
        print("No overlap between selected episodes and rewritten JSON. Exiting.")
        return

    print(f"Checking {len(episodes)} episode(s) across "
          f"{len({e.scan for e in episodes})} scan(s).")

    scenes_dir   = cfg["scenes"]["scenes_dir"]
    needed_scans = sorted({ep.scan for ep in episodes})
    print(f"Loading connectivity: {needed_scans}")
    db = load_connectivity(
        scenes_dir=scenes_dir,
        scans=needed_scans,
        json_dir=cfg["dataset"].get("connectivity_json_dir"),
        pkl_path=cfg["dataset"].get("connectivity_pkl"),
    )

    sensor_h = cfg["env"].get("sensor_height", 1.5)
    env_rgb = cfg["env"].get("rgb", {})
    env_depth = cfg["env"].get("depth", {})
    # Equirectangular: width = full 360° axis (4× the per-view width), height from vfov
    viz_vfov = env_rgb.get("vfov", 90)
    eq_w     = img_w * 4
    eq_h     = round(eq_w * viz_vfov / 360.0)
    rgb_cfg  = {
        "width": eq_w,
        "height": eq_h,
        "depth_width": env_depth.get("width", img_w),
        "depth_height": env_depth.get("height", img_h),
        "depth_hfov": env_depth.get("hfov", 90),
        "min_depth": env_depth.get("min_depth"),
        "max_depth": env_depth.get("max_depth"),
    } if render_obs else None
    obs_dir  = obs_root if render_obs else None
    checker  = VisibilityChecker(scenes_dir, sensor_height=sensor_h, rgb_cfg=rgb_cfg)
    t0 = time.time()

    all_results = run_landmark_uniqueness_check(
        episodes, db, rewritten, checker,
        landmark_mapping=landmark_mapping,
        obs_dir=obs_dir, img_w=img_w, img_h=img_h,
    )
    checker.close()

    all_subs   = [s for ep_d in all_results.values()
                  for s in ep_d["sub_paths"]
                  if "error" not in s and "skipped" not in s and "unique" in s]
    n_total    = len(all_subs)
    n_unique   = sum(1 for s in all_subs if s["unique"] is True)
    n_ambig    = sum(1 for s in all_subs if s.get("unique") is False
                     and s.get("visible_count", 0) > 1)
    n_novis    = sum(1 for s in all_subs if s.get("unique") is False
                     and s.get("visible_count", 0) == 0)
    n_notfound = sum(1 for s in all_subs if s.get("unique") is None)
    n_error    = sum(1 for ep_d in all_results.values()
                     for s in ep_d["sub_paths"] if "error" in s)
    ambig_cats = Counter(s["matched_category"] for s in all_subs
                         if s.get("unique") is False
                         and s.get("visible_count", 0) > 1
                         and s.get("matched_category"))

    summary = {
        "total_sub_paths": n_total,
        "unique":          n_unique,
        "ambiguous":       n_ambig,
        "not_visible":     n_novis,
        "not_found":       n_notfound,
        "errors":          n_error,
        "unique_pct":      round(n_unique / n_total, 4) if n_total else 0,
        "top_ambiguous_categories": dict(ambig_cats.most_common(10)),
    }

    print("\n=== Summary ===")
    print(f"  sub-paths  : {n_total}")
    print(f"  unique     : {n_unique}  ({summary['unique_pct']*100:.1f}%)")
    print(f"  ambiguous  : {n_ambig}")
    print(f"  not_visible: {n_novis}")
    print(f"  not_found  : {n_notfound}")
    if n_error:
        print(f"  errors     : {n_error}")
    print(f"  top ambiguous cats: {dict(ambig_cats.most_common(5))}")

    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w") as f:
        json.dump({"summary": summary, "episodes": all_results}, f, indent=2)
    print(f"\nJSON  → {json_out}")
    print(f"Time  : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
