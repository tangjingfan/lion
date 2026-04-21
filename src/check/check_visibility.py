"""
LION-Bench — Sub-path visibility check  (CLI entry point).

All pipeline logic lives in src/process/visibility.py :: run_visibility_check().

Usage
-----
  python src/check/check_visibility.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml

Output
------
  results/val_unseen/visibility.json
  results/val_unseen/visibility/{instruction_id}/sub_NN.png
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
from src.process.visibility import VisibilityChecker, run_visibility_check


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LION-Bench sub-path visibility check")
    p.add_argument("--config", required=True, help="Path to rollout YAML config")
    p.add_argument("--from_yaml", default=None,
                   help="Selection YAML to override config's selection.from_yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.from_yaml:
        cfg.setdefault("selection", {})["from_yaml"] = args.from_yaml

    episodes = episodes_from_config(cfg)
    if not episodes:
        print("No episodes matched. Exiting.")
        return

    vis_cfg  = cfg.get("visibility", {})
    do_viz   = vis_cfg.get("viz", True)

    print(f"Checking {len(episodes)} episode(s) across "
          f"{len({e.scan for e in episodes})} scan(s).  "
          f"viz={'on' if do_viz else 'off'}")

    scenes_dir   = cfg["scenes"]["scenes_dir"]
    needed_scans = sorted({ep.scan for ep in episodes})
    print(f"Loading connectivity: {needed_scans}")
    db = load_connectivity(
        scenes_dir=scenes_dir,
        scans=needed_scans,
        json_dir=cfg["dataset"].get("connectivity_json_dir"),
        pkl_path=cfg["dataset"].get("connectivity_pkl"),
    )

    out_cfg  = cfg.get("output", {})
    base_dir = Path(out_cfg.get("base_dir", "results")).expanduser()
    run_name = (out_cfg.get("run_name")
                or Path(cfg["dataset"]["data_path"]).stem.replace("LandmarkRxR_", ""))
    out_dir   = base_dir / run_name
    json_path = out_dir / "visibility.json"
    viz_root  = out_dir / "visibility"

    sensor_h = cfg["env"].get("sensor_height", 1.5)
    env_rgb  = cfg["env"].get("rgb", {})
    env_depth = cfg["env"].get("depth", {})
    rgb_cfg  = {
        "width":  env_rgb.get("width",  256),
        "height": env_rgb.get("height", env_rgb.get("width", 256)),
        "hfov":   90,
        "depth_width": env_depth.get("width", env_rgb.get("width", 256)),
        "depth_height": env_depth.get("height", env_depth.get("width", 256)),
        "depth_hfov": env_depth.get("hfov", 90),
        "min_depth": env_depth.get("min_depth"),
        "max_depth": env_depth.get("max_depth"),
    } if do_viz else None

    checker = VisibilityChecker(scenes_dir, sensor_height=sensor_h, rgb_cfg=rgb_cfg)
    t0 = time.time()

    all_results = run_visibility_check(
        episodes, db, checker,
        viz_root=viz_root if rgb_cfg else None,
        rgb_cfg=rgb_cfg,
    )
    checker.close()

    all_sub   = [r for ep_d in all_results.values()
                 for r in ep_d["sub_paths"] if "error" not in r]
    n_total   = len(all_sub)
    n_visible = sum(1 for r in all_sub if r["visible"])
    cat_counts = Counter(r["obstacle"]["semantic_cat"]
                         for r in all_sub if not r["visible"])
    summary = {
        "total_sub_paths":         n_total,
        "visible":                 n_visible,
        "blocked":                 n_total - n_visible,
        "visible_pct":             round(n_visible / n_total, 4) if n_total else 0,
        "top_obstacle_categories": dict(cat_counts.most_common(10)),
    }
    print("\n=== Summary ===")
    print(f"  sub-paths : {n_total}")
    print(f"  visible   : {n_visible}  ({summary['visible_pct']*100:.1f}%)")
    print(f"  blocked   : {summary['blocked']}")
    print(f"  obstacles : {dict(cat_counts.most_common(5))}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "episodes": all_results}, f, indent=2)
    print(f"\nJSON  → {json_path}")
    if rgb_cfg:
        print(f"PNGs  → {viz_root}/{{instruction_id}}/sub_NN.png")
    print(f"Time  : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
