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

    # Aggregate per-scan rewrite + mapping files into the cross-scan
    # ``rewritten`` / ``landmark_mapping`` shapes consumed downstream.
    if not rewrite_dir.exists():
        print(f"ERROR: rewrite dir not found: {rewrite_dir}")
        print("Run src/check/rewrite_subinstructions.py first.")
        sys.exit(1)
    scan_dirs = [d for d in sorted(rewrite_dir.iterdir()) if d.is_dir()]

    chosen_suffix = None
    for cand in ("_filtered", ""):
        if any((d / f"sub_instructions_rewritten{cand}.json").exists()
               for d in scan_dirs):
            chosen_suffix = cand
            break
    if chosen_suffix is None:
        print(f"ERROR: no sub_instructions_rewritten[_filtered].json under {rewrite_dir}/*/")
        sys.exit(1)

    rewritten = {"episodes": {}}
    landmark_mapping: dict = {}
    n_map_loaded = 0
    for scan_dir in scan_dirs:
        rw = scan_dir / f"sub_instructions_rewritten{chosen_suffix}.json"
        lm = scan_dir / f"landmark_mapping{chosen_suffix}.json"
        if rw.exists():
            with open(rw) as f:
                rewritten["episodes"].update(json.load(f).get("episodes", {}))
        if lm.exists():
            with open(lm) as f:
                landmark_mapping[scan_dir.name] = json.load(f) or {}
            n_map_loaded += 1
    print(f"Loaded rewrite from {len(scan_dirs)} per-scan dir(s) "
          f"(suffix={chosen_suffix!r})")
    print(f"Loaded landmark_mapping{chosen_suffix}.json for "
          f"{n_map_loaded}/{len(scan_dirs)} scans")

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

    # VisibilityChecker now follows the rollout's rgbds_agent setup; the sim
    # always has rgb + semantic equirectangular sensors at eye height.
    # ``render_obs`` only toggles whether per-sub-path PNGs are saved.
    obs_dir = obs_root if render_obs else None
    checker = VisibilityChecker(cfg["env"], scenes_dir)
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
