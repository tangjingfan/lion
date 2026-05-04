"""
LION-Bench — Sub-instruction spatial rewriter  (CLI entry point).

All pipeline logic lives in src/process/rewriter.py.

Usage
-----
  GEMINI_API_KEY=your_key \\
  python src/check/rewrite_subinstructions.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml

Output
------
  results/val_unseen/sub_instructions_rewritten.json   — full per-episode rewrite
  results/val_unseen/landmark_mapping.json             — original_mention → semantic_labels
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import get_run_dir, resolve_selection
from src.dataset.landmark_rxr import episodes_from_config
from src.process.rewriter import make_client, run_rewriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rewrite Landmark-RxR sub-instructions into structured landmark guidance"
    )
    p.add_argument("--config", required=True,
                   help="Rollout YAML config (dataset / scenes / selection / output paths)")
    p.add_argument("--rewrite_config",
                   default="configs/rewrite/rewrite_subinstructions.yaml",
                   help="Rewrite-specific YAML config (model, workers, temperature, …)")
    p.add_argument("--from_yaml", default=None,
                   help="Selection YAML to override config's selection.from_yaml")
    p.add_argument("--api_key", default=None,
                   help="Gemini API key (falls back to GEMINI_API_KEY env var)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rw_cfg = {}
    rw_path = Path(args.rewrite_config)
    if rw_path.exists():
        with open(rw_path) as f:
            rw_cfg = yaml.safe_load(f) or {}

    model            = rw_cfg.get("model",            "gemini-2.0-flash")
    max_workers      = rw_cfg.get("max_workers",       4)
    temperature      = float(rw_cfg.get("temperature",      0.2))
    max_tokens       = int(rw_cfg.get("max_tokens",        4096))
    max_retries      = int(rw_cfg.get("max_retries",       3))
    retry_delay      = float(rw_cfg.get("retry_delay",     2.0))
    filter_ambiguous = bool(rw_cfg.get("filter",           False))

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Provide --api_key or set GEMINI_API_KEY environment variable.")
        sys.exit(1)

    resolve_selection(cfg, args.from_yaml)

    episodes = episodes_from_config(cfg)
    if not episodes:
        print("No episodes matched. Exiting.")
        return
    print(f"Rewriting {len(episodes)} episode(s) | model={model} | workers={max_workers} | filter={filter_ambiguous}")

    out_dir      = get_run_dir(cfg) / "rewrite"
    suffix       = "_filtered" if filter_ambiguous else ""
    json_path    = out_dir / f"sub_instructions_rewritten{suffix}.json"
    mapping_path = out_dir / f"landmark_mapping{suffix}.json"
    scenes_dir   = cfg.get("scenes", {}).get("scenes_dir", "")

    client = make_client(api_key)
    t0     = time.time()

    all_results, landmark_mapping = run_rewriter(
        episodes, client, scenes_dir,
        model=model, max_workers=max_workers,
        temperature=temperature, max_tokens=max_tokens,
        max_retries=max_retries, retry_delay=retry_delay,
        filter_ambiguous=filter_ambiguous,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({"model": model, "episodes": all_results}, f, indent=2, ensure_ascii=False)
    with open(mapping_path, "w") as f:
        json.dump(landmark_mapping, f, indent=2, ensure_ascii=False)

    n_total = sum(len(v["sub_paths"]) for v in all_results.values())
    n_err   = sum(1 for v in all_results.values()
                  for s in v["sub_paths"] if "error" in s)
    n_comp  = sum(len(s.get("components", [])) for v in all_results.values()
                  for s in v["sub_paths"])
    print(f"\n=== Summary ===")
    print(f"  episodes   : {len(all_results)}")
    print(f"  sub-paths  : {n_total}  (errors: {n_err})")
    print(f"  components : {n_comp}  ({len(landmark_mapping)} unique mentions)")
    print(f"  time       : {time.time() - t0:.1f}s")
    print(f"  output     → {json_path}")
    print(f"  mapping    → {mapping_path}")


if __name__ == "__main__":
    main()
