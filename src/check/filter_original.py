"""
LION-Bench — Stage 0: record original selected episodes / sub-paths.

This stage does not filter anything.  It materializes the selection YAML into
the filter framework so later stages have a clear baseline:

  00_original.yaml          — all selected instruction_ids and all sub_idx
  00_original_dropped.yaml  — empty dropped set
  audit.json               — initial per-episode/sub-path counts
  current.yaml             — initialized to 00_original.yaml only when missing

Usage
-----
  python src/check/filter_original.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --from_yaml configs/selection/one_scene_partial_val_unseen.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import (
    ensure_episode,
    ensure_sub_path,
    get_filter_dir,
    get_split,
    load_audit,
    register_stage,
    resolve_selection,
    save_audit,
    update_current,
    write_drop_yaml,
    write_keep_yaml,
)
from src.dataset.landmark_rxr import episodes_from_config


STAGE_NUM = 0
STAGE_NAME = "original"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 0: record original selected episodes and sub-paths",
    )
    ap.add_argument("--config", required=True,
                    help="Rollout / dataset YAML config")
    ap.add_argument("--from_yaml", default=None,
                    help="Selection YAML to materialize")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    resolve_selection(cfg, args.from_yaml)

    episodes = episodes_from_config(cfg)
    if not episodes:
        print("No episodes matched. Exiting.")
        return

    filt_dir = get_filter_dir(cfg)
    filt_dir.mkdir(parents=True, exist_ok=True)
    split = get_split(cfg)

    audit = load_audit(filt_dir, split)
    register_stage(audit, STAGE_NAME)

    instruction_ids: List[int] = []
    sub_paths: Dict[int, List[int]] = {}
    n_sub_paths = 0

    for ep in episodes:
        instruction_ids.append(ep.instruction_id)
        ep_sub_idxs = list(range(len(ep.sub_paths)))
        sub_paths[ep.instruction_id] = ep_sub_idxs
        n_sub_paths += len(ep_sub_idxs)

        ep_audit = ensure_episode(audit, ep)
        ep_audit["stages"][STAGE_NAME] = {
            "status": "ok",
            "n_sub_paths": len(ep_sub_idxs),
        }
        for sub_idx in ep_sub_idxs:
            sp_audit = ensure_sub_path(ep_audit, sub_idx)
            sp_audit["stages"][STAGE_NAME] = {"status": "ok"}

    keep_path = write_keep_yaml(
        filt_dir,
        STAGE_NUM,
        STAGE_NAME,
        split,
        instruction_ids=instruction_ids,
        sub_paths=sub_paths,
        cfg=cfg,
    )
    drop_path = write_drop_yaml(
        filt_dir,
        STAGE_NUM,
        STAGE_NAME,
        split,
        dropped={},
        extras={"note": "Stage 0 records the original selected set; no drops."},
    )
    save_audit(audit, filt_dir)
    existing_current = filt_dir / "current.yaml"
    if existing_current.exists() or existing_current.is_symlink():
        current_path = existing_current
        current_note = "(left unchanged)"
    else:
        current_path = update_current(filt_dir, keep_path)
        current_note = f"->  {keep_path.name}"

    print(f"=== Stage {STAGE_NUM} — {STAGE_NAME} ===")
    print(f"  episodes       : {len(episodes)}")
    print(f"  sub-paths      : {n_sub_paths}")
    print(f"  avg sub/episode: {n_sub_paths / len(episodes):.2f}")
    print()
    print("Outputs:")
    print(f"  {keep_path}")
    print(f"  {drop_path}")
    print(f"  {current_path}  {current_note}")
    print(f"  {filt_dir / 'audit.json'}")


if __name__ == "__main__":
    main()
