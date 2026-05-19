"""
LION-Bench — Stage 0: record original selected episodes / sub-paths.

This stage does not filter anything.  It materializes the selection YAML into
the filter framework so later stages have a clear baseline:

  00_original.yaml          — all selected instruction_ids and all sub_idx
  00_original_dropped.yaml  — empty dropped set
  audit.json               — initial per-episode/sub-path counts
  survivor.yaml            — initialized only when missing (never clobbers
                             an in-progress pipeline's narrower state)

Usage
-----
  python src/check/filter_original.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --exp configs/selection/val_unseen/one_scene_partial.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import (
    append_ep_event,
    append_sub_event,
    ensure_episode,
    finalize_audit,
    get_filter_dir,
    get_split,
    get_survivor_path,
    load_audit,
    register_stage,
    resolve_exp,
    save_audit,
    strip_stage_events,
    write_drop_yaml,
    write_survivor,
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
    ap.add_argument("--exp", default=None,
                    help="Selection YAML to materialize as the seed.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    resolve_exp(cfg, args.exp, apply_current=False)

    episodes = episodes_from_config(cfg)
    if not episodes:
        print("No episodes matched. Exiting.")
        return

    filt_dir = get_filter_dir(cfg)
    filt_dir.mkdir(parents=True, exist_ok=True)
    split = get_split(cfg)

    audit = load_audit(filt_dir, split)
    register_stage(audit, STAGE_NAME)
    strip_stage_events(audit, STAGE_NAME)

    instruction_ids: List[int] = []
    sub_paths: Dict[int, List[int]] = {}
    n_sub_paths = 0

    for ep in episodes:
        instruction_ids.append(ep.instruction_id)
        ep_sub_idxs = list(range(len(ep.sub_paths)))
        sub_paths[ep.instruction_id] = ep_sub_idxs
        n_sub_paths += len(ep_sub_idxs)

        ep_audit = ensure_episode(audit, ep)
        append_ep_event(
            ep_audit, stage=STAGE_NAME, action="kept",
            n_sub_paths=len(ep_sub_idxs),
        )
        for sub_idx in ep_sub_idxs:
            append_sub_event(ep_audit, sub_idx, stage=STAGE_NAME, action="kept")

    # Stage 0 only writes survivor.yaml when none exists yet — never
    # clobber an in-progress pipeline's narrower state.
    survivor_path = get_survivor_path(cfg)
    if survivor_path.exists():
        survivor_note = "(left unchanged)"
    else:
        write_survivor(
            cfg, split,
            instruction_ids=instruction_ids,
            sub_paths=sub_paths,
        )
        survivor_note = "(initialized)"
    drop_path = write_drop_yaml(
        filt_dir,
        STAGE_NUM,
        STAGE_NAME,
        split,
        dropped={},
        extras={"note": "Stage 0 records the original selected set; no drops."},
    )
    finalize_audit(audit)
    save_audit(audit, filt_dir)

    print(f"=== Stage {STAGE_NUM} — {STAGE_NAME} ===")
    print(f"  episodes       : {len(episodes)}")
    print(f"  sub-paths      : {n_sub_paths}")
    print(f"  avg sub/episode: {n_sub_paths / len(episodes):.2f}")
    print()
    print("Outputs:")
    print(f"  {survivor_path}  {survivor_note}")
    print(f"  {drop_path}")
    print(f"  {filt_dir / 'audit.json'}")


if __name__ == "__main__":
    main()
