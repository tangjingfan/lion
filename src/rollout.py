"""
LION-Bench — Landmark-RxR rollout entry point.

Usage
-----
  python src/rollout.py --config configs/rollout/rollout_landmark_rxr.yaml

  # Run specific instruction_ids only:
  python src/rollout.py --config configs/rollout/rollout_landmark_rxr.yaml \
      --instruction_ids 19199 19200

  # Restrict to two scenes:
  python src/rollout.py --config configs/rollout/rollout_landmark_rxr.yaml \
      --scenes X7HyMhZNoso S9hNv5qa7GM

  # Replay a previous run exactly:
  python src/rollout.py --config configs/rollout/rollout_landmark_rxr.yaml \
        --from_yaml results/landmark_rxr_20260411_143022/viz/replay.yaml

  # Use a pre-canned selection file (instruction_ids / scenes / languages / max_episodes):
  python src/rollout.py --config configs/rollout/rollout_landmark_rxr.yaml \
      --selection configs/selection/example.yaml

Merge order: rollout config ← --selection YAML ← CLI flags.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# Make sure src/ is on the path when called from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.base_agent import BaseAgent, build_agent
from src.dataset.landmark_rxr import LandmarkRxREpisode, episodes_from_config
from src.env.connectivity import load_connectivity
from src.env.habitat_env import HabitatEnv
from src.metrics import aggregate_metrics, compute_episode_metrics
from src.viz import EpisodeVisualizer


# ---------------------------------------------------------------------------
#  Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def apply_selection_file(cfg: dict, selection_path: str) -> dict:
    """Merge a ``configs/selection/*.yaml`` file into ``cfg`` in place.

    Schema mirrors the files already in ``configs/selection/``:

        split: val_unseen           # informational, ignored by loader
        scans: [X7HyMhZNoso, ...]   # → cfg["scenes"]["include"]
        languages: [en-US]          # → cfg["selection"]["languages"]
        instruction_ids: [19199]    # → cfg["selection"]["instruction_ids"]
        max_episodes: 10            # → cfg["selection"]["max_episodes"]

    Empty lists / null values are treated as "don't restrict" and leave
    the parent config untouched — mirrors the comment convention in the
    existing selection files.
    """
    with open(selection_path) as f:
        sel_yaml = yaml.safe_load(f) or {}

    selection = cfg.setdefault("selection", {})
    for key in ("instruction_ids", "languages"):
        val = sel_yaml.get(key)
        if val:  # non-empty list
            selection[key] = val
    if sel_yaml.get("max_episodes") is not None:
        selection["max_episodes"] = sel_yaml["max_episodes"]

    scans = sel_yaml.get("scans")
    if scans:
        cfg.setdefault("scenes", {})["include"] = scans

    return cfg


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Patch cfg in-place with any non-None CLI arguments."""
    sel = cfg.setdefault("selection", {})
    scene_cfg = cfg.setdefault("scenes", {})

    if args.instruction_ids:
        sel["instruction_ids"] = [int(i) for i in args.instruction_ids]
    if args.from_yaml:
        sel["from_yaml"] = args.from_yaml
    if args.scenes:
        scene_cfg["include"] = list(args.scenes)
    if args.max_episodes is not None:
        sel["max_episodes"] = args.max_episodes
    if args.output_dir:
        cfg.setdefault("output", {})["base_dir"] = args.output_dir
    if args.run_name:
        cfg.setdefault("output", {})["run_name"] = args.run_name

    return cfg


def make_output_dir(cfg: dict) -> Path:
    out_cfg = cfg.get("output", {})
    base = Path(out_cfg.get("base_dir", "results")).expanduser()
    if out_cfg.get("run_name"):
        run_name = out_cfg["run_name"]
    else:
        # derive from the dataset split filename, e.g. "LandmarkRxR_val_unseen" → "val_unseen"
        data_stem = Path(cfg["dataset"]["data_path"]).stem  # e.g. LandmarkRxR_val_unseen
        run_name = data_stem.replace("LandmarkRxR_", "")
    out_dir = base / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
#  Rollout loop
# ---------------------------------------------------------------------------

def run_rollout(
    episodes: List[LandmarkRxREpisode],
    env: HabitatEnv,
    agent: BaseAgent,
    cfg: dict,
    out_dir: Path,
) -> Dict[str, Any]:
    """Run all episodes and return per-episode and aggregate metrics."""
    out_cfg = cfg.get("output", {})
    success_dist = cfg["env"].get("success_distance", 3.0)

    viz_cfg = out_cfg.get("viz", {})
    viz_enabled = viz_cfg.get("enabled", False)
    viz_dir = out_dir / "rollout_viz"
    viz = EpisodeVisualizer(viz_dir, viz_cfg) if viz_enabled else None

    per_episode_metrics: Dict[str, Dict] = {}
    per_episode_results: Dict[str, Dict] = {}

    total = len(episodes)
    for idx, episode in enumerate(episodes):
        print(f"[{idx+1}/{total}]  instr_id={episode.instruction_id}"
              f"  scan={episode.scan}  lang={episode.language}")

        # --- reset ---
        obs = env.reset(episode)
        # OracleAgent (and any future agent with set_env) needs the pathfinder
        # re-bound after each reset because reconfigure() replaces it per scene.
        if hasattr(agent, "set_env"):
            agent.set_env(env)
        agent.reset(episode)
        if viz:
            viz.on_reset(episode, obs)

        agent_positions: List[np.ndarray] = [obs["position"].copy()]

        # --- episode loop ---
        done = False
        info: Dict[str, Any] = {}
        while not done:
            action = agent.step(obs)
            obs, done, info = env.step(action)
            agent_positions.append(obs["position"].copy())
            if viz:
                viz.on_step(action, obs, done, metrics=info if done else None)

        # --- metrics ---
        ep_metrics = compute_episode_metrics(
            agent_positions=agent_positions,
            episode_reference_path=episode.reference_path,
            goal_position=episode.goal_position,
            success_distance=success_dist,
        )
        ep_metrics["steps"] = info.get("steps", len(agent_positions) - 1)

        if viz:
            viz.on_episode_end(ep_metrics)

        per_episode_metrics[episode.path_key] = ep_metrics
        per_episode_results[episode.path_key] = {
            "instruction_id": episode.instruction_id,
            "path_id": episode.path_id,
            "scan": episode.scan,
            "language": episode.language,
            "instruction": episode.instruction,
            **ep_metrics,
        }

        sr_str = "OK" if ep_metrics["sr"] else "FAIL"
        print(f"  → [{sr_str}]  NE={ep_metrics['ne']:.2f}m"
              f"  SPL={ep_metrics['spl']:.3f}"
              f"  NDTW={ep_metrics['ndtw']:.3f}"
              f"  steps={ep_metrics['steps']}")

    # --- aggregate ---
    agg = aggregate_metrics(per_episode_metrics)
    print("\n=== Aggregate metrics ===")
    for k, v in agg.items():
        if k == "num_episodes":
            print(f"  {k}: {int(v)}")
        else:
            print(f"  {k}: {v:.4f}")

    # --- save outputs ---
    if out_cfg.get("save_json", True):
        results_path = viz_dir / "results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(
                {
                    "aggregate": agg,
                    "episodes": per_episode_results,
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to: {results_path}")

    if out_cfg.get("save_replay_yaml", True):
        replay = {
            "split": Path(cfg["dataset"]["data_path"]).stem,
            "scans": sorted({ep.scan for ep in episodes}),
            "languages": sorted({ep.language for ep in episodes}),
            "instruction_ids": [ep.instruction_id for ep in episodes],
        }
        replay_path = viz_dir / "replay.yaml"
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        with open(replay_path, "w") as f:
            yaml.dump(replay, f, default_flow_style=False, sort_keys=False)
        print(f"Replay YAML saved to: {replay_path}")

    # Save the effective config used for this run
    viz_dir.mkdir(parents=True, exist_ok=True)
    with open(viz_dir / "config_used.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return {"aggregate": agg, "episodes": per_episode_results}


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LION-Bench Landmark-RxR rollout")
    p.add_argument("--config", required=True,
                   help="Path to rollout YAML config")

    # Selection overrides
    p.add_argument("--instruction_ids", nargs="+", type=int, default=None,
                   help="Override: specific instruction_ids to run")
    p.add_argument("--scenes", nargs="+", default=None,
                   help="Override: restrict to these scan IDs")
    p.add_argument("--from_yaml", default=None,
                   help="Override: load instruction_ids from replay YAML")
    p.add_argument("--selection", default=None,
                   help="Path to a selection YAML (configs/selection/*.yaml) "
                        "that sets instruction_ids / scenes / languages / max_episodes")
    p.add_argument("--max_episodes", type=int, default=None,
                   help="Override: cap total number of episodes")

    # Output overrides
    p.add_argument("--output_dir", default=None,
                   help="Override: output base directory")
    p.add_argument("--run_name", default=None,
                   help="Override: run name (subdirectory under output_dir)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.selection:
        cfg = apply_selection_file(cfg, args.selection)
    cfg = apply_cli_overrides(cfg, args)

    out_dir = make_output_dir(cfg)
    print(f"Output directory: {out_dir}")

    # Determine which scans to pre-load (avoids loading all MP3D scans)
    episodes = episodes_from_config(cfg)
    if not episodes:
        print("No episodes matched the filters. Exiting.")
        return

    needed_scans = sorted({ep.scan for ep in episodes})
    print(f"Scenes to evaluate ({len(needed_scans)}): {needed_scans}")

    # --- load connectivity graph from .house files (no extra download needed) ---
    print("Loading connectivity from .house files …")
    scenes_dir = cfg["scenes"]["scenes_dir"]
    db = load_connectivity(
        scenes_dir=scenes_dir,
        scans=needed_scans,
        json_dir=cfg["dataset"].get("connectivity_json_dir"),
        pkl_path=cfg["dataset"].get("connectivity_pkl"),
    )

    # --- build env + agent ---
    env = HabitatEnv(cfg["env"], db)
    env.set_scenes_dir(cfg["scenes"]["scenes_dir"])

    agent = build_agent(cfg.get("agent", {}))

    # --- run ---
    t0 = time.time()
    results = run_rollout(episodes, env, agent, cfg, out_dir)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s  ({elapsed/len(episodes):.1f}s per episode)")

    env.close()


if __name__ == "__main__":
    main()
