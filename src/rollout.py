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

from src.agent.base_agent import BaseAgent, OracleAgent, build_agent
from src.check._filter_utils import (
    apply_selection_yaml,
    get_run_dir,
    resolve_selection,
)
from src.dataset.landmark_rxr import LandmarkRxREpisode, episodes_from_config
from src.env.connectivity import load_connectivity
from src.env.habitat_env import HabitatEnv, STOP
from src.metrics import aggregate_metrics, compute_episode_metrics
from src.viz import EpisodeVisualizer


# ---------------------------------------------------------------------------
#  Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


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
    out_dir = get_run_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
#  Rollout loop
# ---------------------------------------------------------------------------

def _sub_path_end_indices(episode: LandmarkRxREpisode) -> List[int]:
    """Return path-node indices that mark the END of each sub-path.

    Sub-paths are stored as overlapping node-id slices of ``episode.path``
    (the last node of sub-path k equals the first node of sub-path k+1),
    so each sub-path of length N contributes N-1 edges.  Walking those
    edge counts along the full path gives the index of the last node of
    each sub-path.
    """
    indices: List[int] = []
    cursor = 0
    for sub in episode.sub_paths:
        cursor += max(0, len(sub) - 1)
        indices.append(cursor)
    return indices


def _node_to_sub_idx(end_indices: List[int]) -> List[int]:
    """Map path-node index → sub-path index.

    A node belongs to the FIRST sub-path whose end-index is ≥ the node
    index.  Boundary nodes (where two sub-paths meet) belong to the
    sub-path that ends at them, so when the OracleAgent is heading TO
    a boundary node it is still finishing the current sub-path.
    """
    if not end_indices:
        return []
    total = end_indices[-1] + 1
    out = [0] * total
    cursor = 0
    for node_idx in range(total):
        while cursor < len(end_indices) - 1 and node_idx > end_indices[cursor]:
            cursor += 1
        out[node_idx] = cursor
    return out


def _drive_oracle(
    env: HabitatEnv,
    agent: OracleAgent,
    viz,
    end_indices: List[int],
    max_steps_per_wp: int,
    agent_positions: List[np.ndarray],
) -> Dict[str, Any]:
    """CL_CoTNav-style per-waypoint rollout for OracleAgent.

    Outer loop iterates over reference-path waypoints (skipping the start).
    Inner loop drives ``follower.get_next_action(wp)`` until the follower
    signals reached (``None`` / STOP) or ``max_steps_per_wp`` is hit.
    Sub-folder index is derived directly from the waypoint's path-node
    position via ``_node_to_sub_idx``.
    """
    follower = agent.follower
    waypoints = agent.waypoints
    if follower is None or not waypoints:
        # No path to follow — terminate immediately.
        _, _, info = env.step(STOP)
        return info

    node_to_sub = _node_to_sub_idx(end_indices)
    n_sub = max(1, len(end_indices))

    done = False
    info: Dict[str, Any] = {}
    for wp_local_i, wp_pos in enumerate(waypoints):
        if done:
            break
        node_idx = wp_local_i + 1
        sub_idx = node_to_sub[node_idx] if node_idx < len(node_to_sub) else n_sub - 1
        is_final_wp = (wp_local_i == len(waypoints) - 1)
        wp_arr = np.asarray(wp_pos, dtype=np.float32)

        steps_this_wp = 0
        while not done:
            action = follower.get_next_action(wp_arr)
            if action is None or int(action) == STOP:
                if is_final_wp:
                    obs, done, info = env.step(STOP)
                    agent_positions.append(obs["position"].copy())
                    if viz:
                        viz.on_step(STOP, obs, done,
                                    metrics=info if done else None,
                                    sub_idx=sub_idx)
                break

            action_int = int(action)
            obs, done, info = env.step(action_int)
            agent_positions.append(obs["position"].copy())
            if viz:
                viz.on_step(action_int, obs, done,
                            metrics=info if done else None,
                            sub_idx=sub_idx)
            steps_this_wp += 1
            if steps_this_wp >= max_steps_per_wp:
                break

    # If we exhausted waypoints without ever sending STOP (e.g. every wp
    # hit its safety cap), terminate now so metrics get a finite distance.
    if not done:
        obs, done, info = env.step(STOP)
        agent_positions.append(obs["position"].copy())
        if viz:
            viz.on_step(STOP, obs, done,
                        metrics=info, sub_idx=n_sub - 1)
    return info


def _drive_generic(
    env: HabitatEnv,
    agent: BaseAgent,
    viz,
    end_indices: List[int],
    ref_path: List[np.ndarray],
    sub_advance_radius: float,
    obs: Dict[str, Any],
    agent_positions: List[np.ndarray],
) -> Dict[str, Any]:
    """Fallback rollout for non-Oracle agents (Dummy, future real agents).

    Uses the original obs→action→env.step loop, with position-based
    sub-path advancement.
    """
    n_sub = len(end_indices)
    sub_idx = 0
    done = False
    info: Dict[str, Any] = {}
    while not done:
        action = agent.step(obs)
        obs, done, info = env.step(action)
        agent_positions.append(obs["position"].copy())

        while sub_idx < n_sub - 1:
            end_node = ref_path[end_indices[sub_idx]]
            if np.linalg.norm(obs["position"] - end_node) < sub_advance_radius:
                sub_idx += 1
            else:
                break

        if viz:
            viz.on_step(action, obs, done,
                        metrics=info if done else None,
                        sub_idx=sub_idx)
    return info


def run_rollout(
    episodes: List[LandmarkRxREpisode],
    env: HabitatEnv,
    agent: BaseAgent,
    cfg: dict,
    out_dir: Path,
) -> Dict[str, Any]:
    """Run all episodes and return per-episode and aggregate metrics."""
    out_cfg = cfg.get("output", {})
    env_cfg = cfg.get("env", {})
    success_dist = env_cfg.get("success_distance", 3.0)
    max_steps_per_wp = int(env_cfg.get("max_steps_per_wp", 100))

    viz_cfg = out_cfg.get("viz", {})
    viz_enabled = viz_cfg.get("enabled", False)
    viz_dir = out_dir / "rollout_viz"
    viz = EpisodeVisualizer(viz_dir, viz_cfg) if viz_enabled else None
    # Distance (m) within which non-Oracle agents are considered to have
    # reached a sub-path boundary node.  OracleAgent ignores this — it
    # derives sub_idx directly from its current waypoint index.
    sub_advance_radius = float(viz_cfg.get("sub_advance_radius", 0.5))

    per_episode_metrics: Dict[str, Dict] = {}
    per_episode_results: Dict[str, Dict] = {}

    total = len(episodes)
    for idx, episode in enumerate(episodes):
        print(f"[{idx+1}/{total}]  instr_id={episode.instruction_id}"
              f"  scan={episode.scan}  lang={episode.language}")

        # --- reset ---
        obs = env.reset(episode)
        # OracleAgent (and any future agent with set_env) needs the follower
        # rebuilt after each reset because reconfigure() replaces the sim's
        # pathfinder per scene.
        if hasattr(agent, "set_env"):
            agent.set_env(env)
        agent.reset(episode)
        if viz:
            viz.on_reset(episode, obs)

        agent_positions: List[np.ndarray] = [obs["position"].copy()]

        end_indices = _sub_path_end_indices(episode)
        ref_path = [np.asarray(p, dtype=np.float32) for p in episode.reference_path]

        # --- episode loop ---
        if isinstance(agent, OracleAgent):
            info = _drive_oracle(
                env=env, agent=agent, viz=viz,
                end_indices=end_indices,
                max_steps_per_wp=max_steps_per_wp,
                agent_positions=agent_positions,
            )
        else:
            info = _drive_generic(
                env=env, agent=agent, viz=viz,
                end_indices=end_indices,
                ref_path=ref_path,
                sub_advance_radius=sub_advance_radius,
                obs=obs,
                agent_positions=agent_positions,
            )

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

    # --- save outputs (per-scan) ---
    eps_by_scan: Dict[str, List[LandmarkRxREpisode]] = {}
    for ep in episodes:
        eps_by_scan.setdefault(ep.scan, []).append(ep)

    save_json   = out_cfg.get("save_json", True)
    save_replay = out_cfg.get("save_replay_yaml", True)

    for scan, scan_eps in eps_by_scan.items():
        scan_dir = viz_dir / scan
        scan_dir.mkdir(parents=True, exist_ok=True)

        scan_results: Dict[str, Dict] = {}
        scan_metrics: Dict[str, Dict] = {}
        for ep in scan_eps:
            key = ep.path_key
            if key in per_episode_results:
                scan_results[key] = per_episode_results[key]
            if key in per_episode_metrics:
                scan_metrics[key] = per_episode_metrics[key]

        if save_json:
            results_path = scan_dir / "results.json"
            with open(results_path, "w") as f:
                json.dump(
                    {
                        "aggregate": aggregate_metrics(scan_metrics),
                        "episodes":  scan_results,
                    },
                    f,
                    indent=2,
                )
            print(f"  [{scan}] results → {results_path}")

        if save_replay:
            replay = {
                "split": Path(cfg["dataset"]["data_path"]).stem,
                "scans": [scan],
                "languages": sorted({ep.language for ep in scan_eps}),
                "instruction_ids": [ep.instruction_id for ep in scan_eps],
            }
            replay_path = scan_dir / "replay.yaml"
            with open(replay_path, "w") as f:
                yaml.dump(replay, f, default_flow_style=False, sort_keys=False)
            print(f"  [{scan}] replay  → {replay_path}")

        # Save the effective config used for this run (one copy per scan).
        with open(scan_dir / "config_used.yaml", "w") as f:
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
        apply_selection_yaml(cfg, args.selection)
    cfg = apply_cli_overrides(cfg, args)
    # Eagerly merge whichever selection ended up active (CLI --from_yaml or
    # cfg.selection.from_yaml from the YAML) so cfg.output.expname etc. are
    # populated before make_output_dir reads them.
    resolve_selection(cfg)

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

    agent = build_agent(cfg)

    # --- run ---
    t0 = time.time()
    results = run_rollout(episodes, env, agent, cfg, out_dir)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s  ({elapsed/len(episodes):.1f}s per episode)")

    env.close()


if __name__ == "__main__":
    main()
