"""
LION-Bench — Visualize sub-path partitions on the MP3D connectivity graph.

For each episode we:
  • compute partition indices per sub-path (see src/process/partition.py)
  • render a top-down PNG showing the full scan's unobstructed graph with
    the episode's path overlaid, sub-paths colour-coded, and partition
    points marked.

Usage
-----
  python src/check/visualize_partition.py \\
      --config           configs/rollout/rollout_landmark_rxr.yaml \\
      --partition_config configs/partition/partition.yaml

Output
------
  results/val_unseen/partition/{instruction_id}.png
  results/val_unseen/partition/partition.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dataset.landmark_rxr import episodes_from_config
from src.env.connectivity import _mp3d_to_habitat, load_connectivity
from src.process.partition import partition_episode


# ── Adjacency loader (for drawing the scan's connectivity graph) ──────────
def load_adjacency(
    json_dir: str, scan: str
) -> List[Tuple[str, str]]:
    """Return list of (node_id_a, node_id_b) unobstructed pairs, undirected."""
    p = Path(json_dir) / f"{scan}_connectivity.json"
    if not p.exists():
        return []
    with open(p) as f:
        nodes = json.load(f)

    included = [n for n in nodes if n.get("included", True)]
    id_of = {i: n["image_id"] for i, n in enumerate(included)}
    idx_of = {n["image_id"]: i for i, n in enumerate(included)}

    edges: set = set()
    for i, node in enumerate(included):
        unob = node.get("unobstructed", [])
        for j, v in enumerate(unob):
            if not v:
                continue
            if j not in id_of:
                continue
            a, b = node["image_id"], id_of[j]
            if a == b:
                continue
            edges.add((a, b) if a < b else (b, a))
    return sorted(edges)


# ── Drawing ───────────────────────────────────────────────────────────────
#
# Layout per episode: a small overview strip at the top + one zoomed panel
# per sub-path in a grid below.  In each zoomed panel:
#   • nodes = numbered circles (0, 1, 2, …, K)
#   • orange fill  = spatial-portion node  (before/at the partition)
#   • teal fill    = landmark-portion node (at/after the partition)
#   • the partition node is outlined with a thick black ring
#   • every edge is drawn as an arrow from node_k → node_{k+1}
#   • a short black arrow on node 0 shows the agent's starting heading
#
_COLOUR_SPATIAL  = "#f28e2b"    # orange
_COLOUR_LANDMARK = "#2fa4a4"    # teal
_COLOUR_ARROW    = "#35363a"    # dark grey for edge arrows


def _xy(pos: np.ndarray) -> Tuple[float, float]:
    """Top-down plot coordinates: (x, −z) so north (−Z in Habitat) points up."""
    return float(pos[0]), float(-pos[2])


def _heading_arrow(ax, x: float, y: float, heading: float,
                   length: float, color: str, lw: float = 1.8) -> None:
    """Arrow in the heading direction (clockwise from north)."""
    dx = length * math.sin(heading)
    dy = length * math.cos(heading)
    ax.annotate(
        "", xy=(x + dx, y + dy), xytext=(x, y),
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                        shrinkA=0, shrinkB=0),
    )


def _draw_tape(ax_tape, K: int, p_idx: int) -> None:
    """Action tape: numbered boxes for each node, partition split visible."""
    ax_tape.set_xlim(-0.5, K + 0.5)
    ax_tape.set_ylim(-0.8, 0.8)
    ax_tape.axis("off")

    # Region labels
    if p_idx >= 0:
        ax_tape.text(p_idx / 2.0, 0.75, "SPATIAL",
                     fontsize=7, color=_COLOUR_SPATIAL, fontweight="bold",
                     ha="center", va="center")
    if p_idx < K:
        ax_tape.text((p_idx + K) / 2.0, 0.75, "LANDMARK",
                     fontsize=7, color=_COLOUR_LANDMARK, fontweight="bold",
                     ha="center", va="center")

    # Connector
    ax_tape.plot([0, K], [0, 0], color="#bbb", lw=1.2, zorder=1)

    # Node boxes
    for i in range(K + 1):
        role_spatial = i <= p_idx
        fc = _COLOUR_SPATIAL if role_spatial else _COLOUR_LANDMARK
        ec = "black" if i == p_idx else "white"
        lw = 2.2   if i == p_idx else 1.0
        ax_tape.scatter([i], [0], s=280, facecolor=fc,
                        edgecolor=ec, lw=lw, zorder=2)
        ax_tape.text(i, 0, str(i), ha="center", va="center",
                     fontsize=8, fontweight="bold", color="white", zorder=3)

    # Partition callout
    if 0 <= p_idx <= K:
        ax_tape.annotate("partition", xy=(p_idx, -0.25),
                         xytext=(p_idx, -0.7),
                         fontsize=7, color="black", ha="center", va="top",
                         arrowprops=dict(arrowstyle="-", color="black", lw=0.8))


def _draw_subpath_map(ax, part: Dict, scan_db: Dict[str, np.ndarray]) -> None:
    """Zoomed top-down 2-D map of a single sub-path."""
    nodes  = part["sub_path_nodes"]
    p_idx  = part["partition_idx"]
    coords = np.array([_xy(scan_db[n]) for n in nodes])
    K      = len(nodes) - 1

    # Edge arrows (dest colored by role of destination node); distance + Δ°
    deltas = part.get("turn_deltas") or []
    for k in range(K):
        x0, y0 = coords[k]
        x1, y1 = coords[k + 1]
        dest_is_spatial = (k + 1) <= p_idx
        ec = _COLOUR_SPATIAL if dest_is_spatial else _COLOUR_LANDMARK
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=ec, lw=2.6,
                            shrinkA=14, shrinkB=14),
            zorder=3,
        )
        dist = float(np.hypot(x1 - x0, y1 - y0))
        d_deg = math.degrees(deltas[k]) if k < len(deltas) else 0.0
        mx = 0.5 * (x0 + x1); my = 0.5 * (y0 + y1)
        ax.text(mx, my, f"{dist:.1f} m\nΔ{d_deg:+.0f}°",
                fontsize=7, color="#333", zorder=6,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#ccc", lw=0.6, alpha=0.9))

    # Node markers: numbered circles, filled by role
    for i, (x, y) in enumerate(coords):
        role_spatial = i <= p_idx
        fc = _COLOUR_SPATIAL if role_spatial else _COLOUR_LANDMARK
        ec = "black" if i == p_idx else "white"
        lw = 2.8   if i == p_idx else 1.4
        ax.scatter([x], [y], s=640, facecolor=fc,
                   edgecolor=ec, lw=lw, zorder=4)
        ax.text(x, y, str(i), ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=5)

    # Start-heading arrow at node 0 (labeled with absolute heading in degrees)
    x0, y0 = coords[0]
    span = max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), 1.0)
    arr_len = 0.28 * span
    _heading_arrow(ax, x0, y0, part["start_heading"],
                   length=arr_len, color="black", lw=2.0)
    h_deg = math.degrees(part["start_heading"]) % 360
    ax.annotate(f"θ₀={h_deg:.0f}°", xy=(x0, y0),
                xytext=(-14, -18), textcoords="offset points",
                fontsize=7, color="black",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="#888", lw=0.5))

    # Landmark target label at the last node (if any)
    lm = (part.get("landmark") or "").strip()
    if lm:
        xK, yK = coords[-1]
        ax.annotate(f"→ {lm}", xy=(xK, yK),
                    xytext=(16, 10), textcoords="offset points",
                    fontsize=8, color=_COLOUR_LANDMARK, fontweight="bold",
                    arrowprops=dict(arrowstyle="->",
                                    color=_COLOUR_LANDMARK, lw=1.2))

    ax.set_aspect("equal")
    ax.grid(True, lw=0.3, alpha=0.35)
    ax.tick_params(labelsize=7)
    pad = max(0.8, 0.20 * span)
    ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)


def _draw_subpath_panel(fig, outer_gs_cell, part: Dict,
                        scan_db: Dict[str, np.ndarray]) -> None:
    """Composite panel: tape strip on top + 2-D map below."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    inner = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs_cell,
        height_ratios=[0.9, 5.0], hspace=0.15,
    )
    ax_tape = fig.add_subplot(inner[0])
    ax_map  = fig.add_subplot(inner[1])

    K     = len(part["sub_path_nodes"]) - 1
    p_idx = part["partition_idx"]
    _draw_tape(ax_tape, K, p_idx)
    _draw_subpath_map(ax_map, part, scan_db)

    kind    = part["kind"]
    spatial = (part.get("spatial_instruction") or "").strip()
    lm      = part.get("landmark") or ""
    title   = f"[{part['sub_idx']}] {spatial}"
    if lm:
        title += f"  →  {lm}"
    title += f"      p={p_idx}/{K}  ·  {kind}"
    ax_tape.set_title(title, fontsize=9, pad=4)


def _draw_overview_panel(ax, episode, scan_db: Dict[str, np.ndarray],
                         partitions: List[Dict]) -> None:
    """Overview: full episode path, numbered sub-paths on episode bbox."""
    all_coords: List[Tuple[float, float]] = []
    for part in partitions:
        if "error" in part:
            continue
        coords = [_xy(scan_db[n]) for n in part["sub_path_nodes"]]
        all_coords.extend(coords)
        xs, ys = zip(*coords)
        ax.plot(xs, ys, color="#888", lw=1.6, zorder=2)
        # sub-path index near start node
        sx, sy = coords[0]
        ax.text(sx, sy, f"[{part['sub_idx']}]", fontsize=8,
                fontweight="bold", color="#333",
                zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="#888", lw=0.6))

    # Start / goal
    sp = _xy(scan_db[episode.path[0]])
    gp = _xy(scan_db[episode.path[-1]])
    ax.scatter(*sp, s=110, marker="o", facecolor="#2ca02c",
               edgecolor="black", lw=1.0, zorder=5)
    ax.scatter(*gp, s=130, marker="X", facecolor="#d62728",
               edgecolor="black", lw=1.0, zorder=5)
    ax.text(sp[0], sp[1], " start", fontsize=8, va="center", color="#2ca02c")
    ax.text(gp[0], gp[1], " goal",  fontsize=8, va="center", color="#d62728")

    ax.set_aspect("equal")
    ax.set_title("overview", fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.35)
    ax.tick_params(labelsize=7)
    if all_coords:
        xs, ys = zip(*all_coords)
        pad = max(0.8, 0.12 * max(max(xs) - min(xs), max(ys) - min(ys)))
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)


def draw_episode(
    episode,
    scan_db: Dict[str, np.ndarray],
    adjacency: List[Tuple[str, str]],   # unused in the new layout (kept for API)
    partitions: List[Dict],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines  import Line2D

    valid = [p for p in partitions if "error" not in p]
    n_sub = len(valid)
    # 2 panels per row for sub-paths
    cols = 2
    rows_sub = max(1, math.ceil(n_sub / cols))

    fig = plt.figure(figsize=(cols * 5.0, 3.6 + rows_sub * 4.2), dpi=120)
    # Top row: overview; below: grid of sub-path panels (each with tape + map)
    gs = fig.add_gridspec(rows_sub + 1, cols, height_ratios=[1.0] + [1.4] * rows_sub,
                          hspace=0.55, wspace=0.30)

    # Overview spans the whole top row
    ax_ov = fig.add_subplot(gs[0, :])
    _draw_overview_panel(ax_ov, episode, scan_db, partitions)

    # Sub-path panels — each cell gets a tape + map via nested gridspec
    for i, part in enumerate(valid):
        r = 1 + i // cols
        c = i % cols
        _draw_subpath_panel(fig, gs[r, c], part, scan_db)

    # Shared legend (single row across the top)
    legend_handles = [
        Patch(facecolor=_COLOUR_SPATIAL,  edgecolor="white", label="spatial node"),
        Patch(facecolor=_COLOUR_LANDMARK, edgecolor="white", label="landmark node"),
        Line2D([0], [0], marker="o", color="white",
               markerfacecolor="white", markeredgecolor="black",
               markeredgewidth=2.4, markersize=10, label="partition node (black ring)"),
        Line2D([0], [0], marker=r"$\rightarrow$", color="black",
               markersize=12, lw=0, label="heading at start"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.995))

    fig.suptitle(f"scan={episode.scan}    instr_id={episode.instruction_id}    "
                 f"{len(episode.sub_paths)} sub-path(s)",
                 fontsize=11, y=0.965)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize sub-path partitions")
    p.add_argument("--config",    required=True)
    p.add_argument("--partition_config",
                   default="configs/partition/partition.yaml",
                   help="YAML with partition hyper-parameters (turn thresholds, forward distance)")
    p.add_argument("--from_yaml", default=None)
    p.add_argument("--limit",     type=int, default=None,
                   help="Cap the number of episodes rendered")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.from_yaml:
        cfg.setdefault("selection", {})["from_yaml"] = args.from_yaml

    part_cfg_path = Path(args.partition_config)
    if part_cfg_path.exists():
        with open(part_cfg_path) as f:
            part_cfg = yaml.safe_load(f) or {}
        print(f"Loaded partition config: {part_cfg_path}  →  {part_cfg}")
    else:
        part_cfg = {}
        print(f"Partition config {part_cfg_path} not found; using built-in defaults.")
    part_kwargs = {
        k: part_cfg[k]
        for k in ("turn_thresh_deg", "around_thresh_deg", "forward_distance_m")
        if k in part_cfg
    }

    episodes = episodes_from_config(cfg)
    if not episodes:
        print("No episodes matched. Exiting.")
        return

    out_cfg  = cfg.get("output", {})
    base_dir = Path(out_cfg.get("base_dir", "results")).expanduser()
    run_name = (out_cfg.get("run_name")
                or Path(cfg["dataset"]["data_path"]).stem.replace("LandmarkRxR_", ""))
    out_dir  = base_dir / run_name
    part_dir = out_dir / "partition"

    # Load rewritten sub-instructions (spatial instructions come from here)
    uniq_cfg       = cfg.get("uniqueness", {})
    rewrite_dir    = out_dir / "rewrite"
    rewritten_path = Path(uniq_cfg["rewritten_path"]) if uniq_cfg.get("rewritten_path") \
                     else rewrite_dir / "sub_instructions_rewritten.json"
    if not rewritten_path.exists():
        filtered = rewrite_dir / "sub_instructions_rewritten_filtered.json"
        if filtered.exists():
            rewritten_path = filtered
    rewritten: Optional[Dict] = None
    if rewritten_path.exists():
        print(f"Loading rewritten instructions: {rewritten_path}")
        with open(rewritten_path) as f:
            rewritten = json.load(f)
        rewritten_ids = set(rewritten["episodes"].keys())
        episodes = [ep for ep in episodes
                    if str(ep.instruction_id) in rewritten_ids]
        print(f"Filtered to {len(episodes)} episode(s) present in rewritten JSON.")
    else:
        print("No rewritten JSON found; treating all sub-paths as forward.")

    if args.limit:
        episodes = episodes[:args.limit]
    if not episodes:
        print("No episodes left after filtering. Exiting.")
        return

    scenes_dir   = cfg["scenes"]["scenes_dir"]
    json_dir     = cfg["dataset"].get("connectivity_json_dir")
    needed_scans = sorted({ep.scan for ep in episodes})
    db = load_connectivity(
        scenes_dir=scenes_dir,
        scans=needed_scans,
        json_dir=json_dir,
        pkl_path=cfg["dataset"].get("connectivity_pkl"),
    )

    adjacency_cache: Dict[str, List[Tuple[str, str]]] = {}

    all_results: Dict[str, Dict] = {}
    for idx, ep in enumerate(episodes, 1):
        scan_db = db.get(ep.scan, {})
        if ep.scan not in adjacency_cache:
            adjacency_cache[ep.scan] = (
                load_adjacency(json_dir, ep.scan) if json_dir else []
            )
        ep_rewritten = None
        if rewritten is not None:
            ep_rewritten = rewritten.get("episodes", {}).get(str(ep.instruction_id))

        partitions = partition_episode(ep, scan_db, ep_rewritten, **part_kwargs)
        all_results[str(ep.instruction_id)] = {
            "scan":       ep.scan,
            "partitions": [
                {k: (v if k != "sub_path_nodes" else list(v))
                 for k, v in p.items()
                 if k not in {"edge_headings", "edge_lengths", "turn_deltas"}}
                for p in partitions
            ],
        }

        out_path = part_dir / f"{ep.instruction_id}.png"
        try:
            draw_episode(ep, scan_db, adjacency_cache[ep.scan],
                         partitions, out_path)
            print(f"  [{idx}/{len(episodes)}] ep={ep.instruction_id}  "
                  f"scan={ep.scan}  → {out_path.name}")
        except Exception as exc:
            print(f"  [{idx}/{len(episodes)}] ep={ep.instruction_id}  FAIL: {exc}")

    part_dir.mkdir(parents=True, exist_ok=True)
    with open(part_dir / "partition.json", "w") as f:
        json.dump({"episodes": all_results}, f, indent=2)
    print(f"\nPNGs  → {part_dir}/{{instruction_id}}.png")
    print(f"JSON  → {part_dir}/partition.json")


if __name__ == "__main__":
    main()
