"""Pipeline attrition report — how many sub-trajectories survived each stage.

Reads the per-experiment ``filters/audit.json`` (cross-stage trace),
the per-stage ``filters/NN_*_dropped.yaml`` (drop reasons), and the
final ``dataset.json`` (post-rescue usability), then prints a funnel
showing how many episodes / sub-paths survived each step and why the
rest got dropped.

Usage
-----
  python src/check/attrition.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --exp configs/selection/val_unseen/one_scene_partial.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import get_filter_dir, get_run_dir, resolve_exp


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> object:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _stage_episode_drops(
    audit: dict, stage_name: str,
) -> Tuple[Set[str], Counter]:
    """For an episode-level filter stage, return the dropped episode-id
    set and a Counter over drop reasons (best-effort)."""
    dropped: Set[str] = set()
    reasons: Counter = Counter()
    for ep_id, ep in audit.get("episodes", {}).items():
        st = ep.get("stages", {}).get(stage_name, {})
        if st.get("status") == "drop":
            dropped.add(ep_id)
            # Pick the most likely "reason" field present.
            for key in ("reason", "y_range_m", "kept_sub"):
                if key in st:
                    reasons[f"{key}={st[key]}"] += 1
                    break
            else:
                reasons["drop"] += 1
    return dropped, reasons


def _sub_drop_reasons(dropped_yaml: dict) -> Tuple[int, Counter]:
    """Sum sub-path-level drops + their reason histogram from a
    NN_*_dropped.yaml payload."""
    total = 0
    reasons: Counter = Counter()
    for _ep, info in (dropped_yaml.get("dropped") or {}).items():
        subs = (info or {}).get("subs") or {}
        for _sub_idx, rec in subs.items():
            reasons[(rec or {}).get("reason", "unknown")] += 1
            total += 1
    return total, reasons


def _episode_alive_after(
    audit: dict, stage_name: str, prior_dead: Set[str],
) -> Set[str]:
    """Episodes still alive after a given sub-path-level stage —
    i.e. they have ≥1 sub_path marked 'ok' under that stage."""
    alive: Set[str] = set()
    for ep_id, ep in audit.get("episodes", {}).items():
        if ep_id in prior_dead:
            continue
        kept = sum(
            1 for sp in (ep.get("sub_paths") or {}).values()
            if (sp.get("stages") or {}).get(stage_name, {}).get("status") == "ok"
        )
        if kept > 0:
            alive.add(ep_id)
    return alive


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Print a per-stage attrition report for one experiment.",
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", default=None,
                    help="Experiment handle (selection YAML path or expname).")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # apply_current=False — we want absolute totals across the whole run,
    # not just whatever survivor.yaml currently happens to point at.
    resolve_exp(cfg, args.exp, apply_current=False)

    run_dir  = get_run_dir(cfg)
    filt_dir = get_filter_dir(cfg)
    audit    = _load_json(filt_dir / "audit.json") or {}
    if not audit:
        raise SystemExit(f"No audit.json under {filt_dir}.")

    dataset  = _load_json(run_dir / "dataset.json") or []

    drop_cross   = _load_yaml(filt_dir / "01_cross_floor_dropped.yaml")
    drop_black   = _load_yaml(filt_dir / "02_blacklist_dropped.yaml")
    drop_part    = _load_yaml(filt_dir / "03_partition_dropped.yaml")

    # ── Stage 00 totals ────────────────────────────────────────────
    n_ep_0  = len(audit.get("episodes", {}))
    n_sub_0 = sum(int(ep.get("n_sub_paths") or 0) for ep in audit["episodes"].values())

    # ── Stage 01 cross_floor (episode-level) ───────────────────────
    cf_dead, cf_reasons = _stage_episode_drops(audit, "cross_floor")
    n_ep_1  = n_ep_0 - len(cf_dead)
    n_sub_1 = sum(
        int(ep.get("n_sub_paths") or 0)
        for eid, ep in audit["episodes"].items()
        if eid not in cf_dead
    )

    # ── Stage 03 blacklist_landmark (sub-path level) ───────────────
    bl_dropped_n, bl_reasons = _sub_drop_reasons(drop_black)
    n_sub_2 = n_sub_1 - bl_dropped_n
    bl_alive = _episode_alive_after(audit, "blacklist", cf_dead)
    n_ep_2 = len(bl_alive)

    # ── Stage 04 partition (sub-path level) ────────────────────────
    pt_dropped_n, pt_reasons = _sub_drop_reasons(drop_part)
    n_sub_3 = n_sub_2 - pt_dropped_n
    pt_alive = _episode_alive_after(audit, "partition", cf_dead | (bl_alive ^ set(audit["episodes"].keys())))
    # Above is a bit loose; lean on what we already know: any episode
    # that lost all its blacklist survivors also has no partition
    # survivors. The exact count we want is "episodes with ≥1 sub_path
    # still ok after partition".
    pt_alive = _episode_alive_after(audit, "partition", cf_dead)
    n_ep_3 = len(pt_alive)

    # ── Final dataset.json snapshot ────────────────────────────────
    final_status: Counter = Counter()
    final_source: Counter = Counter()
    n_with_target  = 0
    n_rescued      = 0
    n_synthesized  = 0
    for r in dataset:
        final_status[r.get("target_status") or "unknown"] += 1
        final_source[r.get("landmark_source") or "unknown"] += 1
        if r.get("target_instance_ids"):
            n_with_target += 1
        if r.get("target_status") == "rescued":
            n_rescued += 1
        if r.get("landmark_source") == "synthesized":
            n_synthesized += 1
    n_final_records = len(dataset)
    n_final_eps     = len({r.get("instruction_id") for r in dataset})

    # ── Print ──────────────────────────────────────────────────────
    pct = lambda x, base: (100.0 * x / base) if base else 0.0

    print(f"\n=== Pipeline attrition: {run_dir.name} ===\n")
    print(f"{'stage':<28} {'ep':>5} {'sub':>5}  {'Δsub':>6}  reasons")
    print("-" * 96)
    print(f"{'00 original (seed)':<28} {n_ep_0:>5} {n_sub_0:>5}  {'─':>6}")
    print(
        f"{'01 cross_floor':<28} {n_ep_1:>5} {n_sub_1:>5}  "
        f"{n_sub_1 - n_sub_0:+6d}  "
        f"{len(cf_dead)} episode(s) cross-floor; "
        f"sub-paths of those episodes counted as dropped"
    )
    bl_txt = ", ".join(f"{k}={v}" for k, v in bl_reasons.most_common()) or "(none)"
    print(
        f"{'03 blacklist_landmark':<28} {n_ep_2:>5} {n_sub_2:>5}  "
        f"{n_sub_2 - n_sub_1:+6d}  {bl_txt}"
    )
    pt_txt = ", ".join(f"{k}={v}" for k, v in pt_reasons.most_common()) or "(none)"
    print(
        f"{'04 partition':<28} {n_ep_3:>5} {n_sub_3:>5}  "
        f"{n_sub_3 - n_sub_2:+6d}  {pt_txt}"
    )
    print("-" * 96)
    print(
        f"{'(records in dataset.json)':<28} {n_final_eps:>5} {n_final_records:>5}"
    )

    print()
    print("Post-pipeline target status (records exist but may be unusable):")
    print(f"  with target_instance_ids : {n_with_target}/{n_final_records}")
    print(f"  rescued from not_visible : {n_rescued}")
    print(f"  synthesized landmarks    : {n_synthesized}  (originally blacklisted, replacement found via step 13)")
    print(f"  status histogram         :")
    for status, count in final_status.most_common():
        print(f"    {status:<25s} {count}")
    print(f"  landmark_source histogram:")
    for src, count in final_source.most_common():
        print(f"    {src:<25s} {count}")

    print()
    print("=== overall ===")
    print(f"  starting sub-paths       : {n_sub_0}")
    print(
        f"  filtered out             : {n_sub_0 - n_final_records}  "
        f"({pct(n_sub_0 - n_final_records, n_sub_0):.1f}%)"
    )
    print(
        f"  in dataset.json          : {n_final_records}  "
        f"({pct(n_final_records, n_sub_0):.1f}%)"
    )
    print(
        f"  with usable target id    : {n_with_target}  "
        f"({pct(n_with_target, n_sub_0):.1f}%)"
    )


if __name__ == "__main__":
    main()
