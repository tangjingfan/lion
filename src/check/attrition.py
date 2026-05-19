"""Pipeline attrition report — read from the lifecycle audit.

Reads ``{run_dir}/filters/audit.json`` (the single source of truth for
per-(ep, sub) events emitted at every stage from 00 record_original
through 12 consolidate) and prints:

  • a funnel showing how many subs entered / exited each stage and
    where they went (kept / dropped / rescued / synthesized);
  • a verdict histogram (final ``verdict`` per sub);
  • a breakdown of rescue activity at stages 09/10/11.

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
from typing import Dict, List, Optional, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import get_filter_dir, get_run_dir, resolve_exp


# Ordered list of stages we expect to see in the audit. Stages that
# haven't run yet simply have zero rows in their summary.
STAGE_ORDER = [
    "original",
    "cross_floor",
    "blacklist",
    "partition",
    "visibility",
    "select",
    "detection",
    "apply_rescue",
    "rescue_blacklist",
    "consolidate",
]


def _load_audit(filt_dir: Path) -> dict:
    p = filt_dir / "audit.json"
    if not p.exists():
        raise SystemExit(f"No audit.json under {filt_dir}.")
    with open(p) as f:
        return json.load(f) or {}


def _stage_breakdown(audit: dict, stage: str) -> Tuple[int, int, Counter]:
    """Return (n_touched, n_dropped_or_failed, reason_counter) for a stage.

    Walks both episode-level and sub-level events. ``n_touched`` is the
    total number of subs (or eps, for episode-only stages like
    cross_floor) the stage emitted at least one event for. The reason
    counter groups dropped / rescue_failed / excluded events.
    """
    touched = 0
    bad = 0
    reasons: Counter = Counter()
    for ep_id, ep in audit.get("episodes", {}).items():
        # Episode-level events (e.g. cross_floor dropping a whole ep).
        for e in ep.get("events", []):
            if e.get("stage") != stage:
                continue
            touched += 1
            act = e.get("action")
            if act == "dropped":
                bad += 1
                reasons[e.get("reason") or "drop"] += 1
        for _sub_idx, sp in (ep.get("sub_paths") or {}).items():
            for e in sp.get("events", []):
                if e.get("stage") != stage:
                    continue
                touched += 1
                act = e.get("action")
                if act in ("dropped", "excluded"):
                    bad += 1
                    reasons[e.get("reason") or act] += 1
                elif act == "rescue_failed":
                    bad += 1
                    reasons[f"rescue_failed:{e.get('reason') or '?'}"] += 1
                elif act == "labeled":
                    # not "bad" per se, but the visibility tag is the
                    # most actionable label — count any non-visible state.
                    v = e.get("visibility")
                    if v and v != "visible":
                        reasons[f"visibility:{v}"] += 1
    return touched, bad, reasons


def _verdict_histogram(audit: dict) -> Counter:
    out: Counter = Counter()
    for ep in audit.get("episodes", {}).values():
        for sp in (ep.get("sub_paths") or {}).values():
            out[sp.get("verdict") or "(unknown)"] += 1
    return out


def _rescue_breakdown(audit: dict) -> Dict[str, Counter]:
    """Per-stage rescue stats: how many subs got rescued / synthesized /
    failed at each of detection / rescue_blacklist."""
    out: Dict[str, Counter] = {
        "detection":        Counter(),
        "rescue_blacklist": Counter(),
    }
    for ep in audit.get("episodes", {}).values():
        for sp in (ep.get("sub_paths") or {}).values():
            for e in sp.get("events", []):
                stage = e.get("stage")
                act = e.get("action")
                if stage in out and act in ("rescued", "synthesized", "rescue_failed"):
                    key = act
                    if act == "rescue_failed":
                        key = f"rescue_failed:{e.get('reason') or '?'}"
                    out[stage][key] += 1
    return out


def _in_dataset_count(audit: dict) -> Tuple[int, int]:
    total = 0
    in_ds = 0
    for ep in audit.get("episodes", {}).values():
        for sp in (ep.get("sub_paths") or {}).values():
            total += 1
            if sp.get("in_dataset"):
                in_ds += 1
    return total, in_ds


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Print a per-stage attrition report from the lifecycle audit.",
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
    audit    = _load_audit(filt_dir)

    n_eps    = len(audit.get("episodes", {}))
    total_subs, in_ds = _in_dataset_count(audit)

    print(f"\n=== Lifecycle attrition: {run_dir.name} ===\n")
    print(f"{'stage':<22} {'touched':>8} {'bad':>6}  details")
    print("-" * 96)
    for stage in STAGE_ORDER:
        touched, bad, reasons = _stage_breakdown(audit, stage)
        if touched == 0:
            continue
        detail = ", ".join(f"{k}={v}" for k, v in reasons.most_common(6)) or "—"
        print(f"{stage:<22} {touched:>8} {bad:>6}  {detail}")
    print("-" * 96)
    print(f"{'TOTAL':<22} {total_subs:>8} {'':>6}  episodes={n_eps}")

    rescue = _rescue_breakdown(audit)
    print()
    print("Rescue stages:")
    for stage in ("detection", "rescue_blacklist"):
        c = rescue[stage]
        if not c:
            continue
        line = ", ".join(f"{k}={v}" for k, v in c.most_common())
        print(f"  {stage:<20s} {line}")

    print()
    print("Final verdicts (per sub-path):")
    verdicts = _verdict_histogram(audit)
    for v, n in verdicts.most_common():
        print(f"  {v:<40s} {n}")

    pct = lambda x, base: (100.0 * x / base) if base else 0.0
    print()
    print("=== overall ===")
    print(f"  sub-paths total          : {total_subs}")
    print(f"  in dataset.json          : {in_ds}  ({pct(in_ds, total_subs):.1f}%)")
    print(f"  filtered out             : {total_subs - in_ds}  ({pct(total_subs - in_ds, total_subs):.1f}%)")


if __name__ == "__main__":
    main()
