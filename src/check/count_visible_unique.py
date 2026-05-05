"""
Count how many visible landmarks are unique.

Reads per-scan visibility annotations under
``{run_dir}/landmark_visibility/{scan}/visibility.json`` (produced by
``annotate_visibility.py``) and reports, among records with
``status == "visible"``, how many have exactly one visible instance.

Usage
-----
  python src/check/count_visible_unique.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --from_yaml configs/selection/one_scene_partial_val_unseen.yaml

  # Or aim at a specific scan's file directly:
  python src/check/count_visible_unique.py \\
      --visibility_json results/<run>/landmark_visibility/<scan>/visibility.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import get_run_dir, resolve_selection


def _iter_records(
    data: Dict[str, Any], scan: str = ""
) -> Iterable[Tuple[str, str, str, Dict[str, Any]]]:
    annotations = data.get("annotations", {})
    s = scan or data.get("scan") or ""
    for ep_id, ep_records in annotations.items():
        if not isinstance(ep_records, dict):
            continue
        for sub_idx, record in ep_records.items():
            if isinstance(record, dict):
                yield str(s), str(ep_id), str(sub_idx), record


def _pct(n: int, d: int) -> str:
    return f"{(n / d):.1%}" if d else "0.0%"


def _gather_visibility_paths(
    args: argparse.Namespace,
) -> Tuple[List[Path], Path]:
    """Resolve which per-scan visibility JSON file(s) to read.

    Returns ``(paths, root)`` where ``root`` is shown in the header.
    """
    if args.visibility_json:
        p = Path(args.visibility_json)
        if not p.exists():
            raise SystemExit(f"Visibility JSON not found: {p}")
        return [p], p

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    resolve_selection(cfg, args.from_yaml)
    root = get_run_dir(cfg) / "landmark_visibility"
    if not root.exists():
        raise SystemExit(f"No landmark_visibility/ at {root}")
    paths = sorted(root.glob("*/visibility.json"))
    if not paths:
        raise SystemExit(
            f"No per-scan visibility files matched {root}/*/visibility.json"
        )
    return paths, root


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Count unique landmarks among visible visibility annotations.",
    )
    ap.add_argument("--config", default="configs/rollout/rollout_landmark_rxr.yaml",
                    help="Rollout YAML; used with --from_yaml to find run dir.")
    ap.add_argument("--from_yaml", default=None,
                    help="Selection YAML carrying expname / run_name.")
    ap.add_argument("--visibility_json", default=None,
                    help="Direct path to a single scan's visibility.json.")
    ap.add_argument("--print_non_unique", action="store_true",
                    help="Print visible records with n_instances != 1.")
    args = ap.parse_args()

    vis_paths, root = _gather_visibility_paths(args)

    status_counts: Counter = Counter()
    visible_total = 0
    unique_total = 0
    non_unique_total = 0
    no_instance_total = 0
    by_category: Dict[str, Counter] = defaultdict(Counter)
    non_unique = []
    by_scan_visible: Counter = Counter()

    for vis_path in vis_paths:
        with open(vis_path) as f:
            data = json.load(f)
        scan = data.get("scan") or vis_path.parent.name
        for s, ep_id, sub_idx, record in _iter_records(data, scan):
            status = record.get("status", "unknown")
            status_counts[status] += 1
            if status != "visible":
                continue

            visible_total += 1
            by_scan_visible[s] += 1
            n_instances = int(record.get("n_instances") or 0)
            category = record.get("matched_category") or "unknown"
            if n_instances == 1:
                unique_total += 1
                by_category[category]["unique"] += 1
            else:
                non_unique_total += 1
                by_category[category]["non_unique"] += 1
                non_unique.append((s, ep_id, sub_idx, record))
                if n_instances == 0:
                    no_instance_total += 1

    print("=== Visible Landmark Uniqueness ===")
    print(f"source          : {root}")
    print(f"files read      : {len(vis_paths)}")
    print(f"annotated       : {sum(status_counts.values())}")
    print("status breakdown:")
    for status, n in status_counts.most_common():
        print(f"  {status:<24s} {n:>5d}  ({_pct(n, sum(status_counts.values()))})")

    print()
    print(f"visible         : {visible_total}")
    print(f"unique visible  : {unique_total}  ({_pct(unique_total, visible_total)})")
    print(f"non-unique      : {non_unique_total}  ({_pct(non_unique_total, visible_total)})")
    if no_instance_total:
        print(f"visible with 0 instances: {no_instance_total}")

    if len(by_scan_visible) > 1:
        print()
        print("by scan (visible only):")
        for s, n in by_scan_visible.most_common():
            print(f"  {s:<20s} visible={n}")

    if by_category:
        print()
        print("by matched_category:")
        for cat, counts in sorted(
            by_category.items(),
            key=lambda kv: -(kv[1]["unique"] + kv[1]["non_unique"]),
        ):
            total = counts["unique"] + counts["non_unique"]
            print(
                f"  {cat:<24s} total={total:>4d}  "
                f"unique={counts['unique']:>4d}  "
                f"non_unique={counts['non_unique']:>4d}"
            )

    if args.print_non_unique and non_unique:
        print()
        print("visible but non-unique:")
        for s, ep_id, sub_idx, record in non_unique:
            print(
                f"  scan={s:<14s} ep={ep_id:<8s} sub={sub_idx:<3s} "
                f"n={int(record.get('n_instances') or 0):<3d} "
                f"cat={record.get('matched_category')!r:<16s} "
                f"landmark={record.get('landmark')!r}"
            )


if __name__ == "__main__":
    main()
