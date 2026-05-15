"""Stage 4 filter: drop targets whose semantic label is only a coarse bucket.

This catches cases such as a landmark mention "stove" being grounded only to
the MPCAT40 label "appliances".  The failure is semantic granularity, not
visibility: the scene taxonomy does not expose a standalone stove category.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import (
    ensure_episode,
    ensure_sub_path,
    get_filter_dir,
    get_split,
    load_audit,
    load_keep,
    register_stage,
    resolve_selection,
    save_audit,
    update_current,
    write_drop_yaml,
    write_keep_yaml,
)
from src.dataset.landmark_rxr import episodes_from_config


STAGE_NUM = 4
STAGE_NAME = "semantic_granularity"

# These MPCAT40 labels are useful for rendering, but too coarse to prove that
# a fine-grained landmark ("stove", "thermostat", "pot", "lamp") is separately
# represented by the semantic annotation.
DEFAULT_COARSE_LABELS = {
    "appliances",
    "objects",
    "furniture",
    "lighting",
}


def _norm(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("_", " ").lower()).strip()


def _label_candidates(rec: Dict) -> List[str]:
    labels: List[str] = []
    target_ids = set(rec.get("target_instance_ids") or [])
    selected_candidate_labels = [
        _norm(cand.get("category"))
        for cand in (rec.get("candidates") or [])
        if cand.get("category") and cand.get("id") in target_ids
    ]
    if selected_candidate_labels:
        labels.extend(selected_candidate_labels)
    else:
        for cand in rec.get("candidates") or []:
            if cand.get("category"):
                labels.append(_norm(cand["category"]))
        for key in ("matched_category",):
            if rec.get(key):
                labels.append(_norm(rec[key]))
        for key in ("semantic_labels", "matched_categories"):
            vals = rec.get(key) or []
            if isinstance(vals, list):
                labels.extend(_norm(v) for v in vals if v)
    seen = set()
    out: List[str] = []
    for label in labels:
        if label and label not in seen:
            out.append(label)
            seen.add(label)
    return out


def _is_only_coarse_label(landmark: str, labels: Iterable[str], coarse: set[str]) -> Tuple[bool, str]:
    """Return whether this record is grounded only to a configured coarse label."""
    lm = _norm(landmark)
    label_set = {_norm(x) for x in labels if _norm(x)}
    hit = sorted(label_set & coarse)
    if not hit:
        return False, ""

    # If a record also has a more specific semantic label, keep it.  The bad
    # case is exactly "all grounded labels are coarse buckets".
    non_coarse = sorted(label_set - coarse)
    if non_coarse:
        return False, ""

    # Mentions that literally ask for the coarse concept are rare but valid.
    coarse_forms = set(hit)
    coarse_forms.update(x[:-1] for x in hit if x.endswith("s"))
    if lm in coarse_forms:
        return False, ""

    return True, hit[0]


def _load_target_db(run_dir: Path) -> Dict[str, Dict]:
    target_root = run_dir / "target_instances"
    out: Dict[str, Dict] = {}
    if not target_root.exists():
        return out
    for scan_dir in sorted(target_root.iterdir()):
        if not scan_dir.is_dir():
            continue
        path = scan_dir / "target_instances.json"
        if not path.exists():
            continue
        with open(path) as f:
            out[scan_dir.name] = json.load(f)
    return out


def _lookup_target(target_db: Dict[str, Dict], scan: str, ep_id: int, sub_idx: int) -> Optional[Dict]:
    data = target_db.get(scan) or {}
    targets = data.get("target_instances") or {}
    ep_entry = targets.get(str(ep_id)) or targets.get(ep_id) or {}
    if not isinstance(ep_entry, dict):
        return None
    return ep_entry.get(str(sub_idx)) or ep_entry.get(sub_idx)


def _load_rescue_db(run_dir: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    root = run_dir / "target_instances"
    if not root.exists():
        return out
    for scan_dir in sorted(root.iterdir()):
        path = scan_dir / "semantic_rescue_categories.json"
        if scan_dir.is_dir() and path.exists():
            with open(path) as f:
                payload = json.load(f) or {}
            out[scan_dir.name] = payload.get("instances") or {}
    return out


def _rescue_for_sub_path(
    rescue_db: Dict[str, Dict],
    scan: str,
    ep_id: int,
    sub_idx: int,
    target_ids: Iterable[int],
) -> Optional[Dict]:
    instances = rescue_db.get(scan) or {}

    # Pixel-grounded rescue records the original (episode, sub_idx) in examples,
    # which also covers cases where the previous selector had no target id.
    for entry in instances.values():
        if not entry.get("is_rescuable"):
            continue
        for ex in entry.get("examples") or []:
            if str(ex.get("episode_id")) == str(ep_id) and str(ex.get("sub_idx")) == str(sub_idx):
                out = dict(entry)
                # YOLO-driven rescue writes ``grounding``; older VLM-driven runs
                # used ``vlm``. Accept either so we don't lose the per-example
                # category override when both producers exist on disk.
                g = ex.get("grounding") or ex.get("vlm") or {}
                if g.get("category"):
                    out["category"] = g.get("category")
                if g.get("confidence"):
                    out["confidence"] = g.get("confidence")
                if ex.get("pixel_instance", {}).get("instance_id") is not None:
                    out["instance_id"] = ex["pixel_instance"]["instance_id"]
                return out

    # Mask-based rescue is keyed by the already-selected instance id.
    for iid in target_ids:
        entry = instances.get(str(iid))
        if entry and entry.get("is_rescuable"):
            return entry
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 4: filter landmarks grounded only to coarse MPCAT40 labels",
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--from_yaml", default=None,
                    help="Selection/current YAML; defaults to filters/current.yaml.")
    ap.add_argument("--coarse_label", action="append", default=None,
                    help="MPCAT40 label to treat as too coarse. May be repeated. "
                         f"Default: {', '.join(sorted(DEFAULT_COARSE_LABELS))}")
    ap.add_argument("--report_only", action="store_true",
                    help="Print counts without writing 04_semantic_granularity.yaml.")
    ap.add_argument("--no_rescue", action="store_true",
                    help="Do not consult target_instances/{scan}/semantic_rescue_categories.json.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    resolve_selection(cfg, args.from_yaml)

    filt_dir = get_filter_dir(cfg)
    if not filt_dir.exists():
        raise SystemExit(f"No filters/ at {filt_dir} -- run prior stages first.")
    run_dir = filt_dir.parent
    split = get_split(cfg)

    prior_path = Path(args.from_yaml).expanduser() if args.from_yaml else (filt_dir / "current.yaml")
    if not prior_path.exists():
        raise SystemExit(f"No survivor YAML at {prior_path}.")
    prior_keep = load_keep(prior_path.resolve())
    prior_subs = prior_keep.get("sub_paths")
    if not prior_subs:
        raise SystemExit("This stage expects sub-path-level survivors in `sub_paths`.")
    allowed_subs = {int(ep): [int(s) for s in subs] for ep, subs in prior_subs.items()}

    resolve_selection(cfg, str(prior_path.resolve()))
    episodes = episodes_from_config(cfg)
    if not episodes:
        raise SystemExit("No episodes loaded from current.yaml.")

    coarse = {_norm(x) for x in (args.coarse_label or sorted(DEFAULT_COARSE_LABELS))}
    target_db = _load_target_db(run_dir)
    if not target_db:
        raise SystemExit(
            "No per-scan target_instances.json files under "
            f"{run_dir / 'target_instances'}.\n\n"
            "Stage 4 is a filter stage, but it must run after target-instance "
            "selection because it reads the selected target ids. Run these "
            "steps first, then rerun stage 4:\n"
            f"  bash scripts/list_scene_categories.sh --from_yaml {prior_path}\n"
            f"  GEMINI_API_KEY=... bash scripts/refine_landmark_mapping.sh --from_yaml {prior_path}\n"
            f"  bash scripts/list_target_instances.sh --from_yaml {prior_path}\n"
            f"  bash scripts/select_target_instances.sh --from_yaml {prior_path}\n"
            f"  bash scripts/filter.sh 4 --from_yaml {prior_path}"
        )
    rescue_db = {} if args.no_rescue else _load_rescue_db(run_dir)

    audit = load_audit(filt_dir, split)
    register_stage(audit, STAGE_NAME, coarse_labels=sorted(coarse))

    keep_sub_paths: Dict[int, List[int]] = {}
    dropped: Dict[str, Dict] = {}
    drop_by_label: Counter[str] = Counter()
    drop_by_landmark: Dict[str, Counter[str]] = defaultdict(Counter)
    rescue_by_category: Counter[str] = Counter()
    n_subs_total = 0
    n_subs_keep = 0
    n_missing_target = 0
    n_rescued = 0

    for ep in episodes:
        ep_id = int(ep.instruction_id)
        ep_id_str = str(ep_id)
        ep_keep: List[int] = []
        ep_drops: Dict[int, Dict] = {}
        ep_audit = ensure_episode(audit, ep)

        for sub_idx in allowed_subs.get(ep_id, []):
            n_subs_total += 1
            rec = _lookup_target(target_db, ep.scan, ep_id, sub_idx)
            sp_audit = ensure_sub_path(ep_audit, sub_idx)

            if rec is None:
                n_missing_target += 1
                payload = {"reason": "missing_target_selection"}
                sp_audit["stages"][STAGE_NAME] = {"status": "drop", **payload}
                ep_drops[sub_idx] = payload
                continue

            landmark = rec.get("landmark") or ""
            labels = _label_candidates(rec)
            is_bad, coarse_label = _is_only_coarse_label(landmark, labels, coarse)
            payload = {
                "landmark": landmark,
                "semantic_labels": labels,
            }
            if is_bad:
                target_ids = rec.get("target_instance_ids") or []
                rescue = _rescue_for_sub_path(
                    rescue_db, ep.scan, ep_id, sub_idx, target_ids
                )
                if rescue is not None:
                    payload.update({
                        "rescue_category": rescue.get("category"),
                        "rescue_confidence": rescue.get("confidence"),
                        "rescue_instance_id": rescue.get("instance_id"),
                        "coarse_label": coarse_label,
                    })
                    sp_audit["stages"][STAGE_NAME] = {"status": "ok_rescued", **payload}
                    ep_keep.append(sub_idx)
                    n_subs_keep += 1
                    n_rescued += 1
                    rescue_by_category[_norm(rescue.get("category"))] += 1
                    continue
                payload.update({
                    "reason": "coarse_semantic_label",
                    "coarse_label": coarse_label,
                    "target_instance_ids": target_ids,
                })
                sp_audit["stages"][STAGE_NAME] = {"status": "drop", **payload}
                ep_drops[sub_idx] = payload
                drop_by_label[coarse_label] += 1
                drop_by_landmark[coarse_label][_norm(landmark)] += 1
            else:
                sp_audit["stages"][STAGE_NAME] = {"status": "ok", **payload}
                ep_keep.append(sub_idx)
                n_subs_keep += 1

        ep_audit["stages"][STAGE_NAME] = {
            "status": "ok" if ep_keep else "drop",
            "kept_sub": len(ep_keep),
            "total_sub": len(allowed_subs.get(ep_id, [])),
        }
        if ep_keep:
            keep_sub_paths[ep_id] = ep_keep
        if ep_drops:
            dropped[ep_id_str] = {
                "scan": ep.scan,
                "subs": {str(k): v for k, v in sorted(ep_drops.items())},
            }

    n_drop = n_subs_total - n_subs_keep
    print(f"=== Stage {STAGE_NUM} -- {STAGE_NAME} ===")
    print(f"  coarse labels : {', '.join(sorted(coarse))}")
    print(f"  episodes in   : {len(episodes)}")
    print(f"  episodes keep : {len(keep_sub_paths)}")
    print(f"  sub-paths in  : {n_subs_total}")
    print(f"  sub-paths keep: {n_subs_keep}")
    print(f"  sub-paths drop: {n_drop}")
    if n_missing_target:
        print(f"    missing target selection: {n_missing_target}")
    if n_rescued:
        print(f"  sub-paths rescued: {n_rescued}")
        common = ", ".join(f"{cat}={n}" for cat, n in rescue_by_category.most_common(8))
        print(f"    rescue categories: {common}")
    if drop_by_label:
        print("  coarse-label drops:")
        for label, count in drop_by_label.most_common():
            common = ", ".join(f"{lm}={n}" for lm, n in drop_by_landmark[label].most_common(8))
            print(f"    {label}: {count}  ({common})")

    if args.report_only:
        return

    keep_path = write_keep_yaml(
        filt_dir, STAGE_NUM, STAGE_NAME, split,
        instruction_ids=sorted(keep_sub_paths.keys()),
        sub_paths=keep_sub_paths,
        cfg=cfg,
    )
    drop_path = write_drop_yaml(
        filt_dir, STAGE_NUM, STAGE_NAME, split,
        dropped=dict(sorted(dropped.items(), key=lambda kv: int(kv[0]))),
        extras={"coarse_labels": sorted(coarse)},
    )
    save_audit(audit, filt_dir)
    current_path = update_current(filt_dir, keep_path)

    print()
    print("Outputs:")
    print(f"  {keep_path}")
    print(f"  {drop_path}")
    print(f"  {current_path}  ->  {keep_path.name}")
    print(f"  {filt_dir / 'audit.json'}")


if __name__ == "__main__":
    main()
