"""Consolidate per-sub-path info into a single dataset file.

Reads every survivor (scan, instruction_id, sub_idx) from
``survivor.yaml`` and assembles one record per sub-trajectory pulling
together:

  • text       — full instruction + per-sub split (landmark / spatial)
  • geometry   — sub_path_nodes / spatial_path / landmark_path / heading
                  / partition kind / direction_mismatch
  • target     — target_instance_ids + status + matched semantic category
                  + landmark_visible_at_partition
  • rescue     — landmark / category / grounding method (when applicable)
  • pointers   — last rollout frame, partition viz PNG, etc.

Inputs are written by earlier stages — this script does not call the
LLM, the simulator, or YOLO. It is a pure aggregation.

Output: ``results/{run}/dataset.json`` — top-level JSON list of records.

Usage
-----
  python src/check/consolidate_dataset.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --exp configs/selection/val_unseen/one_scene_partial.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import (
    append_sub_event,
    discover_rewrite_suffix,
    ensure_episode,
    finalize_audit,
    get_filter_dir,
    get_run_dir,
    get_split,
    load_audit,
    register_stage,
    resolve_exp,
    save_audit,
    strip_stage_events,
    sub_status_for,
)


STAGE_NAME = "consolidate"
from src.dataset.landmark_rxr import episodes_from_config


def _load_partition(run_dir: Path, scan: str, ep_id: int) -> Optional[Dict]:
    p = run_dir / "partition" / scan / str(ep_id) / "partition.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _partition_for_sub(part_json: Dict, sub_idx: int) -> Optional[Dict]:
    for part in (part_json or {}).get("partitions", []):
        if int(part.get("sub_idx", -1)) == sub_idx:
            return part
    return None


def _load_rewrite(run_dir: Path, scan: str, ep_id: int, suffix: str) -> Optional[Dict]:
    p = (
        run_dir
        / "rewrite"
        / scan
        / str(ep_id)
        / f"sub_instructions_rewritten{suffix}.json"
    )
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f) or {}
    return data.get("episode")


def _rewrite_for_sub(rewrite_episode: Dict, sub_idx: int) -> Optional[Dict]:
    for sp in (rewrite_episode or {}).get("sub_paths", []):
        if int(sp.get("sub_idx", -1)) == sub_idx:
            return sp
    return None


def _load_target_db(run_dir: Path, scan: str) -> Optional[Dict]:
    p = run_dir / "target_instances" / scan / "target_instances.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _target_for_sub(target_db: Dict, ep_id: int, sub_idx: int) -> Optional[Dict]:
    """Look up the per-(ep, sub) record. Prefers the new ``annotations``
    section (single source of truth after the step 07/08/10 merge);
    falls back to the legacy ``target_instances`` section if present."""
    db = target_db or {}
    for section in ("annotations", "target_instances"):
        sub = (
            (db.get(section) or {})
            .get(str(ep_id), {})
            .get(str(sub_idx))
        )
        if sub:
            return sub
    return None


def _load_blacklist_rescue(run_dir: Path, scan: str) -> Dict:
    p = run_dir / "target_instances" / scan / "blacklist_rescue.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f) or {}


def _build_record(
    ep,
    sub_idx: int,
    part_json: Optional[Dict],
    rewrite_episode: Optional[Dict],
    target_db: Optional[Dict],
    rescue_rec: Optional[Dict] = None,
    sub_label:  Optional[str]  = None,
) -> Dict[str, Any]:
    """Build one dataset record.

    Both original and synthesized records share an identical field set
    and ordering. The ``synthesized`` boolean and the
    ``synthesized_from`` block are the only places they differ in
    content; ``synthesized_from`` is ``None`` for original records and
    a populated dict for synthesized ones, but the key is always
    present so consumers can ``rec["synthesized_from"]`` without
    branching.

    When ``rescue_rec`` is provided the record is built from a
    11_rescue_blacklist entry (synthesized replacement landmark);
    otherwise it is built from the rewrite + target_instances side-cars
    (original landmark).

    ``sub_label`` is the value from ``survivor.sub_status[ep][sub]`` (e.g.
    ``"blacklist:llm_keep_false"``). For labeled-but-not-rescued subs the
    record still carries rewrite + partition geometry so inspection /
    follow-up rescue tools have everything they need; ``target_status``
    is just the label and ``target_instance_ids`` stays empty.
    """
    is_synth = rescue_rec is not None
    is_labeled = (not is_synth) and bool(sub_label)
    part     = _partition_for_sub(part_json or {}, sub_idx)
    rew      = _rewrite_for_sub(rewrite_episode or {}, sub_idx) or {}
    target   = _target_for_sub(target_db or {}, ep.instruction_id, sub_idx) or {}

    sub_pair = ep.sub_paths[sub_idx] if sub_idx < len(ep.sub_paths) else [None, None]
    if is_synth:
        sub_text = rescue_rec.get("new_sub_instruction") or ""
    else:
        sub_text = ep.sub_instructions[sub_idx] if sub_idx < len(ep.sub_instructions) else ""

    rec: Dict[str, Any] = {
        "scan":            ep.scan,
        "instruction_id":  ep.instruction_id,
        "path_id":         ep.path_id,
        "sub_idx":         sub_idx,
        "language":        ep.language,
        "instruction":     ep.instruction,
        "sub_instruction": sub_text,
        "sub_path":        list(sub_pair),
    }

    # ── Landmark / text block ────────────────────────────────────────
    if is_synth:
        new_landmark = rescue_rec.get("new_landmark") or ""
        rec["landmark"]             = new_landmark
        rec["landmark_category"]    = "object"
        rec["landmark_instruction"] = rescue_rec.get("landmark_instruction")
        rec["spatial_instruction"]  = rescue_rec.get("spatial_instruction")
        rec["components"] = [{
            "original_mention": new_landmark,
            "semantic_label":   rescue_rec.get("new_mpcat40"),
            "description":      f"Synthesized replacement for {rescue_rec.get('original_landmark')!r}.",
        }]
    else:
        rec["landmark"]             = rew.get("landmark")
        rec["landmark_category"]    = rew.get("landmark_category")
        rec["landmark_instruction"] = rew.get("landmark_instruction")
        rec["spatial_instruction"]  = rew.get("spatial_instruction")
        rec["components"] = [
            {
                "original_mention": c.get("original_mention"),
                "semantic_label":   c.get("semantic_label"),
                "description":      c.get("description"),
            }
            for c in (rew.get("components") or [])
        ]

    # ── Partition geometry (same source for both) ────────────────────
    if part:
        rec["sub_path_nodes"]     = part.get("sub_path_nodes")
        rec["spatial_path"]       = part.get("spatial_path")
        rec["landmark_path"]      = part.get("landmark_path")
        rec["partition_kind"]     = part.get("kind")
        rec["instruction_kind"]   = part.get("instruction_kind")
        rec["direction_mismatch"] = part.get("direction_mismatch")

    # ── Target block ─────────────────────────────────────────────────
    if is_synth:
        new_iid = rescue_rec.get("new_instance_id")
        target_ids = [int(new_iid)] if new_iid is not None else []
        rec["target_instance_ids"]         = target_ids
        rec["target_status"]               = "synthesized"
        rec["matched_semantic_category"]   = rescue_rec.get("new_mpcat40")
        rec["matched_semantic_categories"] = (
            [rescue_rec.get("new_mpcat40")] if rescue_rec.get("new_mpcat40") else []
        )
        rec["landmark_visible"]            = bool(target_ids)
        # Synth records carry visibility + uniqueness fields directly
        # from rescue_blacklist (same split schema as step 07/08).
        rec["visibility"] = rescue_rec.get("visibility") or (
            "visible" if target_ids else "not_visible"
        )
        rec["uniqueness"] = (
            rescue_rec["uniqueness"] if "uniqueness" in rescue_rec
            else "not_visible"
        )
        rec["visibility_status"] = rec["visibility"]
    else:
        target_ids = target.get("target_instance_ids") or []
        rec["target_instance_ids"]         = target_ids
        rec["target_status"]               = (
            sub_label if is_labeled and not target.get("status") else target.get("status")
        )
        if is_labeled and not rec["target_status"]:
            rec["target_status"] = sub_label
        rec["matched_semantic_category"]   = target.get("matched_category")
        rec["matched_semantic_categories"] = target.get("matched_categories")
        rec["landmark_visible"]            = bool(target_ids)
        # Split fields from step 07's _classify(). visibility is a string
        # ("visible" / "not_visible" / "no_match" / "partition_pos_unresolvable");
        # uniqueness is True/False when visible, else "not_visible".
        rec["visibility"]                  = target.get("visibility")
        rec["uniqueness"]                  = target.get("uniqueness")
        rec["visibility_status"]           = target.get("visibility_status")

    # ── Provenance (uniform key set) ─────────────────────────────────
    rec["synthesized"] = is_synth
    if is_synth:
        rec["synthesized_from"] = {
            "origin":            rescue_rec.get("origin") or "blacklist",
            "original_landmark": rescue_rec.get("original_landmark"),
            "original_reason":   rescue_rec.get("original_reason"),
        }
    else:
        rec["synthesized_from"] = None

    return rec


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Consolidate surviving sub-trajectory info into one dataset file.",
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", default=None,
                    help="Experiment handle (selection YAML path or expname). "
                         "Auto-merges survivor.yaml.")
    ap.add_argument("--out", default=None,
                    help="Output JSON path. Default: <run_dir>/dataset.json.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    resolve_exp(cfg, args.exp, apply_current=True)

    run_dir = get_run_dir(cfg)
    if not run_dir.exists():
        raise SystemExit(f"No run_dir at {run_dir}")

    filt_dir = get_filter_dir(cfg)
    filt_dir.mkdir(parents=True, exist_ok=True)
    split = get_split(cfg)
    audit = load_audit(filt_dir, split)
    register_stage(audit, STAGE_NAME)
    strip_stage_events(audit, STAGE_NAME)

    episodes = episodes_from_config(cfg)
    if not episodes:
        raise SystemExit("No surviving episodes — run the filter pipeline first.")

    sub_paths_filter = cfg.get("selection", {}).get("sub_paths") or {}
    if not sub_paths_filter:
        print(
            "WARNING: survivor.yaml has no `sub_paths` field — every sub-path "
            "of each surviving episode will be consolidated. Run 03/04 first "
            "for a tighter set."
        )

    suffix = discover_rewrite_suffix(run_dir / "rewrite") or ""

    # Group survivors per (scan, ep_id) so we hit each partition.json /
    # rewrite JSON / target_instances.json once.
    per_scan_target_cache: Dict[str, Optional[Dict]] = {}

    # Pre-load every per-scan rescue file so the synth-records loop
    # below can find them. (Originals are filtered out by the "must
    # have target_instance_ids" rule, not by checking for rescue
    # presence, so we don't need a per-(ep, sub) rescue lookup here.)
    per_scan_rescue_cache: Dict[str, Dict] = {}
    for ep in episodes:
        per_scan_rescue_cache.setdefault(
            ep.scan, _load_blacklist_rescue(run_dir, ep.scan),
        )

    records: List[Dict[str, Any]] = []
    missing: List[str] = []
    n_skipped_unusable = 0
    for ep in episodes:
        ep_subs = sub_paths_filter.get(int(ep.instruction_id))
        if ep_subs is None:
            ep_subs = list(range(len(ep.sub_paths)))
        else:
            ep_subs = sorted({int(s) for s in ep_subs})

        if ep.scan not in per_scan_target_cache:
            per_scan_target_cache[ep.scan] = _load_target_db(run_dir, ep.scan)
        target_db = per_scan_target_cache[ep.scan]

        part_json = _load_partition(run_dir, ep.scan, ep.instruction_id)
        rewrite_episode = _load_rewrite(run_dir, ep.scan, ep.instruction_id, suffix)

        ep_audit = ensure_episode(audit, ep)
        for sub_idx in ep_subs:
            sub_label = sub_status_for(cfg, ep.instruction_id, sub_idx)
            # Drop records where the instruction itself isn't usable:
            #   - blacklist:* labels (instruction text problem — LLM/regex
            #     judged the landmark non-referrable)
            #   - visibility:no_match (the landmark text doesn't match any
            #     MP3D category in this scene's vocabulary — the concept
            #     doesn't ground here, regardless of pose)
            # visibility:not_visible / partition_pos_unresolvable stay —
            # those are valid sub-trajectories where the landmark concept
            # exists but isn't visible from the partition pose.
            if sub_label and sub_label.startswith("blacklist:"):
                n_skipped_unusable += 1
                append_sub_event(
                    ep_audit, sub_idx, stage=STAGE_NAME, action="excluded",
                    reason=sub_label,
                )
                continue
            rec = _build_record(
                ep, sub_idx, part_json, rewrite_episode, target_db,
                sub_label=sub_label,
            )
            if rec.get("visibility") == "no_match":
                n_skipped_unusable += 1
                append_sub_event(
                    ep_audit, sub_idx, stage=STAGE_NAME, action="excluded",
                    reason="visibility:no_match",
                )
                continue
            records.append(rec)
            append_sub_event(
                ep_audit, sub_idx, stage=STAGE_NAME, action="included",
                synthesized=False,
                target_instance_ids=list(rec.get("target_instance_ids") or []),
                target_status=rec.get("target_status"),
            )
            if part_json is None or _partition_for_sub(part_json, sub_idx) is None:
                missing.append(f"{ep.instruction_id}#{sub_idx}: no partition.json")

    # ── Synthesized records from blacklist rescue (step 11) ───────────
    # We pull every episode the rescue file mentions — these episodes
    # may not be in `episodes` if they were entirely blacklisted (e.g.
    # 882 in val_unseen_one_scene_partial). per_scan_rescue_cache was
    # populated above so the labeled-record skip could consult it.
    syn_records: List[Dict[str, Any]] = []
    # Resolve every episode mentioned by any rescue file (covers ones
    # not in `episodes` because the whole episode got blacklist-dropped).
    extra_ep_ids = set()
    for rescue_data in per_scan_rescue_cache.values():
        for ep_id_str in (rescue_data.get("rescues") or {}):
            try:
                extra_ep_ids.add(int(ep_id_str))
            except ValueError:
                continue
    extra_ep_ids -= {int(ep.instruction_id) for ep in episodes}
    extra_episodes: Dict[int, Any] = {}
    if extra_ep_ids:
        side_cfg = dict(cfg)
        side_cfg.setdefault("selection", {}).update({
            "instruction_ids": sorted(extra_ep_ids),
            "sub_paths": {},
        })
        for ep in episodes_from_config(side_cfg):
            extra_episodes[int(ep.instruction_id)] = ep
            per_scan_rescue_cache.setdefault(
                ep.scan, _load_blacklist_rescue(run_dir, ep.scan),
            )

    ep_lookup = {int(ep.instruction_id): ep for ep in episodes}
    ep_lookup.update(extra_episodes)

    for scan, rescue_data in per_scan_rescue_cache.items():
        for ep_id_str, subs in (rescue_data.get("rescues") or {}).items():
            try:
                ep_id = int(ep_id_str)
            except ValueError:
                continue
            ep = ep_lookup.get(ep_id)
            if ep is None:
                continue
            part_json = _load_partition(run_dir, ep.scan, ep.instruction_id)
            for sub_idx_str, rescue_rec in (subs or {}).items():
                try:
                    sub_idx = int(sub_idx_str)
                except ValueError:
                    continue
                syn = _build_record(
                    ep, sub_idx,
                    part_json=part_json,
                    rewrite_episode=None,
                    target_db=None,
                    rescue_rec=rescue_rec,
                )
                syn_records.append(syn)
                ep_audit = ensure_episode(audit, ep)
                append_sub_event(
                    ep_audit, sub_idx, stage=STAGE_NAME, action="included",
                    synthesized=True,
                    new_landmark=rescue_rec.get("new_landmark"),
                    instance_id=rescue_rec.get("new_instance_id"),
                )

    records.extend(syn_records)

    out_path = Path(args.out).expanduser() if args.out else (run_dir / "dataset.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    finalize_audit(audit)
    save_audit(audit, filt_dir)

    # Summary print.
    n = len(records)
    by_status = {}
    n_synth = 0
    n_with_target = 0
    n_visible = 0
    scans = set()
    for r in records:
        scans.add(r["scan"])
        by_status[r.get("target_status") or "unknown"] = by_status.get(r.get("target_status") or "unknown", 0) + 1
        if r.get("synthesized"):
            n_synth += 1
        if r.get("target_instance_ids"):
            n_with_target += 1
        if r.get("landmark_visible"):
            n_visible += 1

    print()
    print(f"=== consolidate summary ===")
    print(f"  records             : {n}")
    print(f"  scans               : {len(scans)}  ({', '.join(sorted(scans))})")
    print(f"  synthesized=true    : {n_synth}")
    print(f"  synthesized=false   : {n - n_synth}")
    print(f"  with target_id      : {n_with_target}")
    print(f"  landmark visible    : {n_visible}")
    if n_skipped_unusable:
        print(f"  dropped (unusable)  : {n_skipped_unusable}  (blacklist:* + visibility:no_match; synth records below replace those with a fit)")
    print(f"  by target_status    :")
    for status, count in sorted(by_status.items(), key=lambda kv: -kv[1]):
        print(f"    {status:<25s} {count}")
    if missing:
        print(f"  ⚠ partition missing : {len(missing)} sub-paths")
        for m in missing[:5]:
            print(f"      {m}")
    print()
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
