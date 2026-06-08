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
from typing import Any, Dict, List, Optional, Tuple

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
from src.process.consolidate import (
    build_record,
    load_blacklist_rescue,
    load_partition,
    load_rewrite,
    load_target_db,
    partition_for_sub,
)


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
            ep.scan, load_blacklist_rescue(run_dir, ep.scan),
        )

    records: List[Dict[str, Any]] = []
    missing: List[str] = []
    n_skipped_unusable = 0
    # Cells whose *original* record made it into the dataset. A synth
    # replacement for the same (ep, sub) is redundant — the original
    # (e.g. grounded by the step-09 detection rescue after a not_visible
    # label) wins over the template rewrite.
    included_original_cells: set = set()
    for ep in episodes:
        ep_subs = sub_paths_filter.get(int(ep.instruction_id))
        if ep_subs is None:
            ep_subs = list(range(len(ep.sub_paths)))
        else:
            ep_subs = sorted({int(s) for s in ep_subs})

        if ep.scan not in per_scan_target_cache:
            per_scan_target_cache[ep.scan] = load_target_db(run_dir, ep.scan)
        target_db = per_scan_target_cache[ep.scan]

        part_json = load_partition(run_dir, ep.scan, ep.instruction_id)
        rewrite_episode = load_rewrite(run_dir, ep.scan, ep.instruction_id, suffix)

        ep_audit = ensure_episode(audit, ep)
        for sub_idx in ep_subs:
            sub_label = sub_status_for(cfg, ep.instruction_id, sub_idx)
            # Drop original records whose landmark isn't usable as-is.
            # Step 11 may have synthesized a replacement; that's added
            # separately in the synth loop below (the sticky-synth
            # verdict resolver in _filter_utils handles the case where
            # excluded + included co-occur on the same audit cell).
            #   - blacklist:* labels — instruction text problem
            #   - visibility:no_match — landmark text has no MP3D match
            #     in this scene's vocabulary
            #   - visibility:not_visible — landmark matched a category
            #     but no instance is visible at the partition pose
            # Keeping originals in the dataset with empty target ids
            # produced records that look "in_dataset" but were
            # practically unusable; the simple dataset only wants
            # records whose landmark is actually visible.
            # partition_pos_unresolvable still passes through (there's
            # no synthesis path for it either).
            if sub_label and sub_label.startswith("blacklist:"):
                n_skipped_unusable += 1
                append_sub_event(
                    ep_audit, sub_idx, stage=STAGE_NAME, action="excluded",
                    reason=sub_label,
                )
                continue
            rec = build_record(
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
            if rec.get("visibility") == "not_visible":
                n_skipped_unusable += 1
                append_sub_event(
                    ep_audit, sub_idx, stage=STAGE_NAME, action="excluded",
                    reason="visibility:not_visible",
                )
                continue
            if rec.get("visibility") == "partition_pos_unresolvable":
                # End-pose couldn't be resolved → empty target and no synthesis
                # path. Exclude rather than emit an unusable empty-target record.
                n_skipped_unusable += 1
                append_sub_event(
                    ep_audit, sub_idx, stage=STAGE_NAME, action="excluded",
                    reason="visibility:partition_pos_unresolvable",
                )
                continue
            records.append(rec)
            included_original_cells.add((int(ep.instruction_id), int(sub_idx)))
            append_sub_event(
                ep_audit, sub_idx, stage=STAGE_NAME, action="included",
                synthesized=False,
                target_instance_ids=list(rec.get("target_instance_ids") or []),
                target_status=rec.get("target_status"),
            )
            if part_json is None or partition_for_sub(part_json, sub_idx) is None:
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
                ep.scan, load_blacklist_rescue(run_dir, ep.scan),
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
            part_json = load_partition(run_dir, ep.scan, ep.instruction_id)
            for sub_idx_str, rescue_rec in (subs or {}).items():
                try:
                    sub_idx = int(sub_idx_str)
                except ValueError:
                    continue
                if (ep_id, sub_idx) in included_original_cells:
                    # The original record for this cell is already in the
                    # dataset (typically rescued by step 09 after the
                    # visibility label that put it on step 11's work
                    # list). Real human language wins over the template
                    # rewrite — skip the synth duplicate.
                    print(
                        f"  [dedup] {scan} ep={ep_id} sub={sub_idx}: "
                        "original already included — skipping synth record"
                    )
                    ep_audit = ensure_episode(audit, ep)
                    append_sub_event(
                        ep_audit, sub_idx, stage=STAGE_NAME,
                        action="synth_superseded",
                        reason="original_already_included",
                        new_landmark=rescue_rec.get("new_landmark"),
                    )
                    continue
                syn = build_record(
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

    # ── Walk audit for the explicit drop / rescue breakdown ────────────
    # Goal: anyone running the pipeline knows exactly which (ep, sub)
    # pairs didn't make it into dataset.json and why, without having to
    # crack open audit.json by hand.
    not_in_by_verdict: Dict[str, List[Tuple[str, str, str]]] = {}
    synth_by_origin:   Dict[str, int] = {}
    rescued_by_method: Dict[str, int] = {}
    total_subs_in_audit  = 0
    in_dataset_count     = 0
    for ep_id_str, ep in (audit.get("episodes") or {}).items():
        for sub_idx_str, sp in (ep.get("sub_paths") or {}).items():
            total_subs_in_audit += 1
            events  = sp.get("events") or []
            verdict = sp.get("verdict") or "(unknown)"
            if sp.get("in_dataset"):
                in_dataset_count += 1
                # Tally synthesis origin / detection rescue method, if any.
                synth_evt = next(
                    (e for e in events if e.get("action") == "synthesized"),
                    None,
                )
                if synth_evt:
                    origin = synth_evt.get("origin") or "?"
                    synth_by_origin[origin] = synth_by_origin.get(origin, 0) + 1
                rescued_evt = next(
                    (e for e in events if e.get("action") == "rescued"),
                    None,
                )
                if rescued_evt:
                    method = rescued_evt.get("method") or "?"
                    rescued_by_method[method] = rescued_by_method.get(method, 0) + 1
                continue
            # Not in dataset — extract the landmark text for context.
            # Cross-floor drops have no landmark (geometric only), so fall
            # back to a Δy hint when present.
            landmark = next(
                (e.get("landmark") for e in events if e.get("landmark")),
                None,
            )
            if not landmark:
                y_range = next(
                    (e.get("y_range_m") for e in events
                     if e.get("stage") == "cross_floor"
                     and e.get("y_range_m") is not None),
                    None,
                )
                landmark = f"(Δy={y_range:.2f}m)" if y_range is not None else "?"
            not_in_by_verdict.setdefault(verdict, []).append(
                (ep_id_str, sub_idx_str, str(landmark)),
            )

    print()
    print(f"=== consolidate summary ===")
    print(f"  records             : {n}")
    print(f"  scans               : {len(scans)}  ({', '.join(sorted(scans))})")
    print(f"  synthesized=true    : {n_synth}")
    print(f"  synthesized=false   : {n - n_synth}")
    print(f"  with target_id      : {n_with_target}")
    print(f"  landmark visible    : {n_visible}")
    if n_skipped_unusable:
        print(f"  dropped (unusable)  : {n_skipped_unusable}  (blacklist:* / visibility:no_match / visibility:not_visible; replaced by synth records when step 11 found a fit)")
    print(f"  by target_status    :")
    for status, count in sorted(by_status.items(), key=lambda kv: -kv[1]):
        print(f"    {status:<25s} {count}")

    # ── Rescue / synth breakdown ──
    if rescued_by_method:
        print()
        print(f"  rescued via detection (step 09):")
        for method, k in sorted(rescued_by_method.items(), key=lambda kv: -kv[1]):
            print(f"    {method:<25s} {k}")
    if synth_by_origin:
        print()
        print(f"  synthesized via landmark rescue (step 11) by origin:")
        for origin, k in sorted(synth_by_origin.items(), key=lambda kv: -kv[1]):
            print(f"    {origin:<25s} {k}")

    # ── Explicit drop breakdown ──
    n_not_in = total_subs_in_audit - in_dataset_count
    if n_not_in:
        print()
        print(f"=== sub-paths NOT in dataset.json ({n_not_in}/{total_subs_in_audit} past cross_floor) ===")
        # Group: hard cross_floor losses first (unrescuable), then
        # excluded-and-no-rescue (could have been rescued but wasn't),
        # then everything else.
        def _sort_key(verdict: str) -> Tuple[int, str]:
            if verdict.startswith("dropped:cross_floor"):
                return (0, verdict)   # permanent loss — list first
            if verdict.startswith("excluded:"):
                return (1, verdict)   # rescue attempted, failed
            return (2, verdict)
        for verdict in sorted(not_in_by_verdict.keys(), key=_sort_key):
            items = not_in_by_verdict[verdict]
            print(f"  [{len(items)}]  {verdict}")
            shown = items[:20]
            for ep_id, sub_idx, landmark in shown:
                print(f"      ep={ep_id:>6s} sub={sub_idx:<3s}  landmark={landmark!r}")
            if len(items) > len(shown):
                print(f"      … +{len(items) - len(shown)} more")

    if missing:
        print()
        print(f"  ⚠ partition missing : {len(missing)} sub-paths")
        for m in missing[:5]:
            print(f"      {m}")
    print()
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
