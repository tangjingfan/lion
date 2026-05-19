"""Apply detection-rescue back into ``target_instances.json``.

The detection step (``09_detection``) writes
``target_instances/<scan>/semantic_rescue_categories.json`` listing every
``(episode_id, sub_idx)`` for which YOLO-World (or the VLM fallback) found
a fine-grained instance for a coarse-bucket target. By default that file
is consumed by downstream readers; this step fans those findings back
into the live ``target_instances/<scan>/target_instances.json`` so that
the canonical target-selection record reflects the rescue:

  • A sub-path that was previously ``not_visible`` (empty
    ``target_instance_ids``) and got a rescue → fill in the rescued
    instance id, set ``status = "rescued"``, and record the landmark
    used as the prompt.
  • A sub-path that already had a target → leave the chosen instance
    untouched, but annotate ``rescue_landmark`` / ``rescue_category``
    so downstream consumers know the rescue confirmed it.

The script is idempotent: re-running it produces the same output.

Usage
-----
  python src/check/apply_rescue.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --exp configs/selection/val_unseen/one_scene_partial.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import (
    append_sub_event,
    finalize_audit,
    get_filter_dir,
    get_run_dir,
    get_split,
    load_audit,
    register_stage,
    resolve_exp,
    save_audit,
    strip_stage_events,
)


STAGE_NAME = "apply_rescue"


def _index_rescue(
    rescue_path: Path,
) -> Dict[Tuple[str, str], Dict]:
    """Return ``{(episode_id, sub_idx): {instance_id, landmark, category,
    grounding_method, semantic_category}}`` for every rescued example.

    When multiple instances claim the same (ep, sub), keep the
    highest-confidence one (rare; usually one rescue per sub-path).
    """
    with open(rescue_path) as f:
        data = json.load(f) or {}
    out: Dict[Tuple[str, str], Dict] = {}
    conf_rank = {"high": 3, "medium": 2, "low": 1, "": 0}
    for iid_str, info in (data.get("instances") or {}).items():
        try:
            iid = int(iid_str)
        except (TypeError, ValueError):
            continue
        method = info.get("grounding_method")
        sem_cat = info.get("semantic_category")
        for ex in info.get("examples", []):
            ep = str(ex.get("episode_id") or "")
            sub = str(ex.get("sub_idx") or "")
            if not ep or not sub:
                continue
            landmark = ex.get("landmark") or info.get("category") or ""
            conf = (ex.get("grounding") or {}).get("confidence") or info.get("confidence") or ""
            new = {
                "instance_id": iid,
                "landmark": landmark,
                "category": landmark.strip().lower() or "unknown",
                "grounding_method": method,
                "semantic_category": sem_cat,
                "_conf_rank": conf_rank.get(conf, 0),
            }
            key = (ep, sub)
            prev = out.get(key)
            if prev is None or new["_conf_rank"] > prev["_conf_rank"]:
                out[key] = new
    for v in out.values():
        v.pop("_conf_rank", None)
    return out


def _apply_to_target_instances(
    ti_path: Path,
    rescue_index: Dict[Tuple[str, str], Dict],
    audit: Optional[Dict] = None,
    scan: str = "",
) -> Tuple[int, int]:
    """Merge rescue hits into a single ``target_instances.json`` in place.

    Returns ``(n_filled, n_confirmed)`` — how many empty sub-paths got
    filled, and how many already-targeted sub-paths got an annotation.
    Also pushes one ``applied`` event per (ep, sub) into ``audit`` when
    provided.
    """
    with open(ti_path) as f:
        data = json.load(f)
    # New layout: one section called ``annotations`` per (ep, sub).
    # Legacy layout used a separate ``target_instances`` section — we
    # still merge into both when present so old data is left consistent
    # for any straggling reader.
    sections = [
        data.get("annotations") or {},
        data.get("target_instances") or {},
    ]
    n_filled = 0
    n_confirmed = 0
    seen_keys: set = set()
    for targets in sections:
        for ep_id, subs in targets.items():
            for sub_idx, rec in subs.items():
                hit = rescue_index.get((str(ep_id), str(sub_idx)))
                if not hit:
                    continue
                # Wipe stale rescue annotations from prior runs first.
                for k in ("rescue_landmark", "rescue_category",
                          "rescue_grounding_method", "rescue_semantic_category",
                          "rescue_instance_id"):
                    rec.pop(k, None)
                rec["rescue_landmark"]          = hit["landmark"]
                rec["rescue_category"]          = hit["category"]
                rec["rescue_grounding_method"]  = hit["grounding_method"]
                rec["rescue_semantic_category"] = hit["semantic_category"]
                rec["rescue_instance_id"]       = hit["instance_id"]
                current_ids = rec.get("target_instance_ids") or []
                key = (str(ep_id), str(sub_idx))
                if not current_ids:
                    rec["target_instance_ids"] = [hit["instance_id"]]
                    rec["instance_id"]        = int(hit["instance_id"])
                    rec["status"]             = "rescued"
                    rec["rescued"]            = True
                    # Update visibility/uniqueness to reflect the
                    # post-rescue state — same split schema as a normal
                    # step 07/08 record. By construction YOLO landed on
                    # a single MP3D instance, so uniqueness=True.
                    rec["visibility"]         = "visible"
                    rec["uniqueness"]         = True
                    rec["visibility_status"]  = "visible"
                    rec["selection_distance"] = None
                    rec["candidate_distances"] = None
                    if key not in seen_keys:
                        n_filled += 1
                else:
                    if key not in seen_keys:
                        n_confirmed += 1
                if audit is not None and key not in seen_keys:
                    ep_audit = audit["episodes"].setdefault(str(ep_id), {
                        "scan": scan, "events": [], "sub_paths": {},
                    })
                    append_sub_event(
                        ep_audit, sub_idx, stage=STAGE_NAME, action="applied",
                        instance_id=int(hit["instance_id"]),
                        filled=(not current_ids),
                        landmark=hit.get("landmark"),
                    )
                seen_keys.add(key)
    with open(ti_path, "w") as f:
        json.dump(data, f, indent=2)
    return n_filled, n_confirmed


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge rescue results back into target_instances.json.",
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", default=None,
                    help="Experiment handle (selection YAML path or expname). "
                         "Auto-merges survivor.yaml on top.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    resolve_exp(cfg, args.exp, apply_current=True)

    run_dir = get_run_dir(cfg)
    ti_root = run_dir / "target_instances"
    if not ti_root.exists():
        raise SystemExit(
            f"No target_instances/ at {ti_root}. Run 07/08 first."
        )

    filt_dir = get_filter_dir(cfg)
    filt_dir.mkdir(parents=True, exist_ok=True)
    split = get_split(cfg)
    audit = load_audit(filt_dir, split)
    register_stage(audit, STAGE_NAME)
    strip_stage_events(audit, STAGE_NAME)

    # scan dirs are those holding a target_instances.json; everything
    # else under ti_root (viz/, viz_partition/, viz_last_frame/, ...) is
    # render output.
    scan_dirs = sorted(
        p for p in ti_root.iterdir()
        if p.is_dir() and (p / "target_instances.json").exists()
    )
    if not scan_dirs:
        raise SystemExit(f"No per-scan target_instances under {ti_root}.")

    total_filled = 0
    total_confirmed = 0
    for scan_dir in scan_dirs:
        rescue_path = scan_dir / "semantic_rescue_categories.json"
        ti_path     = scan_dir / "target_instances.json"
        if not rescue_path.exists():
            print(f"  [{scan_dir.name}] no semantic_rescue_categories.json — skip")
            continue
        rescue_index = _index_rescue(rescue_path)
        if not rescue_index:
            print(f"  [{scan_dir.name}] rescue file has no examples — skip")
            continue
        filled, confirmed = _apply_to_target_instances(
            ti_path, rescue_index, audit=audit, scan=scan_dir.name,
        )
        total_filled    += filled
        total_confirmed += confirmed
        print(
            f"  [{scan_dir.name}] {len(rescue_index)} rescue hit(s)"
            f" → filled {filled}, confirmed {confirmed}"
        )
        print(f"    updated: {ti_path}")

    finalize_audit(audit)
    save_audit(audit, filt_dir)

    print()
    print(f"=== apply_rescue summary ===")
    print(f"  newly filled target_instance_ids : {total_filled}")
    print(f"  rescue-confirmed existing targets: {total_confirmed}")


if __name__ == "__main__":
    main()
