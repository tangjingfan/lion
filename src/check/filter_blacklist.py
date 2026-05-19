"""
LION-Bench — Stage 2 filter: drop non-referrable / unmapped landmarks.

Reads the prior survivor set (via survivor.yaml) and the rewriter JSON, then
drops sub-paths whose landmark cannot be grounded as a
concrete MP3D object.  Four rules:

  1. generic room phrases                       → drop  (e.g. "room with a
                                                         light on"; not a
                                                         named room type)
  2. ``components`` empty / all "unknown"        → drop  (no MP3D mapping)
  3. ``landmark`` text matches blacklist word    → drop  (e.g. "hallway",
                                                          "corridor",
                                                          "door", "window")
  4. mapped semantic label matches blacklist     → drop

The output graduates the sub-path-level survivor set to those with a
trustworthy, MP3D-grounded landmark — ready for the visibility / uniqueness
checks downstream.

Usage
-----
  python src/check/filter_blacklist.py \\
      --config configs/rollout/rollout_landmark_rxr.yaml \\
      --exp configs/selection/exp.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.check._filter_utils import (
    active_subs,
    ensure_episode,
    ensure_sub_path,
    get_filter_dir,
    get_split,
    get_survivor_path,
    load_audit,
    load_rewrite_episodes,
    register_stage,
    resolve_exp,
    save_audit,
    write_drop_yaml,
    write_survivor,
    _sub_status_map,
)
from src.dataset.landmark_rxr import episodes_from_config


STAGE_NUM  = 2
STAGE_NAME = "blacklist"

# Words that, when they appear as a whole word in the landmark phrase,
# signal a non-referrable space rather than a concrete object.  Specific
# room types ("bedroom", "kitchen", "bathroom") are NOT here — those are
# legitimate landmarks.  Generic spatial terms ("area", "space") and
# transition zones ("hallway", "corridor") are.  Note: "stairs" is a real
# MP40 object category, so it stays out of the blacklist.
DEFAULT_BLACKLIST = (
    "hallway", "corridor", "passage", "passageway",
    "doorway", "door way", "archway", "threshold",
    "entrance", "entryway", "opening",
    "area", "space",
    "door", "window", "wall", "floor", "ceiling",
)

SPECIFIC_ROOM_TERMS = (
    "bathroom", "bedroom", "closet", "dining room", "family room",
    "foyer", "garage", "hall", "kitchen", "laundry room", "library",
    "living room", "lounge", "meeting room", "office", "rec room",
    "tv room", "utility room",
)

SAFE_COMPOUND_TERMS = (
    "wall clock", "wall art", "wall painting", "wall picture",
    "floor lamp", "ceiling light", "ceiling fan",
    "window blinds", "window shade", "window curtain",
)


def _is_generic_room_landmark(landmark: str) -> bool:
    """Return true for vague room phrases, but keep named room types."""
    if not re.search(r"\broom\b", landmark):
        return False
    return not any(
        re.search(rf"\b{re.escape(room)}\b", landmark)
        for room in SPECIFIC_ROOM_TERMS
    )


def _term_pattern(term: str) -> re.Pattern:
    """Whole-phrase term matcher with simple plural tolerance.

    The previous exact whole-word regex caught ``door`` but missed ``doors``.
    These labels are noisy LLM strings rather than a controlled ontology, so a
    small amount of morphology buys a lot of robustness without making the
    rules fuzzy.
    """
    words = [re.escape(w) for w in term.split()]
    sep = r"[\s_-]+"
    body = sep.join(words)
    suffix = r"(?:s|es)?" if not term.endswith(("s", "y")) else r""
    return re.compile(rf"(?<![a-z0-9]){body}{suffix}(?![a-z0-9])")


def _matches_blacklist(text: str, blacklist: Tuple[str, ...]) -> str:
    text = (text or "").strip().lower().replace("-", " ").replace("_", " ")
    if not text:
        return ""
    if any(re.search(rf"\b{re.escape(term)}\b", text) for term in SAFE_COMPOUND_TERMS):
        return ""
    for term in blacklist:
        if _term_pattern(term.lower()).search(text):
            return term
    return ""


def _component_labels(components: List[Dict]) -> List[str]:
    return [
        (comp.get("semantic_label") or "").strip().lower()
        for comp in components
        if (comp.get("semantic_label") or "").strip()
    ]


def _classify(rewrite_sub: Dict, blacklist: Tuple[str, ...]) -> Tuple[bool, str]:
    """Return ``(keep, reason)`` for one sub-path's rewrite entry.

    Decision order (primary → fallback):
      1. ``keep`` field from the rewriter (LLM verdict).  This is the
         authoritative signal — a properly tuned rewriter already marks
         non-referrable landmarks (walls, doorways, corridors, ...) as
         ``keep=false``.
      2. ``generic_room`` pattern — vague "room" phrases without a named
         room type.
      3. Regex ``DEFAULT_BLACKLIST`` — defence-in-depth safety net that
         catches generic terms slipping through the LLM.
      4. Mapped semantic label sanity — drop unmapped non-spatial
         landmarks; drop any whose mapped label hits the blacklist.
    """
    category   = (rewrite_sub.get("landmark_category") or "object").lower()
    landmark   = (rewrite_sub.get("landmark") or "").strip().lower()
    components = rewrite_sub.get("components") or []

    # 1. Primary: trust the LLM's keep verdict.
    if not bool(rewrite_sub.get("keep", True)):
        return False, "llm_keep_false"

    # 2. Cheap text pattern that the LLM occasionally misses.
    if _is_generic_room_landmark(landmark):
        return False, "generic_room"

    # 3. Regex safety net on the landmark phrase itself.
    hit = _matches_blacklist(landmark, blacklist)
    if hit:
        return False, f"blacklist:{hit}"

    # Spatial landmarks may not have MP3D components.  Keep them unless the
    # landmark phrase itself is one of the explicitly hard-to-refer terms.
    if category == "spatial":
        return True, "ok"

    mapped = [
        (c.get("semantic_label") or "unknown").strip().lower()
        for c in components
    ]
    if not mapped or all(s in ("", "unknown") for s in mapped):
        return False, "unmapped"

    # 4. Same regex net on mapped labels.  Avoid using descriptions here:
    # they often include contextual text ("teapoy ... facing the glass door")
    # where the blacklisted object is not the target landmark.
    for label in _component_labels(components):
        hit = _matches_blacklist(label, blacklist)
        if hit:
            return False, f"blacklist:{hit}"

    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 2: drop non-referrable / unmapped landmarks",
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", default=None,
                    help="Experiment handle (selection YAML path or expname). "
                         "survivor.yaml is auto-merged on top to "
                         "supply the prior survivor sub_paths.")
    ap.add_argument("--blacklist", nargs="+", default=None,
                    help="Override the default landmark-text blacklist.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    resolve_exp(cfg, args.exp, apply_current=True)

    filt_dir = get_filter_dir(cfg)
    if not filt_dir.exists():
        raise SystemExit(f"No filters/ at {filt_dir} — run prior stages first.")
    out_dir = filt_dir.parent
    split   = get_split(cfg)

    survivor = get_survivor_path(cfg)
    if not survivor.exists():
        raise SystemExit(
            f"No survivor.yaml at {survivor} — run 01_filter_multi_floor first."
        )

    # resolve_exp(apply_current=True) already merged survivor.yaml into
    # cfg.selection. We process the *active* set (un-labeled subs that
    # still need a verdict), but write the *full* set back so labels
    # accumulate across stages.
    #
    # Idempotence: a re-run of this stage strips any labels this stage
    # previously wrote (matching ``blacklist:*``) and re-classifies from
    # scratch. Upstream labels (cross_floor, etc. — none today) are
    # preserved.
    prior_subs   = cfg.get("selection", {}).get("sub_paths")
    raw_prior_status = _sub_status_map(cfg)
    upstream_status: Dict[int, Dict[int, str]] = {}
    for ep_id, labels in raw_prior_status.items():
        keep = {s: lbl for s, lbl in labels.items()
                if not (lbl or "").startswith(f"{STAGE_NAME}:")}
        if keep:
            upstream_status[ep_id] = keep

    episodes = episodes_from_config(cfg)
    if not episodes:
        raise SystemExit("No episodes loaded from survivor.yaml.")
    full_subs: Dict[int, List[int]] = {}     # everything alive past cross_floor
    if prior_subs:
        full_subs = {
            int(ep_id): [int(s) for s in subs]
            for ep_id, subs in prior_subs.items()
        }
    else:
        full_subs = {
            int(ep.instruction_id): list(range(len(ep.sub_paths)))
            for ep in episodes
        }
    # Active = sub_paths minus labels-from-upstream-stages. We strip our
    # own previous labels, so on a re-run we'll see every sub we should
    # classify (not just the leftovers from a partial earlier pass).
    allowed_subs: Dict[int, List[int]] = {}
    for ep_id, subs in full_subs.items():
        upstream_labeled = upstream_status.get(ep_id, {})
        kept = [int(s) for s in subs if int(s) not in upstream_labeled]
        if kept:
            allowed_subs[ep_id] = kept
    episode_by_id = {int(ep.instruction_id): ep for ep in episodes}

    # Locate rewriter output (per-episode).  Pick the ``_filtered`` variant
    # if any per-episode file under rewrite_dir has it; otherwise unfiltered.
    rewrite_dir = out_dir / "rewrite"
    if not rewrite_dir.exists():
        raise SystemExit(f"No rewrite dir under {rewrite_dir}")
    rewrite_episodes, _suffix, used_paths = load_rewrite_episodes(rewrite_dir)
    if not rewrite_episodes:
        raise SystemExit(f"No rewrite JSON under {rewrite_dir}/*/*/")
    print(f"Loaded rewrite from {len(used_paths)} per-episode file(s); "
          f"first: {used_paths[0]}")

    blacklist = tuple(args.blacklist) if args.blacklist else DEFAULT_BLACKLIST
    print(f"Blacklist words: {blacklist}")

    audit = load_audit(filt_dir, split)
    register_stage(audit, STAGE_NAME, blacklist=list(blacklist))

    # Carry forward upstream labels (our own stripped-and-rebuilt).
    new_sub_status: Dict[int, Dict[int, str]] = {
        ep_id: dict(labels) for ep_id, labels in upstream_status.items()
    }
    keep_sub_paths: Dict[int, List[int]] = {}
    dropped:        Dict[str, Dict]      = {}
    n_subs_total = 0
    n_subs_active = 0
    n_subs_labeled = 0
    reason_counts: Dict[str, int] = {}

    for ep_id_raw, sub_idxs in full_subs.items():
        # YAML loads bare int keys as int; rewrite JSON keys are strings.
        ep_id      = int(ep_id_raw)
        ep_id_str  = str(ep_id)
        rewrite_ep = rewrite_episodes.get(ep_id_str)
        ep = episode_by_id.get(ep_id)
        ep_audit = (
            ensure_episode(audit, ep)
            if ep is not None
            else audit["episodes"].setdefault(ep_id_str, {"stages": {}})
        )
        rewrite_subs = (
            {int(s["sub_idx"]): s for s in rewrite_ep.get("sub_paths", [])}
            if rewrite_ep else {}
        )

        # Every sub past cross_floor stays in keep_sub_paths.
        keep_sub_paths[ep_id] = sorted(int(s) for s in sub_idxs)

        ep_active_subs = set(allowed_subs.get(ep_id, []))
        ep_drops: Dict[int, Dict] = {}
        ep_n_labeled = 0

        for sub_idx in sub_idxs:
            n_subs_total += 1
            # Skip subs already labeled by an upstream stage (today: none).
            if int(sub_idx) not in ep_active_subs:
                continue
            rw = rewrite_subs.get(int(sub_idx))
            if rw is None:
                payload = {"reason": "missing_in_rewrite"}
                keep_it = False
            else:
                keep_it, reason = _classify(rw, blacklist)
                payload = {
                    "reason":   reason,
                    "landmark": rw.get("landmark"),
                    "category": rw.get("landmark_category"),
                }

            sp_audit = ensure_sub_path(ep_audit, sub_idx)
            sp_audit["stages"][STAGE_NAME] = {
                "status": "ok" if keep_it else "drop",
                **payload,
            }
            if keep_it:
                n_subs_active += 1
            else:
                ep_drops[sub_idx] = payload
                n_subs_labeled += 1
                ep_n_labeled += 1
                new_sub_status.setdefault(ep_id, {})[int(sub_idx)] = (
                    f"{STAGE_NAME}:{payload['reason']}"
                )
                reason_counts[payload["reason"]] = \
                    reason_counts.get(payload["reason"], 0) + 1

        n_remaining_active = len(ep_active_subs) - ep_n_labeled
        ep_audit["stages"][STAGE_NAME] = {
            "status":    "ok" if n_remaining_active > 0 else "drop",
            "kept_sub":  n_remaining_active,
            "total_sub": len(ep_active_subs),
        }
        if ep_drops:
            dropped[ep_id_str] = {
                "subs": {str(k): v for k, v in sorted(ep_drops.items())},
            }

    survivor_path = write_survivor(
        cfg, split,
        instruction_ids=sorted(keep_sub_paths.keys()),
        sub_paths=keep_sub_paths,
        sub_status=new_sub_status,
    )
    drop_path = write_drop_yaml(
        filt_dir, STAGE_NUM, STAGE_NAME, split,
        dropped=dict(sorted(dropped.items(), key=lambda kv: int(kv[0]))),
        extras={"blacklist": list(blacklist)},
    )
    save_audit(audit, filt_dir)

    n_eps_in   = len(allowed_subs)
    n_eps_keep = len(keep_sub_paths)
    pct_active = (n_subs_active / n_subs_total) if n_subs_total else 0.0

    print(f"=== Stage {STAGE_NUM} — {STAGE_NAME} ===")
    print(f"  episodes in    : {n_eps_in}")
    print(f"  episodes keep  : {n_eps_keep}  (every ep past cross_floor stays — labels only)")
    print(f"  sub-paths in   : {n_subs_total}")
    print(f"  sub-paths active (un-labeled at this stage): {n_subs_active}  ({pct_active:.1%})")
    print(f"  sub-paths labeled this stage              : {n_subs_labeled}")
    if reason_counts:
        print(f"  label reasons:")
        for reason, n in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {reason:<25s} {n}")

    print()
    print("Outputs:")
    print(f"  {survivor_path}")
    print(f"  {drop_path}")
    print(f"  {filt_dir / 'audit.json'}")


if __name__ == "__main__":
    main()
