"""Dataset-record assembly for the consolidate stage (pipeline step 12).

Owns the per-(scan, ep, sub) artifact readers and :func:`build_record`,
which stitches rewrite + partition + target_instances + blacklist_rescue
side-cars into one ``dataset.json`` record.  Pure aggregation — no LLM,
simulator, or detector calls.  The CLI orchestration, audit wiring, and
summary printing stay in ``src/check/consolidate_dataset.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


# ── Per-stage artifact readers ───────────────────────────────────────────
def load_partition(run_dir: Path, scan: str, ep_id: int) -> Optional[Dict]:
    p = run_dir / "partition" / scan / str(ep_id) / "partition.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def partition_for_sub(part_json: Dict, sub_idx: int) -> Optional[Dict]:
    for part in (part_json or {}).get("partitions", []):
        if int(part.get("sub_idx", -1)) == sub_idx:
            return part
    return None


def load_rewrite(run_dir: Path, scan: str, ep_id: int, suffix: str) -> Optional[Dict]:
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


def rewrite_for_sub(rewrite_episode: Dict, sub_idx: int) -> Optional[Dict]:
    for sp in (rewrite_episode or {}).get("sub_paths", []):
        if int(sp.get("sub_idx", -1)) == sub_idx:
            return sp
    return None


def load_target_db(run_dir: Path, scan: str) -> Optional[Dict]:
    p = run_dir / "target_instances" / scan / "target_instances.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def target_for_sub(target_db: Dict, ep_id: int, sub_idx: int) -> Optional[Dict]:
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


def load_blacklist_rescue(run_dir: Path, scan: str) -> Dict:
    p = run_dir / "target_instances" / scan / "blacklist_rescue.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f) or {}


# ── Record assembly ──────────────────────────────────────────────────────
def build_record(
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
    part     = partition_for_sub(part_json or {}, sub_idx)
    rew      = rewrite_for_sub(rewrite_episode or {}, sub_idx) or {}
    target   = target_for_sub(target_db or {}, ep.instruction_id, sub_idx) or {}

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
        # Split fields from step 07's classify_visibility(). visibility is
        # a string ("visible" / "not_visible" / "no_match" /
        # "partition_pos_unresolvable"); uniqueness is True/False when
        # visible, else "not_visible".
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
