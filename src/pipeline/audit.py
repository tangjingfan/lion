"""Per-(ep, sub) lifecycle audit for the filter pipeline.

``filters/audit.json`` is the single trace covering every stage from 00
(record_original) through 12 (consolidate).  Every pipeline stage appends
one event per touched cell describing what it did:

    events: [
      {"stage": "blacklist",   "action": "dropped",   "reason": "llm_keep_false"},
      {"stage": "rescue_blacklist", "action": "synthesized", "new_landmark": "fridge"},
      {"stage": "consolidate", "action": "included"}
    ]

:func:`finalize_audit` resolves the event list into a short ``verdict``
string + ``in_dataset`` bool per cell.  :func:`strip_stage_events` lets a
stage drop its own previous events so re-runs are idempotent.

Label discipline
----------------
``sub_status`` labels and audit verdicts namespace a *reason* under the
*stage* that emitted it (``blacklist:llm_keep_false``).  Always build
them via :func:`make_status_label` — it owns the prefixing, so a reason
that already carries the stage prefix (e.g. the blacklist stage's
term-hit reasons, ``blacklist:door``) is never double-prefixed into
``blacklist:blacklist:door``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple


# ── IO ───────────────────────────────────────────────────────────────────
def load_audit(filt_dir: Path, split: str) -> dict:
    """Load existing audit.json or initialize a new one."""
    audit_path = filt_dir / "audit.json"
    if audit_path.exists():
        with open(audit_path) as f:
            return json.load(f)
    return {"_meta": {"split": split, "stages": []}, "episodes": {}}


def save_audit(audit: dict, filt_dir: Path) -> None:
    with open(filt_dir / "audit.json", "w") as f:
        json.dump(audit, f, indent=2)


def register_stage(audit: dict, stage_name: str, **meta) -> None:
    """Append the stage to the meta list (idempotent) and stash any params."""
    if stage_name not in audit["_meta"]["stages"]:
        audit["_meta"]["stages"].append(stage_name)
    if meta:
        audit["_meta"].setdefault("thresholds", {})[stage_name] = meta


# ── Labels ───────────────────────────────────────────────────────────────
def make_status_label(stage: str, reason: str) -> str:
    """Build the ``sub_status`` label ``{stage}:{reason}``.

    Owns the prefixing so call sites never hand-concatenate: when
    ``reason`` already starts with ``{stage}:`` (e.g. the blacklist
    stage's term-hit reasons look like ``blacklist:door``) the prefix is
    not repeated.

        make_status_label("blacklist", "llm_keep_false")  → "blacklist:llm_keep_false"
        make_status_label("blacklist", "blacklist:door")  → "blacklist:door"
    """
    reason = (reason or "").strip() or "unknown"
    if reason == stage or reason.startswith(f"{stage}:"):
        return reason
    return f"{stage}:{reason}"


# ── Cell helpers ─────────────────────────────────────────────────────────
def ensure_episode(audit: dict, ep) -> dict:
    """Get-or-create the audit slot for one episode."""
    return audit["episodes"].setdefault(str(ep.instruction_id), {
        "scan":        ep.scan,
        "language":    ep.language,
        "n_sub_paths": len(ep.sub_paths),
        "events":      [],
        "sub_paths":   {},
    })


def ensure_sub_path(ep_audit: dict, sub_idx) -> dict:
    """Get-or-create the audit slot for one sub-path within an episode."""
    sp_map = ep_audit.setdefault("sub_paths", {})
    return sp_map.setdefault(str(sub_idx), {"events": []})


def append_ep_event(ep_audit: dict, *, stage: str, action: str, **payload) -> None:
    """Push one event onto an episode's lifecycle log."""
    event = {"stage": stage, "action": action}
    if payload:
        event.update(payload)
    ep_audit.setdefault("events", []).append(event)


def append_sub_event(
    ep_audit: dict, sub_idx, *, stage: str, action: str, **payload,
) -> None:
    """Push one event onto a sub-path's lifecycle log."""
    sp = ensure_sub_path(ep_audit, sub_idx)
    event = {"stage": stage, "action": action}
    if payload:
        event.update(payload)
    sp["events"].append(event)


def strip_stage_events(audit: dict, stage: str) -> None:
    """Drop all events emitted by ``stage`` from every cell.

    Call once at the top of a stage so re-running it doesn't accumulate
    duplicate events. Upstream events from other stages are preserved.
    """
    for ep in audit.get("episodes", {}).values():
        ep["events"] = [e for e in ep.get("events", []) if e.get("stage") != stage]
        for sp in (ep.get("sub_paths") or {}).values():
            sp["events"] = [e for e in sp.get("events", []) if e.get("stage") != stage]


# ── Event queries ────────────────────────────────────────────────────────
def detection_rescued(events: List[dict]) -> bool:
    """True when stage 09 grounded this sub's *original* landmark.

    Such a sub is no longer a synthesis candidate: the detection rescue
    fills its ``target_instance_ids`` (step 10), so the original record
    is usable as-is and must win over any template replacement.
    """
    return any(
        e.get("stage") == "detection" and e.get("action") == "rescued"
        for e in events
    )


# ── Verdict resolution ───────────────────────────────────────────────────
#
# Canonical pipeline order. Events are stable-sorted by this rank before
# resolution, so verdicts don't depend on *when* a stage was (re-)run:
# re-running stage 03 standalone appends its events after stage 11's in
# the raw list, but semantically blacklist still precedes synthesis.
STAGE_ORDER = (
    "original",        # 00 record_original
    "cross_floor",     # 01 filter_multi_floor
    "blacklist",       # 02+03 rewrite + blacklist label
    "partition",       # 04
    "visibility",      # 07 list_target_instances
    "select",          # 08 select_target_instances
    "detection",       # 09 YOLO-World rescue
    "apply_rescue",    # 10
    "rescue_blacklist",# 11 landmark synthesis
    "consolidate",     # 12
)
_STAGE_RANK = {s: i for i, s in enumerate(STAGE_ORDER)}


def _in_pipeline_order(events: List[dict]) -> List[dict]:
    """Stable-sort events into canonical stage order.

    Unknown stages rank after all known ones; within a stage the append
    order is preserved (stable sort).
    """
    return sorted(
        events,
        key=lambda e: _STAGE_RANK.get(e.get("stage"), len(STAGE_ORDER)),
    )


def _sub_verdict(events: List[dict]) -> Tuple[str, bool]:
    """Walk events in order and resolve to ``(verdict, in_dataset)``.

    State machine:
      - ``dropped``        → verdict := dropped:<stage>:<reason>
                             (via :func:`make_status_label`, so a reason
                             that already carries the stage prefix — or
                             equals the stage — isn't repeated)
      - ``labeled``        → verdict := labeled:<visibility>
      - ``selected``       → verdict := kept:<status>  (target found)
                             or labeled:<status>       (no target)
      - ``rescued``        → verdict := rescued:<method>
      - ``synthesized``    → verdict := synthesized:<new_landmark> (sticky:
                             later ``excluded`` events from step 12 refer
                             to the *original* record being skipped, not
                             the synth replacement — they don't override)
      - ``rescue_failed``  → (informational; leaves prior verdict)
      - ``applied``        → (informational)
      - ``included``       → in_dataset := True
      - ``excluded``       → verdict := excluded:<reason>, in_dataset := False
                             (suppressed when a ``synthesized`` event has
                             already fired for the same cell)
      - ``synth_superseded`` → step 12 skipped this cell's synth record
                             because the original was already included
                             (e.g. grounded by the detection rescue);
                             ``synthesized`` events are ignored entirely
                             so the verdict reflects the surviving
                             original (typically ``rescued:*``).
    """
    events = _in_pipeline_order(events)
    verdict = "active"
    in_dataset = False
    synth_seen = False
    superseded = any(e.get("action") == "synth_superseded" for e in events)
    for e in events:
        a = e.get("action")
        if a == "dropped":
            label = make_status_label(e.get("stage") or "?", e.get("reason") or "drop")
            verdict = f"dropped:{label}"
            in_dataset = False
        elif a == "labeled":
            v = e.get("visibility") or "labeled"
            verdict = f"labeled:{v}"
        elif a == "selected":
            s = e.get("status") or "selected"
            if e.get("target_instance_ids"):
                verdict = f"kept:{s}"
            else:
                verdict = f"labeled:{s}"
        elif a == "rescued":
            verdict = f"rescued:{e.get('method') or 'rescued'}"
        elif a == "synthesized":
            if not superseded:
                verdict = f"synthesized:{e.get('new_landmark') or 'synth'}"
                synth_seen = True
        elif a == "included":
            in_dataset = True
        elif a == "excluded":
            # Step 12 emits one excluded event per skipped original
            # record. When the same (ep, sub) was synthesized at step 11,
            # the synth row replaces the original in dataset.json, so the
            # excluded event isn't the final word.
            if not synth_seen:
                verdict = f"excluded:{e.get('reason') or 'excluded'}"
                in_dataset = False
        # `kept`, `rescue_failed`, `applied` are informational; no-op.
    return verdict, in_dataset


def _ep_verdict(events: List[dict]) -> str:
    events = _in_pipeline_order(events)
    for e in events:
        if e.get("action") == "dropped":
            label = make_status_label(e.get("stage") or "?", e.get("reason") or "drop")
            return f"dropped:{label}"
    return "kept"


def finalize_audit(audit: dict) -> None:
    """Compute the convenience ``verdict`` / ``in_dataset`` summary on
    every audit cell from its event list.

    Idempotent — overwrites the summary fields. The events list stays
    the source of truth; the summary is for grep / jq / human reading.
    A sub whose episode was dropped inherits the episode's verdict and
    ``in_dataset = False``.
    """
    for ep_id, ep in audit.get("episodes", {}).items():
        ep_events = ep.get("events", [])
        ep_verdict = _ep_verdict(ep_events)
        ep["verdict"] = ep_verdict
        ep_dropped = ep_verdict.startswith("dropped:")
        for _sub_idx, sp in (ep.get("sub_paths") or {}).items():
            if ep_dropped:
                sp["verdict"]    = ep_verdict
                sp["in_dataset"] = False
            else:
                v, in_ds = _sub_verdict(sp.get("events", []))
                sp["verdict"]    = v
                sp["in_dataset"] = in_ds
