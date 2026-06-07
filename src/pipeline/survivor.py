"""Survivor state: the canonical ``{run_dir}/survivor.yaml`` + label channel.

The pipeline is a chain of numbered stages that progressively narrow
down which (instruction_id, sub_idx) pairs survive.  The current state
is centralized in a single ``survivor.yaml`` at the run dir root; each
stage overwrites it as it narrows the pipeline.

Survivor YAML schema
--------------------
Episode-level (stage 1)::

    split: LandmarkRxR_val_unseen
    instruction_ids: [19199, ...]

Sub-path-level (stage 2+)::

    split: LandmarkRxR_val_unseen
    instruction_ids: [19199, ...]   # episodes with ≥1 alive sub-path
    sub_paths:
      19199: [0, 1, 2, 3]           # every sub_idx past cross_floor —
                                    # labeled drops stay here
    sub_status:                     # label channel; subs absent from this
      19199:                        # dict are "active" (no failure yet)
        2: "blacklist:llm_keep_false"
        3: "partition:rewrite_error"

The hard-drop boundary is cross_floor only — entire episodes vanish when
they cross floors.  Every later stage **labels** drops via ``sub_status``
(built with :func:`src.pipeline.audit.make_status_label`) rather than
removing them, so downstream rescue / inspection can still reach them.
Use :func:`active_subs` to iterate the un-labeled set and
:func:`sub_status_for` to query a specific sub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.pipeline.config import get_run_dir


# ── Survivor / drop YAMLs ────────────────────────────────────────────────
def write_survivor(
    cfg:             dict,
    split:           str,
    instruction_ids: List[int],
    sub_paths:       Optional[Dict[int, List[int]]] = None,
    sub_status:      Optional[Dict[int, Dict[int, str]]] = None,
) -> Path:
    """Overwrite the canonical ``{run_dir}/survivor.yaml``.

    Self-describing: echoes ``expname`` / ``run_name`` so it doubles as
    a valid selection yaml — any tool can read it via
    ``apply_selection_yaml``.

    ``sub_paths`` carries every alive sub-path (everything past cross_floor,
    including labeled drops). ``sub_status``, when provided, attaches a
    failure label to a subset of those subs.  Sub-paths absent from
    ``sub_status`` are "active" (no failure yet).
    """
    payload: dict = {"split": split}
    out_cfg = cfg.get("output", {})
    if out_cfg.get("expname"):
        payload["expname"] = out_cfg["expname"]
    if out_cfg.get("run_name"):
        payload["run_name"] = out_cfg["run_name"]
    payload["scans"]           = []
    payload["languages"]       = []
    payload["instruction_ids"] = sorted(int(x) for x in instruction_ids)
    if sub_paths is not None:
        payload["sub_paths"] = {
            int(k): sorted({int(v) for v in vs})
            for k, vs in sorted(sub_paths.items(), key=lambda kv: int(kv[0]))
        }
    if sub_status:
        payload["sub_status"] = {
            int(k): {int(s): str(lbl) for s, lbl in sorted(
                v.items(), key=lambda kv: int(kv[0]),
            )}
            for k, v in sorted(sub_status.items(), key=lambda kv: int(kv[0]))
            if v
        }
    run_dir = get_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "survivor.yaml"
    with open(path, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
    return path


def write_drop_yaml(
    filt_dir:   Path,
    stage_num:  int,
    stage_name: str,
    split:      str,
    dropped:    dict,
    extras:     Optional[dict] = None,
) -> Path:
    payload: dict = {"split": split, "stage": stage_name}
    if extras:
        payload.update(extras)
    payload["dropped"] = dropped
    path = filt_dir / f"{stage_num:02d}_{stage_name}_dropped.yaml"
    with open(path, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
    return path


# ── Survivor lookup (for downstream consumers) ───────────────────────────
def load_keep(yaml_path: Path) -> dict:
    """Read a keep/survivor YAML.  Returns a dict with keys
    ``instruction_ids``, ``sub_paths`` (may be missing for episode-level
    stages)."""
    with open(yaml_path) as f:
        return yaml.safe_load(f) or {}


# ── sub_status helpers (label channel inside survivor.yaml) ──────────────
#
# After cross_floor, every sub-path past that hard-drop stage stays in
# survivor.yaml. Failures at blacklist / partition / etc. attach a label
# string under ``cfg.selection.sub_status[ep_id][sub_idx]`` rather than
# physically removing the sub. Downstream stages that "do real work"
# (visibility check, target selection, YOLO rescue, etc.) skip labeled
# subs by calling :func:`active_subs` — which mimics the old per-stage
# survivor's contents — while inspection / rescue / consolidate stages
# that *care* about labeled subs iterate the full set and use
# :func:`sub_status_for` to decide what to do per (ep, sub).
def _sub_status_map(cfg: dict) -> Dict[int, Dict[int, str]]:
    raw = (cfg.get("selection") or {}).get("sub_status") or {}
    out: Dict[int, Dict[int, str]] = {}
    for ep_id, subs in raw.items():
        if not isinstance(subs, dict):
            continue
        try:
            ep_key = int(ep_id)
        except (TypeError, ValueError):
            continue
        slot = out.setdefault(ep_key, {})
        for sub_idx, label in subs.items():
            try:
                sub_key = int(sub_idx)
            except (TypeError, ValueError):
                continue
            if label:
                slot[sub_key] = str(label)
    return out


def sub_status_for(
    cfg: dict, ep_id, sub_idx,
) -> Optional[str]:
    """Return the label string for ``(ep_id, sub_idx)`` if it has been
    flagged at any earlier stage; ``None`` for active sub-paths.

    Use this at the top of any per-sub loop in a downstream "real work"
    stage to skip labeled subs without re-deriving from the audit::

        if sub_status_for(cfg, ep.instruction_id, sub_idx):
            continue
    """
    try:
        ep_key  = int(ep_id)
        sub_key = int(sub_idx)
    except (TypeError, ValueError):
        return None
    return _sub_status_map(cfg).get(ep_key, {}).get(sub_key) or None


def active_subs(cfg: dict) -> Dict[int, List[int]]:
    """Return ``{ep_id: [sub_idx, ...]}`` for sub-paths that are still
    actively in the pipeline — i.e. present in ``sub_paths`` AND not
    labeled in ``sub_status``.

    This matches what the old per-stage survivor file used to contain.
    Downstream stages should iterate this view to skip labeled drops.
    """
    sub_paths_raw = (cfg.get("selection") or {}).get("sub_paths") or {}
    status_map    = _sub_status_map(cfg)
    out: Dict[int, List[int]] = {}
    for ep_id, subs in sub_paths_raw.items():
        try:
            ep_key = int(ep_id)
        except (TypeError, ValueError):
            continue
        labeled = status_map.get(ep_key, {})
        kept = [int(s) for s in (subs or []) if int(s) not in labeled]
        if kept:
            out[ep_key] = sorted(kept)
    return out
