"""
Shared helpers for the LION-Bench filter pipeline.

The pipeline is a chain of numbered "stages" that progressively narrow
down which (instruction_id, sub_idx) pairs survive.  Each stage writes:

    {filt_dir}/NN_{name}.yaml          — survivor set after this stage
    {filt_dir}/NN_{name}_dropped.yaml  — what got dropped + why
    {filt_dir}/audit.json              — single source of truth, per-stage
                                          status for every (ep, sub-path)
    {filt_dir}/current.yaml            — symlink to the latest keep file

Survivor YAML schema
--------------------
Episode-level (stage 1):

    split: LandmarkRxR_val_unseen
    instruction_ids: [19199, ...]

Sub-path-level (stage 2+):

    split: LandmarkRxR_val_unseen
    instruction_ids: [19199, ...]   # episodes with ≥1 surviving sub-path
    sub_paths:
      19199: [0, 1, 3]              # which sub_idx survive within each
      ...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ── Path helpers ──────────────────────────────────────────────────────────
def get_split(cfg: dict) -> str:
    return Path(cfg["dataset"]["data_path"]).stem


def resolve_run_name(cfg: dict) -> str:
    """Compute the per-run output folder name.

    Priority:
      1. ``output.run_name``  — explicit override (highest)
      2. ``{split}_{expname}` — when ``output.expname`` is set
      3. ``{split}``          — fallback (legacy behaviour)

    where ``split`` is the dataset filename stem with the ``LandmarkRxR_``
    prefix stripped (e.g. ``LandmarkRxR_val_unseen`` → ``val_unseen``).
    """
    out_cfg = cfg.get("output", {})
    if out_cfg.get("run_name"):
        return out_cfg["run_name"]
    split   = Path(cfg["dataset"]["data_path"]).stem.replace("LandmarkRxR_", "")
    expname = out_cfg.get("expname")
    return f"{split}_{expname}" if expname else split


def get_run_dir(cfg: dict) -> Path:
    """Resolve ``{base_dir}/{run_name}`` from a rollout-style config."""
    out_cfg  = cfg.get("output", {})
    base_dir = Path(out_cfg.get("base_dir", "results")).expanduser()
    return base_dir / resolve_run_name(cfg)


def get_filter_dir(cfg: dict) -> Path:
    """Resolve ``{base_dir}/{run_name}/filters`` from a rollout-style config."""
    return get_run_dir(cfg) / "filters"


# ── Selection YAML merging ───────────────────────────────────────────────
#
# Selection YAMLs are the per-experiment customization layer.  The rollout
# YAML stays simulator-only (dataset paths, env, agent default, output
# base_dir); anything experiment-specific lives in a selection YAML and is
# merged into ``cfg`` at runtime.
#
# Selection YAML schema:
#
#   1) Flat shortcuts — concise aliases for the most common selection-y
#      keys.  Remapped to nested cfg locations:
#
#        scans:           [scan_id, ...]   → cfg.scenes.include
#        languages:       [lang, ...]      → cfg.selection.languages
#        instruction_ids: [int, ...]       → cfg.selection.instruction_ids
#        sub_paths:       {id: [...]}      → cfg.selection.sub_paths
#        max_episodes:    int              → cfg.selection.max_episodes
#        expname:         str              → cfg.output.expname
#        run_name:        str              → cfg.output.run_name
#        split:           str              (informational only)
#
#   2) Anything else is deep-merged into ``cfg``.  This lets the selection
#      override any rollout cfg field with full nesting:
#
#        agent:
#          type: dummy
#        output:
#          viz:
#            enabled: false
#        env:
#          max_steps: 1000
#        visibility:
#          viz: false
_FLAT_SHORTCUTS = {
    # selection key  : (cfg.section, cfg.field)
    "scans":           ("scenes",    "include"),
    "languages":       ("selection", "languages"),
    "instruction_ids": ("selection", "instruction_ids"),
    "sub_paths":       ("selection", "sub_paths"),
    "max_episodes":    ("selection", "max_episodes"),
    "expname":         ("output",    "expname"),
    "run_name":        ("output",    "run_name"),
}

# Informational keys that should not flow into cfg.
_INFO_KEYS = {"split"}


def _deep_merge(dst: dict, src: dict) -> None:
    """Recursively merge ``src`` into ``dst`` in-place.

    Dict values are merged key-by-key; non-dict values overwrite.  Empty
    lists are treated as "no override" so a selection file can leave
    placeholder ``foo: []`` lines without silently clearing rollout
    defaults.
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        elif v == [] and isinstance(dst.get(k), list):
            continue  # empty list = no restriction; leave existing alone
        else:
            dst[k] = v


def apply_selection_yaml(cfg: dict, yaml_path) -> dict:
    """Merge a selection YAML into ``cfg`` in-place.  Returns ``cfg``.

    See the schema described above this function for what's recognised.
    """
    import yaml as _yaml
    with open(yaml_path) as f:
        sel = _yaml.safe_load(f) or {}

    # 1) Flat shortcuts — pop them out of `sel` and remap.
    for key, (section, field) in _FLAT_SHORTCUTS.items():
        if key not in sel:
            continue
        val = sel.pop(key)
        if val is None:
            continue
        if isinstance(val, list) and not val:
            continue   # empty list = "no restriction"
        cfg.setdefault(section, {})[field] = val

    # 2) Drop informational keys.
    for k in _INFO_KEYS:
        sel.pop(k, None)

    # 3) Deep-merge whatever remains.
    _deep_merge(cfg, sel)
    return cfg


def resolve_selection(cfg: dict, from_yaml_cli=None) -> dict:
    """Materialize all selection inputs into ``cfg`` in one place.

    Order of precedence:
      1. ``--from_yaml`` CLI flag (``from_yaml_cli`` argument)
      2. ``cfg.selection.from_yaml`` already set in the loaded YAML

    Whichever is set wins; its contents are merged via
    :func:`apply_selection_yaml`.  Subsequent code can read ``cfg.output.*``
    and ``cfg.selection.*`` without worrying about deferred loading.
    """
    if from_yaml_cli:
        cfg.setdefault("selection", {})["from_yaml"] = from_yaml_cli
    fy = cfg.get("selection", {}).get("from_yaml")
    if fy:
        apply_selection_yaml(cfg, fy)
    return cfg


# ── Audit ────────────────────────────────────────────────────────────────
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


# ── Survivor / drop YAMLs ────────────────────────────────────────────────
def write_keep_yaml(
    filt_dir:        Path,
    stage_num:       int,
    stage_name:      str,
    split:           str,
    instruction_ids: List[int],
    sub_paths:       Optional[Dict[int, List[int]]] = None,
    cfg:             Optional[dict] = None,
) -> Path:
    """Write a stage's survivor YAML.

    When ``cfg`` is provided, ``expname`` / ``run_name`` are echoed into the
    output so the file is **self-describing**: any downstream tool that
    reads it via ``--from_yaml`` or ``apply_selection_yaml`` will pick up
    the right experiment identity automatically (no need to also pass the
    original experiment selection YAML).
    """
    payload: dict = {"split": split}

    # Echo experiment identity so downstream consumers can resolve the
    # correct ``filter_dir`` / ``run_name`` from this file alone.
    if cfg is not None:
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
    path = filt_dir / f"{stage_num:02d}_{stage_name}.yaml"
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


def update_current(filt_dir: Path, keep_path: Path) -> Path:
    """Repoint ``current.yaml`` at the just-written keep file (relative link)."""
    current_path = filt_dir / "current.yaml"
    if current_path.exists() or current_path.is_symlink():
        current_path.unlink()
    current_path.symlink_to(keep_path.name)
    return current_path


# ── Survivor lookup (for downstream consumers) ───────────────────────────
def load_keep(yaml_path: Path) -> dict:
    """Read a keep YAML.  Returns a dict with keys ``instruction_ids``,
    ``sub_paths`` (may be missing for episode-level stages)."""
    with open(yaml_path) as f:
        return yaml.safe_load(f) or {}


def load_sub_path_filter(yaml_path: Path) -> Optional[Dict[int, List[int]]]:
    """Return ``{instruction_id: [sub_idx, ...]}`` if the YAML is sub-path
    level, otherwise ``None``."""
    data = load_keep(yaml_path)
    sp = data.get("sub_paths")
    if not sp:
        return None
    return {int(k): [int(v) for v in vs] for k, vs in sp.items()}


# ── Audit cell helpers ───────────────────────────────────────────────────
def ensure_episode(audit: dict, ep) -> dict:
    """Get-or-create the audit slot for one episode."""
    return audit["episodes"].setdefault(str(ep.instruction_id), {
        "scan":        ep.scan,
        "language":    ep.language,
        "n_sub_paths": len(ep.sub_paths),
        "stages":      {},
    })


def ensure_sub_path(ep_audit: dict, sub_idx: int) -> dict:
    """Get-or-create the audit slot for one sub-path within an episode."""
    sp_map = ep_audit.setdefault("sub_paths", {})
    return sp_map.setdefault(str(sub_idx), {"stages": {}})
