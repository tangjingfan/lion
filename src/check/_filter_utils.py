"""
Shared helpers for the LION-Bench filter pipeline.

The pipeline is a chain of numbered "stages" that progressively narrow
down which (instruction_id, sub_idx) pairs survive.  The current state
is centralized in a single ``survivor.yaml`` at the run dir root; each
stage overwrites it as it narrows the pipeline.  Diagnostics live in
``filters/``:

    {run_dir}/survivor.yaml            — single source of truth, the
                                          latest survivor set (replaces
                                          the old per-stage NN_*.yaml +
                                          current.yaml combo)
    {run_dir}/filters/NN_{name}_dropped.yaml
                                       — per-stage drop reasons (kept
                                          only for debugging; not read
                                          back by downstream tools)
    {run_dir}/filters/audit.json       — per-stage status for every
                                          (ep, sub_idx); cross-stage
                                          trace

Survivor YAML schema
--------------------
Episode-level (stage 1):

    split: LandmarkRxR_val_unseen
    instruction_ids: [19199, ...]

Sub-path-level (stage 2+):

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
they cross floors. Every later stage **labels** drops via ``sub_status``
rather than removing them, so downstream rescue / inspection can still
reach them. Use :func:`active_subs` to iterate the un-labeled set
(matches the old per-stage survivor contents) and :func:`sub_status_for`
to query a specific sub.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

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
    """Resolve ``{base_dir}/{run_name}/filters`` from a rollout-style config.

    The filters/ directory now holds only diagnostic artifacts —
    per-stage ``NN_{name}_dropped.yaml`` and ``audit.json``. The
    canonical survivor state lives at :func:`get_survivor_path`.
    """
    return get_run_dir(cfg) / "filters"


def get_survivor_path(cfg: dict) -> Path:
    """Resolve ``{base_dir}/{run_name}/survivor.yaml``.

    The single canonical survivor file. Each filter stage overwrites it
    with the post-stage survivor set; :func:`resolve_exp` auto-merges it
    on top of any user-passed selection yaml so downstream stages always
    see the latest pipeline state.
    """
    return get_run_dir(cfg) / "survivor.yaml"


# ── Rewrite output discovery ──────────────────────────────────────────────
#
# Rewriter output is per-episode:
#     {rewrite_dir}/{scan}/{instruction_id}/sub_instructions_rewritten[_filtered].json
# Each file holds ``{"model": str, "instruction_id": str, "episode": {...}}``.
# Landmark mappings stay per-scan:
#     {rewrite_dir}/{scan}/landmark_mapping[_filtered].json

def discover_rewrite_suffix(rewrite_dir: Path) -> Optional[str]:
    """Pick the suffix variant present under ``rewrite_dir``.

    Prefers ``_filtered`` over the unfiltered variant.  Returns ``None``
    if no per-episode rewrite file exists anywhere under the directory.
    """
    if not rewrite_dir.exists():
        return None
    for suffix in ("_filtered", ""):
        for scan_dir in rewrite_dir.iterdir():
            if not scan_dir.is_dir():
                continue
            for ep_dir in scan_dir.iterdir():
                if not ep_dir.is_dir():
                    continue
                if (ep_dir / f"sub_instructions_rewritten{suffix}.json").exists():
                    return suffix
    return None


def _ep_dir_sort_key(d: Path):
    try:
        return (0, int(d.name))
    except ValueError:
        return (1, d.name)


def iter_rewrite_files(
    rewrite_dir: Path, suffix: str,
) -> Iterator[Tuple[str, str, Path]]:
    """Yield ``(scan, instruction_id, path)`` for each per-episode rewrite."""
    if not rewrite_dir.exists():
        return
    for scan_dir in sorted(rewrite_dir.iterdir()):
        if not scan_dir.is_dir():
            continue
        ep_dirs = [d for d in scan_dir.iterdir() if d.is_dir()]
        for ep_dir in sorted(ep_dirs, key=_ep_dir_sort_key):
            p = ep_dir / f"sub_instructions_rewritten{suffix}.json"
            if p.exists():
                yield scan_dir.name, ep_dir.name, p


def load_rewrite_episodes(
    rewrite_dir: Path,
    suffix: Optional[str] = None,
) -> Tuple[Dict[str, Dict], Optional[str], List[Path]]:
    """Merge per-episode rewrite JSONs into ``{instruction_id: episode}``.

    If ``suffix`` is ``None``, discovers the preferred suffix automatically.
    Returns ``(episodes, suffix_used, paths_read)``.  When no rewrites
    exist, returns ``({}, None, [])``.
    """
    if suffix is None:
        suffix = discover_rewrite_suffix(rewrite_dir)
    if suffix is None:
        return {}, None, []
    episodes: Dict[str, Dict] = {}
    paths: List[Path] = []
    for _scan, ep_id, p in iter_rewrite_files(rewrite_dir, suffix):
        with open(p) as f:
            data = json.load(f)
        episodes[ep_id] = data["episode"]
        paths.append(p)
    return episodes, suffix, paths


def load_rewrite_by_scan(
    rewrite_dir: Path,
    suffix: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Dict]], Optional[str], List[Path]]:
    """Like :func:`load_rewrite_episodes` but grouped: ``{scan: {ep_id: ep}}``."""
    if suffix is None:
        suffix = discover_rewrite_suffix(rewrite_dir)
    if suffix is None:
        return {}, None, []
    by_scan: Dict[str, Dict[str, Dict]] = {}
    paths: List[Path] = []
    for scan, ep_id, p in iter_rewrite_files(rewrite_dir, suffix):
        with open(p) as f:
            data = json.load(f)
        by_scan.setdefault(scan, {})[ep_id] = data["episode"]
        paths.append(p)
    return by_scan, suffix, paths


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
    "sub_status":      ("selection", "sub_status"),
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


def resolve_exp(cfg: dict, exp_arg=None, apply_current: bool = True) -> dict:
    """Resolve an experiment handle into ``cfg`` + the current pipeline state.

    ``exp_arg`` can be:
      • a path to a selection-style YAML (the original
        ``configs/selection/*.yaml`` OR any ``filters/NN_*.yaml`` survivor
        file) — its fields are merged into ``cfg``.
      • a bare ``expname`` string — set as ``cfg.output.expname`` so
        :func:`get_run_dir` can resolve the experiment directory.

    After applying the user's input, when ``apply_current`` is True, look
    up ``<run_dir>/survivor.yaml`` and merge it on top. This
    automatically restores the latest pipeline survivor set (including
    per-episode ``sub_paths`` drops) for stages 2+, so the user only ever
    has to pass the original selection yaml or the expname — never the
    survivor file.

    Stages that *create* a survivor set (e.g. stage 0 record_original,
    stage 1 cross_floor) should pass ``apply_current=False`` so they read
    the seed input directly.
    """
    if exp_arg:
        as_path = Path(exp_arg)
        if as_path.is_file():
            apply_selection_yaml(cfg, as_path)
        else:
            cfg.setdefault("output", {})["expname"] = exp_arg

    if apply_current:
        try:
            survivor = get_survivor_path(cfg)
        except Exception:
            return cfg
        if survivor.exists():
            apply_selection_yaml(cfg, survivor)
            print(f"[resolve_exp] applied {survivor}")

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
def write_survivor(
    cfg:             dict,
    split:           str,
    instruction_ids: List[int],
    sub_paths:       Optional[Dict[int, List[int]]] = None,
    sub_status:      Optional[Dict[int, Dict[int, str]]] = None,
) -> Path:
    """Overwrite the canonical ``{run_dir}/survivor.yaml``.

    Replaces the legacy per-stage ``filters/NN_<name>.yaml`` + ``current.yaml``
    symlink combo with a single file at the run-dir root that always
    reflects the latest pipeline state. Self-describing: echoes
    ``expname`` / ``run_name`` so it doubles as a valid selection yaml —
    any tool can read it via ``apply_selection_yaml``.

    ``sub_paths`` carries every alive sub-path (everything past cross_floor,
    including labeled drops). ``sub_status``, when provided, attaches a
    failure label to a subset of those subs — e.g.::

        sub_status:
          1239:
            0: "blacklist:llm_keep_false"

    Sub-paths absent from ``sub_status`` are "active" (no failure yet).
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
    stage to skip labeled subs without re-deriving from the audit:

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


# ── Audit cell helpers ───────────────────────────────────────────────────
#
# audit.json is the single per-(ep, sub) lifecycle log. Every pipeline
# stage from 00 (record_original) through 12 (consolidate) appends one
# event per touched cell describing what it did:
#
#   events: [
#     {"stage": "blacklist",   "action": "dropped",   "reason": "llm_keep_false"},
#     {"stage": "rescue_blacklist", "action": "synthesized", "new_landmark": "fridge"},
#     {"stage": "consolidate", "action": "included"}
#   ]
#
# ``finalize_audit`` resolves the event list into a short ``verdict``
# string + ``in_dataset`` bool per cell. ``strip_stage_events`` lets a
# stage drop its own previous events so re-runs are idempotent.
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


def _sub_verdict(events: List[dict]) -> Tuple[str, bool]:
    """Walk events in order and resolve to ``(verdict, in_dataset)``.

    State machine:
      - ``dropped``        → verdict := dropped:<stage>:<reason>
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
    """
    verdict = "active"
    in_dataset = False
    synth_seen = False
    for e in events:
        a = e.get("action")
        if a == "dropped":
            verdict = f"dropped:{e.get('stage') or '?'}:{e.get('reason') or 'drop'}"
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
    for e in events:
        if e.get("action") == "dropped":
            return f"dropped:{e.get('stage') or '?'}:{e.get('reason') or 'drop'}"
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
