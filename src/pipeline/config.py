"""Experiment/config resolution for the filter pipeline.

Owns the mapping from CLI handles (``--exp`` / ``--from_yaml`` /
selection YAMLs) to a fully-merged ``cfg`` dict, plus the derived run
paths.  Selection YAMLs are the per-experiment customization layer; the
rollout YAML stays simulator-only.

Selection YAML schema
---------------------
1) Flat shortcuts — concise aliases for the most common selection-y
   keys.  Remapped to nested cfg locations:

     scans:           [scan_id, ...]   → cfg.scenes.include
     languages:       [lang, ...]      → cfg.selection.languages
     instruction_ids: [int, ...]       → cfg.selection.instruction_ids
     sub_paths:       {id: [...]}      → cfg.selection.sub_paths
     max_episodes:    int              → cfg.selection.max_episodes
     expname:         str              → cfg.output.expname
     run_name:        str              → cfg.output.run_name
     split:           str              (informational only)

2) Anything else is deep-merged into ``cfg``, so a selection file can
   override any rollout cfg field with full nesting.
"""

from __future__ import annotations

from pathlib import Path

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

    The filters/ directory holds only diagnostic artifacts — per-stage
    ``NN_{name}_dropped.yaml`` and ``audit.json``.  The canonical
    survivor state lives at :func:`get_survivor_path`.
    """
    return get_run_dir(cfg) / "filters"


def get_survivor_path(cfg: dict) -> Path:
    """Resolve ``{base_dir}/{run_name}/survivor.yaml``.

    The single canonical survivor file.  Each filter stage overwrites it
    with the post-stage survivor set; :func:`resolve_exp` auto-merges it
    on top of any user-passed selection yaml so downstream stages always
    see the latest pipeline state.
    """
    return get_run_dir(cfg) / "survivor.yaml"


# ── Selection YAML merging ───────────────────────────────────────────────
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

    See the module docstring for the recognised schema.
    """
    with open(yaml_path) as f:
        sel = yaml.safe_load(f) or {}

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
