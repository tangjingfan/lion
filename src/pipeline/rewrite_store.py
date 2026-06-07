"""Discovery + loading of the step-02 rewriter's output artifacts.

Rewriter output is per-episode::

    {rewrite_dir}/{scan}/{instruction_id}/sub_instructions_rewritten[_filtered].json

Each file holds ``{"model": str, "instruction_id": str, "episode": {...}}``.
Landmark mappings stay per-scan::

    {rewrite_dir}/{scan}/landmark_mapping[_filtered].json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


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
