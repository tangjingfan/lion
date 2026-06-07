"""Backward-compat shim — the pipeline framework lives in ``src/pipeline``.

Everything that used to be defined here moved to its proper home:

    src/pipeline/config.py         — exp/selection resolution + run paths
    src/pipeline/survivor.py       — survivor.yaml IO + sub_status label channel
    src/pipeline/audit.py          — per-(ep, sub) lifecycle audit
    src/pipeline/rewrite_store.py  — step-02 rewrite artifact discovery/IO

Existing imports (``from src.check._filter_utils import ...``) keep
working through these re-exports; new code should import from
``src.pipeline.*`` directly.
"""

from __future__ import annotations

from src.pipeline.config import (  # noqa: F401
    apply_selection_yaml,
    get_filter_dir,
    get_run_dir,
    get_split,
    get_survivor_path,
    resolve_exp,
    resolve_run_name,
    resolve_selection,
    _deep_merge,
)
from src.pipeline.survivor import (  # noqa: F401
    active_subs,
    load_keep,
    sub_status_for,
    write_drop_yaml,
    write_survivor,
    _sub_status_map,
)
from src.pipeline.audit import (  # noqa: F401
    append_ep_event,
    append_sub_event,
    detection_rescued,
    ensure_episode,
    ensure_sub_path,
    finalize_audit,
    load_audit,
    make_status_label,
    register_stage,
    save_audit,
    strip_stage_events,
    _ep_verdict,
    _sub_verdict,
)
from src.pipeline.rewrite_store import (  # noqa: F401
    discover_rewrite_suffix,
    iter_rewrite_files,
    load_rewrite_by_scan,
    load_rewrite_episodes,
)
