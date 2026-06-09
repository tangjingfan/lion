"""Pipeline stage registry — the single source of truth for stage order.

Stages are referenced by NAME (not number). The list order *is* the pipeline
order, so inserting/reordering a stage no longer means renumbering files.

v1 wraps the existing ``scripts/NN_*.sh`` (legacy scripts stay working
unchanged). A later step replaces each shell-out with a direct ``run(cfg, args)``
call and drops the wrapper. See the runner in ``src/pipeline/__main__.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Stage:
    name: str                            # canonical name used by --only/--from/--to
    script: str                          # wrapper under scripts/ (v1 shells out to it)
    extra_flags: Tuple[str, ...] = ()    # fixed flags this stage needs
    needs: Tuple[str, ...] = ()          # run-relative inputs (advisory pre-check)
    produces: Tuple[str, ...] = ()       # run-relative outputs (for `list` status)
    desc: str = ""


# Order = pipeline order. This is the ONLY place the order lives.
STAGES: Tuple[Stage, ...] = (
    Stage("record_original", "00_record_original.sh",
          produces=("filters/00_original_dropped.yaml",),
          desc="record the selected episode set as the audit baseline"),
    Stage("cross_floor", "01_filter_multi_floor.sh",
          produces=("survivor.yaml", "filters/01_cross_floor_dropped.yaml"),
          desc="drop cross-floor sub-paths (the only hard drop)"),
    Stage("rewrite", "02_rewrite_subinstruction.sh",
          produces=("rewrite/",),
          desc="LLM: split each sub into spatial + landmark"),
    Stage("blacklist", "03_blacklist_landmark.sh",
          produces=("filters/02_blacklist_dropped.yaml",),
          desc="label non-referrable / unmapped landmarks"),
    Stage("partition", "04_partition.sh",
          needs=("rewrite/",),
          produces=("partition/", "partition_obs/"),
          desc="split spatial/landmark; render partition-pose obs"),
    Stage("object_list", "05_get_object_list.sh",
          extra_flags=("--objects_only",),
          produces=("scene_categories/",),
          desc="per-scan MPCat40 object vocabulary"),
    Stage("refine_mapping", "06_refine_landmark_mapping.sh",
          needs=("rewrite/", "scene_categories/"),
          produces=("rewrite/",),
          desc="LLM: refine the mention→label mapping per scan"),
    Stage("visibility", "07_list_potential_instances.sh",
          needs=("partition/",),
          produces=("target_instances/",),
          desc="annotate visibility/uniqueness at the partition pose"),
    Stage("select_target", "08_get_potential_instance.sh",
          needs=("target_instances/",),
          produces=("target_instances/",),
          desc="pick one MP40 instance id per surviving sub-path"),
    Stage("detection", "09_detection.sh",
          needs=("target_instances/",),
          produces=("detection/",),
          desc="YOLO-World landmark rescue at the partition pose"),
    Stage("apply_rescue", "10_apply_rescue.sh",
          needs=("detection/", "target_instances/"),
          produces=("target_instances/",),
          desc="fold detection hits back into target_instances"),
    Stage("synthesize", "11_rescue_blacklist.sh",
          needs=("target_instances/",),
          produces=("target_instances/",),
          desc="synthesize replacement landmarks visible at the partition pose"),
    Stage("consolidate", "12_consolidate.sh",
          needs=("target_instances/", "partition/"),
          produces=("dataset.json",),
          desc="stitch surviving sub-paths into dataset.json"),
    Stage("attrition", "13_attrition.sh",
          needs=("dataset.json",),
          desc="print the attrition report"),
)

NAMES: Tuple[str, ...] = tuple(s.name for s in STAGES)
_BY_NAME: Dict[str, Stage] = {s.name: s for s in STAGES}


def get(name: str) -> Stage:
    return _BY_NAME[name]


def index(name: str) -> int:
    return NAMES.index(name)
