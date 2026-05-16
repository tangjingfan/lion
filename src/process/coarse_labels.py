"""Helpers for detecting coarse-only MPCAT40 groundings.

Some MPCAT40 labels (``appliances``, ``objects``, ``furniture``, ``lighting``)
are useful for rendering but too coarse to prove that a fine-grained landmark
mention ("stove", "lamp", "thermostat") is separately represented by the
semantic annotation. The VLM rescue stage uses these helpers to decide which
targets to send to YOLO-World for a fine-category recovery attempt.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple


DEFAULT_COARSE_LABELS = {
    "appliances",
    "objects",
    "furniture",
    "lighting",
}


def _norm(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("_", " ").lower()).strip()


def _label_candidates(rec: Dict) -> List[str]:
    """Collect the semantic labels associated with a target-instance record.

    Prefers the labels of the actually selected candidate(s); falls back to
    every visible candidate plus the record's matched / semantic label fields.
    """
    labels: List[str] = []
    target_ids = set(rec.get("target_instance_ids") or [])
    selected_candidate_labels = [
        _norm(cand.get("category"))
        for cand in (rec.get("candidates") or [])
        if cand.get("category") and cand.get("id") in target_ids
    ]
    if selected_candidate_labels:
        labels.extend(selected_candidate_labels)
    else:
        for cand in rec.get("candidates") or []:
            if cand.get("category"):
                labels.append(_norm(cand["category"]))
        for key in ("matched_category",):
            if rec.get(key):
                labels.append(_norm(rec[key]))
        for key in ("semantic_labels", "matched_categories"):
            vals = rec.get(key) or []
            if isinstance(vals, list):
                labels.extend(_norm(v) for v in vals if v)
    seen = set()
    out: List[str] = []
    for label in labels:
        if label and label not in seen:
            out.append(label)
            seen.add(label)
    return out


def _is_only_coarse_label(
    landmark: str, labels: Iterable[str], coarse: set[str],
) -> Tuple[bool, str]:
    """Return ``(is_only_coarse, first_hit)`` for this record.

    A record is "coarse-only" when every grounded semantic label is in
    ``coarse``. If the record also has a more specific label, or if the
    landmark mention is itself the coarse concept (e.g. someone literally
    asking for an "appliance"), the verdict is False.
    """
    lm = _norm(landmark)
    label_set = {_norm(x) for x in labels if _norm(x)}
    hit = sorted(label_set & coarse)
    if not hit:
        return False, ""

    non_coarse = sorted(label_set - coarse)
    if non_coarse:
        return False, ""

    coarse_forms = set(hit)
    coarse_forms.update(x[:-1] for x in hit if x.endswith("s"))
    if lm in coarse_forms:
        return False, ""

    return True, hit[0]
