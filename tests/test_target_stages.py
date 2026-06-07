"""Unit tests for the step-07/08 decision cores:
src/process/target_annotation.py (visibility/uniqueness classification)
and src/process/target_selection.py (target-instance selection rule).

Run:  pytest tests/test_target_stages.py -q
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.process.target_annotation import classify_visibility, resolve_partition_pos
from src.process.target_selection import (
    record_candidates,
    record_is_unique,
    record_visibility_status,
    select_target,
)


# ── classify_visibility (step 07) ────────────────────────────────────────
class TestClassifyVisibility:
    POS = np.zeros(3, dtype=np.float32)

    def test_unresolvable_pose(self):
        assert classify_visibility(None, {"matched_categories": ["bed"], "n_instances": 1}) \
            == ("partition_pos_unresolvable", "not_visible")

    def test_no_match(self):
        assert classify_visibility(self.POS, {"matched_categories": []}) \
            == ("no_match", "not_visible")
        assert classify_visibility(self.POS, None) == ("no_match", "not_visible")

    def test_not_visible(self):
        assert classify_visibility(
            self.POS, {"matched_categories": ["bed"], "n_instances": 0},
        ) == ("not_visible", "not_visible")

    def test_visible_unique(self):
        assert classify_visibility(
            self.POS, {"matched_categories": ["bed"], "n_instances": 1},
        ) == ("visible", True)

    def test_visible_ambiguous(self):
        assert classify_visibility(
            self.POS, {"matched_categories": ["chair"], "n_instances": 3},
        ) == ("visible", False)


# ── resolve_partition_pos ────────────────────────────────────────────────
class TestResolvePartitionPos:
    DB = {"node_a": np.array([1.0, 0.0, 2.0], dtype=np.float32)}
    VIRT = {"virt:0": [3.0, 0.0, 4.0]}

    def test_real_node(self):
        pos = resolve_partition_pos({"spatial_path": ["x", "node_a"]}, {}, self.DB)
        np.testing.assert_allclose(pos, [1.0, 0.0, 2.0])

    def test_virtual_node(self):
        pos = resolve_partition_pos({"spatial_path": ["virt:0"]}, self.VIRT, self.DB)
        np.testing.assert_allclose(pos, [3.0, 0.0, 4.0])

    def test_missing(self):
        assert resolve_partition_pos({"spatial_path": []}, {}, self.DB) is None
        assert resolve_partition_pos({"spatial_path": ["nope"]}, {}, self.DB) is None


# ── record accessors (legacy-schema tolerance) ───────────────────────────
class TestRecordAccessors:
    def test_candidates_sorted_by_pixels(self):
        rec = {"candidates": [{"id": 1, "n_pixels": 10}, {"id": 2, "n_pixels": 99}]}
        assert [c["id"] for c in record_candidates(rec)] == [2, 1]

    def test_legacy_instances_key(self):
        rec = {"instances": [{"id": 7, "n_pixels": 5}]}
        assert [c["id"] for c in record_candidates(rec)] == [7]

    def test_visibility_new_schema(self):
        assert record_visibility_status({"visibility": "not_visible"}) == "not_visible"

    def test_visibility_legacy_uniqueness_strings(self):
        assert record_visibility_status({"uniqueness": "unique"}) == "visible"
        assert record_visibility_status({"uniqueness": "ambiguous"}) == "visible"
        assert record_visibility_status({"uniqueness": "not_visible"}) == "not_visible"

    def test_is_unique(self):
        assert record_is_unique({"uniqueness": True}) is True
        assert record_is_unique({"uniqueness": "unique"}) is True
        assert record_is_unique({"uniqueness": "ambiguous"}) is False
        assert record_is_unique({"uniqueness": "not_visible"}) is None


# ── select_target (step 08) ──────────────────────────────────────────────
def _visible_record(candidates):
    return {
        "landmark": "chair",
        "visibility": "visible",
        "uniqueness": len(candidates) == 1,
        "candidates": candidates,
    }


class TestSelectTarget:
    END = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def test_not_visible_passthrough(self):
        out = select_target({"visibility": "not_visible", "candidates": []})
        assert out["status"] == "visibility:not_visible"
        assert out["target_instance_ids"] == []

    def test_single_candidate_view_unique(self):
        out = select_target(_visible_record([{"id": 5, "n_pixels": 100}]))
        assert out["status"] == "view_unique"
        assert out["target_instance_ids"] == [5]

    def test_multi_picks_nearest_to_end(self):
        cands = [{"id": 1, "n_pixels": 999}, {"id": 2, "n_pixels": 10}]
        centers = {
            1: np.array([5.0, 0.0, 0.0]),   # 5 m from end
            2: np.array([1.0, 0.0, 0.0]),   # 1 m from end → wins despite fewer px
        }
        out = select_target(
            _visible_record(cands), end_pos=self.END, instance_centers=centers,
        )
        assert out["status"] == "view_nearest"
        assert out["target_instance_ids"] == [2]
        assert abs(out["selection_distance"] - 1.0) < 1e-6
        assert set(out["candidate_distances"]) == {"1", "2"}

    def test_multi_without_centers_falls_back_to_largest_pixels(self):
        cands = [{"id": 1, "n_pixels": 999}, {"id": 2, "n_pixels": 10}]
        out = select_target(
            _visible_record(cands), end_pos=self.END, instance_centers={},
        )
        assert out["status"] == "view_nearest_fallback"
        assert out["target_instance_ids"] == [1]
        assert out["fallback_reason"] == "instance_centers_unavailable"

    def test_multi_without_end_pos_reason(self):
        cands = [{"id": 1, "n_pixels": 9}, {"id": 2, "n_pixels": 10}]
        out = select_target(_visible_record(cands), end_pos=None)
        assert out["status"] == "view_nearest_fallback"
        assert out["fallback_reason"] == "subpath_end_pos_unresolvable"
