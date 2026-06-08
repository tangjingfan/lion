"""Unit tests for src/process/visibility_analysis.py — the pure pixel/geometry
cores of the perturb-visibility robustness check (extracted from
src/check/perturb_visibility.py so they run without Habitat).

Run:  pytest tests/test_visibility_analysis.py -q
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.process.visibility_analysis import (
    category_for_instance,
    check_same_category_instances,
    check_targets_visible,
    overlay_target,
    perturbed_positions,
    target_categories,
    target_required_pixels,
)


# ── perturbed_positions ──────────────────────────────────────────────────
class TestPerturbedPositions:
    def test_count_and_keys(self):
        pts = perturbed_positions(np.array([0.0, 0.0, 0.0], np.float32), 0.5, 4)
        assert len(pts) == 4
        assert [p["k"] for p in pts] == [0, 1, 2, 3]
        assert set(pts[0]) == {"k", "angle_deg", "raw_pos", "raw_pos_np"}

    def test_stays_coplanar_in_y(self):
        start = np.array([10.0, 5.0, -3.0], np.float32)
        for p in perturbed_positions(start, 0.7, 6):
            assert p["raw_pos"][1] == 5.0  # Y (height) unchanged

    def test_radius_and_angles(self):
        pts = perturbed_positions(np.array([0.0, 0.0, 0.0], np.float32), 0.5, 4)
        assert [p["angle_deg"] for p in pts] == [0.0, 90.0, 180.0, 270.0]
        # angle 0 → +X ; angle 90 → +Z
        assert np.allclose(pts[0]["raw_pos"], [0.5, 0.0, 0.0], atol=1e-6)
        assert np.allclose(pts[1]["raw_pos"], [0.0, 0.0, 0.5], atol=1e-6)
        # every point is exactly `radius` from the start in the X-Z plane
        for p in pts:
            r = np.hypot(p["raw_pos"][0], p["raw_pos"][2])
            assert abs(r - 0.5) < 1e-6

    def test_raw_pos_np_dtype(self):
        pts = perturbed_positions(np.array([0.0, 0.0, 0.0], np.float32), 0.5, 2)
        assert pts[0]["raw_pos_np"].dtype == np.float32


# ── check_targets_visible ────────────────────────────────────────────────
class TestCheckTargetsVisible:
    def test_empty_targets(self):
        sem = np.zeros((2, 2), dtype=np.int32)
        out = check_targets_visible(sem, [], min_pixel=1)
        assert out["any_visible"] is False
        assert out["hits"] == []
        assert out["target_mask"] is None

    def test_pixel_counting_and_threshold(self):
        sem = np.array([[1, 1, 1], [2, 9, 9]], dtype=np.int32)  # id1×3, id2×1
        out = check_targets_visible(sem, [1, 2], min_pixel=2)
        assert out["n_visible_targets"] == 1          # only id1 clears the bar
        assert out["any_visible"] is True
        assert out["max_pixels"] == 3
        assert out["hits"] == [{"id": 1, "n_pixels": 3, "required_pixels": 2}]
        assert out["target_mask"].tolist() == [[True, True, True],
                                               [False, False, False]]

    def test_required_pixels_override(self):
        sem = np.array([[1, 1, 1], [0, 0, 0]], dtype=np.int32)  # id1×3
        out = check_targets_visible(sem, [1], min_pixel=2, required_pixels={1: 5})
        assert out["any_visible"] is False            # 3 < per-target 5
        assert out["max_pixels"] == 3
        assert out["required_pixels"] == {"1": 5}


# ── target_required_pixels ───────────────────────────────────────────────
class TestTargetRequiredPixels:
    def test_relative_and_floor(self):
        entry = {"candidates": [{"id": 1, "n_pixels": 100},
                                {"id": 2, "n_pixels": 10}]}
        out = target_required_pixels(entry, [1, 2, 3], min_pixel=50,
                                     original_fraction=0.8)
        assert out == {1: 80,   # ceil(100*0.8)=80 > floor 50
                       2: 50,   # ceil(10*0.8)=8  < floor 50
                       3: 50}   # no candidate → relative 0 → floor 50

    def test_zero_fraction_is_floor(self):
        entry = {"candidates": [{"id": 1, "n_pixels": 100}]}
        assert target_required_pixels(entry, [1], 30, 0.0) == {1: 30}


# ── category_for_instance ────────────────────────────────────────────────
class TestCategoryForInstance:
    NAME_MAP = np.array(["", "chair", "table", ""], dtype=object)

    def test_none_map(self):
        assert category_for_instance(None, 1) is None

    def test_out_of_range(self):
        assert category_for_instance(self.NAME_MAP, -1) is None
        assert category_for_instance(self.NAME_MAP, 4) is None

    def test_valid_and_empty(self):
        assert category_for_instance(self.NAME_MAP, 1) == "chair"
        assert category_for_instance(self.NAME_MAP, 0) is None   # "" → None
        assert category_for_instance(self.NAME_MAP, 3) is None


# ── target_categories ────────────────────────────────────────────────────
class TestTargetCategories:
    def test_from_candidates(self):
        entry = {"candidates": [{"id": 1, "category": "Chair"},
                                {"id": 2, "category": "sofa"}]}
        assert target_categories(entry, [1, 2], None) == ["Chair", "sofa"]

    def test_dedup_case_insensitive(self):
        entry = {"candidates": [{"id": 1, "category": "chair"},
                                {"id": 2, "category": "Chair"}]}
        assert target_categories(entry, [1, 2], None) == ["chair"]

    def test_name_map_fallback(self):
        name_map = np.array(["", "", "", "", "", "lamp"], dtype=object)
        out = target_categories({"candidates": []}, [5], name_map)
        assert out == ["lamp"]

    def test_matched_category_fallback(self):
        entry = {"matched_category": "door",
                 "matched_categories": ["door", "window"]}
        assert target_categories(entry, [], None) == ["door", "window"]


# ── check_same_category_instances ────────────────────────────────────────
class TestCheckSameCategoryInstances:
    NAME_MAP = np.array(["", "chair", "chair", "table"], dtype=object)
    SEM = np.array([[1, 1, 2], [2, 3, 3]], dtype=np.int32)  # 1:chair 2:chair 3:table

    def test_finds_other_same_category(self):
        out = check_same_category_instances(
            self.SEM, [1], ["chair"], self.NAME_MAP, min_pixel=1)
        assert out["n_other_same_category"] == 1          # id2 (chair, not target)
        assert out["category_unique"] is False
        assert out["hits"][0]["id"] == 2
        assert out["same_category_mask"].tolist() == [[False, False, True],
                                                      [True, False, False]]

    def test_unique_when_min_pixel_filters_others(self):
        out = check_same_category_instances(
            self.SEM, [1], ["chair"], self.NAME_MAP, min_pixel=3)
        assert out["n_other_same_category"] == 0          # id2 only 2px < 3
        assert out["category_unique"] is True

    def test_empty_categories(self):
        out = check_same_category_instances(
            self.SEM, [1], [], self.NAME_MAP, min_pixel=1)
        assert out["category_unique"] is None
        assert out["same_category_mask"] is None

    def test_none_sem(self):
        out = check_same_category_instances(
            None, [1], ["chair"], self.NAME_MAP, min_pixel=1)
        assert out["category_unique"] is None


# ── overlay_target ───────────────────────────────────────────────────────
class TestOverlayTarget:
    def test_none_mask_returns_unmodified_copy(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        out = overlay_target(rgb, None)
        assert out is not rgb            # a copy, not the same buffer
        assert np.array_equal(out, rgb)

    def test_empty_mask(self):
        rgb = np.full((2, 2, 3), 50, dtype=np.uint8)
        out = overlay_target(rgb, np.zeros((2, 2), dtype=bool))
        assert np.array_equal(out, rgb)

    def test_blends_only_masked_pixels(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        mask = np.array([[True, False], [False, False]])
        out = overlay_target(rgb, mask)
        # 0.55*0 + 0.45*[240,70,70] = [108, 31.5, 31.5] → uint8 [108, 31, 31]
        assert out[0, 0].tolist() == [108, 31, 31]
        assert out[0, 1].tolist() == [0, 0, 0]
        assert np.array_equal(rgb, np.zeros((2, 2, 3), dtype=np.uint8))  # input intact
