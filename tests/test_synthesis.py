"""Unit tests for the pure parts of src/process/synthesis.py and
src/env/geometry.py — candidate ranking, scoring, instruction template,
voxelisation.

Run:  pytest tests/test_synthesis.py -q
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.geometry import unproject_equirect, voxelise_per_instance
from src.process.synthesis import (
    MIN_VISIBILITY_RATIO,
    merge_drops,
    rank_candidates,
    synth_sub_instruction,
    _size_weight,
)


# ── synth_sub_instruction ────────────────────────────────────────────────
class TestSynthSubInstruction:
    def test_with_spatial(self):
        assert synth_sub_instruction("Turn right.", "fridge") == \
            "Turn right. Walk to a fridge."

    def test_vowel_article(self):
        assert synth_sub_instruction("", "oven") == "Walk to an oven."

    def test_strips_trailing_period(self):
        assert synth_sub_instruction("Turn left", "sink") == \
            "Turn left. Walk to a sink."

    def test_empty_spatial(self):
        assert synth_sub_instruction(None, "bed") == "Walk to a bed."


# ── size weight ──────────────────────────────────────────────────────────
class TestSizeWeight:
    def test_saturates_at_one(self):
        assert _size_weight(200) == 1.0
        assert _size_weight(10_000) == 1.0

    def test_linear_below_saturation(self):
        assert _size_weight(100) == 0.5

    def test_clamps_negative(self):
        assert _size_weight(-5) == 0.0


# ── merge_drops ──────────────────────────────────────────────────────────
class TestMergeDrops:
    def test_first_source_wins(self):
        a = [(1, 0, {"origin": "blacklist"})]
        b = [(1, 0, {"origin": "detection_failure"}), (2, 1, {"origin": "detection_failure"})]
        merged = merge_drops(a, b)
        assert len(merged) == 2
        assert merged[0][2]["origin"] == "blacklist"          # (1,0) kept from a
        assert merged[1][:2] == (2, 1)


# ── rank_candidates (geometric filters) ──────────────────────────────────
def _meta(center):
    return {"category": "chair", "center": np.asarray(center, dtype=np.float32)}


class TestRankCandidates:
    P = np.array([0.0, 0.0, 0.0], dtype=np.float32)   # partition pose
    E = np.array([4.0, 0.0, 0.0], dtype=np.float32)   # end pose

    def _rank(self, p_vox, e_vox, meta, table=None):
        return rank_candidates(
            partition_voxels=p_vox, end_voxels=e_vox, inst_meta=meta,
            partition_pos=self.P, end_pos=self.E,
            referrability=table or {},
        )

    def test_approaching_visible_candidate_passes(self):
        # Instance at x=5: dist from P=5, from E=1 → approach +4.
        out = self._rank({7: 300}, {7: 300}, {7: _meta([5, 0, 0])})
        assert [c["instance_id"] for c in out] == [7]
        assert out[0]["approach_m"] > 0

    def test_low_visibility_ratio_filtered(self):
        # ratio = 10/300 < MIN_VISIBILITY_RATIO (0.10)
        assert MIN_VISIBILITY_RATIO == 0.10
        out = self._rank({7: 10}, {7: 300}, {7: _meta([5, 0, 0])})
        assert out == []

    def test_receding_instance_filtered(self):
        # Instance behind the agent: dist from P=1 < dist from E=5.
        out = self._rank({7: 300}, {7: 300}, {7: _meta([-1, 0, 0])})
        assert out == []

    def test_too_generic_tier_filtered(self):
        out = self._rank(
            {7: 300}, {7: 300}, {7: _meta([5, 0, 0])},
            table={"chair": "too_generic"},
        )
        assert out == []

    def test_score_ordering_size_weight_mutes_small(self):
        # iid 1: big approach but tiny (10 voxels) → score 6 × 0.05 = 0.3
        # iid 2: smaller approach, saturated size → score 2 × 1.0 = 2.0
        meta = {
            1: _meta([7, 0, 0]),    # dist P=7, E=3 → approach 4... use bigger
            2: _meta([5, 0, 0]),    # dist P=5, E=1 → approach 4
        }
        # Give iid1 approach 4 with 10 voxels (score 0.2), iid2 approach 4
        # with 300 voxels (score 4.0) — iid2 must rank first.
        out = self._rank({1: 10, 2: 300}, {1: 10, 2: 300}, meta)
        assert [c["instance_id"] for c in out] == [2, 1]


# ── geometry ─────────────────────────────────────────────────────────────
class TestGeometry:
    def test_voxelise_counts_distinct_cells(self):
        pts = np.array([
            [0.01, 0.01, 0.01],   # cell (0,0,0)
            [0.02, 0.02, 0.02],   # same cell
            [0.95, 0.01, 0.01],   # cell (9,0,0)
        ], dtype=np.float32)
        iids = np.array([5, 5, 5], dtype=np.int32)
        out = voxelise_per_instance(pts, iids, 0.10)
        assert out == {5: 2}

    def test_voxelise_empty(self):
        assert voxelise_per_instance(
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int32), 0.1,
        ) == {}

    def test_unproject_center_pixel_points_forward(self):
        # A 4x8 panorama with depth 2 everywhere; the center pixel
        # (theta=0, phi=0) must unproject to (0, 0, -2) + pos at heading 0
        # (agent faces -z).
        H, W = 4, 8
        depth = np.full((H, W), 2.0, dtype=np.float32)
        sem   = np.zeros((H, W), dtype=np.int32)
        pos   = np.array([1.0, 0.5, 1.0], dtype=np.float32)
        pts, iids = unproject_equirect(depth, sem, pos, heading=0.0)
        assert pts.shape == (H * W, 3)
        # center pixel: u = W/2 → theta 0; v = H/2 → phi 0
        idx = (H // 2) * W + (W // 2)
        np.testing.assert_allclose(pts[idx], pos + [0, 0, -2.0], atol=1e-5)

    def test_unproject_filters_invalid_depth(self):
        depth = np.array([[0.0, 5.0]], dtype=np.float32)
        sem   = np.array([[1, 2]], dtype=np.int32)
        pts, iids = unproject_equirect(depth, sem, np.zeros(3, np.float32), 0.0)
        assert len(pts) == 1 and iids.tolist() == [2]
