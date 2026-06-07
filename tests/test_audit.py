"""Unit tests for src/pipeline/audit.py — label discipline + verdict resolver.

These cover the two bug classes observed in the val_unseen_one_scene_full
run (2026-06-07):

  * ``blacklist:blacklist:door`` double-prefixed sub_status labels
  * the (ep, sub) cell that landed in dataset.json twice (original
    rescued by detection 09 *and* synthesized by step 11)

Run:  pytest tests/test_audit.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.audit import (
    _ep_verdict,
    _sub_verdict,
    detection_rescued,
    finalize_audit,
    make_status_label,
    strip_stage_events,
)


# ── make_status_label ────────────────────────────────────────────────────
class TestMakeStatusLabel:
    def test_plain_reason_gets_stage_prefix(self):
        assert make_status_label("blacklist", "llm_keep_false") == "blacklist:llm_keep_false"

    def test_already_prefixed_reason_not_doubled(self):
        # The blacklist stage's term-hit reasons carry the prefix already.
        assert make_status_label("blacklist", "blacklist:door") == "blacklist:door"

    def test_other_stage(self):
        assert make_status_label("partition", "rewrite_error") == "partition:rewrite_error"

    def test_reason_equal_to_stage(self):
        assert make_status_label("blacklist", "blacklist") == "blacklist"

    def test_empty_reason(self):
        assert make_status_label("partition", "") == "partition:unknown"
        assert make_status_label("partition", None) == "partition:unknown"

    def test_prefix_must_match_whole_token(self):
        # "blacklist_extra:..." does NOT start with "blacklist:" — prefix added.
        assert (
            make_status_label("blacklist", "blacklist_extra:x")
            == "blacklist:blacklist_extra:x"
        )


# ── _sub_verdict state machine ───────────────────────────────────────────
class TestSubVerdict:
    def test_active_default(self):
        assert _sub_verdict([]) == ("active", False)

    def test_dropped(self):
        v, in_ds = _sub_verdict([
            {"stage": "cross_floor", "action": "dropped", "reason": "cross_floor"},
        ])
        # reason == stage → not repeated (was "dropped:cross_floor:cross_floor")
        assert v == "dropped:cross_floor"
        assert in_ds is False

    def test_dropped_reason_with_stage_prefix_not_doubled(self):
        v, _ = _sub_verdict([
            {"stage": "blacklist", "action": "dropped", "reason": "blacklist:door"},
        ])
        assert v == "dropped:blacklist:door"

    def test_kept_then_included(self):
        v, in_ds = _sub_verdict([
            {"stage": "select", "action": "selected",
             "status": "view_unique", "target_instance_ids": [7]},
            {"stage": "consolidate", "action": "included"},
        ])
        assert v == "kept:view_unique"
        assert in_ds is True

    def test_selected_without_target_is_labeled(self):
        v, _ = _sub_verdict([
            {"stage": "select", "action": "selected",
             "status": "visibility:not_visible", "target_instance_ids": []},
        ])
        assert v == "labeled:visibility:not_visible"

    def test_synth_sticky_over_excluded(self):
        # Step 12 excludes the original record of a synthesized sub; the
        # synth replacement is what's in the dataset.
        v, in_ds = _sub_verdict([
            {"stage": "blacklist", "action": "dropped", "reason": "llm_keep_false"},
            {"stage": "rescue_blacklist", "action": "synthesized", "new_landmark": "fridge"},
            {"stage": "consolidate", "action": "excluded", "reason": "blacklist:llm_keep_false"},
            {"stage": "consolidate", "action": "included", "synthesized": True},
        ])
        assert v == "synthesized:fridge"
        assert in_ds is True

    def test_excluded_without_synth(self):
        v, in_ds = _sub_verdict([
            {"stage": "visibility", "action": "labeled", "visibility": "not_visible"},
            {"stage": "consolidate", "action": "excluded", "reason": "visibility:not_visible"},
        ])
        assert v == "excluded:visibility:not_visible"
        assert in_ds is False

    def test_rescued(self):
        v, _ = _sub_verdict([
            {"stage": "detection", "action": "rescued", "method": "yolo_world"},
        ])
        assert v == "rescued:yolo_world"

    def test_rescue_failed_is_informational(self):
        v, _ = _sub_verdict([
            {"stage": "visibility", "action": "labeled", "visibility": "not_visible"},
            {"stage": "detection", "action": "rescue_failed", "reason": "no_detection"},
        ])
        assert v == "labeled:not_visible"

    def test_out_of_order_rerun_events_resolve_canonically(self):
        # Re-running stage 03 standalone appends its `dropped` event
        # AFTER stage 11's `synthesized` in the raw list. The resolver
        # must sort by canonical stage order, not append order.
        in_order = [
            {"stage": "blacklist", "action": "dropped", "reason": "llm_keep_false"},
            {"stage": "rescue_blacklist", "action": "synthesized", "new_landmark": "fridge"},
            {"stage": "consolidate", "action": "excluded", "reason": "blacklist:llm_keep_false"},
            {"stage": "consolidate", "action": "included", "synthesized": True},
        ]
        rerun_order = [in_order[1], in_order[0], in_order[2], in_order[3]]
        assert _sub_verdict(rerun_order) == _sub_verdict(in_order)
        assert _sub_verdict(rerun_order) == ("synthesized:fridge", True)

    def test_synth_superseded_restores_rescued_verdict(self):
        # The ep=98672 sub=2 scenario from val_unseen_one_scene_full:
        # visibility not_visible → detection rescued → step 11 synthesized
        # anyway → step 12 included the original and superseded the synth.
        v, in_ds = _sub_verdict([
            {"stage": "visibility", "action": "labeled", "visibility": "not_visible"},
            {"stage": "select", "action": "selected",
             "status": "visibility:not_visible", "target_instance_ids": []},
            {"stage": "detection", "action": "rescued", "method": "yolo_world"},
            {"stage": "apply_rescue", "action": "applied", "filled": True},
            {"stage": "rescue_blacklist", "action": "synthesized", "new_landmark": "chair"},
            {"stage": "consolidate", "action": "included", "synthesized": False,
             "target_status": "rescued"},
            {"stage": "consolidate", "action": "synth_superseded",
             "reason": "original_already_included", "new_landmark": "chair"},
        ])
        assert v == "rescued:yolo_world"
        assert in_ds is True


# ── detection_rescued ────────────────────────────────────────────────────
class TestDetectionRescued:
    def test_true_on_rescued_event(self):
        assert detection_rescued([
            {"stage": "detection", "action": "rescued", "method": "yolo_world"},
        ])

    def test_false_on_rescue_failed(self):
        assert not detection_rescued([
            {"stage": "detection", "action": "rescue_failed", "reason": "no_detection"},
        ])

    def test_false_on_other_stage_rescued(self):
        assert not detection_rescued([
            {"stage": "rescue_blacklist", "action": "synthesized"},
        ])

    def test_empty(self):
        assert not detection_rescued([])


# ── finalize_audit / strip_stage_events ──────────────────────────────────
def _audit_fixture():
    return {
        "_meta": {"split": "test", "stages": []},
        "episodes": {
            "1": {
                "scan": "s", "language": "en", "n_sub_paths": 2,
                "events": [{"stage": "cross_floor", "action": "kept"}],
                "sub_paths": {
                    "0": {"events": [
                        {"stage": "select", "action": "selected",
                         "status": "view_unique", "target_instance_ids": [3]},
                        {"stage": "consolidate", "action": "included"},
                    ]},
                    "1": {"events": [
                        {"stage": "blacklist", "action": "dropped",
                         "reason": "llm_keep_false"},
                    ]},
                },
            },
            "2": {
                "scan": "s", "language": "en", "n_sub_paths": 1,
                "events": [{"stage": "cross_floor", "action": "dropped",
                            "reason": "cross_floor"}],
                "sub_paths": {
                    "0": {"events": [{"stage": "consolidate", "action": "included"}]},
                },
            },
        },
    }


class TestFinalizeAudit:
    def test_summary_fields(self):
        audit = _audit_fixture()
        finalize_audit(audit)
        eps = audit["episodes"]
        assert eps["1"]["verdict"] == "kept"
        assert eps["1"]["sub_paths"]["0"]["verdict"] == "kept:view_unique"
        assert eps["1"]["sub_paths"]["0"]["in_dataset"] is True
        assert eps["1"]["sub_paths"]["1"]["verdict"] == "dropped:blacklist:llm_keep_false"

    def test_dropped_episode_overrides_subs(self):
        audit = _audit_fixture()
        finalize_audit(audit)
        sp = audit["episodes"]["2"]["sub_paths"]["0"]
        # Episode-level cross_floor drop wins over the sub's included event.
        assert sp["verdict"] == "dropped:cross_floor"
        assert sp["in_dataset"] is False

    def test_in_dataset_count_matches_included_cells(self):
        audit = _audit_fixture()
        finalize_audit(audit)
        n = sum(
            1
            for ep in audit["episodes"].values()
            for sp in ep["sub_paths"].values()
            if sp["in_dataset"]
        )
        assert n == 1  # only ep1#0; ep2#0 is killed by the episode drop

    def test_idempotent(self):
        audit = _audit_fixture()
        finalize_audit(audit)
        first = {
            (ep_id, s): (sp["verdict"], sp["in_dataset"])
            for ep_id, ep in audit["episodes"].items()
            for s, sp in ep["sub_paths"].items()
        }
        finalize_audit(audit)
        second = {
            (ep_id, s): (sp["verdict"], sp["in_dataset"])
            for ep_id, ep in audit["episodes"].items()
            for s, sp in ep["sub_paths"].items()
        }
        assert first == second


class TestStripStageEvents:
    def test_strips_only_named_stage(self):
        audit = _audit_fixture()
        strip_stage_events(audit, "consolidate")
        sp0 = audit["episodes"]["1"]["sub_paths"]["0"]
        assert all(e["stage"] != "consolidate" for e in sp0["events"])
        assert len(sp0["events"]) == 1  # select event survives
        # Other cells untouched.
        assert len(audit["episodes"]["1"]["sub_paths"]["1"]["events"]) == 1


# ── _visibility_not_visible_failures source filter (step 11) ─────────────
class TestVisibilityNotVisibleSource:
    def _make_audit(self, *, rescued: bool):
        events = [
            {"stage": "visibility", "action": "labeled",
             "visibility": "not_visible", "landmark": "outdoor furniture"},
        ]
        if rescued:
            events += [
                {"stage": "detection", "action": "rescued",
                 "method": "yolo_world", "instance_id": 353},
                {"stage": "apply_rescue", "action": "applied", "filled": True},
            ]
        return {
            "_meta": {"split": "test", "stages": []},
            "episodes": {
                "98672": {
                    "scan": "s", "language": "en", "n_sub_paths": 3,
                    "events": [],
                    "sub_paths": {"2": {"events": events}},
                },
            },
        }

    def test_unrescued_cell_is_candidate(self):
        from src.check.rescue_blacklist import _visibility_not_visible_failures
        out = _visibility_not_visible_failures(self._make_audit(rescued=False))
        assert [(ep, sub) for ep, sub, _ in out] == [(98672, 2)]
        assert out[0][2]["origin"] == "visibility_not_visible"

    def test_detection_rescued_cell_is_skipped(self):
        from src.check.rescue_blacklist import _visibility_not_visible_failures
        out = _visibility_not_visible_failures(self._make_audit(rescued=True))
        assert out == []
