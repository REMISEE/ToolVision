from __future__ import annotations

import unittest

from offline_sft_pipeline.core.models import JudgeRecord, utc_now
from offline_sft_pipeline.eval.stop_policies import evaluate_stop_policy


def make_judge_record(
    *,
    trajectory_id: str,
    scope_type: str,
    overall_score: float,
    scope_step_idx: int | None = None,
    metadata: dict | None = None,
) -> JudgeRecord:
    return JudgeRecord(
        judge_record_id=f"judge__{trajectory_id}__{scope_type}__{scope_step_idx}",
        sample_id="demo__sample",
        trajectory_id=trajectory_id,
        scope_type=scope_type,
        scope_step_idx=scope_step_idx,
        judge_stage="committee",
        created_at=utc_now(),
        keep_for_frontier=True,
        exportable=False,
        overall_score=overall_score,
        metadata=dict(metadata or {}),
    )


class StopPolicyTests(unittest.TestCase):
    def test_binary_policy_stops_after_two_rounds_without_improvement(self) -> None:
        records = [
            make_judge_record(trajectory_id="traj", scope_type="trajectory", overall_score=0.2),
            make_judge_record(trajectory_id="traj", scope_type="step", scope_step_idx=1, overall_score=0.2),
            make_judge_record(trajectory_id="traj", scope_type="step", scope_step_idx=2, overall_score=0.2),
        ]
        decision = evaluate_stop_policy(
            source_dataset="gqa",
            answer="cat",
            judge_records=records,
        )
        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.stop_reason, "no_improvement_patience_exhausted")
        self.assertEqual(decision.details["policy_kind"], "exact_match_binary_score")

    def test_exact_match_policy_stops_immediately_on_quarter_regression_from_best(self) -> None:
        records = [
            make_judge_record(trajectory_id="traj", scope_type="trajectory", overall_score=0.2),
            make_judge_record(trajectory_id="traj", scope_type="step", scope_step_idx=1, overall_score=0.6),
            make_judge_record(trajectory_id="traj", scope_type="step", scope_step_idx=2, overall_score=0.35),
        ]
        decision = evaluate_stop_policy(
            source_dataset="gqa",
            answer="cat",
            judge_records=records,
        )
        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.stop_reason, "severe_regression")
        self.assertAlmostEqual(decision.details["severe_regression_margin"], 0.25)

    def test_exact_match_policy_allows_one_non_improving_round_when_not_down_by_one_eighth(self) -> None:
        records = [
            make_judge_record(trajectory_id="traj", scope_type="trajectory", overall_score=0.80),
            make_judge_record(trajectory_id="traj", scope_type="step", scope_step_idx=1, overall_score=0.76),
        ]
        decision = evaluate_stop_policy(
            source_dataset="gqa",
            answer="cat",
            judge_records=records,
        )
        self.assertFalse(decision.should_stop)
        self.assertIsNone(decision.stop_reason)

    def test_fsc147_policy_stops_after_two_rounds_without_beating_best(self) -> None:
        records = [
            make_judge_record(trajectory_id="traj", scope_type="trajectory", overall_score=0.89),
            make_judge_record(trajectory_id="traj", scope_type="step", scope_step_idx=1, overall_score=0.82),
            make_judge_record(trajectory_id="traj", scope_type="step", scope_step_idx=2, overall_score=0.83),
        ]
        decision = evaluate_stop_policy(
            source_dataset="fsc147",
            answer="10",
            judge_records=records,
        )
        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.metric_name, "overall_score")
        self.assertAlmostEqual(decision.best_value, 0.89)
        self.assertAlmostEqual(decision.current_value, 0.83)
        self.assertEqual(decision.stop_reason, "no_improvement_patience_exhausted")
        self.assertEqual(decision.details["policy_kind"], "count_relative_error_score")
        self.assertEqual(decision.details["patience"], 2)

    def test_fsc147_policy_allows_one_mild_regression_below_best(self) -> None:
        records = [
            make_judge_record(trajectory_id="traj", scope_type="trajectory", overall_score=0.89),
            make_judge_record(trajectory_id="traj", scope_type="step", scope_step_idx=1, overall_score=0.83),
        ]
        decision = evaluate_stop_policy(
            source_dataset="fsc147",
            answer="10",
            judge_records=records,
        )
        self.assertFalse(decision.should_stop)
        self.assertIsNone(decision.stop_reason)

    def test_fsc147_policy_stops_immediately_on_severe_regression(self) -> None:
        records = [
            make_judge_record(trajectory_id="traj", scope_type="trajectory", overall_score=0.89),
            make_judge_record(trajectory_id="traj", scope_type="step", scope_step_idx=1, overall_score=0.72),
        ]
        decision = evaluate_stop_policy(
            source_dataset="fsc147",
            answer="10",
            judge_records=records,
        )
        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.stop_reason, "severe_regression")
        self.assertAlmostEqual(decision.details["severe_regression_margin"], 0.10)


if __name__ == "__main__":
    unittest.main()
