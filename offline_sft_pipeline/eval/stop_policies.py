from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from offline_sft_pipeline.core.dataset_names import canonicalize_dataset_name, is_high_conf_exact_match_dataset
from offline_sft_pipeline.core.models import JudgeRecord


FSC147_NO_IMPROVEMENT_PATIENCE = 2
FSC147_SEVERE_REGRESSION_MARGIN = 0.10
EXACT_MATCH_NO_IMPROVEMENT_PATIENCE = 2
EXACT_MATCH_SEVERE_REGRESSION_MARGIN = 0.25


@dataclass(frozen=True, slots=True)
class StopPolicyDecision:
    dataset_name: str
    metric_name: str
    current_value: float
    previous_value: float | None
    best_value: float | None
    no_improve_rounds: int
    should_stop: bool
    stop_reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


def evaluate_stop_policy(
    *,
    source_dataset: str,
    answer: Any,
    judge_records: Sequence[JudgeRecord],
) -> StopPolicyDecision:
    dataset_name = canonicalize_dataset_name(source_dataset)
    if dataset_name == "fsc147":
        return _evaluate_fsc147_stop_policy(
            dataset_name=dataset_name,
            judge_records=judge_records,
        )
    if is_high_conf_exact_match_dataset(dataset_name):
        return _evaluate_exact_match_stop_policy(
            dataset_name=dataset_name,
            judge_records=judge_records,
        )
    return _evaluate_score_stop_policy(
        dataset_name=dataset_name,
        judge_records=judge_records,
        policy_kind="binary_score",
    )


def _evaluate_score_stop_policy(
    *,
    dataset_name: str,
    judge_records: Sequence[JudgeRecord],
    policy_kind: str,
) -> StopPolicyDecision:
    values = [float(record.overall_score) for record in judge_records]
    current_value, previous_value, best_value, no_improve_rounds = _history_summary(values)
    should_stop, stop_reason = _shared_stop_decision(
        current_value=current_value,
        previous_value=previous_value,
        no_improve_rounds=no_improve_rounds,
    )
    return StopPolicyDecision(
        dataset_name=dataset_name,
        metric_name="overall_score",
        current_value=current_value,
        previous_value=previous_value,
        best_value=best_value,
        no_improve_rounds=no_improve_rounds,
        should_stop=should_stop,
        stop_reason=stop_reason,
        details={"policy_kind": policy_kind},
    )


def _evaluate_fsc147_stop_policy(
    *,
    dataset_name: str,
    judge_records: Sequence[JudgeRecord],
) -> StopPolicyDecision:
    values = [float(record.overall_score) for record in judge_records]
    current_value, previous_value, best_value, no_improve_rounds = _history_summary(values)
    should_stop, stop_reason = _fsc147_stop_decision(
        current_value=current_value,
        best_value=best_value,
        no_improve_rounds=no_improve_rounds,
        patience=FSC147_NO_IMPROVEMENT_PATIENCE,
        severe_regression_margin=FSC147_SEVERE_REGRESSION_MARGIN,
    )
    return StopPolicyDecision(
        dataset_name=dataset_name,
        metric_name="overall_score",
        current_value=current_value,
        previous_value=previous_value,
        best_value=best_value,
        no_improve_rounds=no_improve_rounds,
        should_stop=should_stop,
        stop_reason=stop_reason,
        details={
            "policy_kind": "count_relative_error_score",
            "patience": FSC147_NO_IMPROVEMENT_PATIENCE,
            "severe_regression_margin": FSC147_SEVERE_REGRESSION_MARGIN,
        },
    )


def _evaluate_exact_match_stop_policy(
    *,
    dataset_name: str,
    judge_records: Sequence[JudgeRecord],
) -> StopPolicyDecision:
    values = [float(record.overall_score) for record in judge_records]
    current_value, previous_value, best_value, no_improve_rounds = _history_summary(values)
    should_stop, stop_reason = _fsc147_stop_decision(
        current_value=current_value,
        best_value=best_value,
        no_improve_rounds=no_improve_rounds,
        patience=EXACT_MATCH_NO_IMPROVEMENT_PATIENCE,
        severe_regression_margin=EXACT_MATCH_SEVERE_REGRESSION_MARGIN,
    )
    return StopPolicyDecision(
        dataset_name=dataset_name,
        metric_name="overall_score",
        current_value=current_value,
        previous_value=previous_value,
        best_value=best_value,
        no_improve_rounds=no_improve_rounds,
        should_stop=should_stop,
        stop_reason=stop_reason,
        details={
            "policy_kind": "exact_match_binary_score",
            "patience": EXACT_MATCH_NO_IMPROVEMENT_PATIENCE,
            "severe_regression_margin": EXACT_MATCH_SEVERE_REGRESSION_MARGIN,
        },
    )


def _shared_stop_decision(
    *,
    current_value: float,
    previous_value: float | None,
    no_improve_rounds: int,
) -> tuple[bool, str | None]:
    if previous_value is not None and current_value < previous_value:
        return True, "regressed"
    if no_improve_rounds >= 2:
        return True, "no_improvement_patience_exhausted"
    return False, None


def _fsc147_stop_decision(
    *,
    current_value: float,
    best_value: float | None,
    no_improve_rounds: int,
    patience: int,
    severe_regression_margin: float,
) -> tuple[bool, str | None]:
    if best_value is not None and (best_value - current_value) >= severe_regression_margin:
        return True, "severe_regression"
    if no_improve_rounds >= patience:
        return True, "no_improvement_patience_exhausted"
    return False, None


def _history_summary(values: Sequence[float]) -> tuple[float, float | None, float | None, int]:
    if not values:
        raise ValueError("stop policy requires at least one judge value.")
    current_value = float(values[-1])
    previous_value = float(values[-2]) if len(values) >= 2 else None
    best_value = max(float(item) for item in values[:-1]) if len(values) >= 2 else None

    best_so_far = float(values[0])
    no_improve_rounds = 0
    for value in values[1:]:
        numeric_value = float(value)
        if numeric_value > best_so_far:
            best_so_far = numeric_value
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
    return current_value, previous_value, best_value, no_improve_rounds


__all__ = [
    "StopPolicyDecision",
    "evaluate_stop_policy",
]
