from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

from offline_sft_pipeline.eval.stop_policies import evaluate_stop_policy
from offline_sft_pipeline.core.dataset_names import (
    canonicalize_dataset_name,
    is_reference_forced_final_answer_dataset,
)
from offline_sft_pipeline.core.models import (
    Budget,
    ConversationMessage,
    ExecutorRuntimeResult,
    ForkProvenance,
    ImageArtifactRef,
    JudgeRecord,
    JudgeStage,
    PendingExecution,
    PlannerOutput,
    PlannerSuggestion,
    PlannerStepSpec,
    RootSample,
    StepRecord,
    TrajectoryErrorInfo,
    TrajectoryRecord,
    utc_now,
)
from offline_sft_pipeline.core.sample_normalization import normalize_root_sample
from offline_sft_pipeline.core.store import DEFAULT_SYSTEM_MESSAGE, OfflineTrajectoryStore
from offline_sft_pipeline.pipelines.executor_client import ExecutorClient
from offline_sft_pipeline.pipelines.judge_client import JudgeClient
from offline_sft_pipeline.pipelines.planner_client import PlannerClient
from offline_sft_pipeline.pipelines.request_models import (
    ExecutorClientRequest,
    JudgeClientRequest,
    PlannerClientRequest,
    ToolCapability,
)
from offline_sft_pipeline.runtime.types import (
    ArtifactRef as RuntimeArtifactRef,
    RuntimeStepOutput,
    RuntimeStepRequest,
)
from collections import Counter

MAX_PLANNER_SUGGESTIONS = 3
TERMINAL_STATUSES = {
    "answered",
    "pruned",
    "failed",
    "stopped_early",
    "max_step_reached",
    "error",
}


class StepRuntime(Protocol):
    def run_step_sync(self, request: RuntimeStepRequest | dict[str, Any]) -> RuntimeStepOutput: ...


@dataclass(slots=True)
class OrchestratorConfig:
    planner_suggestion_count: int = 2
    max_child_trajectories: int = 2
    default_budget: Budget = field(
        default_factory=lambda: Budget(
            remaining_exec_steps=6,
        )
    )
    system_message: str = DEFAULT_SYSTEM_MESSAGE
    judge_stage: JudgeStage = "cheap_filter"
    tool_capabilities: Sequence[ToolCapability | dict[str, Any]] = field(default_factory=tuple)
    stop_unselected_trajectories: bool = True
    enable_root_baseline_judge: bool = True
    enable_count_forced_final_answer: bool = True
    count_forced_final_answer_score_threshold: float = 0.98
    enable_exact_match_forced_final_answer: bool = True
    exact_match_forced_final_answer_score_threshold: float = 0.9
    enable_reference_perfect_sample_early_exit: bool = True
    reference_perfect_sample_early_exit_score_threshold: float = 0.999
    enable_count_perfect_sample_early_exit: bool = True
    count_perfect_sample_early_exit_score_threshold: float = 0.999
    force_first_round_must_suggest: bool = True
    must_suggest_score_threshold: float = 0.25
    must_answer_score_threshold: float = 0.75

    def __post_init__(self) -> None:
        if self.planner_suggestion_count < 1:
            raise ValueError("planner_suggestion_count must be >= 1.")
        if self.max_child_trajectories < 1:
            raise ValueError("max_child_trajectories must be >= 1.")
        if not 0.0 <= self.count_forced_final_answer_score_threshold <= 1.0:
            raise ValueError("count_forced_final_answer_score_threshold must be between 0 and 1.")
        if not 0.0 <= self.exact_match_forced_final_answer_score_threshold <= 1.0:
            raise ValueError("exact_match_forced_final_answer_score_threshold must be between 0 and 1.")
        if not 0.0 <= self.reference_perfect_sample_early_exit_score_threshold <= 1.0:
            raise ValueError("reference_perfect_sample_early_exit_score_threshold must be between 0 and 1.")
        if not 0.0 <= self.count_perfect_sample_early_exit_score_threshold <= 1.0:
            raise ValueError("count_perfect_sample_early_exit_score_threshold must be between 0 and 1.")
        if not 0.0 <= self.must_suggest_score_threshold <= 1.0:
            raise ValueError("must_suggest_score_threshold must be between 0 and 1.")
        if not 0.0 <= self.must_answer_score_threshold <= 1.0:
            raise ValueError("must_answer_score_threshold must be between 0 and 1.")
        if self.must_suggest_score_threshold > self.must_answer_score_threshold:
            raise ValueError("must_suggest_score_threshold must be <= must_answer_score_threshold.")

    @property
    def effective_planner_suggestion_count(self) -> int:
        return min(self.planner_suggestion_count, MAX_PLANNER_SUGGESTIONS)


@dataclass(slots=True)
class OrchestratorRunResult:
    sample_id: str
    root_trajectory_id: str
    all_trajectory_ids: list[str]
    running_trajectory_ids: list[str]
    expanded_trajectory_ids: list[str]
    terminal_trajectory_ids: list[str]


@dataclass(slots=True)
class _PlannerRoundContext:
    trajectory: TrajectoryRecord
    messages: list[ConversationMessage]
    visible_images: list[ImageArtifactRef]
    planner_output: PlannerOutput
    requested_suggestion_count: int
    must_answer_now: bool
    planning_policy: str


@dataclass(slots=True)
class _ForcedFinalAnswerSignal:
    reason: str
    candidate_answer: str
    overall_score: float
    successful_model_count: int
    model_names: list[str]


@dataclass(slots=True)
class _FrontierCandidate:
    parent_trajectory: TrajectoryRecord
    planner_output: PlannerOutput
    suggestion: PlannerSuggestion
    suggestion_order: int


@dataclass(slots=True)
class _ReadyFrontierTrajectory:
    trajectory: TrajectoryRecord
    frontier_metric: float
    judge_record: JudgeRecord


class OrchestratorV01:
    """Minimal offline branching orchestrator with planner/executor/runtime/judge wiring."""

    def __init__(
        self,
        *,
        store: OfflineTrajectoryStore,
        planner_client: PlannerClient,
        executor_client: ExecutorClient,
        judge_client: JudgeClient,
        runtime: StepRuntime,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self.store = store
        self.planner_client = planner_client
        self.executor_client = executor_client
        self.judge_client = judge_client
        self.runtime = runtime
        self.config = config or OrchestratorConfig()
        self.tool_capabilities = self._normalize_tool_capabilities(self.config.tool_capabilities)

    def run(
        self,
        root_sample: RootSample | dict[str, Any],
        *,
        budget: Budget | dict[str, Any] | None = None,
    ) -> OrchestratorRunResult:
        sample = self._coerce_root_sample(root_sample)
        initial_budget = self._coerce_budget(budget) if budget is not None else self.config.default_budget.model_copy(deep=True)
        initialized = self.store.init_root_trajectory(
            sample,
            budget=initial_budget,
            system_message=self.config.system_message,
        )
        root_trajectory = initialized.trajectory
        if self.config.enable_root_baseline_judge:
            root_trajectory = self._run_root_baseline_judge(root_trajectory)

        frontier: list[TrajectoryRecord] = [root_trajectory]
        while frontier:
            sample_finish_frontier: list[TrajectoryRecord] | None = None
            planner_contexts: list[_PlannerRoundContext] = []
            for frontier_item in frontier:
                trajectory = self.store.load_trajectory(frontier_item.sample_id, frontier_item.trajectory_id)
                if trajectory.status != "running":
                    continue
                try:
                    planner_contexts.append(self._plan_trajectory(trajectory))
                except Exception as exc:
                    self._mark_trajectory_error(
                        trajectory,
                        code="planner_error",
                        message=str(exc),
                        round_idx=self._next_planner_round_idx(trajectory),
                    )

            if not planner_contexts:
                break

            ready_frontier_items: list[_ReadyFrontierTrajectory] = []
            for context in planner_contexts:
                parent_candidates = self._planner_candidates_for_context(context)
                refreshed_parent = self.store.register_planner_round(
                    context.planner_output,
                    selected_for_expansion=bool(parent_candidates),
                    pending_execution=None,
                    final_answer=context.planner_output.direct_answer,
                )

                if context.planner_output.can_answer_now:
                    self._append_final_answer_message(
                        refreshed_parent,
                        planner_think=context.planner_output.global_chain_cot,
                        final_answer=context.planner_output.direct_answer or "",
                        planner_round_idx=context.planner_output.round_idx,
                    )
                    self.store.mark_trajectory_status(
                        refreshed_parent.sample_id,
                        refreshed_parent.trajectory_id,
                        status="answered",
                        final_answer=context.planner_output.direct_answer,
                        pending_execution=None,
                    )
                    continue

                if context.must_answer_now:
                    self.store.mark_trajectory_status(
                        refreshed_parent.sample_id,
                        refreshed_parent.trajectory_id,
                        status="max_step_reached",
                        pending_execution=None,
                    )
                    continue

                if not parent_candidates:
                    self._finalize_unselected_parent(refreshed_parent)
                    continue

                child_created = False
                for candidate in parent_candidates:
                    try:
                        child = self._spawn_child_trajectory(candidate, parent_trajectory=refreshed_parent)
                    except Exception as exc:
                        self._mark_trajectory_error(
                            refreshed_parent,
                            code="child_init_error",
                            message=str(exc),
                            round_idx=context.planner_output.round_idx,
                        )
                        continue

                    child_created = True
                    executed_child, frontier_metric, judge_record = self._execute_child_step(child, candidate)
                    if executed_child.status == "running" and frontier_metric is not None and judge_record is not None:
                        perfect_exit_signal = self._maybe_build_perfect_sample_exit_signal(
                            root_sample=sample,
                            trajectory=executed_child,
                            judge_record=judge_record,
                        )
                        if perfect_exit_signal is not None:
                            self.store.mark_trajectory_expanded(
                                refreshed_parent.sample_id,
                                refreshed_parent.trajectory_id,
                            )
                            sample_finish_frontier = self._prepare_sample_for_reference_perfect_finish(
                                winner_trajectory=executed_child,
                            )
                            ready_frontier_items = []
                            break
                        ready_frontier_items.append(
                            _ReadyFrontierTrajectory(
                                trajectory=executed_child,
                                frontier_metric=frontier_metric,
                                judge_record=judge_record,
                            )
                        )

                if sample_finish_frontier is not None:
                    break
                if child_created:
                    self.store.mark_trajectory_expanded(
                        refreshed_parent.sample_id,
                        refreshed_parent.trajectory_id,
                    )
                elif refreshed_parent.status == "running":
                    self._finalize_unselected_parent(refreshed_parent)

            if sample_finish_frontier is not None:
                frontier = sample_finish_frontier
                continue
            frontier = self._select_next_frontier(ready_frontier_items)

        trajectories = self.store.list_trajectories(sample_id=sample.sample_id)
        all_ids = [item.trajectory_id for item in trajectories]
        running_ids = [item.trajectory_id for item in trajectories if item.status == "running"]
        expanded_ids = [item.trajectory_id for item in trajectories if item.status == "expanded"]
        terminal_ids = [item.trajectory_id for item in trajectories if item.status in TERMINAL_STATUSES]
        return OrchestratorRunResult(
            sample_id=sample.sample_id,
            root_trajectory_id=initialized.trajectory.trajectory_id,
            all_trajectory_ids=all_ids,
            running_trajectory_ids=running_ids,
            expanded_trajectory_ids=expanded_ids,
            terminal_trajectory_ids=terminal_ids,
        )

    def _plan_trajectory(self, trajectory: TrajectoryRecord) -> _PlannerRoundContext:
        messages_doc = self.store.load_messages(trajectory.sample_id, trajectory.trajectory_id)
        visible_images = self._select_visible_images(trajectory)
        root_sample = self.store.load_root_sample(trajectory.sample_id)
        next_round_idx = self._next_planner_round_idx(trajectory)
        forced_final_answer = self._maybe_build_forced_final_answer_signal(
            trajectory,
            root_sample=root_sample,
        )
        planning_policy, policy_reason, latest_overall_score = self._determine_planning_policy(
            trajectory,
            next_round_idx=next_round_idx,
            forced_final_answer=forced_final_answer,
        )
        must_answer_now = planning_policy == "must_answer"
        requested_suggestion_count = 0 if must_answer_now else self.config.effective_planner_suggestion_count
        metadata: dict[str, Any] = {
            "orchestrator_version": "v01",
            "planning_policy": planning_policy,
            "planning_policy_reason": policy_reason,
        }
        if latest_overall_score is not None:
            metadata["latest_overall_score"] = latest_overall_score
        if forced_final_answer is not None:
            metadata["forced_final_answer_audit"] = {
                "reason": forced_final_answer.reason,
                "candidate_answer": forced_final_answer.candidate_answer,
                "overall_score": forced_final_answer.overall_score,
                "successful_model_count": forced_final_answer.successful_model_count,
                "model_names": list(forced_final_answer.model_names),
            }
        if must_answer_now:
            judge_consensus_answer = self._maybe_build_judge_consensus_answer_hint(trajectory)
            if judge_consensus_answer is not None:
                metadata["judge_consensus_answer_hint"] = {
                    "reason": judge_consensus_answer.reason,
                    "candidate_answer": judge_consensus_answer.candidate_answer,
                    "overall_score": judge_consensus_answer.overall_score,
                    "successful_model_count": judge_consensus_answer.successful_model_count,
                    "model_names": list(judge_consensus_answer.model_names),
                }
        planner_request = PlannerClientRequest(
            sample_id=trajectory.sample_id,
            trajectory_id=trajectory.trajectory_id,
            round_idx=next_round_idx,
            sample_dir=str(self.store.sample_dir(trajectory.sample_id)),
            trajectory_dir=str(self.store.trajectory_dir(trajectory.sample_id, trajectory.trajectory_id)),
            planner_dir=str(self.store.planner_dir(trajectory.sample_id, trajectory.trajectory_id)),
            steps_dir=str(self.store.steps_dir(trajectory.sample_id, trajectory.trajectory_id)),
            question=trajectory.question,
            answer_instruction=trajectory.answer_instruction,
            messages=list(messages_doc.root),
            visible_images=visible_images,
            budget=trajectory.budget.model_copy(deep=True),
            must_answer_now=must_answer_now,
            tool_capabilities=[item.model_copy(deep=True) for item in self.tool_capabilities],
            latest_runtime_result=self._load_latest_runtime_result(trajectory),
            requested_suggestion_count=requested_suggestion_count,
            metadata=metadata,
        )
        planner_output = self.planner_client.run(planner_request)
        return _PlannerRoundContext(
            trajectory=trajectory,
            messages=list(messages_doc.root),
            visible_images=visible_images,
            planner_output=planner_output,
            requested_suggestion_count=requested_suggestion_count,
            must_answer_now=must_answer_now,
            planning_policy=planning_policy,
        )

    def _determine_planning_policy(
        self,
        trajectory: TrajectoryRecord,
        *,
        next_round_idx: int,
        forced_final_answer: _ForcedFinalAnswerSignal | None,
    ) -> tuple[str, str, float | None]:
        if trajectory.budget.remaining_exec_steps <= 0:
            return "must_answer", "budget_exhausted", None

        is_first_round = next_round_idx == 0 and trajectory.step_idx == 0
        if self.config.force_first_round_must_suggest and is_first_round:
            return "must_suggest", "first_round", None

        if forced_final_answer is not None:
            return "must_answer", forced_final_answer.reason, float(forced_final_answer.overall_score)

        latest_record = self._latest_judge_record_for_trajectory(trajectory)
        if latest_record is None:
            return "may_answer_or_suggest", "no_judge_score", None

        latest_overall_score = float(latest_record.overall_score)
        if latest_overall_score <= self.config.must_suggest_score_threshold:
            return "must_suggest", "score_at_or_below_must_suggest_threshold", latest_overall_score
        if latest_overall_score >= self.config.must_answer_score_threshold:
            return "must_answer", "score_at_or_above_must_answer_threshold", latest_overall_score
        return "may_answer_or_suggest", "score_in_middle_band", latest_overall_score

    def _planner_candidates_for_context(
        self,
        context: _PlannerRoundContext,
    ) -> list[_FrontierCandidate]:
        if context.planner_output.can_answer_now or context.requested_suggestion_count <= 0:
            return []
        usable_suggestions = context.planner_output.suggestions[: context.requested_suggestion_count]
        return [
            _FrontierCandidate(
                parent_trajectory=context.trajectory,
                planner_output=context.planner_output,
                suggestion=suggestion,
                suggestion_order=suggestion_order,
            )
            for suggestion_order, suggestion in enumerate(usable_suggestions)
        ]

    def _select_next_frontier(
        self,
        ready_items: Sequence[_ReadyFrontierTrajectory],
    ) -> list[TrajectoryRecord]:
        ranked_items = sorted(
            ready_items,
            key=lambda item: (
                -item.frontier_metric,
                -float(item.judge_record.overall_score),
                item.trajectory.trajectory_id,
            ),
        )
        selected = ranked_items[: self.config.max_child_trajectories]
        unselected = ranked_items[self.config.max_child_trajectories :]
        if self.config.stop_unselected_trajectories:
            for item in unselected:
                self.store.mark_trajectory_status(
                    item.trajectory.sample_id,
                    item.trajectory.trajectory_id,
                    status="stopped_early",
                    pending_execution=None,
                )
        return [
            self.store.load_trajectory(item.trajectory.sample_id, item.trajectory.trajectory_id)
            for item in selected
        ]

    def _spawn_child_trajectory(
        self,
        candidate: _FrontierCandidate,
        *,
        parent_trajectory: TrajectoryRecord | None = None,
    ) -> TrajectoryRecord:
        selected_step = candidate.suggestion.steps[0]
        source_parent = parent_trajectory or candidate.parent_trajectory
        fork_provenance = ForkProvenance(
            parent_trajectory_id=source_parent.trajectory_id,
            parent_planner_round_idx=candidate.planner_output.round_idx,
            parent_suggestion_id=candidate.suggestion.suggestion_id,
        )
        pending_execution = PendingExecution(
            planner_round_idx=candidate.planner_output.round_idx,
            suggestion_id=candidate.suggestion.suggestion_id,
            suggestion_step_index=0,
            step_id=selected_step.step_id,
        )
        child_budget = self._build_child_budget(source_parent.budget)
        initialized = self.store.init_child_trajectory(
            source_parent,
            fork_provenance=fork_provenance,
            pending_execution=pending_execution,
            budget=child_budget,
        )
        return initialized.trajectory

    def _execute_child_step(
        self,
        child_trajectory: TrajectoryRecord,
        candidate: _FrontierCandidate,
    ) -> tuple[TrajectoryRecord, float | None, JudgeRecord | None]:
        selected_step = candidate.suggestion.steps[0]
        step_idx = child_trajectory.step_idx + 1
        visible_images = self._select_visible_images(child_trajectory)
        input_artifact, runtime_image_index = self._resolve_runtime_input(
            visible_images,
            selected_step.input_image_index,
        )
        messages_doc = self.store.load_messages(child_trajectory.sample_id, child_trajectory.trajectory_id)

        try:
            executor_output = self.executor_client.run(
                ExecutorClientRequest(
                    sample_id=child_trajectory.sample_id,
                    trajectory_id=child_trajectory.trajectory_id,
                    round_idx=candidate.planner_output.round_idx,
                    step_idx=step_idx,
                    sample_dir=str(self.store.sample_dir(child_trajectory.sample_id)),
                    trajectory_dir=str(
                        self.store.trajectory_dir(child_trajectory.sample_id, child_trajectory.trajectory_id)
                    ),
                    planner_dir=str(
                        self.store.planner_dir(child_trajectory.sample_id, child_trajectory.trajectory_id)
                    ),
                    steps_dir=str(self.store.steps_dir(child_trajectory.sample_id, child_trajectory.trajectory_id)),
                    question=child_trajectory.question,
                    messages=list(messages_doc.root),
                    visible_images=visible_images,
                    suggestion_id=candidate.suggestion.suggestion_id,
                    suggestion_step_index=0,
                    step_spec=selected_step,
                    planner_global_chain_cot=candidate.planner_output.global_chain_cot,
                    suggestion_cot=candidate.suggestion.suggestion_cot,
                    tool_capabilities=[item.model_copy(deep=True) for item in self.tool_capabilities],
                    metadata={"orchestrator_version": "v01"},
                )
            )
        except Exception as exc:
            return self._mark_trajectory_error(
                child_trajectory,
                code="executor_error",
                message=str(exc),
                round_idx=candidate.planner_output.round_idx,
                step_idx=step_idx,
            ), None, None

        step_paths = self.store.write_executor_step_files(
            child_trajectory.sample_id,
            child_trajectory.trajectory_id,
            step_idx,
            executor_cot=executor_output.cot,
            executor_code=executor_output.code,
        )

        try:
            runtime_output = self.runtime.run_step_sync(
                RuntimeStepRequest(
                    sample_id=child_trajectory.sample_id,
                    trajectory_id=child_trajectory.trajectory_id,
                    round_idx=candidate.planner_output.round_idx,
                    step_idx=step_idx,
                    executor_cot_path=str(step_paths.executor_cot_path),
                    executor_code_path=str(step_paths.executor_code_path),
                    visible_images=self._to_runtime_visible_images(visible_images),
                    step_output_dir=str(step_paths.step_dir),
                    image_index=runtime_image_index,
                )
            )
            runtime_result = ExecutorRuntimeResult.from_dict(
                runtime_output.runtime_result,
                validate_schema=True,
            )
        except Exception as exc:
            return self._mark_trajectory_error(
                child_trajectory,
                code="runtime_error",
                message=str(exc),
                round_idx=candidate.planner_output.round_idx,
                step_idx=step_idx,
            ), None, None

        assistant_message = self._build_assistant_step_message(
            child_trajectory=child_trajectory,
            planner_round_idx=candidate.planner_output.round_idx,
            step_idx=step_idx,
            suggestion_id=candidate.suggestion.suggestion_id,
            selected_step=selected_step,
            executor_cot=executor_output.cot,
            executor_code=executor_output.code,
            executor_description=executor_output.description,
            runtime_image_index=runtime_image_index,
            input_artifact_id=input_artifact.artifact_id,
            executor_code_path=step_paths.executor_code_path,
        )
        tool_message = self._build_tool_step_message(
            child_trajectory=child_trajectory,
            step_idx=step_idx,
            runtime_result=runtime_result,
            runtime_result_path=step_paths.runtime_result_path,
        )
        updated_messages = self.store.append_messages(
            child_trajectory.sample_id,
            child_trajectory.trajectory_id,
            [assistant_message, tool_message],
        )

        step_record = StepRecord(
            execution_trajectory_id=child_trajectory.trajectory_id,
            step_idx=step_idx,
            planner_round_idx=candidate.planner_output.round_idx,
            suggestion_id=candidate.suggestion.suggestion_id,
            suggestion_step_index=0,
            step_id=selected_step.step_id,
            step_goal=selected_step.step_goal,
            input_image_index=selected_step.input_image_index,
            input_artifact_id=input_artifact.artifact_id,
            capability_plan=[item.model_copy(deep=True) for item in selected_step.capability_plan],
            executor_description=executor_output.description,
            executor_cot_path=str(step_paths.executor_cot_path),
            executor_code_path=str(step_paths.executor_code_path),
            runtime_result_path=str(step_paths.runtime_result_path),
            assistant_message_id=assistant_message.message_id,
            tool_message_id=tool_message.message_id,
            executor_metadata=dict(executor_output.metadata),
        )
        self.store.register_step_record(
            child_trajectory.sample_id,
            child_trajectory.trajectory_id,
            step_record,
            clear_pending_execution=True,
        )

        post_step_trajectory = self.store.load_trajectory(
            child_trajectory.sample_id,
            child_trajectory.trajectory_id,
        )
        post_step_visible_images = self._select_visible_images(post_step_trajectory)
        root_sample = self.store.load_root_sample(post_step_trajectory.sample_id)

        try:
            judge_record = self.judge_client.run(
                JudgeClientRequest(
                    sample_id=post_step_trajectory.sample_id,
                    trajectory_id=post_step_trajectory.trajectory_id,
                    sample_dir=str(self.store.sample_dir(post_step_trajectory.sample_id)),
                    trajectory_dir=str(
                        self.store.trajectory_dir(
                            post_step_trajectory.sample_id,
                            post_step_trajectory.trajectory_id,
                        )
                    ),
                    scope_type="step",
                    scope_step_idx=step_idx,
                    judge_stage=self.config.judge_stage,
                    question=post_step_trajectory.question,
                    answer_instruction=post_step_trajectory.answer_instruction,
                    answer=root_sample.answer,
                    messages=list(updated_messages.root),
                    visible_images=post_step_visible_images,
                    planner_output=candidate.planner_output,
                    step_record=step_record,
                    runtime_result=runtime_result,
                    final_answer=None,
                    metadata={
                        "orchestrator_version": "v01",
                        **dict(root_sample.metadata),
                    },
                )
            )
        except Exception as exc:
            return self._mark_trajectory_error(
                post_step_trajectory,
                code="judge_error",
                message=str(exc),
                round_idx=candidate.planner_output.round_idx,
                step_idx=step_idx,
            ), None, None

        return self._finalize_child_after_step(
            post_step_trajectory,
            root_sample=root_sample,
            runtime_result=runtime_result,
            judge_record=judge_record,
        )

    def _finalize_child_after_step(
        self,
        trajectory: TrajectoryRecord,
        *,
        root_sample: RootSample,
        runtime_result: ExecutorRuntimeResult,
        judge_record: JudgeRecord,
    ) -> tuple[TrajectoryRecord, float | None, JudgeRecord | None]:
        updated = self.store.load_trajectory(trajectory.sample_id, trajectory.trajectory_id)
        frontier_metric: float | None = None

        if not runtime_result.success:
            updated.status = "failed"
        elif not runtime_result.images and not runtime_result.text.strip():
            updated.status = "pruned"
        else:
            stop_decision = self._evaluate_stop_policy(updated, judge_record=judge_record, root_sample=root_sample)
            judge_record.metadata.update(
                {
                    "stop_policy": {
                        "dataset_name": stop_decision.dataset_name,
                        "metric_name": stop_decision.metric_name,
                        "current_value": stop_decision.current_value,
                        "previous_value": stop_decision.previous_value,
                        "best_value": stop_decision.best_value,
                        "no_improve_rounds": stop_decision.no_improve_rounds,
                        "should_stop": stop_decision.should_stop,
                        "stop_reason": stop_decision.stop_reason,
                        "details": dict(stop_decision.details),
                    }
                }
            )
            frontier_metric = float(stop_decision.current_value)
            should_force_answer_next = frontier_metric >= self.config.must_answer_score_threshold
            if stop_decision.should_stop and should_force_answer_next:
                judge_record.metadata["stop_policy_must_answer_override"] = {
                    "reason": "score_at_or_above_must_answer_threshold",
                    "must_answer_score_threshold": self.config.must_answer_score_threshold,
                    "current_value": frontier_metric,
                    "suppressed_stop_reason": stop_decision.stop_reason,
                }
            self.store.register_judge_record(judge_record)
            updated = self.store.load_trajectory(trajectory.sample_id, trajectory.trajectory_id)
            if stop_decision.should_stop and should_force_answer_next:
                updated.status = "running"
            elif stop_decision.should_stop:
                updated.status = "stopped_early"
            else:
                updated.status = "running"

        updated.pending_execution = None
        updated.updated_at = utc_now()
        self.store.save_trajectory(updated)
        return updated, frontier_metric, judge_record

    def _finalize_unselected_parent(self, trajectory: TrajectoryRecord) -> TrajectoryRecord:
        if not self.config.stop_unselected_trajectories:
            return trajectory
        return self.store.mark_trajectory_status(
            trajectory.sample_id,
            trajectory.trajectory_id,
            status="stopped_early",
            pending_execution=None,
        )

    def _append_final_answer_message(
        self,
        trajectory: TrajectoryRecord,
        *,
        planner_think: str,
        final_answer: str,
        planner_round_idx: int,
        final_answer_source: str = "planner",
        final_answer_reason: str | None = None,
    ) -> None:
        think_block = ""
        if str(planner_think or "").strip():
            think_block = f"<think>\n{planner_think}\n</think>\n"
        metadata = {
            "message_kind": "final_answer",
            "planner_round_idx": planner_round_idx,
            "final_answer_source": final_answer_source,
        }
        if final_answer_reason is not None:
            metadata["final_answer_reason"] = final_answer_reason
        final_message = ConversationMessage(
            message_id="m_final_answer",
            role="assistant",
            content=f"{think_block}<answer>\n{final_answer}\n</answer>",
            image_artifact_ids=[],
            metadata=metadata,
        )
        self.store.append_messages(
            trajectory.sample_id,
            trajectory.trajectory_id,
            [final_message],
        )

    def _build_assistant_step_message(
        self,
        *,
        child_trajectory: TrajectoryRecord,
        planner_round_idx: int,
        step_idx: int,
        suggestion_id: str,
        selected_step: PlannerStepSpec,
        executor_cot: str,
        executor_code: str,
        executor_description: str,
        runtime_image_index: int,
        input_artifact_id: str,
        executor_code_path: Path,
    ) -> ConversationMessage:
        tool_call_payload = {
            "name": "code_image_tool",
            "arguments": {
                "code": executor_code,
                "description": executor_description,
                "image_index": runtime_image_index,
            },
        }
        return ConversationMessage(
            message_id=f"m_step_{step_idx:03d}_assistant",
            role="assistant",
            content=(
                f"<think>\n{executor_cot}\n</think>\n"
                f"<tool_call>\n{json.dumps(tool_call_payload, ensure_ascii=False, indent=2)}\n</tool_call>"
            ),
            image_artifact_ids=[],
            metadata={
                "message_kind": "executor_step",
                "step_idx": step_idx,
                "planner_round_idx": planner_round_idx,
                "suggestion_id": suggestion_id,
                "step_id": selected_step.step_id,
                "input_image_index": selected_step.input_image_index,
                "input_artifact_id": input_artifact_id,
                "executor_description": executor_description,
                "runtime_image_index": runtime_image_index,
                "executor_code_path": self._path_relative_to_trajectory(
                    child_trajectory.sample_id,
                    child_trajectory.trajectory_id,
                    executor_code_path,
                ),
            },
        )

    def _build_tool_step_message(
        self,
        *,
        child_trajectory: TrajectoryRecord,
        step_idx: int,
        runtime_result: ExecutorRuntimeResult,
        runtime_result_path: Path,
    ) -> ConversationMessage:
        metadata = {
            "message_kind": "tool_result",
            "step_idx": step_idx,
            "tool_name": "code_image_tool",
            "runtime_result_path": self._path_relative_to_trajectory(
                child_trajectory.sample_id,
                child_trajectory.trajectory_id,
                runtime_result_path,
            ),
        }
        if runtime_result.images:
            metadata["primary_image_artifact_id"] = runtime_result.images[0].artifact_id
        return ConversationMessage(
            message_id=f"m_step_{step_idx:03d}_tool",
            role="tool",
            content=runtime_result.text or "",
            image_artifact_ids=[item.artifact_id for item in runtime_result.images],
            metadata=metadata,
        )

    def _select_visible_images(self, trajectory: TrajectoryRecord) -> list[ImageArtifactRef]:
        visible_images = [item.model_copy(deep=True) for item in self.store.load_root_artifacts(trajectory.sample_id)]
        visible_images.extend(self._load_visible_history_images(trajectory))
        return visible_images

    def _load_visible_history_images(self, trajectory: TrajectoryRecord) -> list[ImageArtifactRef]:
        if trajectory.step_idx <= 0 or not trajectory.steps:
            return []
        history_images: list[ImageArtifactRef] = []
        for step in trajectory.steps:
            try:
                runtime_result = self.store.load_runtime_result(
                    trajectory.sample_id,
                    trajectory.trajectory_id,
                    step.step_idx,
                )
            except FileNotFoundError:
                continue
            if not runtime_result.images:
                continue
            history_images.append(runtime_result.images[0].model_copy(deep=True))
        return history_images

    def _load_latest_runtime_result(self, trajectory: TrajectoryRecord) -> ExecutorRuntimeResult | None:
        if trajectory.step_idx <= 0 or not trajectory.steps:
            return None
        try:
            return self.store.load_runtime_result(
                trajectory.sample_id,
                trajectory.trajectory_id,
                trajectory.step_idx,
            )
        except FileNotFoundError:
            return None

    def _resolve_runtime_input(
        self,
        visible_images: Sequence[ImageArtifactRef],
        input_image_index: int,
    ) -> tuple[ImageArtifactRef, int]:
        if not visible_images:
            raise ValueError("visible_images must not be empty when resolving runtime input.")
        if input_image_index < 0:
            raise ValueError(f"Planner selected invalid input_image_index={input_image_index!r}.")
        if input_image_index >= len(visible_images):
            raise ValueError(
                f"Planner selected input_image_index={input_image_index}, but only "
                f"{len(visible_images)} visible image(s) are available."
            )
        return visible_images[input_image_index].model_copy(deep=True), int(input_image_index)

    def _load_parent_frontier_score(self, trajectory: TrajectoryRecord) -> float:
        for judge_ref in reversed(trajectory.judge_records):
            if judge_ref.judge_stage != self.config.judge_stage:
                continue
            judge_path = self._resolve_trajectory_path(
                trajectory.sample_id,
                trajectory.trajectory_id,
                judge_ref.judge_record_path,
            )
            judge_record = JudgeRecord.from_json_file(judge_path)
            return float(judge_record.overall_score)
        return 0.0

    def _run_root_baseline_judge(self, trajectory: TrajectoryRecord) -> TrajectoryRecord:
        root_sample = self.store.load_root_sample(trajectory.sample_id)
        messages = self.store.load_messages(trajectory.sample_id, trajectory.trajectory_id)
        visible_images = self.store.load_root_artifacts(trajectory.sample_id)
        judge_record = self.judge_client.run(
            JudgeClientRequest(
                sample_id=trajectory.sample_id,
                trajectory_id=trajectory.trajectory_id,
                sample_dir=str(self.store.sample_dir(trajectory.sample_id)),
                trajectory_dir=str(self.store.trajectory_dir(trajectory.sample_id, trajectory.trajectory_id)),
                scope_type="trajectory",
                scope_step_idx=None,
                judge_stage=self.config.judge_stage,
                question=trajectory.question,
                answer_instruction=trajectory.answer_instruction,
                answer=root_sample.answer,
                messages=list(messages.root),
                visible_images=visible_images,
                planner_output=None,
                step_record=None,
                runtime_result=None,
                final_answer=None,
                metadata={
                    "orchestrator_version": "v01",
                    **dict(root_sample.metadata),
                },
            )
        )
        self.store.register_judge_record(judge_record)
        return self.store.load_trajectory(trajectory.sample_id, trajectory.trajectory_id)

    def _maybe_build_forced_final_answer_signal(
        self,
        trajectory: TrajectoryRecord,
        *,
        root_sample: RootSample,
    ) -> _ForcedFinalAnswerSignal | None:
        dataset_name = canonicalize_dataset_name(root_sample.metadata.get("source_dataset"))
        if dataset_name == "fsc147":
            return self._maybe_build_count_forced_final_answer_signal(trajectory)
        if is_reference_forced_final_answer_dataset(dataset_name):
            return self._maybe_build_reference_forced_final_answer_signal(dataset_name, trajectory)
        return None

    def _maybe_build_count_forced_final_answer_signal(
        self,
        trajectory: TrajectoryRecord,
    ) -> _ForcedFinalAnswerSignal | None:
        if not self.config.enable_count_forced_final_answer:
            return None
        latest_record = self._latest_judge_record_for_trajectory(trajectory)
        if latest_record is None:
            return None
        overall_score = float(latest_record.overall_score)
        if overall_score < self.config.count_forced_final_answer_score_threshold:
            return None
        model_results = latest_record.metadata.get("model_results")
        if not isinstance(model_results, list):
            return None

        normalized_answers: list[str] = []
        model_names: list[str] = []
        for item in model_results:
            if not isinstance(item, dict):
                continue
            if item.get("error") is not None:
                continue
            normalized = str(item.get("normalized_answer") or "").strip()
            if not normalized:
                continue
            normalized_answers.append(normalized)
            model_names.append(str(item.get("name") or "").strip())

        if not normalized_answers:
            return None
        if len(set(normalized_answers)) != 1:
            return None

        return _ForcedFinalAnswerSignal(
            reason="count_high_score_consensus",
            candidate_answer=normalized_answers[0],
            overall_score=overall_score,
            successful_model_count=len(normalized_answers),
            model_names=[name for name in model_names if name],
        )

    def _maybe_build_judge_consensus_answer_hint(
        self,
        trajectory: TrajectoryRecord,
    ) -> _ForcedFinalAnswerSignal | None:
        latest_record = self._latest_judge_record_for_trajectory(trajectory)
        if latest_record is None:
            return None
        overall_score = float(latest_record.overall_score)
        if overall_score < self.config.must_answer_score_threshold:
            return None
        model_results = latest_record.metadata.get("model_results")
        if not isinstance(model_results, list):
            return None

        normalized_answers: list[str] = []
        model_names: list[str] = []
        for item in model_results:
            if not isinstance(item, dict):
                continue
            if item.get("error") is not None:
                continue
            normalized = str(item.get("normalized_answer") or "").strip()
            if not normalized:
                continue
            normalized_answers.append(normalized)
            model_names.append(str(item.get("name") or "").strip())

        if not normalized_answers:
            return None
        if len(set(normalized_answers)) != 1:
            return None

        return _ForcedFinalAnswerSignal(
            reason="judge_consensus_answer_hint",
            candidate_answer=normalized_answers[0],
            overall_score=overall_score,
            successful_model_count=len(normalized_answers),
            model_names=[name for name in model_names if name],
        )

    def _maybe_build_reference_forced_final_answer_signal(
        self,
        dataset_name: str,
        trajectory: TrajectoryRecord,
    ) -> _ForcedFinalAnswerSignal | None:
        if not self.config.enable_exact_match_forced_final_answer:
            return None
        latest_record = self._latest_judge_record_for_trajectory(trajectory)
        if latest_record is None:
            return None
        overall_score = float(latest_record.overall_score)
        if overall_score < self.config.exact_match_forced_final_answer_score_threshold:
            return None
        model_results = latest_record.metadata.get("model_results")
        if not isinstance(model_results, list):
            return None

        candidate_answers: list[str] = []
        model_names: list[str] = []
        for item in model_results:
            if not isinstance(item, dict):
                continue
            if item.get("error") is not None:
                continue
            candidate = self._coerce_forced_final_answer_candidate(
                dataset_name=dataset_name,
                normalized_reference=item.get("normalized_reference"),
            )
            if not candidate:
                continue
            candidate_answers.append(candidate)
            model_names.append(str(item.get("name") or "").strip())

        if not candidate_answers:
            return None
        if len(set(candidate_answers)) != 1:
            return None

        return _ForcedFinalAnswerSignal(
            reason=(
                "textvqa_high_score_reference"
                if dataset_name == "textvqa"
                else "exact_match_high_score_reference"
            ),
            candidate_answer=candidate_answers[0],
            overall_score=overall_score,
            successful_model_count=len(candidate_answers),
            model_names=[name for name in model_names if name],
        )

    def _maybe_build_perfect_sample_exit_signal(
        self,
        *,
        root_sample: RootSample,
        trajectory: TrajectoryRecord,
        judge_record: JudgeRecord,
    ) -> _ForcedFinalAnswerSignal | None:
        dataset_name = canonicalize_dataset_name(root_sample.metadata.get("source_dataset"))
        if dataset_name == "fsc147":
            return self._maybe_build_count_perfect_sample_exit_signal(judge_record)
        if is_reference_forced_final_answer_dataset(dataset_name):
            return self._maybe_build_reference_perfect_sample_exit_signal(
                root_sample=root_sample,
                trajectory=trajectory,
                judge_record=judge_record,
            )
        return None

    def _maybe_build_count_perfect_sample_exit_signal(
        self,
        judge_record: JudgeRecord,
    ) -> _ForcedFinalAnswerSignal | None:
        if not self.config.enable_count_perfect_sample_early_exit:
            return None
        overall_score = float(judge_record.overall_score)
        if overall_score < self.config.count_perfect_sample_early_exit_score_threshold:
            return None
        model_results = judge_record.metadata.get("model_results")
        if not isinstance(model_results, list):
            return None

        candidate_answers: list[str] = []
        model_names: list[str] = []
        for item in model_results:
            if not isinstance(item, dict):
                continue
            if item.get("error") is not None:
                continue
            candidate = str(item.get("normalized_answer") or "").strip()
            if not candidate:
                continue
            candidate_answers.append(candidate)
            model_names.append(str(item.get("name") or "").strip())

        if not candidate_answers:
            return None
        if len(set(candidate_answers)) != 1:
            return None

        return _ForcedFinalAnswerSignal(
            reason="count_perfect_sample_early_exit",
            candidate_answer=candidate_answers[0],
            overall_score=overall_score,
            successful_model_count=len(candidate_answers),
            model_names=[name for name in model_names if name],
        )

    def _maybe_build_reference_perfect_sample_exit_signal(
        self,
        *,
        root_sample: RootSample,
        trajectory: TrajectoryRecord,
        judge_record: JudgeRecord,
    ) -> _ForcedFinalAnswerSignal | None:
        if not self.config.enable_reference_perfect_sample_early_exit:
            return None
        dataset_name = canonicalize_dataset_name(root_sample.metadata.get("source_dataset"))
        if not is_reference_forced_final_answer_dataset(dataset_name):
            return None
        overall_score = float(judge_record.overall_score)
        if overall_score < self.config.reference_perfect_sample_early_exit_score_threshold:
            return None
        model_results = judge_record.metadata.get("model_results")
        if not isinstance(model_results, list):
            return None

        candidate_answers: list[str] = []
        model_names: list[str] = []
        for item in model_results:
            if not isinstance(item, dict):
                continue
            if item.get("error") is not None:
                continue
            candidate = self._coerce_forced_final_answer_candidate(
                dataset_name=dataset_name,
                normalized_reference=item.get("normalized_reference"),
            )
            if not candidate:
                continue
            candidate_answers.append(candidate)
            model_names.append(str(item.get("name") or "").strip())

        if not candidate_answers:
            return None
        if len(set(candidate_answers)) != 1:
            return None

        return _ForcedFinalAnswerSignal(
            reason=(
                "textvqa_perfect_reference_sample_early_exit"
                if dataset_name == "textvqa"
                else "perfect_reference_sample_early_exit"
            ),
            candidate_answer=candidate_answers[0],
            overall_score=overall_score,
            successful_model_count=len(candidate_answers),
            model_names=[name for name in model_names if name],
        )

    def _coerce_forced_final_answer_candidate(
        self,
        *,
        dataset_name: str,
        normalized_reference: Any,
    ) -> str | None:
        if dataset_name == "textvqa":
            if not isinstance(normalized_reference, list):
                candidate = str(normalized_reference or "").strip()
                return candidate or None
            refs = [str(item).strip() for item in normalized_reference if str(item).strip()]
            if not refs:
                return None
            counts = Counter(refs)
            best_count = max(counts.values())
            for ref in refs:
                if counts[ref] == best_count:
                    return ref
            return None
        if isinstance(normalized_reference, list):
            return None
        candidate = str(normalized_reference or "").strip()
        return candidate or None

    def _prepare_sample_for_reference_perfect_finish(
        self,
        *,
        winner_trajectory: TrajectoryRecord,
    ) -> list[TrajectoryRecord]:
        refreshed_winner = self.store.load_trajectory(
            winner_trajectory.sample_id,
            winner_trajectory.trajectory_id,
        )
        for trajectory in self.store.list_trajectories(sample_id=refreshed_winner.sample_id):
            if trajectory.trajectory_id == refreshed_winner.trajectory_id:
                continue
            if trajectory.status != "running":
                continue
            self.store.mark_trajectory_status(
                trajectory.sample_id,
                trajectory.trajectory_id,
                status="stopped_early",
                pending_execution=None,
            )
        return [self.store.load_trajectory(refreshed_winner.sample_id, refreshed_winner.trajectory_id)]

    def _latest_judge_record_for_trajectory(self, trajectory: TrajectoryRecord) -> JudgeRecord | None:
        judge_dir = self.store.trajectory_dir(trajectory.sample_id, trajectory.trajectory_id) / "judge"
        if judge_dir.is_dir():
            local_files = sorted(judge_dir.glob("*.json"))
            for judge_path in reversed(local_files):
                judge_record = JudgeRecord.from_json_file(judge_path)
                if judge_record.judge_stage == self.config.judge_stage:
                    return judge_record

        for ref in reversed(trajectory.judge_records):
            if ref.judge_stage != self.config.judge_stage:
                continue
            judge_path = self._resolve_trajectory_path(
                trajectory.sample_id,
                trajectory.trajectory_id,
                ref.judge_record_path,
            )
            return JudgeRecord.from_json_file(judge_path)
        return None

    def _evaluate_stop_policy(
        self,
        trajectory: TrajectoryRecord,
        *,
        judge_record: JudgeRecord,
        root_sample: RootSample,
    ):
        judge_records = self._load_lineage_judge_records(trajectory, current_judge_record=judge_record)
        return evaluate_stop_policy(
            source_dataset=str(root_sample.metadata.get("source_dataset") or ""),
            answer=root_sample.answer,
            judge_records=judge_records,
        )

    def _load_lineage_judge_records(
        self,
        trajectory: TrajectoryRecord,
        *,
        current_judge_record: JudgeRecord,
    ) -> list[JudgeRecord]:
        lineage: list[TrajectoryRecord] = []
        seen: set[str] = set()
        cursor: TrajectoryRecord | None = trajectory
        while cursor is not None and cursor.trajectory_id not in seen:
            seen.add(cursor.trajectory_id)
            lineage.append(cursor)
            parent_id = cursor.parent_trajectory_id
            if not parent_id:
                break
            try:
                cursor = self.store.load_trajectory(cursor.sample_id, parent_id)
            except FileNotFoundError:
                break

        records: list[JudgeRecord] = []
        seen_judge_paths: set[str] = set()
        for item in reversed(lineage):
            for ref in item.judge_records:
                if ref.judge_stage != self.config.judge_stage:
                    continue
                judge_path = self._resolve_trajectory_path(
                    item.sample_id,
                    item.trajectory_id,
                    ref.judge_record_path,
                )
                key = str(judge_path.resolve())
                if key in seen_judge_paths:
                    continue
                seen_judge_paths.add(key)
                records.append(JudgeRecord.from_json_file(judge_path))

        records.append(current_judge_record)
        return records

    def _build_child_budget(self, parent_budget: Budget) -> Budget:
        return Budget(
            remaining_exec_steps=max(0, parent_budget.remaining_exec_steps - 1),
        )

    def _mark_trajectory_error(
        self,
        trajectory: TrajectoryRecord,
        *,
        code: str,
        message: str,
        round_idx: int | None = None,
        step_idx: int | None = None,
    ) -> TrajectoryRecord:
        return self.store.mark_trajectory_status(
            trajectory.sample_id,
            trajectory.trajectory_id,
            status="error",
            pending_execution=None,
            last_error=TrajectoryErrorInfo(
                code=code,
                message=message,
                round_idx=round_idx,
                step_idx=step_idx,
            ),
        )

    def _next_planner_round_idx(self, trajectory: TrajectoryRecord) -> int:
        if trajectory.latest_planner_round_idx is None:
            return 0
        return trajectory.latest_planner_round_idx + 1

    def _normalize_tool_capabilities(
        self,
        items: Sequence[ToolCapability | dict[str, Any]],
    ) -> list[ToolCapability]:
        normalized: list[ToolCapability] = []
        for item in items:
            if isinstance(item, ToolCapability):
                normalized.append(item.model_copy(deep=True))
            else:
                normalized.append(ToolCapability.model_validate(item))
        return normalized

    def _coerce_root_sample(self, root_sample: RootSample | dict[str, Any]) -> RootSample:
        if isinstance(root_sample, RootSample):
            return normalize_root_sample(root_sample)
        return normalize_root_sample(RootSample.model_validate(root_sample))

    def _coerce_budget(self, budget: Budget | dict[str, Any]) -> Budget:
        if isinstance(budget, Budget):
            return budget.model_copy(deep=True)
        return Budget.model_validate(budget)

    def _to_runtime_visible_images(
        self,
        visible_images: Sequence[ImageArtifactRef],
    ) -> list[RuntimeArtifactRef]:
        return [
            RuntimeArtifactRef(
                artifact_id=item.artifact_id,
                path=item.path,
                media_type=item.media_type,
                width=item.width,
                height=item.height,
            )
            for item in visible_images
        ]

    def _path_relative_to_trajectory(
        self,
        sample_id: str,
        trajectory_id: str,
        path: str | Path,
    ) -> str:
        trajectory_dir = self.store.trajectory_dir(sample_id, trajectory_id)
        path_obj = Path(path).resolve()
        try:
            return path_obj.relative_to(trajectory_dir).as_posix()
        except ValueError:
            return path_obj.as_posix()

    def _resolve_trajectory_path(
        self,
        sample_id: str,
        trajectory_id: str,
        path: str | Path,
    ) -> Path:
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return (self.store.trajectory_dir(sample_id, trajectory_id) / path_obj).resolve()


__all__ = [
    "OrchestratorConfig",
    "OrchestratorRunResult",
    "OrchestratorV01",
]
