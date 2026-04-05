from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

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
    planner_suggestion_count: int = 3
    max_child_trajectories: int = 6
    default_budget: Budget = field(
        default_factory=lambda: Budget(
            remaining_rounds=3,
        )
    )
    system_message: str = DEFAULT_SYSTEM_MESSAGE
    judge_stage: JudgeStage = "cheap_filter"
    tool_capabilities: Sequence[ToolCapability | dict[str, Any]] = field(default_factory=tuple)
    stop_unselected_trajectories: bool = True

    def __post_init__(self) -> None:
        if self.planner_suggestion_count < 1:
            raise ValueError("planner_suggestion_count must be >= 1.")
        if self.max_child_trajectories < 1:
            raise ValueError("max_child_trajectories must be >= 1.")

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
    parent_frontier_score: float


@dataclass(slots=True)
class _FrontierCandidate:
    parent_trajectory: TrajectoryRecord
    planner_output: PlannerOutput
    suggestion: PlannerSuggestion
    suggestion_order: int
    parent_frontier_score: float


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

        frontier: list[TrajectoryRecord] = [initialized.trajectory]
        while frontier:
            planner_contexts: list[_PlannerRoundContext] = []
            for frontier_item in frontier:
                trajectory = self.store.load_trajectory(frontier_item.sample_id, frontier_item.trajectory_id)
                if trajectory.status != "running":
                    continue
                if trajectory.budget.remaining_rounds <= 0:
                    self.store.mark_trajectory_status(
                        trajectory.sample_id,
                        trajectory.trajectory_id,
                        status="max_step_reached",
                        pending_execution=None,
                    )
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

            selected_candidates = self._select_frontier_candidates(planner_contexts)
            candidates_by_parent: dict[str, list[_FrontierCandidate]] = {}
            for candidate in selected_candidates:
                candidates_by_parent.setdefault(candidate.parent_trajectory.trajectory_id, []).append(candidate)

            next_frontier: list[TrajectoryRecord] = []
            for context in planner_contexts:
                parent_selected = candidates_by_parent.get(context.trajectory.trajectory_id, [])
                refreshed_parent = self.store.register_planner_round(
                    context.planner_output,
                    selected_for_expansion=bool(parent_selected),
                    pending_execution=None,
                    final_answer=context.planner_output.direct_answer,
                )

                if context.planner_output.can_answer_now:
                    self._append_final_answer_message(
                        refreshed_parent,
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

                if not parent_selected:
                    self._finalize_unselected_parent(refreshed_parent)
                    continue

                child_created = False
                for candidate in parent_selected:
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
                    executed_child = self._execute_child_step(child, candidate)
                    if executed_child.status == "running":
                        next_frontier.append(executed_child)

                if child_created:
                    self.store.mark_trajectory_expanded(
                        refreshed_parent.sample_id,
                        refreshed_parent.trajectory_id,
                    )
                elif refreshed_parent.status == "running":
                    self._finalize_unselected_parent(refreshed_parent)

            frontier = next_frontier

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
        requested_suggestion_count = self.config.effective_planner_suggestion_count
        planner_request = PlannerClientRequest(
            sample_id=trajectory.sample_id,
            trajectory_id=trajectory.trajectory_id,
            round_idx=self._next_planner_round_idx(trajectory),
            sample_dir=str(self.store.sample_dir(trajectory.sample_id)),
            trajectory_dir=str(self.store.trajectory_dir(trajectory.sample_id, trajectory.trajectory_id)),
            planner_dir=str(self.store.planner_dir(trajectory.sample_id, trajectory.trajectory_id)),
            steps_dir=str(self.store.steps_dir(trajectory.sample_id, trajectory.trajectory_id)),
            question=trajectory.question,
            messages=list(messages_doc.root),
            visible_images=visible_images,
            budget=trajectory.budget.model_copy(deep=True),
            tool_capabilities=[item.model_copy(deep=True) for item in self.tool_capabilities],
            latest_runtime_result=self._load_latest_runtime_result(trajectory),
            requested_suggestion_count=requested_suggestion_count,
            metadata={"orchestrator_version": "v01"},
        )
        planner_output = self.planner_client.run(planner_request)
        return _PlannerRoundContext(
            trajectory=trajectory,
            messages=list(messages_doc.root),
            visible_images=visible_images,
            planner_output=planner_output,
            requested_suggestion_count=requested_suggestion_count,
            parent_frontier_score=self._load_parent_frontier_score(trajectory),
        )

    def _select_frontier_candidates(
        self,
        planner_contexts: Sequence[_PlannerRoundContext],
    ) -> list[_FrontierCandidate]:
        top1_candidates: list[_FrontierCandidate] = []
        remaining_candidates: list[_FrontierCandidate] = []
        for context in planner_contexts:
            if context.planner_output.can_answer_now:
                continue
            if context.requested_suggestion_count <= 0:
                continue
            usable_suggestions = context.planner_output.suggestions[: context.requested_suggestion_count]
            for suggestion_order, suggestion in enumerate(usable_suggestions):
                candidate = _FrontierCandidate(
                    parent_trajectory=context.trajectory,
                    planner_output=context.planner_output,
                    suggestion=suggestion,
                    suggestion_order=suggestion_order,
                    parent_frontier_score=context.parent_frontier_score,
                )
                if suggestion_order == 0:
                    top1_candidates.append(candidate)
                else:
                    remaining_candidates.append(candidate)

        top1_candidates.sort(key=self._candidate_sort_key)
        if len(top1_candidates) >= self.config.max_child_trajectories:
            return top1_candidates[: self.config.max_child_trajectories]

        remaining_candidates.sort(key=self._candidate_sort_key)
        selected = list(top1_candidates)
        remaining_slots = self.config.max_child_trajectories - len(selected)
        if remaining_slots > 0:
            selected.extend(remaining_candidates[:remaining_slots])
        return selected

    def _candidate_sort_key(self, candidate: _FrontierCandidate) -> tuple[float, int, str]:
        return (
            -candidate.parent_frontier_score,
            candidate.suggestion_order,
            candidate.parent_trajectory.trajectory_id,
        )

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
    ) -> TrajectoryRecord:
        selected_step = candidate.suggestion.steps[0]
        step_idx = child_trajectory.step_idx + 1
        visible_images = self._select_visible_images(child_trajectory)
        runtime_image_index = self._select_runtime_image_index(child_trajectory, visible_images)
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
            )

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
            )

        assistant_message = self._build_assistant_step_message(
            child_trajectory=child_trajectory,
            planner_round_idx=candidate.planner_output.round_idx,
            step_idx=step_idx,
            suggestion_id=candidate.suggestion.suggestion_id,
            selected_step=selected_step,
            executor_cot=executor_output.cot,
            executor_code=executor_output.code,
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
            step_idx=step_idx,
            planner_round_idx=candidate.planner_output.round_idx,
            suggestion_id=candidate.suggestion.suggestion_id,
            suggestion_step_index=0,
            step_id=selected_step.step_id,
            step_goal=selected_step.step_goal,
            capability_plan=[item.model_copy(deep=True) for item in selected_step.capability_plan],
            executor_cot_path=str(step_paths.executor_cot_path),
            executor_code_path=str(step_paths.executor_code_path),
            runtime_result_path=str(step_paths.runtime_result_path),
            assistant_message_id=assistant_message.message_id,
            tool_message_id=tool_message.message_id,
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

        try:
            judge_record = self.judge_client.run(
                JudgeClientRequest(
                    sample_id=post_step_trajectory.sample_id,
                    trajectory_id=post_step_trajectory.trajectory_id,
                    scope_type="step",
                    scope_step_idx=step_idx,
                    judge_stage=self.config.judge_stage,
                    question=post_step_trajectory.question,
                    messages=list(updated_messages.root),
                    visible_images=post_step_visible_images,
                    planner_output=candidate.planner_output,
                    step_record=step_record,
                    runtime_result=runtime_result,
                    final_answer=None,
                    metadata={"orchestrator_version": "v01"},
                )
            )
            self.store.register_judge_record(judge_record)
        except Exception as exc:
            return self._mark_trajectory_error(
                post_step_trajectory,
                code="judge_error",
                message=str(exc),
                round_idx=candidate.planner_output.round_idx,
                step_idx=step_idx,
            )

        return self._finalize_child_after_step(
            post_step_trajectory,
            runtime_result=runtime_result,
            judge_record=judge_record,
        )

    def _finalize_child_after_step(
        self,
        trajectory: TrajectoryRecord,
        *,
        runtime_result: ExecutorRuntimeResult,
        judge_record: JudgeRecord,
    ) -> TrajectoryRecord:
        updated = self.store.load_trajectory(trajectory.sample_id, trajectory.trajectory_id)

        if not runtime_result.success:
            updated.status = "failed"
        elif not runtime_result.images and not runtime_result.text.strip():
            updated.status = "pruned"
        elif not judge_record.keep_for_frontier:
            updated.status = "pruned"
        elif updated.budget.remaining_rounds <= 0:
            updated.status = "max_step_reached"
        else:
            updated.status = "running"

        updated.pending_execution = None
        updated.updated_at = utc_now()
        self.store.save_trajectory(updated)
        return updated

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
        final_answer: str,
        planner_round_idx: int,
    ) -> None:
        final_message = ConversationMessage(
            message_id="m_final_answer",
            role="assistant",
            content=f"<answer>\n{final_answer}\n</answer>",
            image_artifact_ids=[],
            metadata={
                "message_kind": "final_answer",
                "planner_round_idx": planner_round_idx,
            },
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
        executor_code_path: Path,
    ) -> ConversationMessage:
        return ConversationMessage(
            message_id=f"m_step_{step_idx:03d}_assistant",
            role="assistant",
            content=(
                f"<think>\n{executor_cot}\n</think>\n"
                f"<tool_call name=\"code_image_tool\">\n{executor_code}\n</tool_call>"
            ),
            image_artifact_ids=[],
            metadata={
                "message_kind": "executor_step",
                "step_idx": step_idx,
                "planner_round_idx": planner_round_idx,
                "suggestion_id": suggestion_id,
                "step_id": selected_step.step_id,
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
        latest_primary = self._load_latest_primary_image(trajectory)
        if latest_primary is not None:
            visible_images.append(latest_primary)
        return visible_images

    def _load_latest_primary_image(self, trajectory: TrajectoryRecord) -> ImageArtifactRef | None:
        if trajectory.step_idx <= 0 or not trajectory.steps:
            return None
        try:
            runtime_result = self.store.load_runtime_result(
                trajectory.sample_id,
                trajectory.trajectory_id,
                trajectory.step_idx,
            )
        except FileNotFoundError:
            return None
        if not runtime_result.images:
            return None
        return runtime_result.images[0].model_copy(deep=True)

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

    def _select_runtime_image_index(
        self,
        trajectory: TrajectoryRecord,
        visible_images: Sequence[ImageArtifactRef],
    ) -> int:
        if not visible_images:
            return 0
        latest_primary = self._load_latest_primary_image(trajectory)
        if latest_primary is None:
            return 0
        for idx, image in enumerate(visible_images):
            if image.artifact_id == latest_primary.artifact_id:
                return idx
        return len(visible_images) - 1

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

    def _build_child_budget(self, parent_budget: Budget) -> Budget:
        return Budget(
            remaining_rounds=max(0, parent_budget.remaining_rounds - 1),
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
            return root_sample
        return RootSample.model_validate(root_sample)

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
