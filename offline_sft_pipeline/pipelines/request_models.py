from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from offline_sft_pipeline.core.models import (
    Budget,
    ConversationMessage,
    ExecutorRuntimeResult,
    ImageArtifactRef,
    JudgeScopeType,
    JudgeStage,
    PlannerOutput,
    PlannerStepSpec,
    PipelineBaseModel,
    StepRecord,
)


class ToolCapability(PipelineBaseModel):
    name: str
    description: str
    usage_notes: str | None = None


class PlannerClientRequest(PipelineBaseModel):
    sample_id: str
    trajectory_id: str
    round_idx: int
    sample_dir: str | None = None
    trajectory_dir: str | None = None
    planner_dir: str | None = None
    steps_dir: str | None = None
    question: str
    messages: list[ConversationMessage] = Field(default_factory=list)
    visible_images: list[ImageArtifactRef] = Field(default_factory=list)
    budget: Budget
    tool_capabilities: list[ToolCapability] = Field(default_factory=list)
    latest_runtime_result: ExecutorRuntimeResult | None = None
    requested_suggestion_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_required_fields(self) -> "PlannerClientRequest":
        if not self.question.strip():
            raise ValueError("PlannerClientRequest.question must not be empty.")
        if not self.messages:
            raise ValueError("PlannerClientRequest.messages must not be empty.")
        if not self.visible_images:
            raise ValueError("PlannerClientRequest.visible_images must not be empty.")
        if self.requested_suggestion_count is not None and not 1 <= self.requested_suggestion_count <= 3:
            raise ValueError("PlannerClientRequest.requested_suggestion_count must be between 1 and 3.")
        return self


class ExecutorClientRequest(PipelineBaseModel):
    sample_id: str
    trajectory_id: str
    round_idx: int
    step_idx: int
    sample_dir: str | None = None
    trajectory_dir: str | None = None
    planner_dir: str | None = None
    steps_dir: str | None = None
    question: str
    messages: list[ConversationMessage] = Field(default_factory=list)
    visible_images: list[ImageArtifactRef] = Field(default_factory=list)
    suggestion_id: str
    suggestion_step_index: int
    step_spec: PlannerStepSpec
    planner_global_chain_cot: str | None = None
    suggestion_cot: str | None = None
    tool_capabilities: list[ToolCapability] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_required_fields(self) -> "ExecutorClientRequest":
        if not self.question.strip():
            raise ValueError("ExecutorClientRequest.question must not be empty.")
        if not self.messages:
            raise ValueError("ExecutorClientRequest.messages must not be empty.")
        if not self.visible_images:
            raise ValueError("ExecutorClientRequest.visible_images must not be empty.")
        if not self.suggestion_id.strip():
            raise ValueError("ExecutorClientRequest.suggestion_id must not be empty.")
        return self


class JudgeClientRequest(PipelineBaseModel):
    sample_id: str
    trajectory_id: str
    scope_type: JudgeScopeType
    scope_step_idx: int | None = None
    judge_stage: JudgeStage
    question: str
    messages: list[ConversationMessage] = Field(default_factory=list)
    visible_images: list[ImageArtifactRef] = Field(default_factory=list)
    planner_output: PlannerOutput | None = None
    step_record: StepRecord | None = None
    runtime_result: ExecutorRuntimeResult | None = None
    final_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_required_fields(self) -> "JudgeClientRequest":
        if not self.question.strip():
            raise ValueError("JudgeClientRequest.question must not be empty.")
        if not self.messages:
            raise ValueError("JudgeClientRequest.messages must not be empty.")
        if self.scope_type == "step":
            inferred_step_idx = self.step_record.step_idx if self.step_record is not None else None
            if self.scope_step_idx is None and inferred_step_idx is None:
                raise ValueError(
                    "JudgeClientRequest.scope_step_idx or JudgeClientRequest.step_record is required when scope_type='step'."
                )
        return self


__all__ = [
    "ExecutorClientRequest",
    "JudgeClientRequest",
    "PlannerClientRequest",
    "ToolCapability",
]
