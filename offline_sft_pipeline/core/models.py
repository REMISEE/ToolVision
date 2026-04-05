from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

SCHEMA_VERSION = "0.1.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "offline_sft_pipeline" / "schemas"

MessageRole = Literal["system", "user", "assistant", "tool"]
TrajectoryStatus = Literal[
    "running",
    "expanded",
    "answered",
    "pruned",
    "failed",
    "stopped_early",
    "max_step_reached",
    "error",
]
TerminalTrajectoryStatus = Literal["answered", "pruned", "failed", "stopped_early", "max_step_reached", "error"]
JudgeStage = Literal["cheap_filter", "committee", "final_select"]
JudgeScopeType = Literal["trajectory", "step"]
ObservedHelperStatus = Literal["ok", "error"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_id_component(value: str, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty.")
    if "/" in text or "\\" in text:
        raise ValueError(f"{name} must not contain path separators: {text!r}")
    return text


def build_sample_id(dataset_name: str, question_id: str, *, split: str | None = None) -> str:
    dataset_name = _require_id_component(dataset_name, name="dataset_name")
    question_id = _require_id_component(question_id, name="question_id")
    if split is None:
        return f"{dataset_name}__{question_id}"
    split = _require_id_component(split, name="split")
    return f"{dataset_name}__{split}__{question_id}"


def build_root_trajectory_id(sample_id: str) -> str:
    sample_id = _require_id_component(sample_id, name="sample_id")
    return f"traj__{sample_id}__root"


def build_child_trajectory_id(parent_trajectory_id: str, planner_round_idx: int, suggestion_id: str) -> str:
    parent_trajectory_id = _require_id_component(parent_trajectory_id, name="parent_trajectory_id")
    suggestion_id = _require_id_component(suggestion_id, name="suggestion_id")
    if planner_round_idx < 0:
        raise ValueError("planner_round_idx must be >= 0.")
    return f"{parent_trajectory_id}__r{planner_round_idx:03d}_{suggestion_id}"


@lru_cache(maxsize=None)
def _load_json_schema(schema_filename: str) -> dict[str, Any]:
    schema_path = SCHEMA_DIR / schema_filename
    return json.loads(schema_path.read_text(encoding="utf-8-sig"))


class PipelineBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineBaseModel":
        return cls.model_validate(data)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "PipelineBaseModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls.model_validate(payload)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json_str(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_json_file(self, path: str | Path, *, indent: int = 2) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json_str(indent=indent), encoding="utf-8")
        return output_path


class SchemaBackedModel(PipelineBaseModel):
    schema_filename: ClassVar[str]

    @classmethod
    def schema_path(cls) -> Path:
        return SCHEMA_DIR / cls.schema_filename

    @classmethod
    def validate_payload_against_schema(cls, payload: dict[str, Any]) -> None:
        try:
            import jsonschema
        except ImportError as exc:
            raise RuntimeError("jsonschema is required for schema-backed model validation.") from exc
        jsonschema.validate(payload, _load_json_schema(cls.schema_filename))

    def validate_against_schema(self) -> "SchemaBackedModel":
        self.validate_payload_against_schema(self.to_dict())
        return self

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        validate_schema: bool = False,
    ) -> "SchemaBackedModel":
        if validate_schema:
            cls.validate_payload_against_schema(data)
        return cls.model_validate(data)

    @classmethod
    def from_json_file(
        cls,
        path: str | Path,
        *,
        validate_schema: bool = True,
    ) -> "SchemaBackedModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if validate_schema:
            cls.validate_payload_against_schema(payload)
        return cls.model_validate(payload)

    def to_json_file(
        self,
        path: str | Path,
        *,
        indent: int = 2,
        validate_schema: bool = True,
    ) -> Path:
        payload = self.to_dict()
        if validate_schema:
            self.validate_payload_against_schema(payload)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=indent), encoding="utf-8")
        return output_path


class JsonListDocument(RootModel[list[Any]]):
    model_config = ConfigDict(validate_assignment=True)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "JsonListDocument":
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls.model_validate(payload)

    def to_json_file(self, path: str | Path, *, indent: int = 2) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=indent),
            encoding="utf-8",
        )
        return output_path


class RootImage(PipelineBaseModel):
    image_id: str
    path: str


class RootSample(PipelineBaseModel):
    sample_id: str
    question: str
    images: list[RootImage]
    metadata: dict[str, Any] = Field(default_factory=dict)
    answer: str | None = None

    @model_validator(mode="after")
    def _validate_images(self) -> "RootSample":
        if not self.images:
            raise ValueError("RootSample.images must not be empty.")
        return self


class ToolSpec(PipelineBaseModel):
    name: str
    description: str | None = None
    version: str | None = None


class ConversationMessage(PipelineBaseModel):
    message_id: str
    role: MessageRole
    content: str
    image_artifact_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessagesDocument(JsonListDocument):
    root: list[ConversationMessage]


class ArtifactRef(PipelineBaseModel):
    artifact_id: str
    path: str
    media_type: str | None = None


class ImageArtifactRef(ArtifactRef):
    width: int | None = None
    height: int | None = None


class CapabilityPlanItem(PipelineBaseModel):
    order: int
    capability: str
    instruction: str


class PlannerStepSpec(PipelineBaseModel):
    step_id: str
    step_goal: str
    capability_plan: list[CapabilityPlanItem]
    executor_instruction: str

    @model_validator(mode="after")
    def _validate_capability_plan(self) -> "PlannerStepSpec":
        if not self.capability_plan:
            raise ValueError("PlannerStepSpec.capability_plan must not be empty.")
        return self


class PlannerSuggestion(PipelineBaseModel):
    suggestion_id: str
    suggestion_cot: str
    steps: list[PlannerStepSpec]

    @model_validator(mode="after")
    def _validate_steps(self) -> "PlannerSuggestion":
        if not self.steps:
            raise ValueError("PlannerSuggestion.steps must not be empty.")
        return self


class PlannerOutput(SchemaBackedModel):
    schema_filename: ClassVar[str] = "planner_output_schema.json"

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    sample_id: str
    trajectory_id: str
    round_idx: int
    created_at: datetime
    can_answer_now: bool
    global_chain_cot: str
    direct_answer: str | None = None
    stop_reason: str | None = None
    suggestions: list[PlannerSuggestion] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_answer_mode(self) -> "PlannerOutput":
        if self.can_answer_now:
            if not self.direct_answer:
                raise ValueError("PlannerOutput.direct_answer is required when can_answer_now=True.")
            if self.suggestions:
                raise ValueError("PlannerOutput.suggestions must be empty when can_answer_now=True.")
        elif not self.suggestions:
            raise ValueError("PlannerOutput.suggestions must not be empty when can_answer_now=False.")
        return self


class ExecutorStepOutput(SchemaBackedModel):
    """Structured executor model output before step files/runtime state are written."""

    schema_filename: ClassVar[str] = "executor_step_output_schema.json"

    cot: str = ""
    code: str
    raw_response_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_code(self) -> "ExecutorStepOutput":
        if not self.code.strip():
            raise ValueError("ExecutorStepOutput.code must not be empty.")
        return self


class RuntimeObservedHelperCall(PipelineBaseModel):
    order: int
    name: str
    status: ObservedHelperStatus | None = None


class RuntimeCodeExecution(PipelineBaseModel):
    code_path: str
    exit_code: int
    started_at: datetime
    finished_at: datetime
    elapsed_seconds: float
    stdout_path: str | None = None
    stderr_path: str | None = None


class RuntimeErrorInfo(PipelineBaseModel):
    type: str
    message: str
    traceback_path: str | None = None


class ExecutorRuntimeResult(SchemaBackedModel):
    schema_filename: ClassVar[str] = "executor_runtime_result_schema.json"

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    sample_id: str
    trajectory_id: str
    round_idx: int
    step_idx: int
    created_at: datetime
    success: bool
    images: list[ImageArtifactRef] = Field(default_factory=list)
    text: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)
    observed_helper_call_count: int = 0
    observed_helper_calls: list[RuntimeObservedHelperCall] = Field(default_factory=list)
    code_execution: RuntimeCodeExecution
    error: RuntimeErrorInfo | None = None

    @model_validator(mode="after")
    def _validate_helper_call_count(self) -> "ExecutorRuntimeResult":
        if self.observed_helper_call_count != len(self.observed_helper_calls):
            raise ValueError(
                "ExecutorRuntimeResult.observed_helper_call_count must equal len(observed_helper_calls)."
            )
        if self.success and self.error is not None:
            raise ValueError("ExecutorRuntimeResult.error must be null when success=True.")
        if not self.success and self.error is None:
            raise ValueError("ExecutorRuntimeResult.error must be present when success=False.")
        return self


class JudgeRecord(SchemaBackedModel):
    schema_filename: ClassVar[str] = "judge_record_schema.json"

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    judge_record_id: str
    sample_id: str
    trajectory_id: str
    scope_type: JudgeScopeType
    scope_step_idx: int | None = None
    judge_stage: JudgeStage
    created_at: datetime
    keep_for_frontier: bool
    exportable: bool
    overall_score: float
    answerability_score: float | None = None
    tool_use_quality_score: float | None = None
    trajectory_progress_score: float | None = None
    note: str = ""

    @model_validator(mode="after")
    def _validate_scope(self) -> "JudgeRecord":
        if self.scope_type == "step" and self.scope_step_idx is None:
            raise ValueError("JudgeRecord.scope_step_idx is required when scope_type='step'.")
        return self


class PlannerHistoryItem(PipelineBaseModel):
    round_idx: int
    planner_output_path: str
    can_answer_now: bool
    direct_answer: str | None = None
    selected_for_expansion: bool
    created_at: datetime


class ForkProvenance(PipelineBaseModel):
    parent_trajectory_id: str
    parent_planner_round_idx: int
    parent_suggestion_id: str


class PendingExecution(PipelineBaseModel):
    planner_round_idx: int
    suggestion_id: str
    suggestion_step_index: int
    step_id: str


class StepRecord(PipelineBaseModel):
    step_idx: int
    planner_round_idx: int
    suggestion_id: str
    suggestion_step_index: int
    step_id: str
    step_goal: str
    capability_plan: list[CapabilityPlanItem]
    executor_cot_path: str
    executor_code_path: str
    runtime_result_path: str
    assistant_message_id: str
    tool_message_id: str | None = None

    @model_validator(mode="after")
    def _validate_capability_plan(self) -> "StepRecord":
        if not self.capability_plan:
            raise ValueError("StepRecord.capability_plan must not be empty.")
        return self


class JudgeRecordRef(PipelineBaseModel):
    judge_stage: JudgeStage
    judge_record_path: str


class Budget(PipelineBaseModel):
    remaining_rounds: int


class TrajectoryErrorInfo(PipelineBaseModel):
    code: str
    message: str
    round_idx: int | None = None
    step_idx: int | None = None
    traceback_path: str | None = None


class TrajectoryRecord(SchemaBackedModel):
    schema_filename: ClassVar[str] = "trajectory_schema.json"

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    sample_id: str
    trajectory_id: str
    parent_trajectory_id: str | None = None
    status: TrajectoryStatus
    created_at: datetime
    updated_at: datetime
    round_idx: int
    step_idx: int
    question: str
    original_image_artifact_id: str
    messages_path: str
    planner_history: list[PlannerHistoryItem] = Field(default_factory=list)
    latest_planner_round_idx: int | None = None
    latest_planner_output_path: str | None = None
    fork_provenance: ForkProvenance | None = None
    pending_execution: PendingExecution | None = None
    steps: list[StepRecord] = Field(default_factory=list)
    judge_records: list[JudgeRecordRef] = Field(default_factory=list)
    final_answer: str | None = None
    answer_confidence: float | None = None
    budget: Budget
    last_error: TrajectoryErrorInfo | None = None


class CanonicalJudgeSummary(PipelineBaseModel):
    overall_score: float | None = None
    weak_model_solved_count: int | None = None


class CanonicalSftSample(SchemaBackedModel):
    schema_filename: ClassVar[str] = "canonical_sft_sample_schema.json"

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    sample_id: str
    source_trajectory_id: str
    source_status: TerminalTrajectoryStatus
    export_timestamp: datetime
    thinking_enabled: bool
    tools: list[ToolSpec] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    messages: list[ConversationMessage]
    final_answer: str | None = None
    judge_summary: CanonicalJudgeSummary
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_messages(self) -> "CanonicalSftSample":
        if not self.messages:
            raise ValueError("CanonicalSftSample.messages must not be empty.")
        return self


__all__ = [
    "ArtifactRef",
    "Budget",
    "CanonicalJudgeSummary",
    "CanonicalSftSample",
    "CapabilityPlanItem",
    "ConversationMessage",
    "ExecutorStepOutput",
    "ExecutorRuntimeResult",
    "ForkProvenance",
    "ImageArtifactRef",
    "JudgeRecord",
    "JudgeRecordRef",
    "MessagesDocument",
    "PendingExecution",
    "PlannerHistoryItem",
    "PlannerOutput",
    "PlannerStepSpec",
    "PlannerSuggestion",
    "RootImage",
    "RootSample",
    "RuntimeCodeExecution",
    "RuntimeErrorInfo",
    "RuntimeObservedHelperCall",
    "SCHEMA_VERSION",
    "StepRecord",
    "ToolSpec",
    "TrajectoryErrorInfo",
    "TrajectoryRecord",
    "build_child_trajectory_id",
    "build_root_trajectory_id",
    "build_sample_id",
    "utc_now",
]
