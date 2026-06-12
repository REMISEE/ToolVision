from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

from offline_sft_pipeline.core.models import (
    Budget,
    CanonicalSftSample,
    ConversationMessage,
    ForkProvenance,
    ImageArtifactRef,
    JudgeRecord,
    JudgeRecordRef,
    MessagesDocument,
    PendingExecution,
    PlannerHistoryItem,
    PlannerOutput,
    RootSample,
    StepRecord,
    TrajectoryErrorInfo,
    TrajectoryRecord,
    TrajectoryStatus,
    build_child_trajectory_id,
    build_root_trajectory_id,
    utc_now,
)

DEFAULT_SYSTEM_MESSAGE = "You are a helpful vision tool-use assistant."


@dataclass(slots=True)
class StepFilePaths:
    step_dir: Path
    executor_cot_path: Path
    executor_code_path: Path
    runtime_result_path: Path
    stdout_path: Path
    stderr_path: Path


@dataclass(slots=True)
class InitializedTrajectory:
    root_sample_path: Path
    trajectory_path: Path
    messages_path: Path
    trajectory: TrajectoryRecord
    messages: MessagesDocument
    root_artifacts: list[ImageArtifactRef]


@dataclass(slots=True)
class ResumeState:
    trajectory: TrajectoryRecord
    messages: MessagesDocument
    root_artifacts: list[ImageArtifactRef]


class OfflineTrajectoryStore:
    def __init__(self, run_root: str | Path):
        self.run_root = Path(run_root).expanduser().resolve()
        self.samples_root = self.run_root / "samples"
        self.samples_root.mkdir(parents=True, exist_ok=True)

    def sample_dir(self, sample_id: str) -> Path:
        return self.samples_root / sample_id

    def sample_artifacts_dir(self, sample_id: str) -> Path:
        return self.sample_dir(sample_id) / "artifacts"

    def trajectories_dir(self, sample_id: str) -> Path:
        return self.sample_dir(sample_id) / "trajectories"

    def trajectory_dir(self, sample_id: str, trajectory_id: str) -> Path:
        return self.trajectories_dir(sample_id) / trajectory_id

    def planner_dir(self, sample_id: str, trajectory_id: str) -> Path:
        return self.trajectory_dir(sample_id, trajectory_id) / "planner"

    def steps_dir(self, sample_id: str, trajectory_id: str) -> Path:
        return self.trajectory_dir(sample_id, trajectory_id) / "steps"

    def judge_dir(self, sample_id: str, trajectory_id: str) -> Path:
        return self.trajectory_dir(sample_id, trajectory_id) / "judge"

    def exports_dir(self, sample_id: str, trajectory_id: str) -> Path:
        return self.trajectory_dir(sample_id, trajectory_id) / "exports"

    def root_sample_path(self, sample_id: str) -> Path:
        return self.sample_dir(sample_id) / "root_sample.json"

    def trajectory_path(self, sample_id: str, trajectory_id: str) -> Path:
        return self.trajectory_dir(sample_id, trajectory_id) / "trajectory.json"

    def messages_path(self, sample_id: str, trajectory_id: str) -> Path:
        return self.trajectory_dir(sample_id, trajectory_id) / "messages.json"

    def planner_output_path(self, sample_id: str, trajectory_id: str, round_idx: int) -> Path:
        return self.planner_dir(sample_id, trajectory_id) / f"round_{round_idx:03d}.json"

    def judge_record_path(
        self,
        sample_id: str,
        trajectory_id: str,
        judge_stage: str,
        *,
        scope_step_idx: int | None = None,
    ) -> Path:
        if scope_step_idx is None:
            filename = f"trajectory_{judge_stage}.json"
        else:
            filename = f"step_{scope_step_idx:03d}_{judge_stage}.json"
        return self.judge_dir(sample_id, trajectory_id) / filename

    def export_path(self, sample_id: str, trajectory_id: str) -> Path:
        return self.exports_dir(sample_id, trajectory_id) / "canonical_sft_sample.json"

    def build_step_file_paths(
        self,
        sample_id: str,
        trajectory_id: str,
        step_idx: int,
        *,
        create_dirs: bool = True,
    ) -> StepFilePaths:
        step_dir = self.steps_dir(sample_id, trajectory_id) / f"step_{step_idx:03d}"
        if create_dirs:
            step_dir.mkdir(parents=True, exist_ok=True)
        return StepFilePaths(
            step_dir=step_dir,
            executor_cot_path=step_dir / "executor_cot.md",
            executor_code_path=step_dir / "executor_code.py",
            runtime_result_path=step_dir / "runtime_result.json",
            stdout_path=step_dir / "stdout.txt",
            stderr_path=step_dir / "stderr.txt",
        )

    def ensure_sample_layout(self, sample_id: str) -> None:
        self.sample_dir(sample_id).mkdir(parents=True, exist_ok=True)
        self.sample_artifacts_dir(sample_id).mkdir(parents=True, exist_ok=True)
        self.trajectories_dir(sample_id).mkdir(parents=True, exist_ok=True)

    def ensure_trajectory_layout(self, sample_id: str, trajectory_id: str) -> None:
        trajectory_dir = self.trajectory_dir(sample_id, trajectory_id)
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.planner_dir(sample_id, trajectory_id).mkdir(parents=True, exist_ok=True)
        self.steps_dir(sample_id, trajectory_id).mkdir(parents=True, exist_ok=True)
        self.judge_dir(sample_id, trajectory_id).mkdir(parents=True, exist_ok=True)
        self.exports_dir(sample_id, trajectory_id).mkdir(parents=True, exist_ok=True)

    def _relative_to_trajectory(self, sample_id: str, trajectory_id: str, path: str | Path) -> str:
        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = self.trajectory_dir(sample_id, trajectory_id) / path_obj
        return Path(
            os.path.relpath(path_obj, start=self.trajectory_dir(sample_id, trajectory_id))
        ).as_posix()

    def _resolve_trajectory_path(self, sample_id: str, trajectory_id: str, path: str | Path) -> Path:
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return (self.trajectory_dir(sample_id, trajectory_id) / path_obj).resolve()

    def _rebase_relative_path(
        self,
        path_value: str | None,
        *,
        source_sample_id: str,
        source_trajectory_id: str,
        target_sample_id: str,
        target_trajectory_id: str,
    ) -> str | None:
        if path_value is None:
            return None
        absolute_path = self._resolve_trajectory_path(source_sample_id, source_trajectory_id, path_value)
        return self._relative_to_trajectory(target_sample_id, target_trajectory_id, absolute_path)

    def _materialize_root_artifact(self, source_path: Path, target_path: Path) -> tuple[int | None, int | None]:
        with Image.open(source_path) as image:
            width, height = image.size
            image.save(target_path, format="PNG")
        return int(width), int(height)

    def save_root_sample(self, root_sample: RootSample) -> Path:
        self.ensure_sample_layout(root_sample.sample_id)
        return root_sample.to_json_file(self.root_sample_path(root_sample.sample_id))

    def load_root_sample(self, sample_id: str) -> RootSample:
        return RootSample.from_json_file(self.root_sample_path(sample_id))

    def _copy_root_artifacts(self, root_sample: RootSample) -> list[ImageArtifactRef]:
        artifacts_dir = self.sample_artifacts_dir(root_sample.sample_id)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        saved: list[ImageArtifactRef] = []
        for idx, root_image in enumerate(root_sample.images):
            source_path = Path(root_image.path).expanduser().resolve()
            if not source_path.exists():
                raise FileNotFoundError(f"root image not found: {source_path}")
            target_path = artifacts_dir / f"img_root_{idx}.png"
            width, height = self._materialize_root_artifact(source_path, target_path)
            saved.append(
                ImageArtifactRef(
                    artifact_id=f"img_root_{idx}",
                    path=str(target_path),
                    media_type="image/png",
                    width=width,
                    height=height,
                )
            )
        return saved

    def load_root_artifacts(self, sample_id: str) -> list[ImageArtifactRef]:
        artifacts: list[ImageArtifactRef] = []
        artifacts_dir = self.sample_artifacts_dir(sample_id)
        for artifact_path in sorted(artifacts_dir.glob("img_root_*.png")):
            with Image.open(artifact_path) as image:
                width, height = image.size
            artifact_id = artifact_path.stem
            artifacts.append(
                ImageArtifactRef(
                    artifact_id=artifact_id,
                    path=str(artifact_path),
                    media_type="image/png",
                    width=int(width),
                    height=int(height),
                )
            )
        return artifacts

    def _build_initial_messages(
        self,
        root_sample: RootSample,
        root_artifacts: Sequence[ImageArtifactRef],
        *,
        system_message: str,
    ) -> MessagesDocument:
        return MessagesDocument.model_validate(
            [
                {
                    "message_id": "m_sys",
                    "role": "system",
                    "content": system_message,
                    "image_artifact_ids": [],
                    "metadata": {"message_kind": "system_instruction"},
                },
                {
                    "message_id": "m_user",
                    "role": "user",
                    "content": root_sample.question,
                    "image_artifact_ids": [artifact.artifact_id for artifact in root_artifacts],
                    "metadata": {
                        "message_kind": "user_question",
                        **(
                            {"answer_instruction": root_sample.answer_instruction}
                            if root_sample.answer_instruction
                            else {}
                        ),
                    },
                },
            ]
        )

    def init_root_trajectory(
        self,
        root_sample: RootSample,
        *,
        budget: Budget,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        created_at=None,
        trajectory_id: str | None = None,
    ) -> InitializedTrajectory:
        created_at = created_at or utc_now()
        sample_id = root_sample.sample_id
        trajectory_id = trajectory_id or build_root_trajectory_id(sample_id)
        self.ensure_sample_layout(sample_id)
        if self.trajectory_dir(sample_id, trajectory_id).exists():
            raise FileExistsError(f"trajectory already exists: {self.trajectory_dir(sample_id, trajectory_id)}")

        root_sample_path = self.save_root_sample(root_sample)
        root_artifacts = self._copy_root_artifacts(root_sample)
        self.ensure_trajectory_layout(sample_id, trajectory_id)

        messages = self._build_initial_messages(root_sample, root_artifacts, system_message=system_message)
        messages_path = self.save_messages(sample_id, trajectory_id, messages)

        trajectory = TrajectoryRecord(
            sample_id=sample_id,
            trajectory_id=trajectory_id,
            parent_trajectory_id=None,
            status="running",
            created_at=created_at,
            updated_at=created_at,
            round_idx=0,
            step_idx=0,
            question=root_sample.question,
            answer_instruction=root_sample.answer_instruction,
            original_image_artifact_id=root_artifacts[0].artifact_id,
            messages_path=self._relative_to_trajectory(sample_id, trajectory_id, messages_path),
            planner_history=[],
            latest_planner_round_idx=None,
            latest_planner_output_path=None,
            fork_provenance=None,
            pending_execution=None,
            steps=[],
            judge_records=[],
            final_answer=None,
            answer_confidence=None,
            budget=budget,
            last_error=None,
        )
        trajectory_path = self.save_trajectory(trajectory)

        return InitializedTrajectory(
            root_sample_path=root_sample_path,
            trajectory_path=trajectory_path,
            messages_path=messages_path,
            trajectory=trajectory,
            messages=messages,
            root_artifacts=root_artifacts,
        )

    def init_child_trajectory(
        self,
        parent_trajectory: TrajectoryRecord,
        *,
        fork_provenance: ForkProvenance,
        pending_execution: PendingExecution | None,
        budget: Budget | None = None,
        trajectory_id: str | None = None,
        created_at=None,
    ) -> InitializedTrajectory:
        created_at = created_at or utc_now()
        sample_id = parent_trajectory.sample_id
        trajectory_id = trajectory_id or build_child_trajectory_id(
            parent_trajectory.trajectory_id,
            fork_provenance.parent_planner_round_idx,
            fork_provenance.parent_suggestion_id,
        )
        if self.trajectory_dir(sample_id, trajectory_id).exists():
            raise FileExistsError(f"trajectory already exists: {self.trajectory_dir(sample_id, trajectory_id)}")

        self.ensure_sample_layout(sample_id)
        self.ensure_trajectory_layout(sample_id, trajectory_id)

        parent_messages = self.load_messages(sample_id, parent_trajectory.trajectory_id)
        child_messages = MessagesDocument.model_validate(parent_messages.model_dump(mode="json"))
        messages_path = self.save_messages(sample_id, trajectory_id, child_messages)

        child_trajectory = parent_trajectory.model_copy(deep=True)
        child_trajectory.sample_id = sample_id
        child_trajectory.trajectory_id = trajectory_id
        child_trajectory.parent_trajectory_id = parent_trajectory.trajectory_id
        child_trajectory.status = "running"
        child_trajectory.created_at = created_at
        child_trajectory.updated_at = created_at
        child_trajectory.messages_path = self._relative_to_trajectory(sample_id, trajectory_id, messages_path)
        child_trajectory.fork_provenance = fork_provenance
        child_trajectory.pending_execution = pending_execution
        child_trajectory.budget = budget.model_copy(deep=True) if budget is not None else parent_trajectory.budget.model_copy(deep=True)
        child_trajectory.last_error = None

        for planner_history_item in child_trajectory.planner_history:
            planner_history_item.planner_output_path = self._rebase_relative_path(
                planner_history_item.planner_output_path,
                source_sample_id=sample_id,
                source_trajectory_id=parent_trajectory.trajectory_id,
                target_sample_id=sample_id,
                target_trajectory_id=trajectory_id,
            )

        if child_trajectory.latest_planner_output_path is not None:
            child_trajectory.latest_planner_output_path = self._rebase_relative_path(
                child_trajectory.latest_planner_output_path,
                source_sample_id=sample_id,
                source_trajectory_id=parent_trajectory.trajectory_id,
                target_sample_id=sample_id,
                target_trajectory_id=trajectory_id,
            )

        for step_record in child_trajectory.steps:
            step_record.executor_cot_path = self._rebase_relative_path(
                step_record.executor_cot_path,
                source_sample_id=sample_id,
                source_trajectory_id=parent_trajectory.trajectory_id,
                target_sample_id=sample_id,
                target_trajectory_id=trajectory_id,
            )
            step_record.executor_code_path = self._rebase_relative_path(
                step_record.executor_code_path,
                source_sample_id=sample_id,
                source_trajectory_id=parent_trajectory.trajectory_id,
                target_sample_id=sample_id,
                target_trajectory_id=trajectory_id,
            )
            step_record.runtime_result_path = self._rebase_relative_path(
                step_record.runtime_result_path,
                source_sample_id=sample_id,
                source_trajectory_id=parent_trajectory.trajectory_id,
                target_sample_id=sample_id,
                target_trajectory_id=trajectory_id,
            )

        for judge_record_ref in child_trajectory.judge_records:
            judge_record_ref.judge_record_path = self._rebase_relative_path(
                judge_record_ref.judge_record_path,
                source_sample_id=sample_id,
                source_trajectory_id=parent_trajectory.trajectory_id,
                target_sample_id=sample_id,
                target_trajectory_id=trajectory_id,
            )

        if child_trajectory.last_error and child_trajectory.last_error.traceback_path:
            child_trajectory.last_error.traceback_path = self._rebase_relative_path(
                child_trajectory.last_error.traceback_path,
                source_sample_id=sample_id,
                source_trajectory_id=parent_trajectory.trajectory_id,
                target_sample_id=sample_id,
                target_trajectory_id=trajectory_id,
            )

        trajectory_path = self.save_trajectory(child_trajectory)
        root_sample_path = self.root_sample_path(sample_id)

        return InitializedTrajectory(
            root_sample_path=root_sample_path,
            trajectory_path=trajectory_path,
            messages_path=messages_path,
            trajectory=child_trajectory,
            messages=child_messages,
            root_artifacts=self.load_root_artifacts(sample_id),
        )

    def save_trajectory(self, trajectory: TrajectoryRecord) -> Path:
        self.ensure_trajectory_layout(trajectory.sample_id, trajectory.trajectory_id)
        return trajectory.to_json_file(self.trajectory_path(trajectory.sample_id, trajectory.trajectory_id))

    def load_trajectory(self, sample_id: str, trajectory_id: str) -> TrajectoryRecord:
        return TrajectoryRecord.from_json_file(self.trajectory_path(sample_id, trajectory_id))

    def save_messages(
        self,
        sample_id: str,
        trajectory_id: str,
        messages: MessagesDocument | Sequence[ConversationMessage] | Sequence[dict],
    ) -> Path:
        self.ensure_trajectory_layout(sample_id, trajectory_id)
        if not isinstance(messages, MessagesDocument):
            messages = MessagesDocument.model_validate(list(messages))
        return messages.to_json_file(self.messages_path(sample_id, trajectory_id))

    def load_messages(self, sample_id: str, trajectory_id: str) -> MessagesDocument:
        trajectory = self.load_trajectory(sample_id, trajectory_id)
        return MessagesDocument.from_json_file(
            self._resolve_trajectory_path(sample_id, trajectory_id, trajectory.messages_path)
        )

    def append_messages(
        self,
        sample_id: str,
        trajectory_id: str,
        new_messages: Sequence[ConversationMessage] | Sequence[dict],
    ) -> MessagesDocument:
        existing = self.load_messages(sample_id, trajectory_id)
        existing_payload = existing.model_dump(mode="json")
        existing_payload.extend(
            [
                item.model_dump(mode="json") if isinstance(item, ConversationMessage) else dict(item)
                for item in new_messages
            ]
        )
        updated = MessagesDocument.model_validate(existing_payload)
        self.save_messages(sample_id, trajectory_id, updated)
        return updated

    def write_executor_step_files(
        self,
        sample_id: str,
        trajectory_id: str,
        step_idx: int,
        *,
        executor_cot: str,
        executor_code: str,
    ) -> StepFilePaths:
        step_paths = self.build_step_file_paths(sample_id, trajectory_id, step_idx, create_dirs=True)
        step_paths.executor_cot_path.write_text(executor_cot or "", encoding="utf-8")
        step_paths.executor_code_path.write_text(executor_code or "", encoding="utf-8")
        return step_paths

    def save_planner_output(self, planner_output: PlannerOutput) -> Path:
        self.ensure_trajectory_layout(planner_output.sample_id, planner_output.trajectory_id)
        return planner_output.to_json_file(
            self.planner_output_path(planner_output.sample_id, planner_output.trajectory_id, planner_output.round_idx)
        )

    def load_planner_output(self, sample_id: str, trajectory_id: str, round_idx: int) -> PlannerOutput:
        trajectory = self.load_trajectory(sample_id, trajectory_id)
        for item in trajectory.planner_history:
            if int(item.round_idx) != int(round_idx):
                continue
            planner_path = self._resolve_trajectory_path(sample_id, trajectory_id, item.planner_output_path)
            return PlannerOutput.from_json_file(planner_path)
        return PlannerOutput.from_json_file(self.planner_output_path(sample_id, trajectory_id, round_idx))

    def register_planner_round(
        self,
        planner_output: PlannerOutput,
        *,
        selected_for_expansion: bool,
        pending_execution: PendingExecution | None = None,
        final_answer: str | None = None,
    ) -> TrajectoryRecord:
        planner_path = self.save_planner_output(planner_output)
        trajectory = self.load_trajectory(planner_output.sample_id, planner_output.trajectory_id)
        planner_history_item = PlannerHistoryItem(
            round_idx=planner_output.round_idx,
            planner_output_path=self._relative_to_trajectory(
                planner_output.sample_id,
                planner_output.trajectory_id,
                planner_path,
            ),
            can_answer_now=planner_output.can_answer_now,
            direct_answer=planner_output.direct_answer,
            selected_for_expansion=selected_for_expansion,
            created_at=planner_output.created_at,
        )
        trajectory.planner_history = [
            item for item in trajectory.planner_history if item.round_idx != planner_output.round_idx
        ]
        trajectory.planner_history.append(planner_history_item)
        trajectory.planner_history.sort(key=lambda item: item.round_idx)
        trajectory.latest_planner_round_idx = planner_output.round_idx
        trajectory.latest_planner_output_path = planner_history_item.planner_output_path
        trajectory.round_idx = max(trajectory.round_idx, planner_output.round_idx)
        trajectory.pending_execution = pending_execution
        if planner_output.can_answer_now:
            trajectory.final_answer = final_answer or planner_output.direct_answer
        trajectory.updated_at = utc_now()
        self.save_trajectory(trajectory)
        return trajectory

    def register_step_record(
        self,
        sample_id: str,
        trajectory_id: str,
        step_record: StepRecord,
        *,
        clear_pending_execution: bool = True,
    ) -> TrajectoryRecord:
        trajectory = self.load_trajectory(sample_id, trajectory_id)
        normalized_step_record = step_record.model_copy(deep=True)
        normalized_step_record.executor_cot_path = self._relative_to_trajectory(
            sample_id,
            trajectory_id,
            normalized_step_record.executor_cot_path,
        )
        normalized_step_record.executor_code_path = self._relative_to_trajectory(
            sample_id,
            trajectory_id,
            normalized_step_record.executor_code_path,
        )
        normalized_step_record.runtime_result_path = self._relative_to_trajectory(
            sample_id,
            trajectory_id,
            normalized_step_record.runtime_result_path,
        )
        trajectory.steps = [item for item in trajectory.steps if item.step_idx != normalized_step_record.step_idx]
        trajectory.steps.append(normalized_step_record)
        trajectory.steps.sort(key=lambda item: item.step_idx)
        trajectory.step_idx = max(trajectory.step_idx, normalized_step_record.step_idx)
        trajectory.round_idx = max(trajectory.round_idx, normalized_step_record.planner_round_idx)
        if clear_pending_execution:
            trajectory.pending_execution = None
        trajectory.updated_at = utc_now()
        self.save_trajectory(trajectory)
        return trajectory

    def save_judge_record(self, judge_record: JudgeRecord) -> Path:
        self.ensure_trajectory_layout(judge_record.sample_id, judge_record.trajectory_id)
        return judge_record.to_json_file(
            self.judge_record_path(
                judge_record.sample_id,
                judge_record.trajectory_id,
                judge_record.judge_stage,
                scope_step_idx=judge_record.scope_step_idx,
            )
        )

    def load_judge_record(
        self,
        sample_id: str,
        trajectory_id: str,
        judge_stage: str,
        *,
        scope_step_idx: int | None = None,
    ) -> JudgeRecord:
        trajectory = self.load_trajectory(sample_id, trajectory_id)
        for item in reversed(trajectory.judge_records):
            if item.judge_stage != judge_stage:
                continue
            judge_path = self._resolve_trajectory_path(sample_id, trajectory_id, item.judge_record_path)
            record = JudgeRecord.from_json_file(judge_path)
            if record.scope_step_idx == scope_step_idx:
                return record
        return JudgeRecord.from_json_file(
            self.judge_record_path(sample_id, trajectory_id, judge_stage, scope_step_idx=scope_step_idx)
        )

    def register_judge_record(self, judge_record: JudgeRecord) -> TrajectoryRecord:
        judge_path = self.save_judge_record(judge_record)
        trajectory = self.load_trajectory(judge_record.sample_id, judge_record.trajectory_id)
        record_ref = JudgeRecordRef(
            judge_stage=judge_record.judge_stage,
            judge_record_path=self._relative_to_trajectory(
                judge_record.sample_id,
                judge_record.trajectory_id,
                judge_path,
            ),
        )
        trajectory.judge_records = [
            item
            for item in trajectory.judge_records
            if item.judge_record_path != record_ref.judge_record_path
        ]
        trajectory.judge_records.append(record_ref)
        trajectory.updated_at = utc_now()
        self.save_trajectory(trajectory)
        return trajectory

    def save_canonical_export(self, canonical_sample: CanonicalSftSample) -> Path:
        trajectory = self.load_trajectory(canonical_sample.sample_id, canonical_sample.source_trajectory_id)
        if trajectory.status not in {
            "answered",
            "pruned",
            "failed",
            "stopped_early",
            "max_step_reached",
            "error",
        }:
            raise ValueError(
                f"trajectory status must be terminal before export, got {trajectory.status!r}."
            )
        output_path = self.export_path(canonical_sample.sample_id, canonical_sample.source_trajectory_id)
        return canonical_sample.to_json_file(output_path)

    def load_canonical_export(self, sample_id: str, trajectory_id: str) -> CanonicalSftSample:
        return CanonicalSftSample.from_json_file(self.export_path(sample_id, trajectory_id))

    def set_pending_execution(
        self,
        sample_id: str,
        trajectory_id: str,
        pending_execution: PendingExecution | None,
    ) -> TrajectoryRecord:
        trajectory = self.load_trajectory(sample_id, trajectory_id)
        trajectory.pending_execution = pending_execution
        trajectory.updated_at = utc_now()
        self.save_trajectory(trajectory)
        return trajectory

    def mark_trajectory_expanded(self, sample_id: str, trajectory_id: str) -> TrajectoryRecord:
        return self.mark_trajectory_status(
            sample_id,
            trajectory_id,
            status="expanded",
            pending_execution=None,
        )

    def mark_trajectory_status(
        self,
        sample_id: str,
        trajectory_id: str,
        *,
        status: TrajectoryStatus,
        final_answer: str | None = None,
        answer_confidence: float | None = None,
        pending_execution: PendingExecution | None = None,
        last_error: TrajectoryErrorInfo | None = None,
    ) -> TrajectoryRecord:
        trajectory = self.load_trajectory(sample_id, trajectory_id)
        trajectory.status = status
        trajectory.pending_execution = pending_execution
        if final_answer is not None:
            trajectory.final_answer = final_answer
        if answer_confidence is not None:
            trajectory.answer_confidence = answer_confidence
        if last_error is not None:
            trajectory.last_error = last_error
        trajectory.updated_at = utc_now()
        self.save_trajectory(trajectory)
        return trajectory

    def resolve_artifact_id(
        self,
        sample_id: str,
        trajectory_id: str,
        artifact_id: str,
    ) -> ImageArtifactRef:
        if artifact_id.startswith("img_root_"):
            artifact_path = self.sample_artifacts_dir(sample_id) / f"{artifact_id}.png"
            if not artifact_path.exists():
                raise FileNotFoundError(f"root artifact not found: {artifact_path}")
            with Image.open(artifact_path) as image:
                width, height = image.size
            return ImageArtifactRef(
                artifact_id=artifact_id,
                path=str(artifact_path),
                media_type="image/png",
                width=int(width),
                height=int(height),
            )

        parts = artifact_id.split("_")
        if len(parts) >= 4 and parts[0] == "img" and parts[1] == "step":
            step_idx = int(parts[2])
            runtime_result = self.load_runtime_result(sample_id, trajectory_id, step_idx)
            for artifact in runtime_result.images:
                if artifact.artifact_id == artifact_id:
                    return artifact

        raise KeyError(f"unknown artifact_id: {artifact_id}")

    def resolve_artifact_ids(
        self,
        sample_id: str,
        trajectory_id: str,
        artifact_ids: Iterable[str],
    ) -> list[ImageArtifactRef]:
        return [self.resolve_artifact_id(sample_id, trajectory_id, artifact_id) for artifact_id in artifact_ids]

    def load_runtime_result(self, sample_id: str, trajectory_id: str, step_idx: int):
        from offline_sft_pipeline.core.models import ExecutorRuntimeResult

        trajectory = self.load_trajectory(sample_id, trajectory_id)
        for step_record in trajectory.steps:
            if int(step_record.step_idx) != int(step_idx):
                continue
            runtime_result_path = self._resolve_trajectory_path(
                sample_id,
                trajectory_id,
                step_record.runtime_result_path,
            )
            return ExecutorRuntimeResult.from_json_file(runtime_result_path)
        step_paths = self.build_step_file_paths(sample_id, trajectory_id, step_idx, create_dirs=False)
        return ExecutorRuntimeResult.from_json_file(step_paths.runtime_result_path)

    def load_resume_state(self, sample_id: str, trajectory_id: str) -> ResumeState:
        trajectory = self.load_trajectory(sample_id, trajectory_id)
        messages = self.load_messages(sample_id, trajectory_id)
        return ResumeState(
            trajectory=trajectory,
            messages=messages,
            root_artifacts=self.load_root_artifacts(sample_id),
        )

    def list_trajectories(
        self,
        *,
        sample_id: str | None = None,
    ) -> list[TrajectoryRecord]:
        if sample_id is None:
            trajectory_paths = sorted(self.samples_root.glob("*/trajectories/*/trajectory.json"))
        else:
            trajectory_paths = sorted(self.trajectories_dir(sample_id).glob("*/trajectory.json"))
        return [TrajectoryRecord.from_json_file(path) for path in trajectory_paths]

    def list_resumable_trajectories(
        self,
        *,
        sample_id: str | None = None,
    ) -> list[TrajectoryRecord]:
        trajectories = [
            trajectory
            for trajectory in self.list_trajectories(sample_id=sample_id)
            if trajectory.status == "running"
        ]
        trajectories.sort(key=lambda item: item.updated_at)
        return trajectories


__all__ = [
    "DEFAULT_SYSTEM_MESSAGE",
    "InitializedTrajectory",
    "OfflineTrajectoryStore",
    "ResumeState",
    "StepFilePaths",
]
