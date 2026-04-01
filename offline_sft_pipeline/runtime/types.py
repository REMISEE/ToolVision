from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(slots=True)
class ArtifactRef:
    artifact_id: str
    path: str
    media_type: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactRef":
        return cls(
            artifact_id=str(data["artifact_id"]),
            path=str(data["path"]),
            media_type=data.get("media_type"),
            width=data.get("width"),
            height=data.get("height"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "path": self.path,
            "media_type": self.media_type,
            "width": self.width,
            "height": self.height,
        }

    @property
    def path_obj(self) -> Path:
        return Path(self.path)


@dataclass(slots=True)
class RuntimeStepRequest:
    sample_id: str
    trajectory_id: str
    round_idx: int
    step_idx: int
    executor_code_path: str
    visible_images: list[ArtifactRef]
    step_output_dir: str
    image_index: int = 0
    executor_cot_path: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeStepRequest":
        visible_images = [ArtifactRef.from_dict(item) for item in data.get("visible_images", [])]
        return cls(
            sample_id=str(data["sample_id"]),
            trajectory_id=str(data["trajectory_id"]),
            round_idx=int(data["round_idx"]),
            step_idx=int(data["step_idx"]),
            executor_code_path=str(data["executor_code_path"]),
            visible_images=visible_images,
            step_output_dir=str(data["step_output_dir"]),
            image_index=int(data.get("image_index", 0)),
            executor_cot_path=data.get("executor_cot_path"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "trajectory_id": self.trajectory_id,
            "round_idx": self.round_idx,
            "step_idx": self.step_idx,
            "executor_cot_path": self.executor_cot_path,
            "executor_code_path": self.executor_code_path,
            "visible_images": [item.to_dict() for item in self.visible_images],
            "image_index": self.image_index,
            "step_output_dir": self.step_output_dir,
        }

    @property
    def executor_code_path_obj(self) -> Path:
        return Path(self.executor_code_path)

    @property
    def executor_cot_path_obj(self) -> Optional[Path]:
        if self.executor_cot_path is None:
            return None
        return Path(self.executor_cot_path)

    @property
    def step_output_dir_obj(self) -> Path:
        return Path(self.step_output_dir)


@dataclass(slots=True)
class RuntimeStepOutput:
    runtime_result: dict[str, Any]
    saved_artifacts: list[ArtifactRef] = field(default_factory=list)
    runtime_result_path: Optional[str] = None
    tool_metrics: dict[str, Any] = field(default_factory=dict)

