from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


DEFAULT_FAKE_PLANNER_TEXT = """<think>
We still need tool use before answering. First localize the relevant target, then inspect the cropped region.
</think>
<suggestions>
[
  {
    "suggestion_id": "s1",
    "suggestion_cot": "Ground the most relevant target and crop it for close inspection.",
    "steps": [
      {
        "step_id": "step_a",
        "step_goal": "Locate the target object mentioned in the question.",
        "capability_plan": [
          {
            "order": 1,
            "capability": "ground_box",
            "instruction": "Ground the object or region most relevant to answering the question."
          }
        ],
        "executor_instruction": "Write code that grounds the most relevant target and preserves the resulting image for the next reasoning step."
      }
    ]
  }
]
</suggestions>
"""

DEFAULT_FAKE_EXECUTOR_TEXT = """<think>
First localize the relevant target, then crop it for closer inspection and keep the latest image active.
</think>
<code>
box = _call_ground_box("target object")
crop = _call_dino_crop("target object", image_obj=box["image"], based_on="box", max_crops=1, padding=8)
print(crop.get("text", ""))
result = crop["image"]
</code>
"""


@dataclass(slots=True)
class BackendResponse:
    text: str
    raw_payload: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JudgeBackendResult:
    overall_score: float
    per_model_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    note: str = ""


class TextGenerationBackend(Protocol):
    def generate(
        self,
        *,
        stage: Literal["planner", "executor"],
        system_prompt: str,
        user_prompt: str,
    ) -> BackendResponse: ...


class JudgeBackend(Protocol):
    def score(self, request: Any) -> JudgeBackendResult: ...


class FakeTextBackend:
    def __init__(
        self,
        *,
        stage_responses: dict[str, str] | None = None,
        stage_metadata: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.stage_responses = dict(stage_responses or {})
        self.stage_metadata = {key: dict(value) for key, value in (stage_metadata or {}).items()}

    def generate(
        self,
        *,
        stage: Literal["planner", "executor"],
        system_prompt: str,
        user_prompt: str,
    ) -> BackendResponse:
        if stage in self.stage_responses:
            text = self.stage_responses[stage]
        elif stage == "planner":
            text = DEFAULT_FAKE_PLANNER_TEXT
        elif stage == "executor":
            text = DEFAULT_FAKE_EXECUTOR_TEXT
        else:
            raise NotImplementedError(f"FakeTextBackend has no default response for stage={stage!r}.")

        metadata = {
            "backend": "fake_text",
            "stage": stage,
            "system_prompt_chars": len(system_prompt),
            "user_prompt_chars": len(user_prompt),
        }
        metadata.update(self.stage_metadata.get(stage, {}))
        return BackendResponse(text=text, metadata=metadata)


class ApiTextBackend:
    def generate(
        self,
        *,
        stage: Literal["planner", "executor"],
        system_prompt: str,
        user_prompt: str,
    ) -> BackendResponse:
        raise NotImplementedError(
            "ApiTextBackend is not implemented yet. Wire it to the real model API in a later step."
        )


class FakeJudgeBackend:
    def __init__(
        self,
        *,
        overall_score: float = 0.62,
        per_model_scores: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
        note: str = "Fake judge backend default score.",
    ) -> None:
        self.overall_score = float(overall_score)
        self.per_model_scores = dict(
            per_model_scores
            or {
                "judge_model_01": self.overall_score,
                "judge_model_02": self.overall_score,
                "judge_model_03": self.overall_score,
            }
        )
        self.metadata = dict(metadata or {"backend": "fake_judge"})
        self.note = note

    def score(self, request: Any) -> JudgeBackendResult:
        metadata = dict(self.metadata)
        metadata.update(
            {
                "sample_id": getattr(request, "sample_id", None),
                "trajectory_id": getattr(request, "trajectory_id", None),
                "judge_stage": getattr(request, "judge_stage", None),
            }
        )
        return JudgeBackendResult(
            overall_score=self.overall_score,
            per_model_scores=self.per_model_scores,
            metadata=metadata,
            note=self.note,
        )


class CommitteeJudgeBackend:
    def score(self, request: Any) -> JudgeBackendResult:
        raise NotImplementedError(
            "CommitteeJudgeBackend is not implemented yet. Wire it to the real multi-model judge stack later."
        )


__all__ = [
    "ApiTextBackend",
    "BackendResponse",
    "CommitteeJudgeBackend",
    "DEFAULT_FAKE_EXECUTOR_TEXT",
    "DEFAULT_FAKE_PLANNER_TEXT",
    "FakeJudgeBackend",
    "FakeTextBackend",
    "JudgeBackend",
    "JudgeBackendResult",
    "TextGenerationBackend",
]
