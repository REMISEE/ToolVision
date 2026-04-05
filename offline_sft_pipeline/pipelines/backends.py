from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

from offline_sft_pipeline.pipelines.api_text_multimodal import (
    DEFAULT_QWEN_BASE_URL,
    DEFAULT_QWEN_MODEL,
    chat_completions_text,
    coerce_planner_request,
    env_planner_debug_enabled,
    env_qwen_config,
    is_placeholder_api_key,
    planner_to_openai_messages,
    sanitize_messages_for_debug,
    summarize_openai_message_for_debug,
)


DEFAULT_FAKE_PLANNER_TEXT = """{
  "mode": "suggestions",
  "think": "We still need tool use before answering. First localize the relevant target, then inspect the cropped region.",
  "suggestions": [
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
}
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
        context: dict[str, Any] | None = None,
    ) -> BackendResponse: ...


class JudgeBackend(Protocol):
    def score(self, request: Any) -> JudgeBackendResult: ...


@dataclass(slots=True)
class ApiTextBackendConfig:
    api_key: str | None = None
    base_url: str = DEFAULT_QWEN_BASE_URL
    model: str = DEFAULT_QWEN_MODEL
    timeout_s: float = 120.0
    dry_run: bool = False


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
        context: dict[str, Any] | None = None,
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
            "has_context": context is not None,
        }
        metadata.update(self.stage_metadata.get(stage, {}))
        return BackendResponse(text=text, metadata=metadata)


class ApiTextBackend:
    """Real planner HTTP backend. Executor stage still relies on fake/scripted backends."""

    def __init__(self, *, config: ApiTextBackendConfig | None = None) -> None:
        env = env_qwen_config()
        if config is None:
            self._cfg = ApiTextBackendConfig(
                api_key=env["api_key"],
                base_url=str(env["base_url"]),
                model=str(env["model"]),
                timeout_s=float(env["timeout_s"]),
                dry_run=bool(env["dry_run"]),
            )
        else:
            self._cfg = replace(
                config,
                api_key=config.api_key if config.api_key is not None else env["api_key"],
                dry_run=config.dry_run or bool(env["dry_run"]),
            )

    def generate(
        self,
        *,
        stage: Literal["planner", "executor"],
        system_prompt: str,
        user_prompt: str,
        context: dict[str, Any] | None = None,
    ) -> BackendResponse:
        if stage == "executor":
            raise NotImplementedError(
                "ApiTextBackend does not implement executor stage yet; use FakeTextBackend or ScriptedTextBackend."
            )
        if stage != "planner":
            raise NotImplementedError(f"ApiTextBackend unsupported stage={stage!r}.")

        if self._cfg.dry_run:
            return BackendResponse(
                text=DEFAULT_FAKE_PLANNER_TEXT,
                metadata={
                    "backend": "api_text",
                    "dry_run": True,
                    "stage": "planner",
                },
            )

        if is_placeholder_api_key(self._cfg.api_key):
            raise RuntimeError(
                "OFFLINE_SFT_QWEN_API_KEY is missing or placeholder. Set a real key or set OFFLINE_SFT_API_DRY_RUN=1."
            )

        request_obj = None
        if context is not None:
            request_obj = coerce_planner_request(context.get("request"))
        if request_obj is None:
            raise ValueError(
                "ApiTextBackend(planner) requires context['request'] to be a PlannerClientRequest (or dict)."
            )

        messages, missing_ids = planner_to_openai_messages(system_prompt=system_prompt, req=request_obj)
        if env_planner_debug_enabled():
            safe_messages = sanitize_messages_for_debug(messages)
            print(
                "[OFFLINE_SFT_PLANNER_DEBUG] OpenAI-style messages (base64 images shortened). "
                "PlannerClient's rendered user_prompt is not sent directly; payload is rebuilt from context['request'].",
                file=sys.stderr,
            )
            print(json.dumps(safe_messages, ensure_ascii=False, indent=2), file=sys.stderr)
            if missing_ids:
                print(f"[OFFLINE_SFT_PLANNER_DEBUG] missing_artifact_ids: {missing_ids}", file=sys.stderr)

        text, raw_payload = chat_completions_text(
            base_url=self._cfg.base_url,
            api_key=self._cfg.api_key or "",
            model=self._cfg.model,
            messages=messages,
            timeout_s=self._cfg.timeout_s,
        )
        if env_planner_debug_enabled():
            choice0 = (raw_payload.get("choices") or [{}])[0]
            message = choice0.get("message")
            if isinstance(message, dict):
                print(
                    "[OFFLINE_SFT_PLANNER_DEBUG] choices[0].message (content replaced by len; check reasoning/extra fields):",
                    file=sys.stderr,
                )
                print(
                    json.dumps(summarize_openai_message_for_debug(message), ensure_ascii=False, indent=2),
                    file=sys.stderr,
                )
            print("[OFFLINE_SFT_PLANNER_DEBUG] Assistant message content (full text):", file=sys.stderr)
            print(text, file=sys.stderr)

        return BackendResponse(
            text=text,
            raw_payload=raw_payload,
            metadata={
                "backend": "api_text",
                "stage": "planner",
                "model": self._cfg.model,
                "missing_artifact_ids": missing_ids,
            },
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
    "ApiTextBackendConfig",
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
