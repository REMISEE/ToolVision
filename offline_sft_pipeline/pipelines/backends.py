from __future__ import annotations

import os
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from offline_sft_pipeline.eval.scorers import score_answer_for_dataset
from offline_sft_pipeline.pipelines.api_text_multimodal import (
    DEFAULT_QWEN_BASE_URL,
    DEFAULT_QWEN_MODEL,
    chat_completions_text,
    coerce_executor_request,
    build_judge_control_user_text,
    coerce_planner_request,
    env_executor_debug_enabled,
    env_planner_debug_enabled,
    executor_to_openai_messages,
    env_qwen_config,
    is_placeholder_api_key,
    judge_to_openai_messages,
    planner_to_openai_messages,
    sanitize_messages_for_debug,
    summarize_openai_message_for_debug,
)


def _normalize_token_usage(raw_payload: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(raw_payload, dict):
        return {}
    usage = raw_payload.get("usage")
    if not isinstance(usage, dict):
        return {}
    normalized: dict[str, int] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "cached_tokens",
    ):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            normalized[key] = int(value)
    return normalized


def _empty_token_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
    }


def _accumulate_token_usage(target: dict[str, int], source: dict[str, int]) -> None:
    for key in _empty_token_usage():
        target[key] = int(target.get(key, 0)) + int(source.get(key, 0))


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
          "input_image_index": 0,
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

DEFAULT_FAKE_EXECUTOR_TEXT = """{
  "think": "First localize the relevant target, then crop it for closer inspection and keep the latest image active.",
  "tool_call": {
    "name": "code_image_tool",
    "arguments": {
      "code": "box = _call_ground_box(\\"target object\\")\\ncrop = _call_dino_crop(\\"target object\\", image_obj=box[\\"image\\"], based_on=\\"box\\", max_crops=1, padding=8)\\nprint(crop.get(\\"text\\", \\"\\"))\\nresult = crop[\\"image\\"]",
      "description": "Ground the target object and keep a crop for the next reasoning step."
    }
  }
}
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
    timeout_s: float = 200.0
    dry_run: bool = False


@dataclass(slots=True)
class JudgeModelConfig:
    name: str
    model: str
    base_url: str
    api_key_env: str
    timeout_s: float = 240.0
    enabled: bool = True
    request_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JudgeModelConfig":
        request_body = payload.get("request_body")
        if request_body is None:
            normalized_request_body: dict[str, Any] = {}
        elif isinstance(request_body, dict):
            normalized_request_body = dict(request_body)
        else:
            raise ValueError(f"judge model config request_body must be an object: {payload!r}")
        return cls(
            name=str(payload.get("name", "")).strip(),
            model=str(payload.get("model", "")).strip(),
            base_url=str(payload.get("base_url", "")).strip().rstrip("/"),
            api_key_env=str(payload.get("api_key_env", "")).strip(),
            timeout_s=float(payload.get("timeout_s", 240.0)),
            enabled=bool(payload.get("enabled", True)),
            request_body=normalized_request_body,
        )


DEFAULT_JUDGE_MODELS_PATH = Path(__file__).resolve().parents[1] / "judge_models.json"
DEFAULT_JUDGE_PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"
DEFAULT_JUDGE_SYSTEM_PROMPT_FILE = "judge_system_v01.txt"
DEFAULT_JUDGE_MAX_CONCURRENCY = 5
DEFAULT_JUDGE_MAX_RETRIES = 2


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
    """Real planner/executor HTTP backend."""

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
        if stage not in {"planner", "executor"}:
            raise NotImplementedError(f"ApiTextBackend unsupported stage={stage!r}.")

        if self._cfg.dry_run:
            fake_text = DEFAULT_FAKE_PLANNER_TEXT if stage == "planner" else DEFAULT_FAKE_EXECUTOR_TEXT
            return BackendResponse(
                text=fake_text,
                metadata={
                    "backend": "api_text",
                    "dry_run": True,
                    "stage": stage,
                },
            )

        if is_placeholder_api_key(self._cfg.api_key):
            raise RuntimeError(
                "OFFLINE_SFT_QWEN_API_KEY is missing or placeholder. Set a real key or set OFFLINE_SFT_API_DRY_RUN=1."
            )

        request_obj = None
        if context is not None:
            request_obj = (
                coerce_planner_request(context.get("request"))
                if stage == "planner"
                else coerce_executor_request(context.get("request"))
            )
        if request_obj is None:
            if stage == "planner":
                raise ValueError(
                    "ApiTextBackend(planner) requires context['request'] to be a PlannerClientRequest (or dict)."
                )
            raise ValueError(
                "ApiTextBackend(executor) requires context['request'] to be an ExecutorClientRequest (or dict)."
            )

        if stage == "planner":
            messages, missing_ids = planner_to_openai_messages(system_prompt=system_prompt, req=request_obj)
            debug_enabled = env_planner_debug_enabled()
            debug_prefix = "OFFLINE_SFT_PLANNER_DEBUG"
            rebuild_note = (
                "PlannerClient's rendered user_prompt is not sent directly; payload is rebuilt from context['request']."
            )
        else:
            messages, missing_ids = executor_to_openai_messages(system_prompt=system_prompt, req=request_obj)
            debug_enabled = env_executor_debug_enabled()
            debug_prefix = "OFFLINE_SFT_EXECUTOR_DEBUG"
            rebuild_note = (
                "ExecutorClient's rendered user_prompt is not sent directly; payload is rebuilt from context['request']."
            )

        if debug_enabled:
            safe_messages = sanitize_messages_for_debug(messages)
            print(
                f"[{debug_prefix}] OpenAI-style messages (base64 images shortened). {rebuild_note}",
                file=sys.stderr,
            )
            print(json.dumps(safe_messages, ensure_ascii=False, indent=2), file=sys.stderr)
            if missing_ids:
                print(f"[{debug_prefix}] missing_artifact_ids: {missing_ids}", file=sys.stderr)

        text, raw_payload = chat_completions_text(
            base_url=self._cfg.base_url,
            api_key=self._cfg.api_key or "",
            model=self._cfg.model,
            messages=messages,
            timeout_s=self._cfg.timeout_s,
        )
        if debug_enabled:
            choice0 = (raw_payload.get("choices") or [{}])[0]
            message = choice0.get("message")
            if isinstance(message, dict):
                print(
                    f"[{debug_prefix}] choices[0].message (content replaced by len; check reasoning/extra fields):",
                    file=sys.stderr,
                )
                print(
                    json.dumps(summarize_openai_message_for_debug(message), ensure_ascii=False, indent=2),
                    file=sys.stderr,
                )
            print(f"[{debug_prefix}] Assistant message content (full text):", file=sys.stderr)
            print(text, file=sys.stderr)

        return BackendResponse(
            text=text,
            raw_payload=raw_payload,
            metadata={
                "backend": "api_text",
                "stage": stage,
                "model": self._cfg.model,
                "missing_artifact_ids": missing_ids,
                "token_usage": _normalize_token_usage(raw_payload),
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
    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        prompt_root: str | Path | None = None,
        system_prompt_filename: str = DEFAULT_JUDGE_SYSTEM_PROMPT_FILE,
        max_concurrency: int = DEFAULT_JUDGE_MAX_CONCURRENCY,
        max_retries: int = DEFAULT_JUDGE_MAX_RETRIES,
    ) -> None:
        self.config_path = Path(config_path) if config_path else DEFAULT_JUDGE_MODELS_PATH
        self.prompt_root = Path(prompt_root) if prompt_root else DEFAULT_JUDGE_PROMPT_ROOT
        self.system_prompt_path = self.prompt_root / system_prompt_filename
        self.max_concurrency = max(1, int(max_concurrency))
        self.max_retries = max(0, int(max_retries))
        self.model_configs = self._load_model_configs()
        self.enabled_model_configs = [item for item in self.model_configs if item.enabled]
        if not self.enabled_model_configs:
            raise ValueError(f"No enabled judge models found in {self.config_path}.")
        self.system_prompt = self._load_prompt(self.system_prompt_path)

    def _load_model_configs(self) -> list[JudgeModelConfig]:
        payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"judge model config must be a JSON array: {self.config_path}")
        configs = [JudgeModelConfig.from_dict(dict(item)) for item in payload if isinstance(item, dict)]
        for config in configs:
            if not config.name:
                raise ValueError(f"judge model config missing name: {self.config_path}")
            if not config.model:
                raise ValueError(f"judge model config missing model for {config.name}")
            if not config.base_url:
                raise ValueError(f"judge model config missing base_url for {config.name}")
            if not config.api_key_env:
                raise ValueError(f"judge model config missing api_key_env for {config.name}")
        return configs

    def _load_prompt(self, path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(f"judge system prompt not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    def _call_single_model(
        self,
        *,
        request: Any,
        model_config: JudgeModelConfig,
        messages: list[dict[str, Any]],
        source_dataset: str,
    ) -> dict[str, Any]:
        api_key = os.environ.get(model_config.api_key_env)
        if is_placeholder_api_key(api_key):
            raise RuntimeError(
                f"Missing or placeholder API key for judge model {model_config.name}: env {model_config.api_key_env}"
            )

        last_error: Exception | None = None
        for _attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                text, raw_payload = chat_completions_text(
                    base_url=model_config.base_url,
                    api_key=api_key or "",
                    model=model_config.model,
                    messages=messages,
                    request_body=model_config.request_body,
                    timeout_s=model_config.timeout_s,
                )
                latency_s = round(time.perf_counter() - started, 3)
                raw_answer = self._clean_judge_answer_text(text)
                score_result = score_answer_for_dataset(
                    source_dataset=source_dataset,
                    pred_answer=raw_answer,
                    answer=getattr(request, "answer", None),
                    metadata=dict(getattr(request, "metadata", {}) or {}),
                )
                choice0 = (raw_payload.get("choices") or [{}])[0] if isinstance(raw_payload, dict) else {}
                response_message = choice0.get("message") if isinstance(choice0, dict) else None
                return {
                    "name": model_config.name,
                    "model": model_config.model,
                    "raw_answer": raw_answer,
                    "normalized_answer": score_result.normalized_prediction,
                    "normalized_reference": score_result.normalized_reference,
                    "score": float(score_result.score),
                    "matcher_name": score_result.matcher_name,
                    "latency_s": latency_s,
                    "request_body": dict(model_config.request_body),
                    "token_usage": _normalize_token_usage(raw_payload),
                    "usage_raw": dict(raw_payload.get("usage", {})) if isinstance(raw_payload, dict) else {},
                    "response_message_summary": (
                        summarize_openai_message_for_debug(response_message)
                        if isinstance(response_message, dict)
                        else None
                    ),
                    "error": None,
                }
            except Exception as exc:  # noqa: PERF203
                last_error = exc
        raise RuntimeError(f"judge model {model_config.name} failed after retries: {last_error}") from last_error

    def score(self, request: Any) -> JudgeBackendResult:
        source_dataset = str(getattr(request, "metadata", {}).get("source_dataset") or "").strip().lower()
        messages, missing_ids = judge_to_openai_messages(system_prompt=self.system_prompt, req=request)
        model_results: list[dict[str, Any]] = []
        per_model_scores: dict[str, float] = {}
        aggregate_usage = _empty_token_usage()

        worker_count = min(self.max_concurrency, len(self.enabled_model_configs))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    self._call_single_model,
                    request=request,
                    model_config=model_config,
                    messages=messages,
                    source_dataset=source_dataset,
                ): model_config
                for model_config in self.enabled_model_configs
            }
            for future in as_completed(futures):
                model_config = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: PERF203
                    result = {
                        "name": model_config.name,
                        "model": model_config.model,
                        "raw_answer": None,
                        "normalized_answer": None,
                        "normalized_reference": None,
                        "score": 0.0,
                        "matcher_name": "error",
                        "latency_s": None,
                        "request_body": dict(model_config.request_body),
                        "token_usage": {},
                        "error": str(exc),
                    }
                model_results.append(result)
                if result["error"] is None:
                    per_model_scores[result["name"]] = float(result["score"])
                    _accumulate_token_usage(aggregate_usage, dict(result["token_usage"]))

        successful_scores = [float(item["score"]) for item in model_results if item["error"] is None]
        overall_score = (
            sum(successful_scores) / len(successful_scores)
            if successful_scores
            else 0.0
        )
        metadata = {
            "backend": "committee_judge",
            "source_dataset": source_dataset,
            "enabled_model_count": len(self.enabled_model_configs),
            "successful_model_count": len(successful_scores),
            "missing_artifact_ids": missing_ids,
            "model_results": sorted(model_results, key=lambda item: str(item["name"])),
            "token_usage": aggregate_usage,
            "judge_control_user_text": build_judge_control_user_text(request),
        }
        note = (
            f"committee_judge {len(successful_scores)}/{len(self.enabled_model_configs)} models scored; "
            f"overall_score={overall_score:.4f}"
        )
        return JudgeBackendResult(
            overall_score=overall_score,
            per_model_scores=per_model_scores,
            metadata=metadata,
            note=note,
        )

    def _clean_judge_answer_text(self, text: str) -> str:
        normalized = str(text or "").strip()
        if normalized.startswith("<answer>") and normalized.endswith("</answer>"):
            normalized = normalized[len("<answer>") : -len("</answer>")].strip()
        normalized = normalized.strip("`").strip()
        return normalized


__all__ = [
    "ApiTextBackend",
    "ApiTextBackendConfig",
    "BackendResponse",
    "CommitteeJudgeBackend",
    "DEFAULT_JUDGE_MAX_CONCURRENCY",
    "DEFAULT_JUDGE_MAX_RETRIES",
    "DEFAULT_JUDGE_MODELS_PATH",
    "DEFAULT_FAKE_EXECUTOR_TEXT",
    "DEFAULT_FAKE_PLANNER_TEXT",
    "FakeJudgeBackend",
    "FakeTextBackend",
    "JudgeBackend",
    "JudgeBackendResult",
    "JudgeModelConfig",
    "TextGenerationBackend",
]
