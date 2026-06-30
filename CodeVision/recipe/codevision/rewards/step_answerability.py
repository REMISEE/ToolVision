from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from .common import extract_answer
from .router import compute_toolvision_score


STEP_REWARD_VERSION = "step_answerability_delta_v1"


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def as_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def as_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def coerce_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return coerce_json_list(value.tolist())
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def compute_step_answerability_delta(
    scores: Any,
    valid_steps: Any = None,
    *,
    tau: float = 0.1,
    cap: float = 0.5,
) -> dict[str, Any]:
    """Compute capped positive answerability improvement from V0,V1,... scores.

    ``scores`` is expected to contain baseline V0 followed by one score per tool
    step. ``valid_steps`` applies only to post-step scores.
    """

    raw_scores = coerce_json_list(scores)
    numeric_scores: list[float | None] = []
    for item in raw_scores:
        try:
            numeric_scores.append(float(item))
        except Exception:
            numeric_scores.append(None)

    if not numeric_scores:
        return {
            "v0": None,
            "step_scores": [],
            "step_gains": [],
            "raw_delta": 0.0,
            "capped_delta": 0.0,
            "best_score": None,
            "scored_count": 0,
            "valid_count": 0,
            "version": STEP_REWARD_VERSION,
        }

    valid_raw = coerce_json_list(valid_steps)
    step_count = max(0, len(numeric_scores) - 1)
    if valid_raw:
        valid_flags = [as_bool(item, False) for item in valid_raw[:step_count]]
        if len(valid_flags) < step_count:
            valid_flags.extend([False] * (step_count - len(valid_flags)))
    else:
        valid_flags = [True] * step_count

    v0 = numeric_scores[0]
    if v0 is None:
        v0 = 0.0
    best_score = float(v0)
    gains: list[float] = []
    raw_delta = 0.0
    scored_count = 0
    valid_count = 0

    for idx, score in enumerate(numeric_scores[1:]):
        is_valid = bool(valid_flags[idx]) if idx < len(valid_flags) else False
        if score is None:
            gains.append(0.0)
            continue
        scored_count += 1
        if not is_valid:
            gains.append(0.0)
            continue
        valid_count += 1
        gain = max(0.0, float(score) - best_score - float(tau))
        gains.append(gain)
        raw_delta += gain
        best_score = max(best_score, float(score))

    capped_delta = min(float(cap), max(0.0, raw_delta))
    return {
        "v0": float(v0),
        "step_scores": [None if item is None else float(item) for item in numeric_scores[1:]],
        "step_gains": gains,
        "raw_delta": raw_delta,
        "capped_delta": capped_delta,
        "best_score": best_score,
        "scored_count": scored_count,
        "valid_count": valid_count,
        "version": STEP_REWARD_VERSION,
    }


@dataclass(slots=True)
class StepAnswerabilityConfig:
    enable: bool = False
    base_url: str = ""
    model: str = ""
    api_key_env: str = "STEP_JUDGE_API_KEY"
    timeout_s: float = 60.0
    max_retries: int = 1
    max_images: int = 4
    max_observation_chars: int = 4000
    request_body: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, cfg: Any) -> "StepAnswerabilityConfig":
        cfg = cfg or {}
        get = cfg.get if hasattr(cfg, "get") else lambda key, default=None: default
        try:
            request_body = dict(get("request_body", {}) or {})
        except Exception:
            request_body = {}
        return cls(
            enable=as_bool(get("enable", os.getenv("STEP_REWARD_ENABLE", "0"))),
            base_url=str(get("base_url", os.getenv("STEP_JUDGE_BASE_URL", "")) or "").strip().rstrip("/"),
            model=str(get("model", os.getenv("STEP_JUDGE_MODEL", "")) or "").strip(),
            api_key_env=str(get("api_key_env", os.getenv("STEP_JUDGE_API_KEY_ENV", "STEP_JUDGE_API_KEY")) or "").strip(),
            timeout_s=as_float(get("timeout_s", os.getenv("STEP_JUDGE_TIMEOUT", "60")), 60.0),
            max_retries=as_int(get("max_retries", os.getenv("STEP_JUDGE_MAX_RETRIES", "1")), 1),
            max_images=as_int(get("max_images", os.getenv("STEP_JUDGE_MAX_IMAGES", "4")), 4),
            max_observation_chars=as_int(
                get("max_observation_chars", os.getenv("STEP_JUDGE_MAX_OBSERVATION_CHARS", "4000")),
                4000,
            ),
            request_body=request_body,
        )


class StepAnswerabilityJudgeClient:
    """OpenAI-compatible client for optional per-step answerability scoring."""

    def __init__(self, config: StepAnswerabilityConfig) -> None:
        self.config = config

    @classmethod
    def from_mapping(cls, cfg: Any) -> "StepAnswerabilityJudgeClient":
        return cls(StepAnswerabilityConfig.from_mapping(cfg))

    @property
    def enabled(self) -> bool:
        return bool(self.config.enable and self.config.base_url and self.config.model)

    def score_state(
        self,
        *,
        data_source: str,
        ground_truth: Any,
        extra_info: dict[str, Any],
        question: str,
        answer_instruction: str | None,
        state_label: str,
        observation_text: str,
        images: list[Any],
    ) -> dict[str, Any]:
        started = time.perf_counter()
        record: dict[str, Any] = {
            "state_label": state_label,
            "enabled": self.enabled,
            "score": None,
            "raw_answer": "",
            "final_answer": "",
            "error": None,
            "latency_s": None,
        }
        if not self.enabled:
            record["error"] = "step answerability judge disabled or missing endpoint"
            return record

        try:
            raw_answer, raw_payload = self._call_model(
                question=question,
                answer_instruction=answer_instruction,
                state_label=state_label,
                observation_text=observation_text,
                images=images,
            )
            final_answer = extract_answer(raw_answer) or self._clean_answer_text(raw_answer)
            result = compute_toolvision_score(
                data_source=data_source,
                solution_str=f"<answer>{final_answer}</answer>",
                ground_truth=ground_truth,
                extra_info=extra_info,
                extracted_answer=final_answer,
            )
            if result is None:
                score = 0.0
            else:
                score = float(result.get("score", 0.0) or 0.0)
            record.update(
                {
                    "score": score,
                    "raw_answer": raw_answer,
                    "final_answer": final_answer,
                    "usage": raw_payload.get("usage", {}) if isinstance(raw_payload, dict) else {},
                    "error": None,
                }
            )
        except Exception as exc:
            record["error"] = str(exc)
        finally:
            record["latency_s"] = round(time.perf_counter() - started, 3)
        return record

    def _call_model(
        self,
        *,
        question: str,
        answer_instruction: str | None,
        state_label: str,
        observation_text: str,
        images: list[Any],
    ) -> tuple[str, dict[str, Any]]:
        api_key = os.environ.get(self.config.api_key_env, os.environ.get("OPENAI_API_KEY", "EMPTY"))
        api_root = self.config.base_url.rstrip("/")
        if api_root.endswith("/chat/completions"):
            url = api_root
        elif api_root.endswith("/v1"):
            url = f"{api_root}/chat/completions"
        else:
            url = f"{api_root}/v1/chat/completions"
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": self._build_messages(
                question=question,
                answer_instruction=answer_instruction,
                state_label=state_label,
                observation_text=observation_text,
                images=images,
            ),
            "temperature": 0.0,
            "max_tokens": 256,
        }
        body.update(self.config.request_body or {})

        last_error: Exception | None = None
        for _attempt in range(max(0, self.config.max_retries) + 1):
            try:
                response = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=body,
                    timeout=float(self.config.timeout_s),
                )
                response.raise_for_status()
                payload = response.json()
                choice = (payload.get("choices") or [{}])[0]
                message = choice.get("message") if isinstance(choice, dict) else {}
                text = message.get("content", "") if isinstance(message, dict) else ""
                return str(text or ""), payload
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"step answerability judge request failed: {last_error}") from last_error

    def _build_messages(
        self,
        *,
        question: str,
        answer_instruction: str | None,
        state_label: str,
        observation_text: str,
        images: list[Any],
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Original question:\n"
                    f"{question}\n\n"
                    f"Answer instruction: {answer_instruction or 'Answer concisely.'}\n"
                    f"State: {state_label}\n\n"
                    "Use the visible images and tool observations from this state only. "
                    "Return exactly one final answer inside <answer>...</answer>.\n\n"
                    f"Tool observations:\n{observation_text[: self.config.max_observation_chars]}"
                ),
            }
        ]
        for image in self._select_images(images):
            encoded = self._encode_image(image)
            if encoded:
                content.append({"type": "image_url", "image_url": {"url": encoded}})
        return [
            {
                "role": "system",
                "content": (
                    "You are a strict visual question answering judge. Solve the user question "
                    "from the provided state and output only <answer>...</answer>."
                ),
            },
            {"role": "user", "content": content},
        ]

    def _select_images(self, images: list[Any]) -> list[Any]:
        if not images:
            return []
        max_images = max(1, int(self.config.max_images))
        if len(images) <= max_images:
            return list(images)
        return [images[0], *images[-(max_images - 1) :]]

    def _encode_image(self, image: Any) -> str:
        try:
            if not hasattr(image, "save"):
                return ""
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            data = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/png;base64,{data}"
        except Exception:
            return ""

    def _clean_answer_text(self, text: str) -> str:
        normalized = str(text or "").strip().strip("`").strip()
        match = re.search(r"<answer>(.*?)</answer>", normalized, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return normalized


__all__ = [
    "STEP_REWARD_VERSION",
    "StepAnswerabilityConfig",
    "StepAnswerabilityJudgeClient",
    "compute_step_answerability_delta",
    "coerce_json_list",
]
