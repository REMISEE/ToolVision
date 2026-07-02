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


STEP_REWARD_VERSION = "step_answerability_context_delta_v2"


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
    num_judgments: int = 1
    aggregation: str = "mean"
    prompt_mode: str = "context"
    max_images: int = 8
    max_observation_chars: int = 12000
    max_context_chars: int = 60000
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
            num_judgments=max(1, as_int(get("num_judgments", os.getenv("STEP_JUDGE_NUM_JUDGMENTS", "1")), 1)),
            aggregation=str(get("aggregation", os.getenv("STEP_JUDGE_AGGREGATION", "mean")) or "mean").strip().lower(),
            prompt_mode=str(get("prompt_mode", os.getenv("STEP_JUDGE_PROMPT_MODE", "context")) or "context")
            .strip()
            .lower(),
            max_images=as_int(get("max_images", os.getenv("STEP_JUDGE_MAX_IMAGES", "8")), 8),
            max_observation_chars=as_int(
                get("max_observation_chars", os.getenv("STEP_JUDGE_MAX_OBSERVATION_CHARS", "12000")),
                12000,
            ),
            max_context_chars=as_int(
                get("max_context_chars", os.getenv("STEP_JUDGE_MAX_CONTEXT_CHARS", "60000")),
                60000,
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
        context_messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
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
            "prompt_mode": self.config.prompt_mode,
        }
        if not self.enabled:
            record["error"] = "step answerability judge disabled or missing endpoint"
            return record

        try:
            judgments: list[dict[str, Any]] = []
            for judgment_idx in range(max(1, int(self.config.num_judgments))):
                judgment_record: dict[str, Any] = {
                    "judgment_idx": judgment_idx,
                    "score": None,
                    "raw_answer": "",
                    "final_answer": "",
                    "usage": {},
                    "error": None,
                }
                try:
                    raw_answer, raw_payload = self._call_model(
                        question=question,
                        answer_instruction=answer_instruction,
                        state_label=state_label,
                        observation_text=observation_text,
                        images=images,
                        context_messages=context_messages,
                        tools=tools,
                    )
                    committee_judgments = self._score_committee_payload(
                        raw_answer=raw_answer,
                        raw_payload=raw_payload,
                        data_source=data_source,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    )
                    committee_scores = [
                        float(item["score"])
                        for item in committee_judgments
                        if item.get("score") is not None
                    ]
                    if committee_judgments and committee_scores:
                        first_valid_committee = next(
                            (item for item in committee_judgments if item.get("score") is not None),
                            committee_judgments[0],
                        )
                        judgment_record.update(
                            {
                                "score": self._aggregate_scores(committee_scores),
                                "raw_answer": raw_answer,
                                "final_answer": first_valid_committee.get("final_answer", ""),
                                "usage": raw_payload.get("usage", {}) if isinstance(raw_payload, dict) else {},
                                "committee_judgments": committee_judgments,
                                "committee_success_count": len(committee_scores),
                                "committee_total_count": len(
                                    raw_payload.get("committee_judgments", [])
                                    if isinstance(raw_payload, dict)
                                    else []
                                ),
                                "error": None,
                            }
                        )
                    elif committee_judgments:
                        judgment_record.update(
                            {
                                "committee_judgments": committee_judgments,
                                "committee_success_count": 0,
                                "committee_total_count": len(
                                    raw_payload.get("committee_judgments", [])
                                    if isinstance(raw_payload, dict)
                                    else []
                                ),
                                "error": "all committee judge calls failed",
                            }
                        )
                    else:
                        final_answer, score = self._score_answer(
                            raw_answer=raw_answer,
                            data_source=data_source,
                            ground_truth=ground_truth,
                            extra_info=extra_info,
                        )
                        judgment_record.update(
                            {
                                "score": score,
                                "raw_answer": raw_answer,
                                "final_answer": final_answer,
                                "usage": raw_payload.get("usage", {}) if isinstance(raw_payload, dict) else {},
                                "error": None,
                            }
                        )
                except Exception as exc:
                    judgment_record["error"] = str(exc)
                judgments.append(judgment_record)

            valid_judgments = [item for item in judgments if item.get("score") is not None]
            if valid_judgments:
                scores = [float(item["score"]) for item in valid_judgments]
                score = self._aggregate_scores(scores)
                first_valid = valid_judgments[0]
                record.update(
                    {
                        "score": score,
                        "raw_answer": first_valid.get("raw_answer", ""),
                        "final_answer": first_valid.get("final_answer", ""),
                        "usage": first_valid.get("usage", {}),
                        "judgments": judgments,
                        "judgment_count": len(valid_judgments),
                        "score_std": self._score_std(scores),
                        "error": None,
                    }
                )
            else:
                record["judgments"] = judgments
                record["judgment_count"] = 0
                record["score_std"] = 0.0
                errors = [str(item.get("error")) for item in judgments if item.get("error")]
                record["error"] = "; ".join(errors) or "all step answerability judgments failed"
        finally:
            record["latency_s"] = round(time.perf_counter() - started, 3)
        return record

    def _aggregate_scores(self, scores: list[float]) -> float:
        if not scores:
            return 0.0
        mode = self.config.aggregation
        if mode == "max":
            return max(scores)
        if mode == "min":
            return min(scores)
        return sum(scores) / len(scores)

    def _score_std(self, scores: list[float]) -> float:
        if len(scores) <= 1:
            return 0.0
        mean = sum(scores) / len(scores)
        return (sum((score - mean) ** 2 for score in scores) / len(scores)) ** 0.5

    def _score_answer(
        self,
        *,
        raw_answer: str,
        data_source: str,
        ground_truth: Any,
        extra_info: dict[str, Any],
    ) -> tuple[str, float]:
        final_answer = extract_answer(raw_answer) or self._clean_answer_text(raw_answer)
        result = compute_toolvision_score(
            data_source=data_source,
            solution_str=f"<answer>{final_answer}</answer>",
            ground_truth=ground_truth,
            extra_info=extra_info,
            extracted_answer=final_answer,
        )
        if result is None:
            return final_answer, 0.0
        return final_answer, float(result.get("score", 0.0) or 0.0)

    def _score_committee_payload(
        self,
        *,
        raw_answer: str,
        raw_payload: dict[str, Any],
        data_source: str,
        ground_truth: Any,
        extra_info: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not isinstance(raw_payload, dict):
            return []
        raw_items = raw_payload.get("committee_judgments")
        if not isinstance(raw_items, list):
            return []

        scored_items: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            scored_item = dict(item)
            scored_item.setdefault("committee_idx", idx)
            raw_member_answer = str(
                item.get("raw_answer")
                or item.get("content")
                or item.get("answer")
                or item.get("message")
                or ""
            )
            if item.get("error") or not raw_member_answer.strip():
                scored_item.setdefault("score", None)
                scored_item.setdefault("final_answer", "")
                scored_items.append(scored_item)
                continue
            final_answer, score = self._score_answer(
                raw_answer=raw_member_answer,
                data_source=data_source,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            scored_item.update(
                {
                    "score": score,
                    "raw_answer": raw_member_answer,
                    "final_answer": final_answer,
                }
            )
            scored_items.append(scored_item)

        valid_items = [item for item in scored_items if item.get("score") is not None]
        if valid_items:
            return scored_items

        # Fall back to the top-level answer if a gateway returned committee metadata
        # without usable per-member text.
        if raw_answer.strip():
            final_answer, score = self._score_answer(
                raw_answer=raw_answer,
                data_source=data_source,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            return [{"committee_idx": 0, "score": score, "raw_answer": raw_answer, "final_answer": final_answer}]
        return scored_items

    def _call_model(
        self,
        *,
        question: str,
        answer_instruction: str | None,
        state_label: str,
        observation_text: str,
        images: list[Any],
        context_messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
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
                context_messages=context_messages,
                tools=tools,
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
        context_messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if self.config.prompt_mode in {"context", "trajectory"} and context_messages:
            return self._build_context_messages(
                context_messages=context_messages,
                tools=tools,
                state_label=state_label,
                answer_instruction=answer_instruction,
                images=images,
            )
        return self._build_snapshot_messages(
            question=question,
            answer_instruction=answer_instruction,
            state_label=state_label,
            observation_text=observation_text,
            images=images,
        )

    def _build_snapshot_messages(
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

    def _build_context_messages(
        self,
        *,
        context_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        state_label: str,
        answer_instruction: str | None,
        images: list[Any],
    ) -> list[dict[str, Any]]:
        image_state = self._context_image_state(images)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a strict visual question answering judge. You are given the original "
                    "rollout context up to the current step. Continue from that context, do not call "
                    "any tools, and output only one final answer inside <answer>...</answer>."
                ),
            }
        ]
        if tools:
            tool_text = json.dumps(tools, ensure_ascii=False)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The following tool schemas were available to the original rollout model. "
                        "They are provided only so you can interpret prior tool calls; you must not "
                        f"call tools now.\n{tool_text[: self.config.max_observation_chars]}"
                    ),
                }
            )

        for raw_message in context_messages:
            if not isinstance(raw_message, dict):
                continue
            role = str(raw_message.get("role") or "user").strip().lower()
            if role not in {"system", "user", "assistant", "tool"}:
                role = "user"
            content = self._materialize_context_content(raw_message.get("content"), image_state)
            if role == "tool":
                content = self._prefix_tool_response(content)
                role = "user"
            messages.append({"role": role, "content": content})

        messages.append(
            {
                "role": "user",
                "content": (
                    f"State: {state_label}\n"
                    f"Answer instruction: {answer_instruction or 'Answer concisely.'}\n"
                    "Based only on the context above, answer the original question directly now. "
                    "Do not call tools. Return exactly one final answer inside <answer>...</answer>."
                ),
            }
        )
        return self._truncate_context_messages(messages)

    def _context_image_state(self, images: list[Any]) -> dict[str, Any]:
        image_list = list(images or [])
        max_images = max(1, int(self.config.max_images))
        if len(image_list) <= max_images:
            keep_indices = set(range(len(image_list)))
        else:
            keep_indices = {0, *range(len(image_list) - (max_images - 1), len(image_list))}
        return {"images": image_list, "keep_indices": keep_indices, "idx": 0}

    def _materialize_context_content(self, content: Any, image_state: dict[str, Any]) -> Any:
        if isinstance(content, str):
            if "<image>" not in content:
                return content
            converted: list[dict[str, Any]] = []
            parts = re.split(r"(<image>)", content)
            for part in parts:
                if not part:
                    continue
                if part == "<image>":
                    encoded = self._encode_next_context_image(image_state)
                    if encoded:
                        converted.append({"type": "image_url", "image_url": {"url": encoded}})
                    continue
                converted.append({"type": "text", "text": part})
            return converted
        if isinstance(content, list):
            converted: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    converted.append({"type": "text", "text": str(item.get("text", ""))})
                elif item_type == "image_url":
                    converted.append(item)
                elif item_type == "image":
                    encoded = self._encode_next_context_image(image_state)
                    if encoded:
                        converted.append({"type": "image_url", "image_url": {"url": encoded}})
                elif item_type:
                    converted.append({"type": "text", "text": f"[Unsupported content type: {item_type}]"})
            return converted
        return str(content or "")

    def _prefix_tool_response(self, content: Any) -> Any:
        prefix = {"type": "text", "text": "Tool response from the current rollout context:"}
        if isinstance(content, list):
            return [prefix, *content]
        text = str(content or "")
        return f"Tool response from the current rollout context:\n{text}"

    def _encode_next_context_image(self, image_state: dict[str, Any]) -> str:
        try:
            idx = int(image_state.get("idx", 0))
            image_state["idx"] = idx + 1
            images = image_state.get("images") or []
            keep_indices = image_state.get("keep_indices") or set()
            if idx not in keep_indices or idx >= len(images):
                return ""
            return self._encode_image(images[idx])
        except Exception:
            return ""

    def _truncate_context_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        budget = max(1000, int(self.config.max_context_chars))
        if self._messages_text_chars(messages) <= budget:
            return messages

        # Keep the judge system instruction and the final direct-answer request,
        # then fill remaining budget with the original prompt and latest context.
        head = messages[:1]
        tail = messages[-1:]
        middle = messages[1:-1]
        remaining = max(0, budget - self._messages_text_chars(head) - self._messages_text_chars(tail))
        kept: list[dict[str, Any]] = []
        for message in reversed(middle):
            msg_chars = self._messages_text_chars([message])
            if msg_chars <= remaining:
                kept.append(message)
                remaining -= msg_chars
                continue
            if remaining > 1000:
                kept.append(self._clip_message_text(message, remaining, keep_suffix=True))
                remaining = 0
            break
        kept.reverse()
        return head + [{"role": "system", "content": "[Earlier rollout context was truncated for judge input length.]"}] + kept + tail

    def _messages_text_chars(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total += len(str(item.get("text", "")))
        return total

    def _clip_message_text(self, message: dict[str, Any], limit: int, *, keep_suffix: bool) -> dict[str, Any]:
        clipped = dict(message)
        content = clipped.get("content")
        if isinstance(content, str):
            clipped["content"] = self._clip_text(content, limit, keep_suffix=keep_suffix)
        elif isinstance(content, list):
            out = []
            remaining = limit
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "text":
                    out.append(item)
                    continue
                text = str(item.get("text", ""))
                clipped_text = self._clip_text(text, remaining, keep_suffix=keep_suffix)
                out.append({"type": "text", "text": clipped_text})
                remaining = max(0, remaining - len(clipped_text))
            clipped["content"] = out
        return clipped

    def _clip_text(self, text: str, limit: int, *, keep_suffix: bool) -> str:
        if len(text) <= limit:
            return text
        marker = "\n[...truncated...]\n"
        keep = max(0, limit - len(marker))
        if keep_suffix:
            return marker + text[-keep:]
        return text[:keep] + marker

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
            width, height = getattr(image, "size", (0, 0))
            if width and height and (width <= 10 or height <= 10) and hasattr(image, "resize"):
                scale = max(32 / max(1, width), 32 / max(1, height))
                image = image.resize((max(11, int(round(width * scale))), max(11, int(round(height * scale)))))
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
