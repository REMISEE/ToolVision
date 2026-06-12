from __future__ import annotations

import json
import re
from typing import Any


class ModelResponseParseError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        stage: str,
        tag: str | None = None,
        preview: str | None = None,
    ) -> None:
        details = [message, f"stage={stage}"]
        if tag is not None:
            details.append(f"tag={tag}")
        if preview:
            details.append(f"preview={preview!r}")
        super().__init__(" | ".join(details))
        self.stage = stage
        self.tag = tag
        self.preview = preview


def _preview_text(text: str, *, limit: int = 500) -> str:
    normalized = " ".join(text.strip().split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "..."


def extract_tag_block(text: str, tag: str) -> str | None:
    pattern = re.compile(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", flags=re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple <{tag}> blocks found.")
    return matches[0].strip()


def extract_required_tag(text: str, tag: str, *, stage: str) -> str:
    value = extract_tag_block(text, tag)
    if value is None:
        raise ModelResponseParseError(
            f"Required <{tag}> block not found.",
            stage=stage,
            tag=tag,
            preview=_preview_text(text),
        )
    if not value.strip():
        raise ModelResponseParseError(
            f"Required <{tag}> block is empty.",
            stage=stage,
            tag=tag,
            preview=_preview_text(text),
        )
    return value


def ensure_tag_order(text: str, *, stage: str, first_tag: str, second_tag: str) -> None:
    first_idx = text.find(f"<{first_tag}>")
    second_idx = text.find(f"<{second_tag}>")
    if first_idx == -1 or second_idx == -1:
        return
    if first_idx > second_idx:
        raise ModelResponseParseError(
            f"<{first_tag}> must appear before <{second_tag}>.",
            stage=stage,
            preview=_preview_text(text),
        )


def parse_json_text(text: str, *, stage: str, tag: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelResponseParseError(
            f"Invalid JSON inside <{tag}> block: {exc.msg}.",
            stage=stage,
            tag=tag,
            preview=_preview_text(text),
        ) from exc


def strip_markdown_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped, count=1)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _repair_common_json_issues(text: str) -> str:
    repaired = str(text)
    # Some reasoning models emit apostrophes inside double-quoted JSON strings as \'
    # which is invalid JSON but semantically unambiguous.
    repaired = repaired.replace("\\'", "'")
    return repaired


def try_parse_json_object_from_text(text: str) -> dict[str, Any] | None:
    stripped = strip_markdown_json_fence(text)
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        repaired = _repair_common_json_issues(stripped)
        try:
            payload = json.loads(repaired)
        except json.JSONDecodeError:
            start = repaired.find("{")
            end = repaired.rfind("}")
            if start == -1 or end <= start:
                return None
            try:
                payload = json.loads(repaired[start : end + 1])
            except json.JSONDecodeError:
                return None
    if not isinstance(payload, dict):
        return None
    return payload


def looks_like_planner_json_payload(payload: dict[str, Any]) -> bool:
    return "mode" in payload and "think" in payload


def try_parse_planner_json_payload(text: str) -> dict[str, Any] | None:
    payload = try_parse_json_object_from_text(text)
    if payload is None or not looks_like_planner_json_payload(payload):
        return None
    return payload


def looks_like_executor_json_payload(payload: dict[str, Any]) -> bool:
    return "think" in payload and "tool_call" in payload


def try_parse_executor_json_payload(text: str) -> dict[str, Any] | None:
    payload = try_parse_json_object_from_text(text)
    if payload is None or not looks_like_executor_json_payload(payload):
        return None
    return payload


__all__ = [
    "ModelResponseParseError",
    "ensure_tag_order",
    "extract_required_tag",
    "extract_tag_block",
    "looks_like_executor_json_payload",
    "looks_like_planner_json_payload",
    "parse_json_text",
    "strip_markdown_json_fence",
    "try_parse_executor_json_payload",
    "try_parse_json_object_from_text",
    "try_parse_planner_json_payload",
]
