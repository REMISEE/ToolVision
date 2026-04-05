"""OpenAI-compatible multimodal chat for DashScope (Qwen) planner calls.

POST {base_url}/chat/completions with Authorization: Bearer.

Environment (do not hardcode secrets):
- OFFLINE_SFT_QWEN_API_KEY
- OFFLINE_SFT_QWEN_BASE_URL (default: https://dashscope.aliyuncs.com/compatible-mode/v1)
- OFFLINE_SFT_QWEN_MODEL
- OFFLINE_SFT_QWEN_TIMEOUT_S (seconds, default 120)
- OFFLINE_SFT_API_DRY_RUN (1/true/yes: skip HTTP, return local fake planner text from caller)
- OFFLINE_SFT_PLANNER_DEBUG (1/true/yes: print to stderr the JSON messages sent to the API with base64
  image URLs shortened; choices[0].message with content replaced by length (to spot reasoning/extra fields);
  then the full assistant text; dry_run path does not call HTTP so no debug dump)
- OFFLINE_SFT_PLANNER_USE_TOOL_ROLE (1/true/yes: send internal tool-result messages as native role="tool"
  instead of wrapping them as user-visible environment observations)

Reference: https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-by-calling-api
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from offline_sft_pipeline.core.models import ConversationMessage, ImageArtifactRef
from .request_models import PlannerClientRequest, ToolCapability

DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen3.5-122b-a10b"
DEFAULT_QWEN_TIMEOUT_S = 120.0

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def env_qwen_config() -> dict[str, Any]:
    return {
        "api_key": os.environ.get("OFFLINE_SFT_QWEN_API_KEY"),
        "base_url": os.environ.get("OFFLINE_SFT_QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL).rstrip("/"),
        "model": os.environ.get("OFFLINE_SFT_QWEN_MODEL", DEFAULT_QWEN_MODEL),
        "timeout_s": float(os.environ.get("OFFLINE_SFT_QWEN_TIMEOUT_S", str(DEFAULT_QWEN_TIMEOUT_S))),
        "dry_run": os.environ.get("OFFLINE_SFT_API_DRY_RUN", "").strip().lower() in {"1", "true", "yes"},
    }


def env_planner_debug_enabled() -> bool:
    return os.environ.get("OFFLINE_SFT_PLANNER_DEBUG", "").strip().lower() in {"1", "true", "yes"}


def env_planner_use_tool_role() -> bool:
    return os.environ.get("OFFLINE_SFT_PLANNER_USE_TOOL_ROLE", "").strip().lower() in {"1", "true", "yes"}


def sanitize_messages_for_debug(payload: Any) -> Any:
    """Recursively shorten data:...;base64,... strings so debug logs stay readable."""

    def _shorten_data_url(s: str) -> str:
        if "data:" in s and "base64," in s and len(s) > 160:
            return s[:80] + f"...[base64 omitted, total_len={len(s)}]"
        return s

    if isinstance(payload, str):
        return _shorten_data_url(payload)
    if isinstance(payload, dict):
        return {k: sanitize_messages_for_debug(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [sanitize_messages_for_debug(v) for v in payload]
    return payload


def is_placeholder_api_key(api_key: str | None) -> bool:
    if api_key is None:
        return True
    k = api_key.strip().lower()
    return k in {"", "dummykey", "dummy", "placeholder"}


def _guess_media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    ext = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "application/octet-stream")


def file_to_data_url(path: Path, *, media_type: str | None = None) -> str:
    raw = path.read_bytes()
    mt = media_type or _guess_media_type(path)
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:{mt};base64,{b64}"


def build_artifact_path_index(
    *,
    sample_dir: str | None,
    trajectory_dir: str | None,
    visible_images: list[ImageArtifactRef],
) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for ref in visible_images:
        if not ref.artifact_id:
            continue
        p = Path(ref.path)
        if p.is_file():
            index[ref.artifact_id] = p.resolve()

    if sample_dir:
        art = Path(sample_dir) / "artifacts"
        if art.is_dir():
            for f in art.iterdir():
                if not f.is_file() or f.suffix.lower() not in _IMAGE_SUFFIXES:
                    continue
                index.setdefault(f.stem, f.resolve())

    if trajectory_dir:
        steps_root = Path(trajectory_dir) / "steps"
        if steps_root.is_dir():
            for step_dir in sorted(steps_root.iterdir()):
                if not step_dir.is_dir() or not step_dir.name.startswith("step_"):
                    continue
                rr_path = step_dir / "runtime_result.json"
                if not rr_path.is_file():
                    continue
                try:
                    data = json.loads(rr_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                for img in data.get("images") or []:
                    if not isinstance(img, dict):
                        continue
                    aid = img.get("artifact_id")
                    raw_path = img.get("path")
                    if not aid or not raw_path:
                        continue
                    p = Path(str(raw_path))
                    if not p.is_file():
                        p = step_dir / p.name
                    if p.is_file():
                        index[str(aid)] = p.resolve()
    return index


def _finalize_openai_content(parts: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    if len(parts) == 1 and parts[0].get("type") == "text":
        return str(parts[0].get("text", ""))
    return parts


def _openai_content_parts_for_message(
    message: ConversationMessage,
    index: dict[str, Path],
    missing: list[str],
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for aid in message.image_artifact_ids:
        p = index.get(aid)
        if p is None or not p.is_file():
            missing.append(aid)
            parts.append({"type": "text", "text": f"[missing image artifact_id={aid!r}]"})
            continue
        parts.append({"type": "image_url", "image_url": {"url": file_to_data_url(p)}})
    text = message.content or ""
    if text.strip() or not parts:
        parts.append({"type": "text", "text": text})
    return parts


def _format_capabilities(caps: list[ToolCapability]) -> str:
    if not caps:
        return (
            "(none configured — use only capability names that your runtime exposes; "
            "load example/tool_capabilities.json for production runs.)"
        )
    lines: list[str] = []
    for c in caps:
        line = f"- {c.name}: {c.description}"
        if c.usage_notes:
            line += f" Notes: {c.usage_notes}"
        lines.append(line)
    return "\n".join(lines)


def build_planner_control_user_text(req: PlannerClientRequest) -> str:
    b = req.budget
    k = req.requested_suggestion_count

    suggestion_block = (
        "Planner constraints for this round:\n"
        "- You must choose exactly one mode: answer mode OR suggestions mode.\n"
        "- If the question is already answerable from visible evidence, return answer mode.\n"
        "- Otherwise, return suggestions mode.\n"
        f"- In suggestions mode, the top-level `suggestions` array must contain exactly {k} branch objects.\n"
        "- Each suggestion must be a complete alternative strategy branch.\n"
    )

    budget_block = (
        "Depth limit for the current trajectory:\n"
        f"- remaining_rounds = {b.remaining_rounds}: you have at most {b.remaining_rounds} more planner rounds left on this trajectory, so keep plans short.\n"
    )

    guidance_block = (
        "Planning guidance:\n"
        "- A suggestion is a full strategy branch, not just a single tool call.\n"
        "- A suggestion may contain multiple ordered steps when appropriate.\n"
        "- Diversification is at the full-path strategy level, not at the single-tool level.\n"
        "- When remaining_rounds is small, prefer short, high-value strategies.\n"
    )

    return (
        "The conversation above is the executed trajectory so far "
        "(user question, prior assistant actions, and tool outputs).\n\n"
        f"{suggestion_block}\n"
        f"{budget_block}\n"
        f"{guidance_block}\n"
        "Available capabilities (use only these names in capability_plan):\n"
        f"{_format_capabilities(req.tool_capabilities)}"
    )


def planner_to_openai_messages(
    *,
    system_prompt: str,
    req: PlannerClientRequest,
) -> tuple[list[dict[str, Any]], list[str]]:
    missing: list[str] = []
    index = build_artifact_path_index(
        sample_dir=req.sample_dir,
        trajectory_dir=req.trajectory_dir,
        visible_images=req.visible_images,
    )
    out: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    for msg in req.messages:
        if msg.role == "system":
            continue
        if msg.role == "tool":
            parts = _openai_content_parts_for_message(msg, index, missing)
            if env_planner_use_tool_role():
                out.append({"role": "tool", "content": _finalize_openai_content(parts)})
                continue
            wrapped: list[dict[str, Any]] = [
                {"type": "text", "text": "[Tool output — environment observation, not user speech]\n"},
            ]
            wrapped.extend(parts)
            out.append({"role": "user", "content": wrapped})
            continue
        parts = _openai_content_parts_for_message(msg, index, missing)
        out.append({"role": msg.role, "content": _finalize_openai_content(parts)})

    out.append({"role": "user", "content": build_planner_control_user_text(req)})
    deduped_missing = list(dict.fromkeys(missing))
    return out, deduped_missing


def assistant_text_from_chat_response(raw: dict[str, Any]) -> str:
    choice0 = (raw.get("choices") or [{}])[0]
    msg = choice0.get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                chunks.append(str(part.get("text", "")))
        return "".join(chunks)
    return str(content or "")


def summarize_openai_message_for_debug(msg: dict[str, Any]) -> dict[str, Any]:
    """JSON-friendly view of `choices[].message`: omit full `content`, show length or part count only."""

    out: dict[str, Any] = {}
    for k, v in msg.items():
        if k == "content":
            if isinstance(v, str):
                out["content"] = f"<string, len={len(v)}>"
            elif isinstance(v, list):
                out["content"] = f"<list of {len(v)} parts>"
            else:
                out["content"] = repr(v)[:400]
        elif isinstance(v, str) and len(v) > 800:
            out[k] = v[:800] + f"... (truncated, len={len(v)})"
        else:
            out[k] = v
    return out


def chat_completions_text(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout_s: float = DEFAULT_QWEN_TIMEOUT_S,
) -> tuple[str, dict[str, Any]]:
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({"model": model, "messages": messages}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as resp:
            raw: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Chat API HTTP {exc.code}: {detail[:2000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Chat API connection error: {exc}") from exc

    return assistant_text_from_chat_response(raw), raw


def coerce_planner_request(obj: Any) -> PlannerClientRequest | None:
    if isinstance(obj, PlannerClientRequest):
        return obj
    if isinstance(obj, dict):
        try:
            return PlannerClientRequest.model_validate(obj)
        except Exception:
            return None
    return None


__all__ = [
    "DEFAULT_QWEN_BASE_URL",
    "DEFAULT_QWEN_MODEL",
    "assistant_text_from_chat_response",
    "build_artifact_path_index",
    "build_planner_control_user_text",
    "chat_completions_text",
    "coerce_planner_request",
    "env_planner_debug_enabled",
    "env_planner_use_tool_role",
    "env_qwen_config",
    "file_to_data_url",
    "is_placeholder_api_key",
    "planner_to_openai_messages",
    "sanitize_messages_for_debug",
    "summarize_openai_message_for_debug",
]
