"""HTTP backend for easy-question planner: uses `planner_to_openai_messages_easy` and optional I/O dumps."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from offline_sft_pipeline.pipelines.api_text_multimodal import (
    chat_completions_text,
    coerce_planner_request,
    env_planner_debug_enabled,
    env_qwen_config,
    is_placeholder_api_key,
    sanitize_messages_for_debug,
    summarize_openai_message_for_debug,
)
from offline_sft_pipeline.pipelines.backends import ApiTextBackendConfig, BackendResponse, _normalize_token_usage
from offline_sft_pipeline.pipelines.easy_question_pipeline.messages import planner_to_openai_messages_easy


EASY_SAVE_FULL_BASE64_ENV = "OFFLINE_SFT_EASY_SAVE_FULL_BASE64"

DEFAULT_FAKE_EASY_PLANNER_TEXT = """{
  "mode": "answer",
  "think": "From the image and question, the count matches the visible items.",
  "answer": "0"
}
"""


def env_easy_save_full_base64() -> bool:
    return os.environ.get(EASY_SAVE_FULL_BASE64_ENV, "").strip().lower() in {"1", "true", "yes"}


class EasyApiTextPlannerBackend:
    """Planner-only backend: builds messages via `planner_to_openai_messages_easy` and POSTs chat/completions."""

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
        if stage != "planner":
            raise NotImplementedError("EasyApiTextPlannerBackend supports only stage='planner'.")
        if context is None:
            raise ValueError("EasyApiTextPlannerBackend requires context with request and reference_answer.")

        reference_answer = context.get("reference_answer")
        if reference_answer is None or not str(reference_answer).strip():
            raise ValueError("context['reference_answer'] must be a non-empty string.")

        artifact_dir = context.get("easy_artifact_dir")
        artifact_path: Path | None = Path(str(artifact_dir)).resolve() if artifact_dir else None

        req = coerce_planner_request(context.get("request"))
        if req is None:
            raise ValueError("EasyApiTextPlannerBackend requires context['request'] as PlannerClientRequest.")

        messages, missing_ids = planner_to_openai_messages_easy(
            system_prompt=system_prompt,
            req=req,
            reference_answer=str(reference_answer).strip(),
        )

        file_paths: dict[str, str] = {}
        if artifact_path is not None:
            artifact_path.mkdir(parents=True, exist_ok=True)
            save_full = env_easy_save_full_base64()
            payload_for_disk = messages if save_full else sanitize_messages_for_debug(messages)
            msg_path = artifact_path / (
                "planner_request_messages.full.json" if save_full else "planner_request_messages.json"
            )
            msg_path.write_text(json.dumps(payload_for_disk, ensure_ascii=False, indent=2), encoding="utf-8")
            file_paths["planner_request_messages"] = str(msg_path)

        if self._cfg.dry_run:
            fake = DEFAULT_FAKE_EASY_PLANNER_TEXT
            if artifact_path:
                raw_path = artifact_path / "planner_openai_raw_response.json"
                raw_path.write_text(
                    json.dumps({"dry_run": True, "note": "OFFLINE_SFT_API_DRY_RUN enabled"}, indent=2),
                    encoding="utf-8",
                )
                txt_path = artifact_path / "planner_assistant_text.txt"
                txt_path.write_text(fake, encoding="utf-8")
                file_paths["planner_openai_raw_response"] = str(raw_path)
                file_paths["planner_assistant_text"] = str(txt_path)
            return BackendResponse(
                text=fake,
                raw_payload={"dry_run": True},
                metadata={
                    "backend": "easy_api_text_planner",
                    "dry_run": True,
                    "stage": "planner",
                    "missing_artifact_ids": missing_ids,
                    "easy_artifact_files": file_paths,
                },
            )

        if is_placeholder_api_key(self._cfg.api_key):
            raise RuntimeError(
                "OFFLINE_SFT_QWEN_API_KEY is missing or placeholder. Set a real key or set OFFLINE_SFT_API_DRY_RUN=1."
            )

        debug_enabled = env_planner_debug_enabled()
        if debug_enabled:
            safe_messages = sanitize_messages_for_debug(messages)
            print(
                "[OFFLINE_SFT_PLANNER_DEBUG][easy] OpenAI-style messages (base64 shortened). "
                "Payload uses planner_to_openai_messages_easy.",
                file=sys.stderr,
            )
            print(json.dumps(safe_messages, ensure_ascii=False, indent=2), file=sys.stderr)
            if missing_ids:
                print(f"[OFFLINE_SFT_PLANNER_DEBUG][easy] missing_artifact_ids: {missing_ids}", file=sys.stderr)

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
                    "[OFFLINE_SFT_PLANNER_DEBUG][easy] choices[0].message (content summarized):",
                    file=sys.stderr,
                )
                print(
                    json.dumps(summarize_openai_message_for_debug(message), ensure_ascii=False, indent=2),
                    file=sys.stderr,
                )
            print("[OFFLINE_SFT_PLANNER_DEBUG][easy] Assistant message content (full text):", file=sys.stderr)
            print(text, file=sys.stderr)

        if artifact_path is not None:
            raw_path = artifact_path / "planner_openai_raw_response.json"
            raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            txt_path = artifact_path / "planner_assistant_text.txt"
            txt_path.write_text(text, encoding="utf-8")
            file_paths["planner_openai_raw_response"] = str(raw_path)
            file_paths["planner_assistant_text"] = str(txt_path)

        return BackendResponse(
            text=text,
            raw_payload=raw_payload,
            metadata={
                "backend": "easy_api_text_planner",
                "stage": "planner",
                "model": self._cfg.model,
                "missing_artifact_ids": missing_ids,
                "token_usage": _normalize_token_usage(raw_payload),
                "easy_artifact_files": file_paths,
            },
        )


__all__ = [
    "DEFAULT_FAKE_EASY_PLANNER_TEXT",
    "EASY_SAVE_FULL_BASE64_ENV",
    "EasyApiTextPlannerBackend",
    "env_easy_save_full_base64",
]
