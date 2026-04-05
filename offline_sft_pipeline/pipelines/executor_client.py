from __future__ import annotations

import json
from pathlib import Path
from string import Template
from typing import Any

from offline_sft_pipeline.core.models import ExecutorStepOutput
from offline_sft_pipeline.pipelines.backends import TextGenerationBackend
from offline_sft_pipeline.pipelines.parsing import (
    ModelResponseParseError,
    try_parse_executor_json_payload,
)
from offline_sft_pipeline.pipelines.request_models import ExecutorClientRequest

PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"


class ExecutorClient:
    def __init__(
        self,
        backend: TextGenerationBackend,
        *,
        prompt_root: str | Path | None = None,
        system_prompt_filename: str = "executor_system_v01.txt",
        user_prompt_filename: str = "executor_user_v01.txt",
    ) -> None:
        self.backend = backend
        self.prompt_root = Path(prompt_root) if prompt_root else PROMPT_ROOT
        self.system_prompt_path = self.prompt_root / system_prompt_filename
        self.user_prompt_path = self.prompt_root / user_prompt_filename

    def run(self, request: ExecutorClientRequest | dict[str, Any]) -> ExecutorStepOutput:
        req = self._coerce_request(request)
        system_prompt = self._load_prompt(self.system_prompt_path)
        user_prompt = self._build_user_prompt(req)
        backend_response = self.backend.generate(
            stage="executor",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=self._build_backend_context(req),
        )
        executor_output = self._parse_model_text(backend_response.text, metadata=backend_response.metadata)
        executor_output.validate_against_schema()
        return executor_output

    def _coerce_request(self, request: ExecutorClientRequest | dict[str, Any]) -> ExecutorClientRequest:
        if isinstance(request, ExecutorClientRequest):
            return request
        return ExecutorClientRequest.model_validate(request)

    def _load_prompt(self, path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"executor prompt template not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    def _build_user_prompt(self, request: ExecutorClientRequest) -> str:
        template = Template(self._load_prompt(self.user_prompt_path))
        return template.safe_substitute(
            sample_id=request.sample_id,
            trajectory_id=request.trajectory_id,
            round_idx=str(request.round_idx),
            step_idx=str(request.step_idx),
            question=request.question,
            suggestion_id=request.suggestion_id,
            suggestion_step_index=str(request.suggestion_step_index),
            planner_global_chain_cot=request.planner_global_chain_cot or "",
            suggestion_cot=request.suggestion_cot or "",
            input_image=request.step_spec.input_image,
            selectable_input_images_json=self._to_pretty_json(self._build_selectable_input_images(request)),
            step_spec_json=self._to_pretty_json(request.step_spec.model_dump(mode="json")),
            messages_json=self._to_pretty_json([item.model_dump(mode="json") for item in request.messages]),
            visible_images_json=self._to_pretty_json([item.model_dump(mode="json") for item in request.visible_images]),
            tool_capabilities_json=self._to_pretty_json(
                [item.model_dump(mode="json") for item in request.tool_capabilities]
            ),
        )

    def _build_backend_context(self, request: ExecutorClientRequest) -> dict[str, Any]:
        return {
            "request": request.model_copy(deep=True),
            "paths": {
                "sample_dir": request.sample_dir,
                "trajectory_dir": request.trajectory_dir,
                "planner_dir": request.planner_dir,
                "steps_dir": request.steps_dir,
            },
        }

    def _parse_model_text(self, text: str, *, metadata: dict[str, Any]) -> ExecutorStepOutput:
        payload = try_parse_executor_json_payload(text)
        if payload is None:
            raise ModelResponseParseError(
                "Executor output must be a JSON object with think + tool_call.",
                stage="executor",
                preview=self._preview_executor(text),
            )

        think = str(payload.get("think", "")).strip()
        if not think:
            raise ModelResponseParseError(
                "Executor JSON output field 'think' must not be empty.",
                stage="executor",
                tag="think",
                preview=self._preview_executor(text),
            )

        tool_call = payload.get("tool_call")
        if not isinstance(tool_call, dict):
            raise ModelResponseParseError(
                "Executor JSON output field 'tool_call' must be an object.",
                stage="executor",
                tag="tool_call",
                preview=self._preview_executor(text),
            )
        if str(tool_call.get("name", "")).strip() != "code_image_tool":
            raise ModelResponseParseError(
                "Executor JSON tool_call.name must be 'code_image_tool'.",
                stage="executor",
                tag="tool_call.name",
                preview=self._preview_executor(text),
            )
        arguments = tool_call.get("arguments")
        if not isinstance(arguments, dict):
            raise ModelResponseParseError(
                "Executor JSON tool_call.arguments must be an object.",
                stage="executor",
                tag="tool_call.arguments",
                preview=self._preview_executor(text),
            )
        code = str(arguments.get("code", "")).strip()
        if not code:
            raise ModelResponseParseError(
                "Executor JSON tool_call.arguments.code must not be empty.",
                stage="executor",
                tag="tool_call.arguments.code",
                preview=self._preview_executor(text),
            )
        description = str(arguments.get("description", "")).strip()
        if not description:
            raise ModelResponseParseError(
                "Executor JSON tool_call.arguments.description must not be empty.",
                stage="executor",
                tag="tool_call.arguments.description",
                preview=self._preview_executor(text),
            )
        return ExecutorStepOutput(
            cot=think,
            code=code,
            description=description,
            raw_response_text=text,
            metadata=dict(metadata),
        )

    def _to_pretty_json(self, payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _build_selectable_input_images(self, request: ExecutorClientRequest) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        total = len(request.visible_images)
        for idx, image in enumerate(request.visible_images):
            role = "current" if total >= 2 and idx == total - 1 else "root"
            label = "latest previous-step image" if role == "current" else "original image"
            out.append(
                {
                    "index": idx,
                    "role": role,
                    "label": label,
                }
            )
        return out

    def _preview_executor(self, text: str, *, limit: int = 500) -> str:
        normalized = " ".join(text.strip().split())
        if len(normalized) <= limit:
            return normalized
        return normalized[:limit] + "..."


__all__ = [
    "ExecutorClient",
]
