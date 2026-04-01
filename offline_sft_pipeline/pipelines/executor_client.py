from __future__ import annotations

import json
from pathlib import Path
from string import Template
from typing import Any

from offline_sft_pipeline.core.models import ExecutorStepOutput
from offline_sft_pipeline.pipelines.backends import TextGenerationBackend
from offline_sft_pipeline.pipelines.parsing import ensure_tag_order, extract_required_tag
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
            step_spec_json=self._to_pretty_json(request.step_spec.model_dump(mode="json")),
            messages_json=self._to_pretty_json([item.model_dump(mode="json") for item in request.messages]),
            visible_images_json=self._to_pretty_json([item.model_dump(mode="json") for item in request.visible_images]),
            tool_capabilities_json=self._to_pretty_json(
                [item.model_dump(mode="json") for item in request.tool_capabilities]
            ),
        )

    def _parse_model_text(self, text: str, *, metadata: dict[str, Any]) -> ExecutorStepOutput:
        ensure_tag_order(text, stage="executor", first_tag="think", second_tag="code")
        think = extract_required_tag(text, "think", stage="executor")
        code = extract_required_tag(text, "code", stage="executor")
        return ExecutorStepOutput(
            cot=think,
            code=code,
            raw_response_text=text,
            metadata=dict(metadata),
        )

    def _to_pretty_json(self, payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = [
    "ExecutorClient",
]
