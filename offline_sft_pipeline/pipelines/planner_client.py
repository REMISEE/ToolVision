from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any

from offline_sft_pipeline.core.models import PlannerOutput, PlannerSuggestion, utc_now
from offline_sft_pipeline.pipelines.backends import TextGenerationBackend
from offline_sft_pipeline.pipelines.parsing import (
    ModelResponseParseError,
    ensure_tag_order,
    extract_required_tag,
    extract_tag_block,
    parse_json_text,
)
from offline_sft_pipeline.pipelines.request_models import PlannerClientRequest

PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"


class PlannerClient:
    def __init__(
        self,
        backend: TextGenerationBackend,
        *,
        prompt_root: str | Path | None = None,
        system_prompt_filename: str = "planner_system_v01.txt",
        user_prompt_filename: str = "planner_user_v01.txt",
    ) -> None:
        self.backend = backend
        self.prompt_root = Path(prompt_root) if prompt_root else PROMPT_ROOT
        self.system_prompt_path = self.prompt_root / system_prompt_filename
        self.user_prompt_path = self.prompt_root / user_prompt_filename

    def run(self, request: PlannerClientRequest | dict[str, Any]) -> PlannerOutput:
        req = self._coerce_request(request)
        system_prompt = self._load_prompt(self.system_prompt_path)
        user_prompt = self._build_user_prompt(req)
        backend_response = self.backend.generate(
            stage="planner",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        parsed = self._parse_model_text(backend_response.text)
        planner_output = PlannerOutput(
            sample_id=req.sample_id,
            trajectory_id=req.trajectory_id,
            round_idx=req.round_idx,
            created_at=utc_now(),
            can_answer_now=parsed["can_answer_now"],
            global_chain_cot=parsed["global_chain_cot"],
            direct_answer=parsed["direct_answer"],
            stop_reason=parsed["stop_reason"],
            suggestions=parsed["suggestions"],
        )
        planner_output.validate_against_schema()
        return planner_output

    def _coerce_request(self, request: PlannerClientRequest | dict[str, Any]) -> PlannerClientRequest:
        if isinstance(request, PlannerClientRequest):
            return request
        return PlannerClientRequest.model_validate(request)

    def _load_prompt(self, path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"planner prompt template not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    def _build_user_prompt(self, request: PlannerClientRequest) -> str:
        template = Template(self._load_prompt(self.user_prompt_path))
        latest_runtime_result = None
        if request.latest_runtime_result is not None:
            latest_runtime_result = request.latest_runtime_result.model_dump(mode="json")
        return template.safe_substitute(
            sample_id=request.sample_id,
            trajectory_id=request.trajectory_id,
            round_idx=str(request.round_idx),
            requested_suggestion_count=(
                str(request.requested_suggestion_count)
                if request.requested_suggestion_count is not None
                else ""
            ),
            question=request.question,
            messages_json=self._to_pretty_json([item.model_dump(mode="json") for item in request.messages]),
            visible_images_json=self._to_pretty_json([item.model_dump(mode="json") for item in request.visible_images]),
            budget_json=self._to_pretty_json(request.budget.model_dump(mode="json")),
            tool_capabilities_json=self._to_pretty_json(
                [item.model_dump(mode="json") for item in request.tool_capabilities]
            ),
            latest_runtime_result_json=self._to_pretty_json(latest_runtime_result),
            now_iso=datetime.utcnow().isoformat() + "Z",
        )

    def _parse_model_text(self, text: str) -> dict[str, Any]:
        ensure_tag_order(text, stage="planner", first_tag="think", second_tag="answer")
        ensure_tag_order(text, stage="planner", first_tag="think", second_tag="suggestions")

        think = extract_required_tag(text, "think", stage="planner")
        answer = extract_tag_block(text, "answer")
        suggestions_text = extract_tag_block(text, "suggestions")

        if answer is not None and suggestions_text is not None:
            raise ModelResponseParseError(
                "Planner output must contain either <answer> or <suggestions>, not both.",
                stage="planner",
            )
        if answer is None and suggestions_text is None:
            raise ModelResponseParseError(
                "Planner output must contain <answer> or <suggestions> after <think>.",
                stage="planner",
            )

        if answer is not None:
            answer = answer.strip()
            if not answer:
                raise ModelResponseParseError(
                    "<answer> block must not be empty.",
                    stage="planner",
                    tag="answer",
                )
            return {
                "can_answer_now": True,
                "global_chain_cot": think,
                "direct_answer": answer,
                "stop_reason": None,
                "suggestions": [],
            }

        suggestions_payload = parse_json_text(suggestions_text or "", stage="planner", tag="suggestions")
        if not isinstance(suggestions_payload, list):
            raise ModelResponseParseError(
                "<suggestions> must contain a JSON array.",
                stage="planner",
                tag="suggestions",
            )
        suggestions = [PlannerSuggestion.model_validate(item) for item in suggestions_payload]
        return {
            "can_answer_now": False,
            "global_chain_cot": think,
            "direct_answer": None,
            "stop_reason": None,
            "suggestions": suggestions,
        }

    def _to_pretty_json(self, payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = [
    "PlannerClient",
]
