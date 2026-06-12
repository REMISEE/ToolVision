from __future__ import annotations

from pathlib import Path
from typing import Any

from offline_sft_pipeline.core.models import PlannerOutput, PlannerSuggestion, utc_now
from offline_sft_pipeline.pipelines.backends import TextGenerationBackend
from offline_sft_pipeline.pipelines.parsing import (
    ModelResponseParseError,
    ensure_tag_order,
    extract_required_tag,
    extract_tag_block,
    parse_json_text,
    try_parse_planner_json_payload,
)
from offline_sft_pipeline.pipelines.request_models import PlannerClientRequest

PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"


class PlannerClient:
    def __init__(
        self,
        backend: TextGenerationBackend,
        *,
        prompt_root: str | Path | None = None,
        system_prompt_filename: str = "planner_system_v07.txt",
    ) -> None:
        self.backend = backend
        self.prompt_root = Path(prompt_root) if prompt_root else PROMPT_ROOT
        self.system_prompt_path = self.prompt_root / system_prompt_filename

    def run(self, request: PlannerClientRequest | dict[str, Any]) -> PlannerOutput:
        req = self._coerce_request(request)
        system_prompt = self._load_prompt(self.system_prompt_path)
        backend_response = self.backend.generate(
            stage="planner",
            system_prompt=system_prompt,
            user_prompt="",
            context=self._build_backend_context(req),
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
            metadata=dict(backend_response.metadata),
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

    def _build_backend_context(self, request: PlannerClientRequest) -> dict[str, Any]:
        return {
            "request": request.model_copy(deep=True),
            "paths": {
                "sample_dir": request.sample_dir,
                "trajectory_dir": request.trajectory_dir,
                "planner_dir": request.planner_dir,
                "steps_dir": request.steps_dir,
            },
        }

    def _parse_model_text(self, text: str) -> dict[str, Any]:
        json_payload = try_parse_planner_json_payload(text)
        if json_payload is not None:
            return self._parse_json_contract(json_payload, raw_text=text)

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
                "Planner output must contain <answer> or <suggestions> after <think>, "
                "or a JSON object with mode/think/answer|suggestions.",
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

    def _parse_json_contract(self, payload: dict[str, Any], *, raw_text: str) -> dict[str, Any]:
        mode = str(payload.get("mode", "")).strip().lower()
        think_raw = payload.get("think")
        stop_reason = payload.get("stop_reason")

        if think_raw is None:
            think = "(Model omitted 'think' in JSON; placeholder.)"
        elif isinstance(think_raw, str) and not think_raw.strip():
            think = "(Empty think in JSON; placeholder.)"
        else:
            think = str(think_raw).strip()
        if stop_reason is not None and not isinstance(stop_reason, str):
            raise ModelResponseParseError(
                "Planner JSON output field 'stop_reason' must be a string or null.",
                stage="planner",
                tag="stop_reason",
                preview=self._preview_planner(raw_text),
            )

        if mode == "answer":
            answer = payload.get("answer")
            if payload.get("suggestions") is not None:
                raise ModelResponseParseError(
                    "Planner JSON output with mode='answer' must not contain 'suggestions'.",
                    stage="planner",
                    tag="suggestions",
                    preview=self._preview_planner(raw_text),
                )
            if answer is None:
                raise ModelResponseParseError(
                    "Planner JSON output with mode='answer' must contain an 'answer' field.",
                    stage="planner",
                    tag="answer",
                    preview=self._preview_planner(raw_text),
                )
            answer_text = str(answer).strip()
            if not answer_text:
                raise ModelResponseParseError(
                    "Planner JSON output field 'answer' must not be empty.",
                    stage="planner",
                    tag="answer",
                    preview=self._preview_planner(raw_text),
                )
            return {
                "can_answer_now": True,
                "global_chain_cot": think,
                "direct_answer": answer_text,
                "stop_reason": stop_reason,
                "suggestions": [],
            }

        if mode == "suggestions":
            if "answer" in payload and payload.get("answer") not in (None, ""):
                raise ModelResponseParseError(
                    "Planner JSON output with mode='suggestions' must not contain 'answer'.",
                    stage="planner",
                    tag="answer",
                    preview=self._preview_planner(raw_text),
                )
            suggestions_payload = payload.get("suggestions")
            if not isinstance(suggestions_payload, list):
                raise ModelResponseParseError(
                    "Planner JSON output with mode='suggestions' must contain a 'suggestions' array.",
                    stage="planner",
                    tag="suggestions",
                    preview=self._preview_planner(raw_text),
                )
            suggestions = [PlannerSuggestion.model_validate(item) for item in suggestions_payload]
            return {
                "can_answer_now": False,
                "global_chain_cot": think,
                "direct_answer": None,
                "stop_reason": stop_reason,
                "suggestions": suggestions,
            }

        raise ModelResponseParseError(
            "Planner JSON output field 'mode' must be either 'answer' or 'suggestions'.",
            stage="planner",
            tag="mode",
            preview=self._preview_planner(raw_text),
        )

    def _preview_planner(self, text: str, *, limit: int = 500) -> str:
        normalized = " ".join(text.strip().split())
        if len(normalized) <= limit:
            return normalized
        return normalized[:limit] + "..."


__all__ = [
    "PlannerClient",
]
