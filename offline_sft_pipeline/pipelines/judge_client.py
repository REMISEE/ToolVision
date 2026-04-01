from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from offline_sft_pipeline.core.models import JudgeRecord, utc_now
from offline_sft_pipeline.pipelines.backends import JudgeBackend
from offline_sft_pipeline.pipelines.request_models import JudgeClientRequest


@dataclass(slots=True)
class JudgePolicyConfig:
    keep_threshold: float = 0.25
    export_threshold: float = 0.85


class JudgeClient:
    def __init__(
        self,
        backend: JudgeBackend,
        *,
        policy: JudgePolicyConfig | None = None,
    ) -> None:
        self.backend = backend
        self.policy = policy or JudgePolicyConfig()

    def run(self, request: JudgeClientRequest | dict[str, Any]) -> JudgeRecord:
        req = self._coerce_request(request)
        backend_result = self.backend.score(req)
        scope_step_idx = req.scope_step_idx
        if scope_step_idx is None and req.step_record is not None:
            scope_step_idx = req.step_record.step_idx
        overall_score = self._clamp_score(backend_result.overall_score)
        judge_record = JudgeRecord(
            judge_record_id=self._build_judge_record_id(req, scope_step_idx),
            sample_id=req.sample_id,
            trajectory_id=req.trajectory_id,
            scope_type=req.scope_type,
            scope_step_idx=scope_step_idx,
            judge_stage=req.judge_stage,
            created_at=utc_now(),
            keep_for_frontier=overall_score >= self.policy.keep_threshold,
            exportable=req.scope_type == "trajectory" and overall_score >= self.policy.export_threshold,
            overall_score=overall_score,
            answerability_score=None,
            tool_use_quality_score=None,
            trajectory_progress_score=None,
            note=(backend_result.note or "").strip(),
        )
        judge_record.validate_against_schema()
        return judge_record

    def _coerce_request(self, request: JudgeClientRequest | dict[str, Any]) -> JudgeClientRequest:
        if isinstance(request, JudgeClientRequest):
            return request
        return JudgeClientRequest.model_validate(request)

    def _build_judge_record_id(self, request: JudgeClientRequest, scope_step_idx: int | None) -> str:
        scope_label = "traj" if request.scope_type == "trajectory" else f"step_{int(scope_step_idx or 0):03d}"
        timestamp = utc_now().strftime("%Y%m%dT%H%M%S%fZ")
        return (
            f"judge__{request.sample_id}__{request.trajectory_id}__"
            f"{request.judge_stage}__{scope_label}__{timestamp}"
        )

    def _clamp_score(self, score: float) -> float:
        return max(0.0, min(1.0, float(score)))


__all__ = [
    "JudgeClient",
    "JudgePolicyConfig",
]
