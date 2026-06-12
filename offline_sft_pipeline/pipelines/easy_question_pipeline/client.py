"""Planner client wrapper: passes reference answer into the easy planner backend context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from offline_sft_pipeline.core.models import PlannerOutput
from offline_sft_pipeline.pipelines.backends import TextGenerationBackend
from offline_sft_pipeline.pipelines.easy_question_pipeline.backend import EasyApiTextPlannerBackend
from offline_sft_pipeline.pipelines.planner_client import PlannerClient
from offline_sft_pipeline.pipelines.request_models import PlannerClientRequest


@dataclass(frozen=True, slots=True)
class EasyPlannerRunArtifacts:
    """Paths written by `EasyApiTextPlannerBackend` (when `artifact_dir` was set)."""

    artifact_dir: Path | None
    files: dict[str, str]


class EasyPlannerClient(PlannerClient):
    """Same as `PlannerClient`, but `run()` accepts `reference_answer` and optional `artifact_dir` for I/O dumps."""

    def __init__(
        self,
        backend: TextGenerationBackend | None = None,
        *,
        prompt_root: str | Path | None = None,
        system_prompt_filename: str = "planner_system_v05.txt",
    ) -> None:
        resolved = backend if backend is not None else EasyApiTextPlannerBackend()
        super().__init__(resolved, prompt_root=prompt_root, system_prompt_filename=system_prompt_filename)
        self._easy_ref: str = ""
        self._easy_dir: Path | None = None

    def _build_backend_context(self, request: PlannerClientRequest) -> dict[str, Any]:
        ctx = super()._build_backend_context(request)
        if self._easy_ref:
            ctx["reference_answer"] = self._easy_ref
        if self._easy_dir is not None:
            ctx["easy_artifact_dir"] = str(self._easy_dir)
        return ctx

    def run(
        self,
        request: PlannerClientRequest | dict[str, Any],
        *,
        reference_answer: str,
        artifact_dir: Path | str | None = None,
    ) -> tuple[PlannerOutput, EasyPlannerRunArtifacts]:
        ref = str(reference_answer).strip()
        if not ref:
            raise ValueError("reference_answer must be non-empty.")
        adir = Path(str(artifact_dir)).resolve() if artifact_dir else None
        self._easy_ref = ref
        self._easy_dir = adir
        try:
            planner_output = super().run(request)
        finally:
            self._easy_ref = ""
            self._easy_dir = None
        files = dict(planner_output.metadata.get("easy_artifact_files") or {})
        return planner_output, EasyPlannerRunArtifacts(artifact_dir=adir, files=files)


__all__ = [
    "EasyPlannerClient",
    "EasyPlannerRunArtifacts",
]
