from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
CODEVISION_ROOT = REPO_ROOT / "CodeVision"
if str(CODEVISION_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEVISION_ROOT))

import ray

from verl.tools.code_image_tool import CodeImageTool, SUCCESS_FOLLOWUP_TEXT
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from offline_sft_pipeline.runtime.types import ArtifactRef, RuntimeStepOutput, RuntimeStepRequest

logger = logging.getLogger(__name__)


def build_default_code_image_tool_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "code_image_tool",
                "description": "Offline runtime wrapper schema for one executor step.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to process image."},
                        "description": {"type": "string", "description": "Step description."},
                        "image_index": {"type": "integer", "description": "Index of image."},
                    },
                    "required": ["code", "description", "image_index"],
                },
            },
        }
    )


def build_default_code_image_tool_config(
    *,
    external_services: Optional[dict[str, Any]] = None,
    enable_external_model_functions: bool = True,
    ocr_model_name: str = "paddleocr",
    **overrides: Any,
) -> dict[str, Any]:
    config = {
        "type": "native",
        "num_workers": 1,
        "rate_limit": 10,
        "timeout": 180,
        "max_code_length": 4096,
        "enable_global_rate_limit": False,
        "enable_external_model_functions": enable_external_model_functions,
        "external_call_mode": "service",
        "ocr_model_name": ocr_model_name,
        "external_services": external_services or {},
    }
    config.update(overrides)
    return config


class CodeImageRuntimeWrapper:
    """Single-step runtime wrapper around CodeImageTool."""

    def __init__(
        self,
        tool_config: dict[str, Any],
        *,
        tool_schema: Optional[OpenAIFunctionToolSchema] = None,
        schema_path: Optional[str | Path] = None,
        validate_runtime_result: bool = True,
        ray_init_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.tool_config = tool_config
        self.tool_schema = tool_schema or build_default_code_image_tool_schema()
        self.schema_path = Path(schema_path) if schema_path else REPO_ROOT / "offline_sft_pipeline" / "schemas" / "executor_runtime_result_schema.json"
        self.validate_runtime_result = validate_runtime_result
        self.ray_init_kwargs = dict(ray_init_kwargs or {})
        self._tool: Optional[CodeImageTool] = None
        self._owns_ray = False
        self._runtime_schema: Optional[dict[str, Any]] = None

    def _ensure_ray(self) -> None:
        if ray.is_initialized():
            return
        init_kwargs = {
            "ignore_reinit_error": True,
            "include_dashboard": False,
            "logging_level": "ERROR",
        }
        init_kwargs.update(self.ray_init_kwargs)
        ray.init(**init_kwargs)
        self._owns_ray = True

    def _ensure_tool(self) -> CodeImageTool:
        self._ensure_ray()
        if self._tool is None:
            self._tool = CodeImageTool(config=self.tool_config, tool_schema=self.tool_schema)
        return self._tool

    async def close(self) -> None:
        self._tool = None
        if self._owns_ray and ray.is_initialized():
            ray.shutdown()
            self._owns_ray = False

    def close_sync(self) -> None:
        asyncio.run(self.close())

    def _load_runtime_schema(self) -> dict[str, Any]:
        if self._runtime_schema is None:
            self._runtime_schema = json.loads(self.schema_path.read_text(encoding="utf-8-sig"))
        return self._runtime_schema

    def _validate_runtime_result(self, runtime_result: dict[str, Any]) -> None:
        if not self.validate_runtime_result:
            return
        try:
            import jsonschema
        except ImportError as exc:
            raise RuntimeError("jsonschema is required when validate_runtime_result=True") from exc
        jsonschema.validate(runtime_result, self._load_runtime_schema())

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _format_dt(value: datetime) -> str:
        return value.isoformat().replace("+00:00", "Z")

    @staticmethod
    def _coerce_request(request: RuntimeStepRequest | dict[str, Any]) -> RuntimeStepRequest:
        if isinstance(request, RuntimeStepRequest):
            return request
        return RuntimeStepRequest.from_dict(request)

    @staticmethod
    def _canonicalize_text(tool_response: ToolResponse, tool_metrics: dict[str, Any]) -> str:
        helper_text = str(tool_metrics.get("helper_text") or "").strip()
        if helper_text:
            return helper_text

        response_text = str(getattr(tool_response, "text", None) or "").strip()
        if response_text.endswith(SUCCESS_FOLLOWUP_TEXT):
            response_text = response_text[: -len(SUCCESS_FOLLOWUP_TEXT)].rstrip()
        if response_text:
            return response_text

        return str(tool_metrics.get("message") or "").strip()

    @staticmethod
    def _canonicalize_meta(tool_response: ToolResponse, tool_metrics: dict[str, Any]) -> dict[str, Any]:
        if isinstance(getattr(tool_response, "meta", None), dict):
            return dict(tool_response.meta or {})
        raw_meta = tool_metrics.get("helper_meta")
        if isinstance(raw_meta, dict):
            return dict(raw_meta)
        return {}

    @staticmethod
    def _ensure_output_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        path.write_text(text or "", encoding="utf-8")

    @staticmethod
    def _build_error(tool_response: ToolResponse, tool_metrics: dict[str, Any]) -> Optional[dict[str, Any]]:
        success = bool(tool_metrics.get("success"))
        if success:
            return None
        error_type = str(tool_metrics.get("error") or "runtime_error")
        message = str(tool_metrics.get("message") or getattr(tool_response, "text", None) or "").strip()
        if not message:
            message = "Runtime execution failed."
        return {
            "type": error_type,
            "message": message,
            "traceback_path": None,
        }

    @staticmethod
    def _build_exit_code(tool_metrics: dict[str, Any]) -> int:
        return 0 if bool(tool_metrics.get("success")) else 1

    def _save_output_images(
        self,
        response_images: list[Any] | None,
        request: RuntimeStepRequest,
    ) -> list[ArtifactRef]:
        saved_artifacts: list[ArtifactRef] = []
        if not response_images:
            return saved_artifacts

        for idx, image in enumerate(response_images):
            if not hasattr(image, "save"):
                raise TypeError(f"Output image at index {idx} is not a PIL-compatible image object.")
            output_path = request.step_output_dir_obj / f"output_{idx}.png"
            image.save(output_path)
            width, height = image.size
            saved_artifacts.append(
                ArtifactRef(
                    artifact_id=f"img_step_{request.step_idx:03d}_{idx}",
                    path=str(output_path),
                    media_type="image/png",
                    width=int(width),
                    height=int(height),
                )
            )
        return saved_artifacts

    async def run_step(self, request: RuntimeStepRequest | dict[str, Any]) -> RuntimeStepOutput:
        req = self._coerce_request(request)
        if not req.visible_images:
            raise ValueError("RuntimeStepRequest.visible_images must not be empty.")

        code_path = req.executor_code_path_obj
        if not code_path.exists():
            raise FileNotFoundError(f"executor_code_path not found: {code_path}")

        visible_image_paths: list[str] = []
        for item in req.visible_images:
            image_path = item.path_obj
            if not image_path.exists():
                raise FileNotFoundError(f"visible image not found: {image_path}")
            visible_image_paths.append(str(image_path))

        self._ensure_output_dir(req.step_output_dir_obj)
        code = code_path.read_text(encoding="utf-8")
        stdout_path = req.step_output_dir_obj / "stdout.txt"
        stderr_path = req.step_output_dir_obj / "stderr.txt"
        runtime_result_path = req.step_output_dir_obj / "runtime_result.json"

        started_at = self._utc_now()
        started_monotonic = time.perf_counter()

        tool = self._ensure_tool()
        instance_id = None
        tool_response = ToolResponse()
        tool_metrics: dict[str, Any] = {}
        try:
            instance_id, _ = await tool.create(create_kwargs={"image": visible_image_paths})
            tool_response, _, tool_metrics = await tool.execute(
                instance_id,
                {
                    "code": code,
                    "description": f"offline_runtime_step_{req.step_idx}",
                    "image_index": req.image_index,
                },
            )
        finally:
            if instance_id is not None:
                await tool.release(instance_id)

        finished_at = self._utc_now()
        elapsed_seconds = time.perf_counter() - started_monotonic

        stdout_text = str(tool_metrics.get("stdout_text") or "")
        stderr_text = str(tool_metrics.get("stderr_text") or "")
        self._write_text(stdout_path, stdout_text)
        self._write_text(stderr_path, stderr_text)

        saved_artifacts = self._save_output_images(getattr(tool_response, "image", None), req)
        observed_helper_calls = tool_metrics.get("observed_helper_calls")
        if not isinstance(observed_helper_calls, list):
            observed_helper_calls = []
        observed_helper_call_count = tool_metrics.get("observed_helper_call_count")
        if not isinstance(observed_helper_call_count, int):
            observed_helper_call_count = len(observed_helper_calls)

        runtime_result = {
            "schema_version": "0.1.0",
            "sample_id": req.sample_id,
            "trajectory_id": req.trajectory_id,
            "round_idx": req.round_idx,
            "step_idx": req.step_idx,
            "created_at": self._format_dt(finished_at),
            "success": bool(tool_metrics.get("success")),
            "images": [artifact.to_dict() for artifact in saved_artifacts],
            "text": self._canonicalize_text(tool_response, tool_metrics),
            "meta": self._canonicalize_meta(tool_response, tool_metrics),
            "observed_helper_call_count": observed_helper_call_count,
            "observed_helper_calls": observed_helper_calls,
            "code_execution": {
                "code_path": str(code_path),
                "exit_code": self._build_exit_code(tool_metrics),
                "started_at": self._format_dt(started_at),
                "finished_at": self._format_dt(finished_at),
                "elapsed_seconds": elapsed_seconds,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            },
            "error": self._build_error(tool_response, tool_metrics),
        }

        self._validate_runtime_result(runtime_result)
        runtime_result_path.write_text(json.dumps(runtime_result, ensure_ascii=False, indent=2), encoding="utf-8")

        return RuntimeStepOutput(
            runtime_result=runtime_result,
            saved_artifacts=saved_artifacts,
            runtime_result_path=str(runtime_result_path),
            tool_metrics=tool_metrics,
        )

    def run_step_sync(self, request: RuntimeStepRequest | dict[str, Any]) -> RuntimeStepOutput:
        return asyncio.run(self.run_step(request))

