from __future__ import annotations

"""Scripted fake components for local orchestrator demos and regression tests.

Everything in this module is intentionally deterministic and fake:

- planner outputs are pre-authored
- executor outputs are pre-authored
- runtime outputs are synthesized locally
- judge scores are pre-authored

This module exists so the repo can run the multi-round pipeline semantics
without requiring real model APIs or deployed helper services.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from offline_sft_pipeline.core.models import (
    Budget,
    CapabilityPlanItem,
    ExecutorRuntimeResult,
    ExecutorStepOutput,
    ImageArtifactRef,
    PlannerOutput,
    PlannerStepSpec,
    PlannerSuggestion,
    RootImage,
    RootSample,
    RuntimeCodeExecution,
    RuntimeErrorInfo,
    RuntimeObservedHelperCall,
    build_child_trajectory_id,
    build_root_trajectory_id,
    utc_now,
)
from offline_sft_pipeline.pipelines.backends import BackendResponse, JudgeBackendResult
from offline_sft_pipeline.pipelines.orchestrator_v01 import OrchestratorConfig
from offline_sft_pipeline.pipelines.request_models import ExecutorClientRequest, PlannerClientRequest
from offline_sft_pipeline.runtime.types import ArtifactRef, RuntimeStepOutput, RuntimeStepRequest


def write_demo_image(path: Path, *, label: str, color: str) -> None:
    image = Image.new("RGB", (200, 120), color=color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, 188, 108), outline="black", width=2)
    draw.text((20, 48), label, fill="black")
    image.save(path)


def build_demo_root_sample(root_dir: Path, *, sample_id: str = "demo__train__0001") -> RootSample:
    image_path = root_dir / "root.png"
    write_demo_image(image_path, label="root", color="white")
    return RootSample(
        sample_id=sample_id,
        question="What number is written on the hanging tag?",
        images=[RootImage(image_id="root_0", path=str(image_path))],
    )


def make_step(
    step_id: str,
    step_goal: str,
    capability_names: list[str],
    *,
    input_image_index: int = 0,
    executor_instruction: str | None = None,
) -> PlannerStepSpec:
    return PlannerStepSpec(
        step_id=step_id,
        step_goal=step_goal,
        input_image_index=input_image_index,
        capability_plan=[
            CapabilityPlanItem(
                order=index,
                capability=capability_name,
                instruction=f"Use {capability_name} for {step_goal.lower()}",
            )
            for index, capability_name in enumerate(capability_names, start=1)
        ],
        executor_instruction=executor_instruction or f"Execute {step_goal.lower()}",
    )


def make_suggestion(
    suggestion_id: str,
    suggestion_cot: str,
    steps: list[PlannerStepSpec],
) -> PlannerSuggestion:
    return PlannerSuggestion(
        suggestion_id=suggestion_id,
        suggestion_cot=suggestion_cot,
        steps=steps,
    )


def make_planner_output(
    *,
    sample_id: str,
    trajectory_id: str,
    round_idx: int,
    global_chain_cot: str,
    suggestions: list[PlannerSuggestion] | None = None,
    direct_answer: str | None = None,
) -> PlannerOutput:
    return PlannerOutput(
        sample_id=sample_id,
        trajectory_id=trajectory_id,
        round_idx=round_idx,
        created_at=utc_now(),
        can_answer_now=direct_answer is not None,
        global_chain_cot=global_chain_cot,
        direct_answer=direct_answer,
        stop_reason=None,
        suggestions=list(suggestions or []),
    )


def make_executor_output(cot: str, code: str, *, description: str | None = None) -> ExecutorStepOutput:
    tool_description = description or "Execute the planned image-processing step."
    raw_payload = {
        "think": cot,
        "tool_call": {
            "name": "code_image_tool",
            "arguments": {
                "code": code,
                "description": tool_description,
            },
        },
    }
    return ExecutorStepOutput(
        cot=cot,
        code=code,
        description=tool_description,
        raw_response_text=json.dumps(raw_payload, ensure_ascii=False, indent=2),
        metadata={"backend": "scripted_executor"},
    )


def render_planner_output_as_model_text(planner_output: PlannerOutput) -> str:
    payload: dict[str, Any] = {
        "think": planner_output.global_chain_cot,
    }
    if planner_output.stop_reason is not None:
        payload["stop_reason"] = planner_output.stop_reason
    if planner_output.can_answer_now:
        payload["mode"] = "answer"
        payload["answer"] = planner_output.direct_answer or ""
    else:
        payload["mode"] = "suggestions"
        payload["suggestions"] = [item.model_dump(mode="json") for item in planner_output.suggestions]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_executor_output_as_model_text(executor_output: ExecutorStepOutput) -> str:
    if executor_output.raw_response_text and executor_output.raw_response_text.strip():
        return executor_output.raw_response_text
    return json.dumps(
        {
            "think": executor_output.cot,
            "tool_call": {
                "name": "code_image_tool",
                "arguments": {
                    "code": executor_output.code,
                    "description": executor_output.description,
                },
            },
        },
        ensure_ascii=False,
        indent=2,
    )


class ScriptedPlannerClient:
    """Fake planner client used by docs/demo/tests.

    It does not call a model. Responses are keyed by `(trajectory_id, round_idx)`.
    """

    def __init__(self, responses: dict[tuple[str, int], PlannerOutput]) -> None:
        self.responses = dict(responses)
        self.requests: list[PlannerClientRequest] = []

    def run(self, request: PlannerClientRequest | dict[str, Any]) -> PlannerOutput:
        req = request if isinstance(request, PlannerClientRequest) else PlannerClientRequest.model_validate(request)
        self.requests.append(req.model_copy(deep=True))
        key = (req.trajectory_id, req.round_idx)
        if key not in self.responses:
            raise KeyError(f"Missing scripted planner response for {key}.")
        return self.responses[key].model_copy(deep=True)


class ScriptedExecutorClient:
    """Fake executor client used by docs/demo/tests."""

    def __init__(self, responses: dict[tuple[str, int], ExecutorStepOutput]) -> None:
        self.responses = dict(responses)
        self.requests: list[ExecutorClientRequest] = []

    def run(self, request: ExecutorClientRequest | dict[str, Any]) -> ExecutorStepOutput:
        req = request if isinstance(request, ExecutorClientRequest) else ExecutorClientRequest.model_validate(request)
        self.requests.append(req.model_copy(deep=True))
        key = (req.trajectory_id, req.step_idx)
        if key not in self.responses:
            raise KeyError(f"Missing scripted executor response for {key}.")
        return self.responses[key].model_copy(deep=True)


class ScriptedTextBackend:
    """Fake text backend that still routes through real planner/executor clients.

    It extracts ids from the built prompt text, then returns pre-authored planner
    or executor model text for that request key.
    """

    _FIELD_PATTERNS = {
        "trajectory_id": re.compile(r"(?m)^trajectory_id:\s*(.+?)\s*$"),
        "round_idx": re.compile(r"(?m)^round_idx:\s*(.+?)\s*$"),
        "step_idx": re.compile(r"(?m)^step_idx:\s*(.+?)\s*$"),
    }

    def __init__(
        self,
        *,
        planner_outputs: dict[tuple[str, int], PlannerOutput],
        executor_outputs: dict[tuple[str, int], ExecutorStepOutput],
    ) -> None:
        self.planner_outputs = {
            key: value.model_copy(deep=True)
            for key, value in planner_outputs.items()
        }
        self.executor_outputs = {
            key: value.model_copy(deep=True)
            for key, value in executor_outputs.items()
        }
        self.requests: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        context: dict[str, Any] | None = None,
    ) -> BackendResponse:
        request_obj = self._extract_context_request(context)
        trajectory_id = self._extract_request_field(
            request_obj,
            "trajectory_id",
            fallback_prompt=user_prompt,
        )
        round_idx = int(
            self._extract_request_field(
                request_obj,
                "round_idx",
                fallback_prompt=user_prompt,
            )
        )
        step_idx = None
        response_text = ""
        metadata = {
            "backend": "scripted_text_backend",
            "stage": stage,
            "trajectory_id": trajectory_id,
            "round_idx": round_idx,
            "system_prompt_chars": len(system_prompt),
            "user_prompt_chars": len(user_prompt),
            "has_context": context is not None,
        }
        if stage == "planner":
            key = (trajectory_id, round_idx)
            if key not in self.planner_outputs:
                raise KeyError(f"Missing scripted planner text response for {key}.")
            response_text = render_planner_output_as_model_text(self.planner_outputs[key])
        elif stage == "executor":
            step_idx = int(
                self._extract_request_field(
                    request_obj,
                    "step_idx",
                    fallback_prompt=user_prompt,
                )
            )
            key = (trajectory_id, step_idx)
            if key not in self.executor_outputs:
                raise KeyError(f"Missing scripted executor text response for {key}.")
            response_text = render_executor_output_as_model_text(self.executor_outputs[key])
            metadata["step_idx"] = step_idx
        else:
            raise NotImplementedError(f"Unsupported stage for ScriptedTextBackend: {stage!r}")
        self.requests.append(
            {
                "stage": stage,
                "trajectory_id": trajectory_id,
                "round_idx": round_idx,
                "step_idx": step_idx,
                "system_prompt_chars": len(system_prompt),
                "user_prompt_chars": len(user_prompt),
                "has_context": context is not None,
            }
        )
        return BackendResponse(text=response_text, metadata=metadata)

    def _extract_context_request(self, context: dict[str, Any] | None) -> Any | None:
        if not context:
            return None
        return context.get("request")

    def _extract_request_field(
        self,
        request_obj: Any | None,
        field_name: str,
        *,
        fallback_prompt: str,
    ) -> str:
        if request_obj is not None:
            if isinstance(request_obj, dict) and field_name in request_obj:
                return str(request_obj[field_name]).strip()
            if hasattr(request_obj, field_name):
                return str(getattr(request_obj, field_name)).strip()
        return self._extract_prompt_field(fallback_prompt, field_name)

    def _extract_prompt_field(self, prompt: str, field_name: str) -> str:
        pattern = self._FIELD_PATTERNS[field_name]
        match = pattern.search(prompt)
        if match is None:
            raise KeyError(f"Missing field {field_name!r} in scripted backend prompt.")
        return match.group(1).strip()


class ScriptedJudgeBackend:
    """Fake judge backend used by docs/demo/tests."""

    def __init__(self, scores: dict[tuple[str, int], float]) -> None:
        self.scores = dict(scores)
        self.requests: list[Any] = []

    def score(self, request: Any) -> JudgeBackendResult:
        self.requests.append(request)
        step_idx = request.scope_step_idx
        if step_idx is None and request.step_record is not None:
            step_idx = request.step_record.step_idx
        key = (request.trajectory_id, int(step_idx or 0))
        if key not in self.scores:
            raise KeyError(f"Missing scripted judge score for {key}.")
        score = float(self.scores[key])
        return JudgeBackendResult(
            overall_score=score,
            metadata={"backend": "scripted_judge"},
            note=f"score={score:.2f}",
        )


@dataclass(slots=True)
class RuntimeSpec:
    text: str
    helper_names: list[str]
    image_label: str
    success: bool = True
    error_message: str = ""


class ScriptedRuntime:
    """Fake runtime used by docs/demo/tests.

    It does not execute executor code. It only writes synthetic runtime artifacts
    to the same locations that the real runtime wrapper would use.
    """

    def __init__(
        self,
        responses: dict[tuple[str, int], RuntimeSpec],
        *,
        default_spec: RuntimeSpec | None = None,
    ) -> None:
        self.responses = dict(responses)
        self.default_spec = default_spec
        self.requests: list[RuntimeStepRequest] = []

    def run_step_sync(self, request: RuntimeStepRequest | dict[str, Any]) -> RuntimeStepOutput:
        req = request if isinstance(request, RuntimeStepRequest) else RuntimeStepRequest.from_dict(request)
        self.requests.append(req)
        key = (req.trajectory_id, req.step_idx)
        spec = self.responses.get(key)
        if spec is None:
            if self.default_spec is None:
                raise KeyError(f"Missing scripted runtime response for {key}.")
            spec = self.default_spec

        step_dir = Path(req.step_output_dir)
        step_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = step_dir / "stdout.txt"
        stderr_path = step_dir / "stderr.txt"
        runtime_result_path = step_dir / "runtime_result.json"

        stdout_path.write_text(f"scripted runtime step {req.step_idx}: {spec.image_label}\n", encoding="utf-8")
        stderr_path.write_text("" if spec.success else f"{spec.error_message}\n", encoding="utf-8")

        images: list[ImageArtifactRef] = []
        saved_artifacts: list[ArtifactRef] = []
        if spec.success:
            output_path = step_dir / "output_0.png"
            write_demo_image(output_path, label=spec.image_label, color="lightyellow")
            images.append(
                ImageArtifactRef(
                    artifact_id=f"img_step_{req.step_idx:03d}_0",
                    path=str(output_path),
                    media_type="image/png",
                    width=200,
                    height=120,
                )
            )
            saved_artifacts.append(
                ArtifactRef(
                    artifact_id=f"img_step_{req.step_idx:03d}_0",
                    path=str(output_path),
                    media_type="image/png",
                    width=200,
                    height=120,
                )
            )

        result = ExecutorRuntimeResult(
            sample_id=req.sample_id,
            trajectory_id=req.trajectory_id,
            round_idx=req.round_idx,
            step_idx=req.step_idx,
            created_at=utc_now(),
            success=spec.success,
            images=images,
            text=spec.text,
            meta={"runtime_label": spec.image_label, "runtime_mode": "scripted_fake"},
            observed_helper_call_count=len(spec.helper_names),
            observed_helper_calls=[
                RuntimeObservedHelperCall(order=index, name=helper_name, status="ok" if spec.success else "error")
                for index, helper_name in enumerate(spec.helper_names, start=1)
            ],
            code_execution=RuntimeCodeExecution(
                code_path=req.executor_code_path,
                exit_code=0 if spec.success else 1,
                started_at=utc_now(),
                finished_at=utc_now(),
                elapsed_seconds=0.01,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
            ),
            error=None if spec.success else RuntimeErrorInfo(type="mock_runtime_error", message=spec.error_message),
        )
        result.to_json_file(runtime_result_path)
        return RuntimeStepOutput(
            runtime_result=result.to_dict(),
            saved_artifacts=saved_artifacts,
            runtime_result_path=str(runtime_result_path),
            tool_metrics={"scripted": True},
        )


@dataclass(slots=True)
class ScriptedScenario:
    sample: RootSample
    planner_client: ScriptedPlannerClient
    executor_client: ScriptedExecutorClient
    runtime: ScriptedRuntime
    judge_backend: ScriptedJudgeBackend
    config: OrchestratorConfig
    fake_components: dict[str, str]
    scenario_notes: list[str]


@dataclass(slots=True)
class ThreeRoundDemoSpec:
    sample: RootSample
    planner_outputs: dict[tuple[str, int], PlannerOutput]
    executor_outputs: dict[tuple[str, int], ExecutorStepOutput]
    runtime_specs: dict[tuple[str, int], RuntimeSpec]
    judge_scores: dict[tuple[str, int], float]
    config: OrchestratorConfig
    scenario_notes: list[str]


def build_three_round_demo_spec(
    root_dir: Path,
    *,
    sample_id: str = "demo__train__0001",
) -> ThreeRoundDemoSpec:
    """Build the shared fake 3-round scenario data used by demos/tests."""

    sample = build_demo_root_sample(root_dir, sample_id=sample_id)

    root_id = build_root_trajectory_id(sample_id)
    traj_s1 = build_child_trajectory_id(root_id, 0, "s1")
    traj_s2 = build_child_trajectory_id(root_id, 0, "s2")
    traj_s3 = build_child_trajectory_id(root_id, 0, "s3")
    traj_s21 = build_child_trajectory_id(traj_s2, 1, "s21")
    traj_s22 = build_child_trajectory_id(traj_s2, 1, "s22")
    traj_s31 = build_child_trajectory_id(traj_s3, 1, "s31")
    traj_s221 = build_child_trajectory_id(traj_s22, 2, "s221")

    planner_outputs = {
        (root_id, 0): make_planner_output(
                sample_id=sample_id,
                trajectory_id=root_id,
                round_idx=0,
                global_chain_cot=(
                    "Round 0 global rethink: inspect the tag region first, then branch into OCR-heavy "
                    "and segmentation-heavy trajectories."
                ),
                suggestions=[
                    make_suggestion(
                        "s1",
                        "Trajectory s1: ground the tag, crop it, then decide whether OCR is already enough.",
                        [
                            make_step(
                                "step_crop_tag",
                                "Localize the tag and keep a clean crop for later inspection.",
                                ["ground_box", "dino_crop"],
                                input_image_index=0,
                            )
                        ],
                    ),
                    make_suggestion(
                        "s2",
                        "Trajectory s2: ground the price tag and keep the crop so the next round can continue from it.",
                        [
                            make_step(
                                "step_crop_serial",
                                "Ground the tag and crop the likely number region.",
                                ["ground_box", "dino_crop"],
                                input_image_index=0,
                            )
                        ],
                    ),
                    make_suggestion(
                        "s3",
                        "Trajectory s3: use segmentation plus OCR to inspect a competing region.",
                        [
                            make_step(
                                "step_mask_then_ocr",
                                "Mask the candidate region and read the visible text.",
                                ["sam_mask", "ocr_assist"],
                                input_image_index=0,
                            )
                        ],
                    ),
                ],
            ),
        (traj_s1, 1): make_planner_output(
                sample_id=sample_id,
                trajectory_id=traj_s1,
                round_idx=1,
                global_chain_cot="Round 1 rethink on s1: the crop already isolates the answer region.",
                direct_answer="249",
            ),
        (traj_s2, 1): make_planner_output(
                sample_id=sample_id,
                trajectory_id=traj_s2,
                round_idx=1,
                global_chain_cot=(
                    "Round 1 rethink on s2: the prior crop looks good. One branch should continue the "
                    "existing crop with OCR; another should re-ground and recrop in case the crop is biased."
                ),
                suggestions=[
                    make_suggestion(
                        "s21",
                        "Continue from the previous crop and run OCR directly because the prior tool call is still visible.",
                        [
                            make_step(
                                "step_continue_ocr",
                                "Continue with OCR on the latest crop.",
                                ["ocr_assist"],
                                input_image_index=1,
                            )
                        ],
                    ),
                    make_suggestion(
                        "s22",
                        "Revisit the earlier crop by grounding again, then OCR the refreshed view.",
                        [
                            make_step(
                                "step_reground_then_ocr",
                                "Reground the tag, recrop it, and OCR the new crop.",
                                ["ground_box", "dino_crop", "ocr_assist"],
                                input_image_index=0,
                            )
                        ],
                    ),
                ],
            ),
        (traj_s3, 1): make_planner_output(
                sample_id=sample_id,
                trajectory_id=traj_s3,
                round_idx=1,
                global_chain_cot=(
                    "Round 1 rethink on s3: keep one branch on the masked region, but keep a replan branch "
                    "available if the mask was off."
                ),
                suggestions=[
                    make_suggestion(
                        "s31",
                        "Stay on the current masked region and retry with the same family of tools.",
                        [
                            make_step(
                                "step_retry_mask_path",
                                "Retry the masked region with another visual pass.",
                                ["sam_mask", "dino_crop"],
                                input_image_index=1,
                            )
                        ],
                    ),
                    make_suggestion(
                        "s32",
                        "Drop the masked path and start over with a different grounding strategy.",
                        [
                            make_step(
                                "step_switch_strategy",
                                "Switch away from the mask and reground the tag.",
                                ["ground_box", "dino_crop"],
                                input_image_index=0,
                            )
                        ],
                    ),
                ],
            ),
        (traj_s21, 2): make_planner_output(
                sample_id=sample_id,
                trajectory_id=traj_s21,
                round_idx=2,
                global_chain_cot="Round 2 rethink on s21: OCR is now sufficient to answer.",
                direct_answer="249",
            ),
        (traj_s22, 2): make_planner_output(
                sample_id=sample_id,
                trajectory_id=traj_s22,
                round_idx=2,
                global_chain_cot="Round 2 rethink on s22: one more OCR attempt is worth trying before stopping.",
                suggestions=[
                    make_suggestion(
                        "s221",
                        "Do one final OCR attempt on the refreshed crop.",
                        [
                            make_step(
                                "step_final_ocr",
                                "Run one final OCR pass on the refreshed crop.",
                                ["ocr_assist"],
                                input_image_index=2,
                            )
                        ],
                    )
                ],
            ),
        (traj_s221, 3): make_planner_output(
                sample_id=sample_id,
                trajectory_id=traj_s221,
                round_idx=3,
                global_chain_cot="Forced final-answer round on s221: the refreshed crop remains ambiguous, but the best supported answer is 249.",
                direct_answer="249",
            ),
    }

    executor_outputs = {
        (traj_s1, 1): make_executor_output(
                "Use grounding and cropping so the next planner round can answer directly if the crop is clean.",
                (
                    'box = _call_ground_box("hanging tag")\n'
                    'crop = _call_dino_crop("hanging tag", image_obj=box["image"], based_on="box", max_crops=1, padding=6)\n'
                    'print(crop.get("text", ""))\n'
                    'result = crop["image"]\n'
                ),
            ),
        (traj_s2, 1): make_executor_output(
                "Ground the tag and keep the crop so the next branch can continue from the visible tool output.",
                (
                    'box = _call_ground_box("price tag")\n'
                    'crop = _call_dino_crop("price tag", image_obj=box["image"], based_on="box", max_crops=1, padding=4)\n'
                    'print(crop.get("text", ""))\n'
                    'result = crop["image"]\n'
                ),
            ),
        (traj_s3, 1): make_executor_output(
                "Mask the competing region first, then use OCR on top of the masked result.",
                (
                    'mask = _call_sam_mask("candidate tag", multimask_output=False)\n'
                    'ocr = _call_ocr_assist(image_obj=mask["image"])\n'
                    'print(ocr.get("text", ""))\n'
                    'result = mask["image"]\n'
                ),
            ),
        (traj_s21, 2): make_executor_output(
                "Continue from the previous crop and OCR it directly because the latest tool result is visible.",
                (
                    'ocr = _call_ocr_assist(image_obj=image)\n'
                    'print(ocr.get("text", ""))\n'
                    'result = image\n'
                ),
            ),
        (traj_s22, 2): make_executor_output(
                "Reground the tag, recrop it, and OCR the refreshed crop to revise the earlier tool choice.",
                (
                    'box = _call_ground_box("price tag")\n'
                    'crop = _call_dino_crop("price tag", image_obj=box["image"], based_on="box", max_crops=1, padding=2)\n'
                    'ocr = _call_ocr_assist(image_obj=crop["image"])\n'
                    'print(ocr.get("text", ""))\n'
                    'result = crop["image"]\n'
                ),
            ),
        (traj_s31, 2): make_executor_output(
                "Retry the mask-guided path once more and see whether the branch should die.",
                (
                    'mask = _call_sam_mask("candidate tag", multimask_output=False)\n'
                    'crop = _call_dino_crop("candidate tag", image_obj=mask["image"], based_on="mask", max_crops=1, padding=4)\n'
                    'result = crop["image"]\n'
                ),
            ),
        (traj_s221, 3): make_executor_output(
                "Use one last OCR pass on the refreshed crop before step budget runs out.",
                (
                    'ocr = _call_ocr_assist(image_obj=image)\n'
                    'print(ocr.get("text", ""))\n'
                    'result = image\n'
                ),
            ),
    }

    runtime_specs = {
        (traj_s1, 1): RuntimeSpec(
                text="root crop isolated the tag",
                helper_names=["ground_box", "dino_crop"],
                image_label="s1_step1",
            ),
        (traj_s2, 1): RuntimeSpec(
                text="cropped serial region",
                helper_names=["ground_box", "dino_crop"],
                image_label="s2_step1",
            ),
        (traj_s3, 1): RuntimeSpec(
                text="masked competing region",
                helper_names=["sam_mask", "ocr_assist"],
                image_label="s3_step1",
            ),
        (traj_s21, 2): RuntimeSpec(
                text="ocr=249",
                helper_names=["ocr_assist"],
                image_label="s21_step2",
            ),
        (traj_s22, 2): RuntimeSpec(
                text="reframed crop still ambiguous",
                helper_names=["ground_box", "dino_crop", "ocr_assist"],
                image_label="s22_step2",
            ),
        (traj_s31, 2): RuntimeSpec(
                text="",
                helper_names=["sam_mask", "dino_crop"],
                image_label="s31_step2",
                success=False,
                error_message="mock runtime failure on retry path",
            ),
        (traj_s221, 3): RuntimeSpec(
                text="still ambiguous after final OCR",
                helper_names=["ocr_assist"],
                image_label="s221_step3",
            ),
    }

    judge_scores = {
        (root_id, 0): 0.20,
        (traj_s1, 1): 0.95,
        (traj_s2, 1): 0.82,
        (traj_s3, 1): 0.41,
        (traj_s21, 2): 0.91,
        (traj_s22, 2): 0.88,
        (traj_s31, 2): 0.10,
        (traj_s221, 3): 0.94,
    }

    config = OrchestratorConfig(
        planner_suggestion_count=3,
        max_child_trajectories=2,
        must_suggest_score_threshold=0.6,
        must_answer_score_threshold=0.9,
        default_budget=Budget(
            remaining_exec_steps=6,
        ),
    )

    return ThreeRoundDemoSpec(
        sample=sample,
        planner_outputs=planner_outputs,
        executor_outputs=executor_outputs,
        runtime_specs=runtime_specs,
        judge_scores=judge_scores,
        config=config,
        scenario_notes=[
            "Round 0 forks three child trajectories from the root, then post-judge frontier keeps only top-2.",
            "Round 1 keeps the direct-answer path and the strongest continuing branch.",
            "Round 2 exercises a continuing branch that improves again under the new stop policy.",
        ],
    )


def build_three_round_demo_scenario(
    root_dir: Path,
    *,
    sample_id: str = "demo__train__0001",
) -> ScriptedScenario:
    """Build the shared fake 3-round scenario used by docs/tests/scripted entry."""

    spec = build_three_round_demo_spec(root_dir, sample_id=sample_id)
    return ScriptedScenario(
        sample=spec.sample,
        planner_client=ScriptedPlannerClient(spec.planner_outputs),
        executor_client=ScriptedExecutorClient(spec.executor_outputs),
        runtime=ScriptedRuntime(spec.runtime_specs),
        judge_backend=ScriptedJudgeBackend(spec.judge_scores),
        config=spec.config,
        fake_components={
            "planner_client": "ScriptedPlannerClient returns pre-authored PlannerOutput objects.",
            "executor_client": "ScriptedExecutorClient returns pre-authored ExecutorStepOutput objects.",
            "runtime": "ScriptedRuntime synthesizes runtime_result.json, stdout/stderr, and output images without executing tools.",
            "judge_backend": "ScriptedJudgeBackend returns pre-authored overall_score values.",
        },
        scenario_notes=list(spec.scenario_notes),
    )


__all__ = [
    "RuntimeSpec",
    "ScriptedTextBackend",
    "ThreeRoundDemoSpec",
    "build_three_round_demo_spec",
    "ScriptedExecutorClient",
    "ScriptedJudgeBackend",
    "ScriptedPlannerClient",
    "ScriptedRuntime",
    "ScriptedScenario",
    "build_demo_root_sample",
    "build_three_round_demo_scenario",
    "make_executor_output",
    "make_planner_output",
    "make_step",
    "make_suggestion",
    "write_demo_image",
]
