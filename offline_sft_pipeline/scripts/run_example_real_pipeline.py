#!/usr/bin/env python3
"""Run OrchestratorV01 with real planner + real executor + real runtime.

This is the first practical end-to-end integration path:

- planner: real HTTP API via ``ApiTextBackend``
- executor: real HTTP API via ``ApiTextBackend``
- runtime: ``CodeImageRuntimeWrapper`` backed by OCR + GroundedSAM2 HTTP services
- judge: fake backend for now

It reads ``offline_sft_pipeline/example/question.json`` and materializes one sample
into the offline trajectory store, then runs the orchestrator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offline_sft_pipeline.core.models import Budget, RootImage, RootSample
from offline_sft_pipeline.core.store import OfflineTrajectoryStore
from offline_sft_pipeline.pipelines.backends import ApiTextBackend, FakeJudgeBackend
from offline_sft_pipeline.pipelines.executor_client import ExecutorClient
from offline_sft_pipeline.pipelines.judge_client import JudgeClient
from offline_sft_pipeline.pipelines.orchestrator_v01 import OrchestratorConfig, OrchestratorV01
from offline_sft_pipeline.pipelines.planner_client import PlannerClient
from offline_sft_pipeline.pipelines.tool_capabilities_io import load_tool_capabilities_from_file

EXAMPLE_DIR = PIPELINE_ROOT / "example"
DEFAULT_QUESTION_FILE = "question.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run example/question.json through real planner + real executor + real runtime."
    )
    p.add_argument(
        "--planner-debug",
        action="store_true",
        help="Set OFFLINE_SFT_PLANNER_DEBUG=1 for this process.",
    )
    p.add_argument(
        "--executor-debug",
        action="store_true",
        help="Set OFFLINE_SFT_EXECUTOR_DEBUG=1 for this process.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(PIPELINE_ROOT / "outputs" / "example_real_pipeline_runs"),
        help="Base directory; one run folder is created inside.",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run folder name. Default: utc timestamp prefix example_real_pipeline__YYYYMMDDTHHMMSSZ",
    )
    p.add_argument(
        "--sample-id",
        type=str,
        default="example__weight_plates",
        help="sample_id stored under store/samples/<sample_id>/",
    )
    p.add_argument(
        "--question-file",
        type=str,
        default=DEFAULT_QUESTION_FILE,
        help="Question JSON filename under offline_sft_pipeline/example/.",
    )
    p.add_argument(
        "--planner-system-prompt-file",
        type=str,
        default="planner_system_v07.txt",
        help="Planner system prompt filename under offline_sft_pipeline/prompts/.",
    )
    p.add_argument(
        "--executor-system-prompt-file",
        type=str,
        default="executor_system_v05.txt",
        help="Executor system prompt filename under offline_sft_pipeline/prompts/.",
    )
    p.add_argument("--ocr-base-url", type=str, default="http://127.0.0.1:8080")
    p.add_argument("--grounded-sam2-base-url", type=str, default="http://127.0.0.1:8081")
    p.add_argument("--ocr-model-name", type=str, default="paddleocr")
    p.add_argument("--service-timeout", type=int, default=180)
    p.add_argument(
        "--enable-external-model-functions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether CodeImageRuntimeWrapper should expose OCR and grounded_sam2 helpers.",
    )
    p.add_argument(
        "--judge-score",
        type=float,
        default=0.75,
        help="Fake judge overall score to stamp into JudgeRecord outputs.",
    )
    return p.parse_args()


def build_run_id(raw: str) -> str:
    t = str(raw or "").strip()
    if t:
        return t
    return "example_real_pipeline__" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_example_question_and_image(question_file: str) -> tuple[Path, str, Path]:
    question_path = (EXAMPLE_DIR / str(question_file)).resolve()
    if not question_path.is_file():
        raise FileNotFoundError(f"missing {question_path}")
    payload = json.loads(question_path.read_text(encoding="utf-8"))
    q = str(payload.get("question") or "").strip()
    rel = payload.get("image")
    if not q or not rel:
        raise ValueError(f"{question_path} must contain non-empty 'question' and 'image'")
    img = (EXAMPLE_DIR / str(rel)).resolve()
    if not img.is_file():
        raise FileNotFoundError(f"example image not found: {img}")
    return question_path, q, img


def build_external_services_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "paddleocr": {
            "base_url": args.ocr_base_url,
            "request_timeout": args.service_timeout,
        },
        "grounded_sam2": {
            "base_url": args.grounded_sam2_base_url,
            "request_timeout": args.service_timeout,
        },
    }


def main() -> None:
    args = parse_args()
    if args.planner_debug:
        os.environ["OFFLINE_SFT_PLANNER_DEBUG"] = "1"
    if args.executor_debug:
        os.environ["OFFLINE_SFT_EXECUTOR_DEBUG"] = "1"

    run_id = build_run_id(args.run_id)
    output_base = Path(args.output_dir).expanduser().resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    run_root = output_base / run_id
    if run_root.exists():
        raise FileExistsError(f"run directory already exists: {run_root}")
    run_root.mkdir(parents=True)

    store_root = run_root / "store"
    question_path, question, image_path = load_example_question_and_image(args.question_file)

    sample = RootSample(
        sample_id=args.sample_id,
        question=question,
        images=[RootImage(image_id="root_0", path=str(image_path))],
        metadata={"source": f"offline_sft_pipeline/example/{Path(question_path).name}"},
    )

    caps = load_tool_capabilities_from_file()
    config = OrchestratorConfig(
        tool_capabilities=caps,
        planner_suggestion_count=2,
        default_budget=Budget(remaining_exec_steps=6),
    )

    from offline_sft_pipeline.runtime.code_image_runtime_wrapper import (
        CodeImageRuntimeWrapper,
        build_default_code_image_tool_config,
    )

    runtime_component = CodeImageRuntimeWrapper(
        build_default_code_image_tool_config(
            enable_external_model_functions=args.enable_external_model_functions,
            external_services=build_external_services_config(args),
            ocr_model_name=args.ocr_model_name,
        )
    )

    try:
        orchestrator = OrchestratorV01(
            store=OfflineTrajectoryStore(store_root),
            planner_client=PlannerClient(
                backend=ApiTextBackend(),
                system_prompt_filename=args.planner_system_prompt_file,
            ),
            executor_client=ExecutorClient(
                backend=ApiTextBackend(),
                system_prompt_filename=args.executor_system_prompt_file,
            ),
            judge_client=JudgeClient(backend=FakeJudgeBackend(overall_score=args.judge_score)),
            runtime=runtime_component,
            config=config,
        )
        result = orchestrator.run(sample)
    finally:
        runtime_component.close_sync()

    sample_dir = store_root / "samples" / sample.sample_id
    summary = {
        "run_id": run_id,
        "run_root": str(run_root),
        "store_root": str(store_root),
        "example_question_path": str(question_path),
        "example_image": str(image_path),
        "sample_id": result.sample_id,
        "root_trajectory_id": result.root_trajectory_id,
        "all_trajectory_ids": result.all_trajectory_ids,
        "terminal_trajectory_ids": result.terminal_trajectory_ids,
        "component_wiring": {
            "planner_client": "PlannerClient(ApiTextBackend)",
            "executor_client": "ExecutorClient(ApiTextBackend)",
            "runtime": "CodeImageRuntimeWrapper",
            "judge_client": "JudgeClient(FakeJudgeBackend)",
        },
        "runtime_services": {
            "ocr_base_url": args.ocr_base_url,
            "grounded_sam2_base_url": args.grounded_sam2_base_url,
            "ocr_model_name": args.ocr_model_name,
            "service_timeout": args.service_timeout,
        },
        "prompt_files": {
            "planner_system_prompt_file": args.planner_system_prompt_file,
            "executor_system_prompt_file": args.executor_system_prompt_file,
        },
        "inspect": {
            "samples": str(sample_dir),
            "hint_planner_round_json": "Under each trajectories/<traj_id>/planner/round_*.json",
            "hint_messages_json": "Under each trajectories/<traj_id>/messages.json",
            "hint_step_runtime_result_json": "Under each trajectories/<traj_id>/steps/step_*/runtime_result.json",
        },
    }
    out_json = run_root / "run_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
