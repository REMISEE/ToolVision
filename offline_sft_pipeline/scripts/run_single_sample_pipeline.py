from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offline_sft_pipeline.core.store import OfflineTrajectoryStore
from offline_sft_pipeline.pipelines.executor_client import ExecutorClient
from offline_sft_pipeline.pipelines.judge_client import JudgeClient
from offline_sft_pipeline.pipelines.orchestrator_v01 import OrchestratorV01
from offline_sft_pipeline.pipelines.planner_client import PlannerClient
from offline_sft_pipeline.pipelines.scripted_components import (
    ScriptedJudgeBackend,
    ScriptedRuntime,
    ScriptedTextBackend,
    build_three_round_demo_scenario,
    build_three_round_demo_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one single-sample offline pipeline demo. "
            "Supported modes: scripted and client_fake_backend."
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="client_fake_backend",
        choices=["scripted", "client_fake_backend"],
        help=(
            "scripted uses fake planner/executor/runtime/judge components; "
            "client_fake_backend uses real planner/executor/judge clients with fake backend responses."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "offline_sft_pipeline" / "outputs" / "scripted_sample_pipeline_runs"),
        help="Base directory that will contain one run folder.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional fixed run id. If omitted, a UTC timestamp-based id is generated.",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default="demo__train__0001",
        help="Sample id to stamp into the scripted demo sample.",
    )
    parser.add_argument(
        "--print-tree",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include a relative file tree preview in stdout and summary.json.",
    )
    parser.add_argument(
        "--tree-max-depth",
        type=int,
        default=6,
        help="Maximum depth to include in the tree preview.",
    )
    parser.add_argument("--ocr-base-url", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--grounded-sam2-base-url", type=str, default="http://127.0.0.1:8081")
    parser.add_argument("--ocr-model-name", type=str, default="paddleocr")
    parser.add_argument("--service-timeout", type=int, default=180)
    parser.add_argument(
        "--enable-external-model-functions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the runtime wrapper should expose OCR and grounded_sam2 helper functions.",
    )
    parser.add_argument(
        "--runtime-mode",
        type=str,
        default="scripted",
        choices=["scripted", "code_image_tool"],
        help=(
            "Runtime wiring for client_fake_backend mode. "
            "scripted skips real runtime execution; code_image_tool uses CodeImageRuntimeWrapper."
        ),
    )
    return parser.parse_args()


def build_run_id(raw_run_id: str, *, mode: str) -> str:
    text = str(raw_run_id or "").strip()
    if text:
        return text
    prefix = f"{mode}__"
    return prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def collect_tree_preview(root: Path, *, max_depth: int) -> list[str]:
    preview: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if len(relative.parts) > max_depth:
            continue
        suffix = "/" if path.is_dir() else ""
        preview.append(relative.as_posix() + suffix)
    return preview


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

    output_base = Path(args.output_dir).expanduser().resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    run_id = build_run_id(args.run_id, mode=args.mode)
    run_root = output_base / run_id
    if run_root.exists():
        raise FileExistsError(f"run directory already exists: {run_root}")

    inputs_dir = run_root / "inputs"
    store_run_root = run_root / "store"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    store = OfflineTrajectoryStore(store_run_root)
    runtime_component = None
    sample = None
    config = None
    planner_client = None
    executor_client = None
    judge_client = None
    fake_components: dict[str, str]
    component_wiring: dict[str, str]
    scenario_notes: list[str]

    try:
        if args.mode == "scripted":
            # This scenario is fully fake/scripted. It is useful for verifying pipeline
            # structure and multi-round orchestration semantics without model APIs.
            scenario = build_three_round_demo_scenario(inputs_dir, sample_id=args.sample_id)
            sample = scenario.sample
            planner_client = scenario.planner_client
            executor_client = scenario.executor_client
            judge_client = JudgeClient(backend=scenario.judge_backend)
            runtime_component = scenario.runtime
            config = scenario.config
            fake_components = dict(scenario.fake_components)
            component_wiring = {
                "planner_client": type(planner_client).__name__,
                "executor_client": type(executor_client).__name__,
                "judge_client": type(judge_client).__name__,
                "runtime": type(runtime_component).__name__,
            }
            scenario_notes = list(scenario.scenario_notes)
        else:
            spec = build_three_round_demo_spec(inputs_dir, sample_id=args.sample_id)
            text_backend = ScriptedTextBackend(
                planner_outputs=spec.planner_outputs,
                executor_outputs=spec.executor_outputs,
            )
            planner_client = PlannerClient(backend=text_backend)
            executor_client = ExecutorClient(backend=text_backend)
            judge_client = JudgeClient(backend=ScriptedJudgeBackend(spec.judge_scores))
            if args.runtime_mode == "scripted":
                runtime_component = ScriptedRuntime(spec.runtime_specs)
            else:
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
            sample = spec.sample
            config = spec.config
            fake_components = {
                "planner_backend": "ScriptedTextBackend returns pre-authored planner text to PlannerClient.",
                "executor_backend": "ScriptedTextBackend returns pre-authored executor text to ExecutorClient.",
                "judge_backend": "ScriptedJudgeBackend returns pre-authored overall_score values to JudgeClient.",
            }
            if args.runtime_mode == "scripted":
                fake_components["runtime"] = "ScriptedRuntime synthesizes runtime_result.json and artifacts without executing helpers."
            component_wiring = {
                "planner_client": type(planner_client).__name__,
                "executor_client": type(executor_client).__name__,
                "judge_client": type(judge_client).__name__,
                "runtime": type(runtime_component).__name__,
            }
            scenario_notes = list(spec.scenario_notes) + [
                "Planner/executor prompts and parsers now go through real clients.",
            ]
            if args.runtime_mode == "scripted":
                scenario_notes.append(
                    "Runtime is intentionally kept scripted in this run so the focus stays on backend->client->orchestrator wiring."
                )
            else:
                scenario_notes.append(
                    "Runtime is wired through CodeImageRuntimeWrapper, so this mode expects the runtime helper stack to be available."
                )

        orchestrator = OrchestratorV01(
            store=store,
            planner_client=planner_client,
            executor_client=executor_client,
            judge_client=judge_client,
            runtime=runtime_component,
            config=config,
        )
        result = orchestrator.run(sample)
    finally:
        close_sync = getattr(runtime_component, "close_sync", None)
        if callable(close_sync):
            close_sync()

    tree_preview = collect_tree_preview(run_root, max_depth=max(1, int(args.tree_max_depth))) if args.print_tree else []
    summary = {
        "mode": args.mode,
        "run_id": run_id,
        "run_root": str(run_root),
        "inputs_dir": str(inputs_dir),
        "store_run_root": str(store_run_root),
        "sample_id": sample.sample_id,
        "question": sample.question,
        "runtime_mode": args.runtime_mode if args.mode == "client_fake_backend" else "scripted",
        "component_wiring": component_wiring,
        "fake_components": fake_components,
        "scenario_notes": scenario_notes,
        "result": {
            "sample_id": result.sample_id,
            "root_trajectory_id": result.root_trajectory_id,
            "all_trajectory_ids": result.all_trajectory_ids,
            "running_trajectory_ids": result.running_trajectory_ids,
            "expanded_trajectory_ids": result.expanded_trajectory_ids,
            "terminal_trajectory_ids": result.terminal_trajectory_ids,
        },
        "inspect_files": {
            "root_sample": str(store.root_sample_path(sample.sample_id)),
            "root_trajectory": str(store.trajectory_path(sample.sample_id, result.root_trajectory_id)),
            "root_messages": str(store.messages_path(sample.sample_id, result.root_trajectory_id)),
        },
        "tree_preview": tree_preview,
    }

    summary_path = run_root / "scripted_run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
