#!/usr/bin/env python3
"""Run OrchestratorV01 over a RootSample JSONL dataset."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offline_sft_pipeline.core.models import Budget, JudgeRecord, RootSample, build_root_trajectory_id
from offline_sft_pipeline.core.dataset_names import canonicalize_dataset_name, is_high_conf_exact_match_dataset
from offline_sft_pipeline.core.sample_normalization import normalize_root_sample
from offline_sft_pipeline.core.store import OfflineTrajectoryStore
from offline_sft_pipeline.eval.scorers import score_answer_for_dataset
from offline_sft_pipeline.pipelines.backends import (
    DEFAULT_JUDGE_MAX_CONCURRENCY,
    DEFAULT_JUDGE_MODELS_PATH,
    ApiTextBackend,
    CommitteeJudgeBackend,
    FakeJudgeBackend,
)
from offline_sft_pipeline.pipelines.executor_client import ExecutorClient
from offline_sft_pipeline.pipelines.judge_client import JudgeClient
from offline_sft_pipeline.pipelines.orchestrator_v01 import OrchestratorConfig, OrchestratorV01
from offline_sft_pipeline.pipelines.planner_client import PlannerClient
from offline_sft_pipeline.pipelines.tool_capabilities_io import load_tool_capabilities_from_file


def round_seconds(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def build_sample_timing_summary(
    *,
    run_id: str,
    run_root: Path,
    input_jsonl: Path,
    run_started_at: datetime,
    run_finished_at: datetime,
    sample_timing_records: list[dict[str, object]],
    counts: dict[str, int],
) -> dict[str, object]:
    started_records = [record for record in sample_timing_records if record.get("elapsed_seconds") is not None]
    elapsed_values = [
        float(record["elapsed_seconds"])
        for record in started_records
        if isinstance(record.get("elapsed_seconds"), (int, float))
    ]
    total_sample_elapsed_seconds = sum(elapsed_values)
    wall_clock_seconds = max(0.0, (run_finished_at - run_started_at).total_seconds())
    slowest_examples = sorted(
        started_records,
        key=lambda record: float(record.get("elapsed_seconds") or 0.0),
        reverse=True,
    )[:10]
    return {
        "run_id": run_id,
        "run_root": str(run_root),
        "input_jsonl": str(input_jsonl),
        "run_timing": {
            "started_at": run_started_at.isoformat(),
            "finished_at": run_finished_at.isoformat(),
            "wall_clock_seconds": round_seconds(wall_clock_seconds),
            "wall_clock_minutes": round_seconds(wall_clock_seconds / 60.0),
            "total_sample_elapsed_seconds": round_seconds(total_sample_elapsed_seconds),
            "total_sample_elapsed_minutes": round_seconds(total_sample_elapsed_seconds / 60.0),
            "average_sample_elapsed_seconds": round_seconds(
                total_sample_elapsed_seconds / len(elapsed_values) if elapsed_values else None
            ),
            "average_sample_elapsed_minutes": round_seconds(
                (total_sample_elapsed_seconds / len(elapsed_values)) / 60.0 if elapsed_values else None
            ),
        },
        "counts": counts,
        "slowest_examples": slowest_examples,
        "examples": sample_timing_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a RootSample JSONL file through real planner + real executor + real runtime."
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        required=True,
        help="Path to a JSONL file where each line validates as RootSample.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PIPELINE_ROOT / "outputs" / "dataset_pipeline_runs"),
        help="Base directory containing run folders.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run folder name. Defaults to dataset_pipeline__YYYYMMDDTHHMMSSZ.",
    )
    parser.add_argument(
        "--sample-ids-file",
        type=str,
        default="",
        help="Optional newline-delimited list of sample_id values to run.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on the number of selected samples to start. 0 means no cap.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse an existing run_root and skip samples whose root trajectory already exists.",
    )
    parser.add_argument(
        "--planner-debug",
        action="store_true",
        help="Set OFFLINE_SFT_PLANNER_DEBUG=1 for this process.",
    )
    parser.add_argument(
        "--executor-debug",
        action="store_true",
        help="Set OFFLINE_SFT_EXECUTOR_DEBUG=1 for this process.",
    )
    parser.add_argument(
        "--planner-system-prompt-file",
        type=str,
        default="planner_system_v07.txt",
        help="Planner system prompt filename under offline_sft_pipeline/prompts/.",
    )
    parser.add_argument(
        "--executor-system-prompt-file",
        type=str,
        default="executor_system_v05.txt",
        help="Executor system prompt filename under offline_sft_pipeline/prompts/.",
    )
    parser.add_argument("--ocr-base-url", type=str, default="http://127.0.0.1:28080")
    parser.add_argument("--grounded-sam2-base-url", type=str, default="http://127.0.0.1:28081")
    parser.add_argument("--depth-base-url", type=str, default="http://127.0.0.1:28082")
    parser.add_argument("--countgd-base-url", type=str, default="http://127.0.0.1:28083")
    parser.add_argument("--ocr-model-name", type=str, default="paddleocr")
    parser.add_argument("--service-timeout", type=int, default=200)
    parser.add_argument(
        "--enable-external-model-functions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether CodeImageRuntimeWrapper should expose OCR, grounded_sam2, depth, and countgd helpers.",
    )
    parser.add_argument(
        "--judge-backend",
        type=str,
        default="committee",
        choices=("committee", "fake"),
        help="Judge backend to use for hot-path scoring.",
    )
    parser.add_argument(
        "--judge-models-file",
        type=str,
        default=str(DEFAULT_JUDGE_MODELS_PATH),
        help="JSON file describing enabled judge models for CommitteeJudgeBackend.",
    )
    parser.add_argument(
        "--judge-max-concurrency",
        type=int,
        default=DEFAULT_JUDGE_MAX_CONCURRENCY,
        help="Maximum number of committee judge model calls to execute in parallel.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=6,
        help="Maximum executor tool-use turns per trajectory.",
    )
    parser.add_argument(
        "--ray-num-cpus",
        type=int,
        default=8,
        help="CPU resources exposed to the per-process Ray runtime used by CodeImageRuntimeWrapper.",
    )
    parser.add_argument(
        "--judge-score",
        type=float,
        default=0.75,
        help="Fake judge overall score to stamp into JudgeRecord outputs when --judge-backend=fake.",
    )
    return parser.parse_args()


def build_run_id(raw_run_id: str) -> str:
    text = str(raw_run_id or "").strip()
    if text:
        return text
    return "dataset_pipeline__" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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
        "depth": {
            "base_url": args.depth_base_url,
            "request_timeout": args.service_timeout,
        },
        "countgd": {
            "base_url": args.countgd_base_url,
            "request_timeout": args.service_timeout,
        },
    }


def load_selected_sample_ids(path: str) -> set[str]:
    if not str(path or "").strip():
        return set()
    sample_ids: set[str] = set()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if text:
            sample_ids.add(text)
    return sample_ids


def budget_for_sample(sample: RootSample, *, default_budget: Budget) -> Budget:
    dataset_name = canonicalize_dataset_name(sample.metadata.get("source_dataset"))
    if dataset_name == "fsc147":
        return Budget(remaining_exec_steps=default_budget.remaining_exec_steps)
    if is_high_conf_exact_match_dataset(dataset_name):
        return Budget(remaining_exec_steps=default_budget.remaining_exec_steps)
    return default_budget.model_copy(deep=True)


def _resolve_image_path(raw_path: str, *, base_dir: Path) -> Path:
    path_obj = Path(raw_path).expanduser()
    if path_obj.is_absolute():
        return path_obj.resolve()

    candidate_roots: list[Path] = [base_dir, *base_dir.parents]
    seen: set[Path] = set()
    for root in candidate_roots:
        if root in seen:
            continue
        seen.add(root)
        candidate = (root / path_obj).resolve()
        if candidate.exists():
            return candidate
    return (base_dir / path_obj).resolve()


def _resolve_root_sample_image_paths(payload: dict[str, object], *, base_dir: Path) -> dict[str, object]:
    images = payload.get("images")
    if not isinstance(images, list):
        return payload
    normalized = json.loads(json.dumps(payload, ensure_ascii=False))
    norm_images = normalized.get("images")
    if not isinstance(norm_images, list):
        return normalized
    for image in norm_images:
        if not isinstance(image, dict):
            continue
        raw_path = str(image.get("path", "")).strip()
        if not raw_path:
            continue
        image["path"] = str(_resolve_image_path(raw_path, base_dir=base_dir))
    return normalized


def iter_root_samples(path: Path) -> Iterator[RootSample]:
    base_dir = path.parent.resolve()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue
            payload = json.loads(text)
            payload = _resolve_root_sample_image_paths(payload, base_dir=base_dir)
            if "answer" in payload and payload["answer"] is not None:
                if isinstance(payload["answer"], list):
                    payload["answer"] = [str(item) for item in payload["answer"]]
                else:
                    payload["answer"] = str(payload["answer"])
            try:
                yield normalize_root_sample(RootSample.model_validate(payload))
            except Exception as exc:
                raise ValueError(f"Invalid RootSample JSONL line {line_no} in {path}: {exc}") from exc


def root_trajectory_exists(store_root: Path, sample_id: str) -> bool:
    trajectory_id = build_root_trajectory_id(sample_id)
    trajectory_path = (
        store_root
        / "samples"
        / sample_id
        / "trajectories"
        / trajectory_id
        / "trajectory.json"
    )
    return trajectory_path.is_file()


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def terminal_answered_ids(store: OfflineTrajectoryStore, sample_id: str) -> list[str]:
    answered: list[str] = []
    for trajectory in store.list_trajectories(sample_id=sample_id):
        if trajectory.status == "answered":
            answered.append(trajectory.trajectory_id)
    return sorted(answered)


def terminal_trajectories(store: OfflineTrajectoryStore, sample_id: str) -> list[object]:
    terminal_statuses = {"answered", "pruned", "failed", "stopped_early", "max_step_reached", "error"}
    return [
        trajectory
        for trajectory in store.list_trajectories(sample_id=sample_id)
        if trajectory.status in terminal_statuses
    ]


def _latest_judge_score_for_trajectory(
    store: OfflineTrajectoryStore,
    *,
    sample_id: str,
    trajectory_id: str,
) -> float | None:
    trajectory_dir = store.trajectory_dir(sample_id, trajectory_id)
    local_judge_dir = trajectory_dir / "judge"
    if local_judge_dir.is_dir():
        local_judge_files = sorted(local_judge_dir.glob("*.json"))
        if local_judge_files:
            judge_record = JudgeRecord.from_json_file(local_judge_files[-1])
            return float(judge_record.overall_score)

    trajectory = store.load_trajectory(sample_id, trajectory_id)
    for judge_ref in reversed(trajectory.judge_records):
        judge_path = _resolve_trajectory_relative_path(
            store,
            sample_id=sample_id,
            trajectory_id=trajectory_id,
            path_value=judge_ref.judge_record_path,
        )
        judge_record = JudgeRecord.from_json_file(judge_path)
        return float(judge_record.overall_score)
    return None


def build_answered_results_record(
    store: OfflineTrajectoryStore,
    *,
    sample: RootSample,
    terminal_trajectory_ids: list[str],
    answered_trajectory_ids: list[str],
) -> dict[str, object]:
    preds: list[str | None] = []
    final_scores: list[float | None] = []
    latest_judge_scores: list[float | None] = []
    for trajectory_id in answered_trajectory_ids:
        trajectory = store.load_trajectory(sample.sample_id, trajectory_id)
        preds.append(trajectory.final_answer)
        latest_judge_scores.append(
            _latest_judge_score_for_trajectory(
                store,
                sample_id=sample.sample_id,
                trajectory_id=trajectory_id,
            )
        )
        if trajectory.final_answer is None or sample.answer is None:
            final_scores.append(None)
        else:
            source_dataset = str(sample.metadata.get("source_dataset") or "")
            score_result = score_answer_for_dataset(
                source_dataset,
                trajectory.final_answer,
                sample.answer,
                metadata=dict(sample.metadata),
            )
            final_scores.append(float(score_result.score))
    return {
        "sample_id": sample.sample_id,
        "terminal_trajectory_ids": list(terminal_trajectory_ids),
        "answered_trajectory_ids": list(answered_trajectory_ids),
        "pred": preds,
        "answer": sample.answer,
        "score": latest_judge_scores,
        "latest_judge_score": latest_judge_scores,
        "final_score": final_scores,
    }


def _resolve_trajectory_relative_path(
    store: OfflineTrajectoryStore,
    *,
    sample_id: str,
    trajectory_id: str,
    path_value: str,
) -> Path:
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return (store.trajectory_dir(sample_id, trajectory_id) / path_obj).resolve()


def _normalize_token_usage(metadata: dict[str, object] | None) -> dict[str, int]:
    if not isinstance(metadata, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
    raw_usage = metadata.get("token_usage")
    if not isinstance(raw_usage, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}

    def _coerce_int(key: str, fallback_key: str | None = None) -> int:
        value = raw_usage.get(key)
        if not isinstance(value, (int, float)) and fallback_key is not None:
            value = raw_usage.get(fallback_key)
        if not isinstance(value, (int, float)):
            return 0
        return int(value)

    prompt_tokens = _coerce_int("prompt_tokens", "input_tokens")
    completion_tokens = _coerce_int("completion_tokens", "output_tokens")
    total_tokens = _coerce_int("total_tokens")
    cached_tokens = _coerce_int("cached_tokens")
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
    }


def _empty_usage_totals() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
    }


def _accumulate_usage(target: dict[str, int], usage: dict[str, int]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"):
        target[key] = int(target.get(key, 0)) + int(usage.get(key, 0))


def build_sample_token_usage_summary(store: OfflineTrajectoryStore, sample_id: str) -> dict[str, object]:
    stage_totals = {
        "planner": _empty_usage_totals(),
        "executor": _empty_usage_totals(),
        "judge": _empty_usage_totals(),
    }
    sample_total = _empty_usage_totals()
    trajectory_totals: dict[str, dict[str, object]] = {}

    seen_planner: set[tuple[str, int]] = set()
    seen_executor: set[tuple[str, int]] = set()
    seen_judge: set[str] = set()

    for trajectory in store.list_trajectories(sample_id=sample_id):
        per_traj_stages = {
            "planner": _empty_usage_totals(),
            "executor": _empty_usage_totals(),
            "judge": _empty_usage_totals(),
        }
        per_traj_total = _empty_usage_totals()

        for planner_item in trajectory.planner_history:
            planner_output = store.load_planner_output(sample_id, trajectory.trajectory_id, planner_item.round_idx)
            usage = _normalize_token_usage(planner_output.metadata)
            _accumulate_usage(per_traj_stages["planner"], usage)
            _accumulate_usage(per_traj_total, usage)
            planner_key = (planner_output.trajectory_id, planner_output.round_idx)
            if planner_key not in seen_planner:
                seen_planner.add(planner_key)
                _accumulate_usage(stage_totals["planner"], usage)
                _accumulate_usage(sample_total, usage)

        for step_record in trajectory.steps:
            usage = _normalize_token_usage(step_record.executor_metadata)
            _accumulate_usage(per_traj_stages["executor"], usage)
            _accumulate_usage(per_traj_total, usage)
            executor_key = (step_record.execution_trajectory_id, step_record.step_idx)
            if executor_key not in seen_executor:
                seen_executor.add(executor_key)
                _accumulate_usage(stage_totals["executor"], usage)
                _accumulate_usage(sample_total, usage)

        for judge_ref in trajectory.judge_records:
            judge_path = _resolve_trajectory_relative_path(
                store,
                sample_id=sample_id,
                trajectory_id=trajectory.trajectory_id,
                path_value=judge_ref.judge_record_path,
            )
            judge_record = JudgeRecord.from_json_file(judge_path)
            usage = _normalize_token_usage(judge_record.metadata)
            _accumulate_usage(per_traj_stages["judge"], usage)
            _accumulate_usage(per_traj_total, usage)
            if judge_record.judge_record_id not in seen_judge:
                seen_judge.add(judge_record.judge_record_id)
                _accumulate_usage(stage_totals["judge"], usage)
                _accumulate_usage(sample_total, usage)

        trajectory_totals[trajectory.trajectory_id] = {
            "status": trajectory.status,
            "stages": per_traj_stages,
            "total": per_traj_total,
        }

    return {
        "sample_id": sample_id,
        "sample_total": sample_total,
        "stages": stage_totals,
        "trajectory_totals": trajectory_totals,
    }


def main() -> None:
    args = parse_args()
    if args.planner_debug:
        os.environ["OFFLINE_SFT_PLANNER_DEBUG"] = "1"
    if args.executor_debug:
        os.environ["OFFLINE_SFT_EXECUTOR_DEBUG"] = "1"

    input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    if not input_jsonl.is_file():
        raise FileNotFoundError(f"input JSONL not found: {input_jsonl}")

    run_id = build_run_id(args.run_id)
    output_base = Path(args.output_dir).expanduser().resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    run_root = output_base / run_id
    if run_root.exists() and not args.resume:
        raise FileExistsError(f"run directory already exists: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)

    store_root = run_root / "store"
    logs_root = run_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    sample_results_path = run_root / "sample_results.jsonl"
    answered_results_path = run_root / "answered_results.jsonl"

    selected_sample_ids = load_selected_sample_ids(args.sample_ids_file)
    caps = load_tool_capabilities_from_file()
    config = OrchestratorConfig(
        tool_capabilities=caps,
        planner_suggestion_count=2,
        default_budget=Budget(remaining_exec_steps=args.max_turns),
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
        ),
        ray_init_kwargs={"num_cpus": args.ray_num_cpus},
    )

    if args.judge_backend == "committee":
        judge_backend = CommitteeJudgeBackend(
            config_path=Path(args.judge_models_file).expanduser().resolve(),
            max_concurrency=args.judge_max_concurrency,
        )
    else:
        judge_backend = FakeJudgeBackend(overall_score=args.judge_score)

    store = OfflineTrajectoryStore(store_root)
    orchestrator = OrchestratorV01(
        store=store,
        planner_client=PlannerClient(
            backend=ApiTextBackend(),
            system_prompt_filename=args.planner_system_prompt_file,
        ),
        executor_client=ExecutorClient(
            backend=ApiTextBackend(),
            system_prompt_filename=args.executor_system_prompt_file,
        ),
        judge_client=JudgeClient(backend=judge_backend),
        runtime=runtime_component,
        config=config,
    )

    total_seen = 0
    total_selected = 0
    total_started = 0
    total_skipped_resume = 0
    total_finished = 0
    total_errors = 0
    total_no_answer = 0
    run_started_at = datetime.now(timezone.utc)
    sample_timing_records: list[dict[str, object]] = []
    run_token_usage_totals = _empty_usage_totals()

    try:
        for sample in iter_root_samples(input_jsonl):
            total_seen += 1
            if selected_sample_ids and sample.sample_id not in selected_sample_ids:
                continue
            if args.max_samples > 0 and total_selected >= args.max_samples:
                break
            total_selected += 1

            if args.resume and root_trajectory_exists(store_root, sample.sample_id):
                total_skipped_resume += 1
                sample_timing_records.append(
                    {
                        "sample_id": sample.sample_id,
                        "status": "skipped_resume",
                        "started_at": None,
                        "finished_at": None,
                        "elapsed_seconds": None,
                        "elapsed_minutes": None,
                        "log_path": None,
                        "error": None,
                    }
                )
                answered_ids = terminal_answered_ids(store, sample.sample_id)
                append_jsonl(
                    sample_results_path,
                    {
                        "sample_id": sample.sample_id,
                        "status": "skipped_resume",
                        "root_trajectory_id": build_root_trajectory_id(sample.sample_id),
                        "answered_trajectory_ids": answered_ids,
                        "error": None,
                        "log_path": None,
                        "started_at": None,
                        "finished_at": None,
                        "elapsed_seconds": None,
                        "elapsed_minutes": None,
                    },
                )
                if answered_ids:
                    append_jsonl(
                        answered_results_path,
                        build_answered_results_record(
                            store,
                            sample=sample,
                            terminal_trajectory_ids=[
                                item.trajectory_id for item in terminal_trajectories(store, sample.sample_id)
                            ],
                            answered_trajectory_ids=answered_ids,
                        ),
                    )
                continue

            total_started += 1
            sample_log_path = logs_root / f"{sample.sample_id}.log"
            error_text: str | None = None
            result = None
            sample_started_at = datetime.now(timezone.utc)
            with sample_log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(
                    f"\n===== sample_id={sample.sample_id} started_at={sample_started_at.isoformat()} =====\n"
                )
                with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                    try:
                        result = orchestrator.run(
                            sample,
                            budget=budget_for_sample(sample, default_budget=config.default_budget),
                        )
                    except Exception:
                        error_text = traceback.format_exc()
                        print(error_text)
            sample_finished_at = datetime.now(timezone.utc)
            sample_elapsed_seconds = max(0.0, (sample_finished_at - sample_started_at).total_seconds())

            if error_text is not None:
                total_errors += 1
                sample_timing_records.append(
                    {
                        "sample_id": sample.sample_id,
                        "status": "error",
                        "started_at": sample_started_at.isoformat(),
                        "finished_at": sample_finished_at.isoformat(),
                        "elapsed_seconds": round_seconds(sample_elapsed_seconds),
                        "elapsed_minutes": round_seconds(sample_elapsed_seconds / 60.0),
                        "log_path": str(sample_log_path),
                        "error": error_text,
                    }
                )
                append_jsonl(
                    sample_results_path,
                    {
                        "sample_id": sample.sample_id,
                        "status": "error",
                        "root_trajectory_id": None,
                        "answered_trajectory_ids": [],
                        "error": error_text,
                        "log_path": str(sample_log_path),
                        "started_at": sample_started_at.isoformat(),
                        "finished_at": sample_finished_at.isoformat(),
                        "elapsed_seconds": round_seconds(sample_elapsed_seconds),
                        "elapsed_minutes": round_seconds(sample_elapsed_seconds / 60.0),
                    },
                )
                continue

            total_finished += 1
            answered_ids = terminal_answered_ids(store, sample.sample_id)
            terminal_records = terminal_trajectories(store, sample.sample_id)
            terminal_statuses = {item.trajectory_id: item.status for item in terminal_records}
            token_usage_summary = build_sample_token_usage_summary(store, sample.sample_id)
            _accumulate_usage(run_token_usage_totals, token_usage_summary["sample_total"])
            token_usage_summary_path = store.sample_dir(sample.sample_id) / "token_usage_summary.json"
            token_usage_summary_path.write_text(
                json.dumps(token_usage_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            all_terminal_are_error = bool(terminal_records) and all(item.status == "error" for item in terminal_records)
            terminal_error_messages = [
                str(item.last_error.message)
                for item in terminal_records
                if item.last_error is not None and str(item.last_error.message).strip()
            ]
            sample_status = "ok"
            sample_error_text: str | None = None
            if not answered_ids and all_terminal_are_error:
                sample_status = "error"
                sample_error_text = terminal_error_messages[0] if terminal_error_messages else "All terminal trajectories ended in error."
                total_errors += 1
                total_finished -= 1
            elif not answered_ids:
                sample_status = "no_answer"
                total_no_answer += 1
            sample_timing_records.append(
                {
                    "sample_id": sample.sample_id,
                    "status": sample_status,
                    "started_at": sample_started_at.isoformat(),
                    "finished_at": sample_finished_at.isoformat(),
                    "elapsed_seconds": round_seconds(sample_elapsed_seconds),
                    "elapsed_minutes": round_seconds(sample_elapsed_seconds / 60.0),
                    "log_path": str(sample_log_path),
                    "error": sample_error_text,
                    "token_usage_total_tokens": int(token_usage_summary["sample_total"]["total_tokens"]),
                }
            )
            terminal_trajectory_ids = result.terminal_trajectory_ids if result is not None else []
            append_jsonl(
                sample_results_path,
                {
                    "sample_id": sample.sample_id,
                    "status": sample_status,
                    "root_trajectory_id": result.root_trajectory_id if result is not None else None,
                    "terminal_trajectory_ids": terminal_trajectory_ids,
                    "terminal_trajectory_statuses": terminal_statuses,
                    "answered_trajectory_ids": answered_ids,
                    "error": sample_error_text,
                    "log_path": str(sample_log_path),
                    "started_at": sample_started_at.isoformat(),
                    "finished_at": sample_finished_at.isoformat(),
                    "elapsed_seconds": round_seconds(sample_elapsed_seconds),
                    "elapsed_minutes": round_seconds(sample_elapsed_seconds / 60.0),
                    "token_usage": token_usage_summary["sample_total"],
                    "token_usage_summary_path": str(token_usage_summary_path),
                },
            )
            if answered_ids:
                append_jsonl(
                    answered_results_path,
                    build_answered_results_record(
                        store,
                        sample=sample,
                        terminal_trajectory_ids=terminal_trajectory_ids,
                        answered_trajectory_ids=answered_ids,
                    ),
                )
    finally:
        runtime_component.close_sync()
    run_finished_at = datetime.now(timezone.utc)
    counts = {
        "total_seen": total_seen,
        "total_selected": total_selected,
        "total_started": total_started,
        "total_skipped_resume": total_skipped_resume,
        "total_finished": total_finished,
        "total_errors": total_errors,
        "total_no_answer": total_no_answer,
    }
    timing_summary = build_sample_timing_summary(
        run_id=run_id,
        run_root=run_root,
        input_jsonl=input_jsonl,
        run_started_at=run_started_at,
        run_finished_at=run_finished_at,
        sample_timing_records=sample_timing_records,
        counts=counts,
    )
    timing_summary_path = run_root / "sample_timing_summary.json"
    timing_summary_path.write_text(
        json.dumps(timing_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "run_id": run_id,
        "run_root": str(run_root),
        "store_root": str(store_root),
        "input_jsonl": str(input_jsonl),
        "sample_ids_file": str(Path(args.sample_ids_file).expanduser().resolve()) if args.sample_ids_file else None,
        "planner_debug": bool(args.planner_debug),
        "executor_debug": bool(args.executor_debug),
        "prompt_files": {
            "planner_system_prompt_file": args.planner_system_prompt_file,
            "executor_system_prompt_file": args.executor_system_prompt_file,
        },
        "judge": {
            "backend": args.judge_backend,
            "judge_models_file": str(Path(args.judge_models_file).expanduser().resolve())
            if args.judge_backend == "committee"
            else None,
        },
        "runtime_services": {
            "ocr_base_url": args.ocr_base_url,
            "grounded_sam2_base_url": args.grounded_sam2_base_url,
            "depth_base_url": args.depth_base_url,
            "countgd_base_url": args.countgd_base_url,
            "ocr_model_name": args.ocr_model_name,
            "service_timeout": args.service_timeout,
        },
        "timing": {
            "started_at": run_started_at.isoformat(),
            "finished_at": run_finished_at.isoformat(),
            "wall_clock_seconds": timing_summary["run_timing"]["wall_clock_seconds"],
            "wall_clock_minutes": timing_summary["run_timing"]["wall_clock_minutes"],
            "total_sample_elapsed_seconds": timing_summary["run_timing"]["total_sample_elapsed_seconds"],
            "total_sample_elapsed_minutes": timing_summary["run_timing"]["total_sample_elapsed_minutes"],
        },
        "counts": counts,
        "token_usage": run_token_usage_totals,
        "artifacts": {
            "sample_results_jsonl": str(sample_results_path),
            "answered_results_jsonl": str(answered_results_path),
            "sample_timing_summary_json": str(timing_summary_path),
            "logs_dir": str(logs_root),
        },
    }
    (run_root / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
