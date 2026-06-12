from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offline_sft_pipeline.core.models import ConversationMessage, ImageArtifactRef
from offline_sft_pipeline.core.models import MessagesDocument
from offline_sft_pipeline.core.store import OfflineTrajectoryStore
from offline_sft_pipeline.pipelines.api_text_multimodal import (
    build_judge_control_user_text,
    judge_to_openai_messages,
    sanitize_messages_for_debug,
)
from offline_sft_pipeline.pipelines.backends import (
    DEFAULT_JUDGE_MAX_CONCURRENCY,
    DEFAULT_JUDGE_MODELS_PATH,
    DEFAULT_JUDGE_PROMPT_ROOT,
    DEFAULT_JUDGE_SYSTEM_PROMPT_FILE,
    CommitteeJudgeBackend,
    FakeJudgeBackend,
)
from offline_sft_pipeline.pipelines.judge_client import JudgeClient
from offline_sft_pipeline.pipelines.request_models import JudgeClientRequest
from offline_sft_pipeline.core.models import TrajectoryRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test judge in isolation from an existing store trajectory. "
            "Can render the real committee prompt, run fake/committee backend, and optionally save JudgeRecord."
        )
    )
    parser.add_argument("--store-root", type=str, required=True, help="Run store root, e.g. .../outputs/.../store")
    parser.add_argument("--sample-id", type=str, required=True)
    parser.add_argument("--trajectory-id", type=str, required=True)
    parser.add_argument(
        "--scope-type",
        type=str,
        default="step",
        choices=["step", "trajectory"],
        help="Judge a specific step or the whole trajectory.",
    )
    parser.add_argument(
        "--step-idx",
        type=int,
        default=None,
        help="Required for --scope-type=step. Defaults to the latest step on the trajectory.",
    )
    parser.add_argument(
        "--judge-backend",
        type=str,
        default="fake",
        choices=["fake", "committee"],
        help="Which backend to execute after request/prompt inspection.",
    )
    parser.add_argument(
        "--judge-stage",
        type=str,
        default="committee",
        help="Judge stage to stamp into JudgeClientRequest / JudgeRecord.",
    )
    parser.add_argument(
        "--judge-models-file",
        type=str,
        default=str(DEFAULT_JUDGE_MODELS_PATH),
        help="JSON config for CommitteeJudgeBackend.",
    )
    parser.add_argument(
        "--judge-max-concurrency",
        type=int,
        default=DEFAULT_JUDGE_MAX_CONCURRENCY,
        help="Maximum number of committee judge model calls to execute in parallel.",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=str,
        default=str(DEFAULT_JUDGE_PROMPT_ROOT / DEFAULT_JUDGE_SYSTEM_PROMPT_FILE),
        help="System prompt file used when rendering prompt preview.",
    )
    parser.add_argument(
        "--fake-score",
        type=float,
        default=0.62,
        help="Overall score used by FakeJudgeBackend.",
    )
    parser.add_argument(
        "--print-request-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the constructed JudgeClientRequest.",
    )
    parser.add_argument(
        "--print-control-user-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the control user text appended by judge_to_openai_messages.",
    )
    parser.add_argument(
        "--print-openai-messages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print rendered OpenAI-compatible messages.",
    )
    parser.add_argument(
        "--save-record",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist the returned JudgeRecord under trajectory/judge/ after the smoke run succeeds.",
    )
    parser.add_argument(
        "--include-final-answer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether trajectory-scope smoke should include the existing final-answer assistant message.",
    )
    parser.add_argument(
        "--answer-instruction-override",
        type=str,
        default="",
        help="Optional temporary answer instruction override for smoke runs, e.g. 'Answer with only an integer.'",
    )
    return parser.parse_args()


def _load_system_prompt(path_value: str) -> str:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"judge system prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def _normalize_legacy_trajectory_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    budget = normalized.get("budget")
    if isinstance(budget, dict) and "remaining_exec_steps" not in budget and "remaining_rounds" in budget:
        migrated_budget = dict(budget)
        migrated_budget["remaining_exec_steps"] = int(migrated_budget.pop("remaining_rounds"))
        normalized["budget"] = migrated_budget
    steps = normalized.get("steps")
    trajectory_id = normalized.get("trajectory_id")
    if isinstance(steps, list):
        migrated_steps: list[dict[str, Any]] = []
        for item in steps:
            if not isinstance(item, dict):
                migrated_steps.append(item)
                continue
            migrated_item = dict(item)
            if trajectory_id and "execution_trajectory_id" not in migrated_item:
                migrated_item["execution_trajectory_id"] = trajectory_id
            migrated_steps.append(migrated_item)
        normalized["steps"] = migrated_steps
    return normalized


def _load_trajectory_with_legacy_compat(
    store: OfflineTrajectoryStore,
    *,
    sample_id: str,
    trajectory_id: str,
) -> TrajectoryRecord:
    trajectory_path = store.trajectory_path(sample_id, trajectory_id)
    raw_payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
    normalized_payload = _normalize_legacy_trajectory_payload(raw_payload)
    return TrajectoryRecord.model_validate(normalized_payload)


def _load_messages_for_trajectory(
    store: OfflineTrajectoryStore,
    *,
    trajectory: TrajectoryRecord,
) -> MessagesDocument:
    messages_path = store.trajectory_dir(trajectory.sample_id, trajectory.trajectory_id) / trajectory.messages_path
    return MessagesDocument.from_json_file(messages_path)


def _select_latest_step_idx(trajectory) -> int:
    if trajectory.step_idx <= 0:
        raise ValueError(
            f"trajectory {trajectory.trajectory_id!r} has no executed step; cannot infer --step-idx for scope_type=step."
        )
    return int(trajectory.step_idx)


def _load_step_record(trajectory, *, step_idx: int):
    for step_record in trajectory.steps:
        if int(step_record.step_idx) == int(step_idx):
            return step_record.model_copy(deep=True)
    raise ValueError(f"step_idx={step_idx} not found on trajectory {trajectory.trajectory_id!r}.")


def _messages_up_to_step(messages: list[ConversationMessage], *, step_idx: int) -> list[ConversationMessage]:
    selected: list[ConversationMessage] = []
    for message in messages:
        metadata = dict(message.metadata or {})
        message_kind = str(metadata.get("message_kind") or "").strip()
        metadata_step_idx = metadata.get("step_idx")
        if message_kind == "final_answer":
            continue
        if isinstance(metadata_step_idx, int) and metadata_step_idx > int(step_idx):
            continue
        selected.append(message.model_copy(deep=True))
    return selected


def _messages_without_final_answer(messages: list[ConversationMessage]) -> list[ConversationMessage]:
    selected: list[ConversationMessage] = []
    for message in messages:
        metadata = dict(message.metadata or {})
        if str(metadata.get("message_kind") or "").strip() == "final_answer":
            continue
        selected.append(message.model_copy(deep=True))
    return selected


def _visible_images_for_step(
    store: OfflineTrajectoryStore,
    *,
    sample_id: str,
    trajectory_id: str,
    step_idx: int,
) -> tuple[list[ImageArtifactRef], object]:
    visible_images = [item.model_copy(deep=True) for item in store.load_root_artifacts(sample_id)]
    runtime_result = None
    for current_step_idx in range(1, int(step_idx) + 1):
        candidate = store.load_runtime_result(sample_id, trajectory_id, current_step_idx)
        if candidate.images:
            visible_images.append(candidate.images[0].model_copy(deep=True))
        if current_step_idx == int(step_idx):
            runtime_result = candidate
    if runtime_result is None:
        raise FileNotFoundError(
            f"Missing runtime_result for sample_id={sample_id!r}, trajectory_id={trajectory_id!r}, step_idx={step_idx}."
        )
    return visible_images, runtime_result


def _visible_images_for_trajectory(
    store: OfflineTrajectoryStore,
    *,
    sample_id: str,
    trajectory_id: str,
    trajectory,
) -> tuple[list[ImageArtifactRef], object | None]:
    visible_images = [item.model_copy(deep=True) for item in store.load_root_artifacts(sample_id)]
    runtime_result = None
    for current_step_idx in range(1, int(trajectory.step_idx) + 1):
        candidate = store.load_runtime_result(sample_id, trajectory_id, current_step_idx)
        if candidate.images:
            visible_images.append(candidate.images[0].model_copy(deep=True))
        runtime_result = candidate
    return visible_images, runtime_result


def build_judge_request(args: argparse.Namespace) -> JudgeClientRequest:
    store = OfflineTrajectoryStore(args.store_root)
    root_sample = store.load_root_sample(args.sample_id)
    trajectory = _load_trajectory_with_legacy_compat(
        store,
        sample_id=args.sample_id,
        trajectory_id=args.trajectory_id,
    )
    messages_doc = _load_messages_for_trajectory(store, trajectory=trajectory)

    if args.scope_type == "step":
        step_idx = args.step_idx if args.step_idx is not None else _select_latest_step_idx(trajectory)
        step_record = _load_step_record(trajectory, step_idx=step_idx)
        visible_images, runtime_result = _visible_images_for_step(
            store,
            sample_id=args.sample_id,
            trajectory_id=args.trajectory_id,
            step_idx=step_idx,
        )
        messages = _messages_up_to_step(list(messages_doc.root), step_idx=step_idx)
        final_answer = None
    else:
        step_idx = None
        step_record = None
        visible_images, runtime_result = _visible_images_for_trajectory(
            store,
            sample_id=args.sample_id,
            trajectory_id=args.trajectory_id,
            trajectory=trajectory,
        )
        messages = [item.model_copy(deep=True) for item in messages_doc.root]
        if not args.include_final_answer:
            messages = _messages_without_final_answer(messages)
        final_answer = trajectory.final_answer

    answer_instruction = (
        str(args.answer_instruction_override).strip()
        if str(args.answer_instruction_override).strip()
        else trajectory.answer_instruction
    )

    return JudgeClientRequest(
        sample_id=args.sample_id,
        trajectory_id=args.trajectory_id,
        sample_dir=str(store.sample_dir(args.sample_id)),
        trajectory_dir=str(store.trajectory_dir(args.sample_id, args.trajectory_id)),
        scope_type=args.scope_type,
        scope_step_idx=step_idx,
        judge_stage=args.judge_stage,
        question=trajectory.question,
        answer_instruction=answer_instruction,
        answer=root_sample.answer,
        messages=messages,
        visible_images=visible_images,
        planner_output=None,
        step_record=step_record,
        runtime_result=runtime_result,
        final_answer=final_answer,
        metadata=dict(root_sample.metadata),
    )


def build_backend(args: argparse.Namespace):
    if args.judge_backend == "fake":
        return FakeJudgeBackend(overall_score=args.fake_score)
    return CommitteeJudgeBackend(
        config_path=Path(args.judge_models_file).expanduser().resolve(),
        max_concurrency=args.judge_max_concurrency,
    )


def main() -> None:
    args = parse_args()
    store = OfflineTrajectoryStore(args.store_root)
    request = build_judge_request(args)
    system_prompt = _load_system_prompt(args.system_prompt_file)
    control_user_text = build_judge_control_user_text(request)
    openai_messages, missing_ids = judge_to_openai_messages(system_prompt=system_prompt, req=request)

    preview = {
        "store_root": str(Path(args.store_root).expanduser().resolve()),
        "sample_id": args.sample_id,
        "trajectory_id": args.trajectory_id,
        "scope_type": args.scope_type,
        "scope_step_idx": request.scope_step_idx,
        "judge_backend": args.judge_backend,
        "judge_stage": args.judge_stage,
        "missing_artifact_ids": missing_ids,
        "visible_image_artifact_ids": [item.artifact_id for item in request.visible_images],
        "message_count": len(request.messages),
    }
    print("=== JUDGE PREVIEW ===")
    print(json.dumps(preview, ensure_ascii=False, indent=2))

    if args.print_request_json:
        print("\n=== JUDGE REQUEST ===")
        print(json.dumps(request.model_dump(mode="json"), ensure_ascii=False, indent=2))

    if args.print_control_user_text:
        print("\n=== JUDGE CONTROL USER TEXT ===")
        print(control_user_text)

    if args.print_openai_messages:
        print("\n=== OPENAI MESSAGES ===")
        print(json.dumps(sanitize_messages_for_debug(openai_messages), ensure_ascii=False, indent=2))

    backend = build_backend(args)
    client = JudgeClient(backend=backend)
    judge_record = client.run(request)

    print("\n=== JUDGE RECORD ===")
    print(json.dumps(judge_record.model_dump(mode="json"), ensure_ascii=False, indent=2))

    model_results = judge_record.metadata.get("model_results")
    if isinstance(model_results, list) and model_results:
        summary_payload: list[dict[str, Any]] = []
        for item in model_results:
            if not isinstance(item, dict):
                continue
            summary_payload.append(
                {
                    "name": item.get("name"),
                    "raw_answer": item.get("raw_answer"),
                    "token_usage": item.get("token_usage"),
                    "usage_raw": item.get("usage_raw"),
                    "response_message_summary": item.get("response_message_summary"),
                    "error": item.get("error"),
                }
            )
        print("\n=== MODEL RESPONSE SUMMARIES ===")
        print(json.dumps(summary_payload, ensure_ascii=False, indent=2))

    if args.save_record:
        judge_path = store.save_judge_record(judge_record)
        print("\n=== SAVED JUDGE RECORD PATH ===")
        print(str(judge_path))
        try:
            store.register_judge_record(judge_record)
        except Exception as exc:
            print("\n=== SAVE WARNING ===")
            print(
                "JudgeRecord JSON was written, but trajectory index update failed. "
                f"This usually means the source store uses an older trajectory schema. {exc}"
            )
            return
        registered_path = store.judge_record_path(
            judge_record.sample_id,
            judge_record.trajectory_id,
            judge_record.judge_stage,
            scope_step_idx=judge_record.scope_step_idx,
        )
        if registered_path != judge_path:
            print("\n=== REGISTERED JUDGE RECORD PATH ===")
            print(str(registered_path))


if __name__ == "__main__":
    main()
