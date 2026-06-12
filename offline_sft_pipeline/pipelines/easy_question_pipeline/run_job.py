"""Single easy-planner HTTP job (no CLI). Used by run_easy_planner_sample and batch scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from offline_sft_pipeline.core.models import Budget, ConversationMessage, ImageArtifactRef
from offline_sft_pipeline.pipelines.easy_question_pipeline.client import EasyPlannerClient
from offline_sft_pipeline.pipelines.request_models import PlannerClientRequest
from offline_sft_pipeline.pipelines.tool_capabilities_io import load_tool_capabilities_from_file


def _try_image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
            return int(w), int(h)
    except Exception:
        return None, None


def run_easy_planner_job(
    *,
    image_path: Path,
    question: str,
    reference_answer: str,
    sample_id: str,
    output_dir: Path,
    answer_instruction: str | None = None,
    system_prompt_file: str = "planner_system_v05.txt",
    prompt_root: Path | None = None,
    print_summary_json: bool = True,
) -> tuple[int, dict[str, object]]:
    """Call planner once; write artifacts under ``output_dir``. Returns (exit_code, summary_dict)."""
    if not image_path.is_file():
        return 2, {"error": f"Image not found: {image_path}"}

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    w, h = _try_image_size(image_path)
    artifact_id = "img_root_0"
    visible = [
        ImageArtifactRef(
            artifact_id=artifact_id,
            path=str(image_path.resolve()),
            width=w,
            height=h,
        )
    ]
    messages = [
        ConversationMessage(
            message_id="m_user_q0",
            role="user",
            content=question,
            image_artifact_ids=[artifact_id],
        )
    ]

    req = PlannerClientRequest(
        sample_id=sample_id,
        trajectory_id=f"traj__{sample_id}__root",
        round_idx=0,
        sample_dir=str(image_path.parent),
        trajectory_dir=None,
        planner_dir=None,
        steps_dir=None,
        question=question,
        answer_instruction=answer_instruction,
        messages=messages,
        visible_images=visible,
        budget=Budget(remaining_exec_steps=6),
        must_answer_now=True,
        requested_suggestion_count=0,
        tool_capabilities=load_tool_capabilities_from_file(),
        metadata={"pipeline": "easy_question_sample"},
    )

    client = EasyPlannerClient(
        prompt_root=str(prompt_root) if prompt_root else None,
        system_prompt_filename=system_prompt_file,
    )
    planner_output, artifacts = client.run(
        req,
        reference_answer=reference_answer,
        artifact_dir=output_dir,
    )

    result_path = output_dir / "planner_output.json"
    result_path.write_text(planner_output.to_json_str(), encoding="utf-8")

    summary: dict[str, object] = {
        "sample_id": planner_output.sample_id,
        "can_answer_now": planner_output.can_answer_now,
        "direct_answer": planner_output.direct_answer,
        "artifact_files": artifacts.files,
        "planner_output_path": str(result_path),
    }
    if print_summary_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stdout)
    return 0, summary


__all__ = ["run_easy_planner_job"]
