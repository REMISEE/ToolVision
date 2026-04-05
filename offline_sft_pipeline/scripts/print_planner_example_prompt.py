#!/usr/bin/env python3
"""Print the same ``system_prompt`` and ``user_prompt`` that ``PlannerClient.run`` passes to the backend.

These are exactly:

- ``system_prompt``: contents of ``prompts/planner_system_*.txt`` (see ``PlannerClient`` ctor).
- ``user_prompt``: ``planner_user_*.txt`` after ``Template.safe_substitute`` with the request fields.

This matches the ``TextGenerationBackend.generate(..., system_prompt=..., user_prompt=...)`` contract.

Note: ``ApiTextBackend`` **does not** POST this ``user_prompt`` string as-is; it rebuilds OpenAI
``messages`` from ``context["request"]`` via ``planner_to_openai_messages``. Use ``--openai-messages``
only when you need that HTTP-shaped payload.

Usage::

    cd ToolVision_Copy
    PYTHONPATH=. python offline_sft_pipeline/scripts/print_planner_example_prompt.py
    PYTHONPATH=. python offline_sft_pipeline/scripts/print_planner_example_prompt.py --json
    PYTHONPATH=. python offline_sft_pipeline/scripts/print_planner_example_prompt.py --openai-messages

"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PIPELINE_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = PIPELINE_ROOT.parent
EXAMPLE_DIR = PIPELINE_ROOT / "example"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offline_sft_pipeline.core.models import Budget, ConversationMessage, ImageArtifactRef
from offline_sft_pipeline.pipelines.planner_client import PlannerClient
from offline_sft_pipeline.pipelines.backends import FakeTextBackend
from offline_sft_pipeline.pipelines.request_models import PlannerClientRequest
from offline_sft_pipeline.pipelines.tool_capabilities_io import load_tool_capabilities_from_file


def _load_example() -> tuple[str, Path]:
    qpath = EXAMPLE_DIR / "question.json"
    if not qpath.is_file():
        raise SystemExit(f"missing {qpath}")
    payload = json.loads(qpath.read_text(encoding="utf-8"))
    rel = payload.get("image")
    if not rel:
        raise SystemExit("question.json has no 'image' field")
    img_path = (EXAMPLE_DIR / str(rel)).resolve()
    if not img_path.is_file():
        raise SystemExit(f"missing example image: {img_path}")
    return str(payload["question"]), img_path


def build_example_planner_request(*, trajectory_dir: str) -> PlannerClientRequest:
    question, img_path = _load_example()
    vis = [ImageArtifactRef(artifact_id="img_root_0", path=str(img_path), media_type="image/png")]
    return PlannerClientRequest(
        sample_id="example",
        trajectory_id="traj_example",
        round_idx=0,
        sample_dir=str(EXAMPLE_DIR),
        trajectory_dir=trajectory_dir,
        question=question,
        messages=[
            ConversationMessage(
                message_id="m_user",
                role="user",
                content=question,
                image_artifact_ids=["img_root_0"],
                metadata={},
            ),
        ],
        visible_images=vis,
        budget=Budget(remaining_rounds=3),
        tool_capabilities=load_tool_capabilities_from_file(),
        requested_suggestion_count=3,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Print PlannerClient system_prompt + user_prompt for example/question.json.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON object with keys system_prompt, user_prompt, paths.",
    )
    p.add_argument(
        "--openai-messages",
        action="store_true",
        help="Also include ApiTextBackend-style messages[] (sanitized base64).",
    )
    p.add_argument(
        "--full-images",
        action="store_true",
        help="With --openai-messages, do not shorten data: URLs (huge).",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Write output to this file instead of stdout.",
    )
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        traj = Path(tmp) / "traj"
        traj.mkdir()
        req = build_example_planner_request(trajectory_dir=str(traj))

    pc = PlannerClient(backend=FakeTextBackend())
    system_prompt = pc._load_prompt(pc.system_prompt_path)
    user_prompt = pc._build_user_prompt(req)

    out_lines: list[str] = []
    if not args.json and not args.openai_messages:
        out_lines.extend(
            [
                f"system_prompt_path: {pc.system_prompt_path}",
                f"user_prompt_path: {pc.user_prompt_path}",
                "",
                "========== system_prompt (passed to backend.generate) ==========",
                system_prompt,
                "",
                "========== user_prompt (passed to backend.generate) ==========",
                user_prompt,
                "",
                "========== note ==========",
                "ApiTextBackend ignores this user_prompt for HTTP; it builds messages[] from context['request'].",
                "Use --openai-messages to print that payload.",
            ]
        )
        text = "\n".join(out_lines)
    else:
        payload: dict[str, object] = {
            "system_prompt_path": str(pc.system_prompt_path),
            "user_prompt_path": str(pc.user_prompt_path),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "note_api_text_backend": (
                "ApiTextBackend does not POST user_prompt as a single string; "
                "see planner_to_openai_messages in api_text_multimodal.py."
            ),
        }
        if args.openai_messages:
            from offline_sft_pipeline.pipelines.api_text_multimodal import (
                planner_to_openai_messages,
                sanitize_messages_for_debug,
            )

            messages, missing = planner_to_openai_messages(system_prompt=system_prompt, req=req)
            payload["missing_artifact_ids"] = missing
            payload["openai_messages"] = (
                messages if args.full_images else sanitize_messages_for_debug(messages)
            )
        text = json.dumps(payload, ensure_ascii=False, indent=2)

    if args.out:
        Path(args.out).expanduser().resolve().write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
