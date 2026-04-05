#!/usr/bin/env python3
"""Run one single real planner API call without orchestrator.

This script exists for prompt/protocol debugging when you do NOT want to run the
full branching pipeline. It:

1. builds one ``PlannerClientRequest`` from ``example/question.json``
2. optionally prints the exact OpenAI-style ``messages[]`` payload that ``ApiTextBackend`` would send
3. calls ``PlannerClient(backend=ApiTextBackend())``
4. prints the parsed ``PlannerOutput`` JSON

Use ``--planner-debug`` to let ``ApiTextBackend`` dump:

- sanitized HTTP payload
- raw assistant text

to stderr.
"""

from __future__ import annotations

import argparse
import json
import os
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
from offline_sft_pipeline.pipelines.api_text_multimodal import (
    planner_to_openai_messages,
    sanitize_messages_for_debug,
)
from offline_sft_pipeline.pipelines.backends import ApiTextBackend
from offline_sft_pipeline.pipelines.planner_client import PlannerClient
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one real planner API call without orchestrator.")
    p.add_argument(
        "--planner-debug",
        action="store_true",
        help="Enable OFFLINE_SFT_PLANNER_DEBUG=1 for this process.",
    )
    p.add_argument(
        "--print-openai-messages",
        action="store_true",
        help="Print the exact OpenAI-style messages[] payload before the API call.",
    )
    p.add_argument(
        "--full-images",
        action="store_true",
        help="With --print-openai-messages, do not shorten data URLs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.planner_debug:
        os.environ["OFFLINE_SFT_PLANNER_DEBUG"] = "1"

    with tempfile.TemporaryDirectory() as tmp:
        traj = Path(tmp) / "traj"
        traj.mkdir()
        req = build_example_planner_request(trajectory_dir=str(traj))
        planner = PlannerClient(backend=ApiTextBackend())
        system_prompt = planner._load_prompt(planner.system_prompt_path)
        if args.print_openai_messages:
            messages, missing = planner_to_openai_messages(system_prompt=system_prompt, req=req)
            payload = messages if args.full_images else sanitize_messages_for_debug(messages)
            print(
                json.dumps(
                    {
                        "missing_artifact_ids": missing,
                        "openai_messages": payload,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        out = planner.run(req)
        print(json.dumps(out.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
