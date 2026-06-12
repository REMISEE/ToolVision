"""Tests for easy_question_pipeline message builder and dry-run backend I/O."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from offline_sft_pipeline.core.models import Budget, ConversationMessage, ImageArtifactRef
from offline_sft_pipeline.pipelines.easy_question_pipeline.backend import EasyApiTextPlannerBackend
from offline_sft_pipeline.pipelines.easy_question_pipeline.jsonl_samples import reference_answer_from_row
from offline_sft_pipeline.pipelines.easy_question_pipeline.messages import (
    build_easy_reference_answer_block,
    planner_to_openai_messages_easy,
)
from offline_sft_pipeline.pipelines.request_models import PlannerClientRequest, ToolCapability


class EasyQuestionPipelineMessagesTests(unittest.TestCase):
    def test_reference_answer_textvqa_uses_model_filtered_resps(self) -> None:
        row = {
            "answer": ["a", "b"],
            "metadata": {"source_dataset": "textvqa", "model_filtered_resps": "1664", "log_exact_match": 1.0},
        }
        self.assertEqual(reference_answer_from_row(row, dataset_dir_name="textvqa"), "1664")

    def test_reference_answer_fsc147_uses_scalar_answer(self) -> None:
        row = {"answer": "8", "metadata": {"source_dataset": "fsc147"}}
        self.assertEqual(reference_answer_from_row(row, dataset_dir_name="fsc147"), "8")

    def test_build_easy_reference_answer_block_contains_constraints(self) -> None:
        text = build_easy_reference_answer_block(reference_answer="42", answer_instruction="Answer with only an integer.")
        self.assertIn("42", text)
        self.assertIn("Return `mode=\"answer\"`", text)
        self.assertIn("aligned with that exact answer", text)

    def test_planner_to_openai_messages_easy_appends_block(self) -> None:
        root = Path(__file__).resolve().parents[2]
        # Use repo README or any small existing file as fake image path — tests only need path strings for last message text.
        placeholder = root / "offline_sft_pipeline" / "prompts" / "planner_system_v05.txt"
        self.assertTrue(placeholder.is_file(), f"missing {placeholder}")
        req = PlannerClientRequest(
            sample_id="t__x",
            trajectory_id="traj__t__x__root",
            round_idx=0,
            sample_dir=str(placeholder.parent),
            question="How many?",
            answer_instruction="Answer with only an integer.",
            messages=[
                ConversationMessage(
                    message_id="u0",
                    role="user",
                    content="How many?",
                    image_artifact_ids=["img_root_0"],
                )
            ],
            visible_images=[
                ImageArtifactRef(artifact_id="img_root_0", path=str(placeholder), width=1, height=1)
            ],
            budget=Budget(remaining_exec_steps=0),
            must_answer_now=True,
            requested_suggestion_count=0,
            tool_capabilities=[
                ToolCapability(name="count_assist", description="Count", usage_notes=None),
            ],
        )
        messages, missing = planner_to_openai_messages_easy(
            system_prompt="You are a planner.",
            req=req,
            reference_answer="7",
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1]["role"], "user")
        last = messages[-1]["content"]
        self.assertIsInstance(last, str)
        assert isinstance(last, str)
        self.assertIn("Easy no-tool final-answer constraint", last)
        self.assertIn("'7'", last)
        self.assertFalse(missing)

    def test_easy_backend_dry_run_writes_artifacts(self) -> None:
        prev = os.environ.get("OFFLINE_SFT_API_DRY_RUN")
        os.environ["OFFLINE_SFT_API_DRY_RUN"] = "1"
        try:
            root = Path(__file__).resolve().parents[2]
            placeholder = root / "offline_sft_pipeline" / "prompts" / "planner_system_v05.txt"
            req = PlannerClientRequest(
                sample_id="t__dry",
                trajectory_id="traj__t__dry__root",
                round_idx=0,
                sample_dir=str(placeholder.parent),
                question="Q?",
                messages=[
                    ConversationMessage(
                        message_id="u0",
                        role="user",
                        content="Q?",
                        image_artifact_ids=["img_root_0"],
                    )
                ],
                visible_images=[
                    ImageArtifactRef(artifact_id="img_root_0", path=str(placeholder), width=1, height=1)
                ],
                budget=Budget(remaining_exec_steps=0),
                must_answer_now=True,
                requested_suggestion_count=0,
                tool_capabilities=[],
            )
            with tempfile.TemporaryDirectory() as tmp:
                backend = EasyApiTextPlannerBackend()
                backend.generate(
                    stage="planner",
                    system_prompt="sys",
                    user_prompt="",
                    context={
                        "request": req,
                        "reference_answer": "ok",
                        "easy_artifact_dir": tmp,
                    },
                )
                msg_path = Path(tmp) / "planner_request_messages.json"
                self.assertTrue(msg_path.is_file())
                data = json.loads(msg_path.read_text(encoding="utf-8"))
                self.assertIsInstance(data, list)
                last_user = data[-1]["content"]
                self.assertIn("Easy no-tool final-answer constraint", last_user)
        finally:
            if prev is None:
                os.environ.pop("OFFLINE_SFT_API_DRY_RUN", None)
            else:
                os.environ["OFFLINE_SFT_API_DRY_RUN"] = prev


if __name__ == "__main__":
    unittest.main()
