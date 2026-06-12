from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from offline_sft_pipeline.core.models import (
    Budget,
    ConversationMessage,
    CapabilityPlanItem,
    ImageArtifactRef,
    PlannerStepSpec,
    build_child_trajectory_id,
    build_root_trajectory_id,
)
from offline_sft_pipeline.core.store import OfflineTrajectoryStore
from offline_sft_pipeline.pipelines.backends import FakeTextBackend, JudgeBackendResult
from offline_sft_pipeline.pipelines.executor_client import ExecutorClient
from offline_sft_pipeline.pipelines.judge_client import JudgeClient
from offline_sft_pipeline.pipelines.orchestrator_v01 import OrchestratorConfig, OrchestratorV01
from offline_sft_pipeline.pipelines.planner_client import PlannerClient
from offline_sft_pipeline.pipelines.request_models import ExecutorClientRequest, PlannerClientRequest, ToolCapability
from offline_sft_pipeline.pipelines.scripted_components import (
    RuntimeSpec,
    ScriptedExecutorClient,
    ScriptedJudgeBackend,
    ScriptedPlannerClient,
    ScriptedRuntime,
    ScriptedTextBackend,
    build_demo_root_sample,
    build_three_round_demo_scenario,
    build_three_round_demo_spec,
    make_executor_output,
    make_planner_output,
    make_step,
    make_suggestion,
)


class _ConsensusJudgeBackend:
    def __init__(self, scores: dict[tuple[str, int], float], *, candidate_answer: str) -> None:
        self.scores = dict(scores)
        self.candidate_answer = candidate_answer

    def score(self, request):
        step_idx = request.scope_step_idx
        if step_idx is None and request.step_record is not None:
            step_idx = request.step_record.step_idx
        key = (request.trajectory_id, int(step_idx or 0))
        score = float(self.scores[key])
        model_results = [
            {"name": "judge_a", "normalized_answer": self.candidate_answer},
            {"name": "judge_b", "normalized_answer": self.candidate_answer},
            {"name": "judge_c", "normalized_answer": self.candidate_answer},
        ]
        return JudgeBackendResult(
            overall_score=score,
            metadata={"model_results": model_results},
            note=f"score={score:.2f}",
        )


class _ReferenceAwareJudgeBackend:
    def __init__(self, scores: dict[tuple[str, int], float], *, normalized_reference) -> None:
        self.scores = dict(scores)
        self.normalized_reference = normalized_reference

    def score(self, request):
        step_idx = request.scope_step_idx
        if step_idx is None and request.step_record is not None:
            step_idx = request.step_record.step_idx
        key = (request.trajectory_id, int(step_idx or 0))
        score = float(self.scores[key])
        model_results = [
            {"name": "judge_a", "normalized_answer": "dog", "normalized_reference": self.normalized_reference},
            {"name": "judge_b", "normalized_answer": "cat", "normalized_reference": self.normalized_reference},
            {"name": "judge_c", "normalized_answer": "cat", "normalized_reference": self.normalized_reference},
        ]
        return JudgeBackendResult(
            overall_score=score,
            metadata={"model_results": model_results},
            note=f"score={score:.3f}",
        )


class OrchestratorV01SmokeTest(unittest.TestCase):
    maxDiff = None

    def test_planner_client_parses_new_json_contract(self) -> None:
        planner = PlannerClient(
            backend=FakeTextBackend(
                stage_responses={
                    "planner": """{
  "mode": "suggestions",
  "think": "The current crop is ambiguous, so I should localize the likely tag region first.",
  "suggestions": [
    {
      "suggestion_id": "s1",
      "suggestion_cot": "Localize the tag before OCR.",
      "steps": [
        {
          "step_id": "step_1",
          "step_goal": "Locate the tag region.",
          "input_image_index": 0,
          "capability_plan": [
            {
              "order": 1,
              "capability": "ground_box",
              "instruction": "Find the tag region."
            }
          ],
          "executor_instruction": "Use grounding to localize the tag."
        }
      ]
    }
  ]
}"""
                }
            )
        )
        output = planner.run(self._build_planner_request())
        self.assertFalse(output.can_answer_now)
        self.assertEqual(
            output.global_chain_cot,
            "The current crop is ambiguous, so I should localize the likely tag region first.",
        )
        self.assertEqual(output.suggestions[0].suggestion_id, "s1")

    def test_planner_client_keeps_legacy_tag_contract_compatibility(self) -> None:
        planner = PlannerClient(
            backend=FakeTextBackend(
                stage_responses={
                    "planner": "<think>\nThe visible image already contains the answer.\n</think>\n<answer>\n249\n</answer>"
                }
            )
        )
        output = planner.run(self._build_planner_request())
        self.assertTrue(output.can_answer_now)
        self.assertEqual(output.global_chain_cot, "The visible image already contains the answer.")
        self.assertEqual(output.direct_answer, "249")

    def test_executor_client_parses_json_tool_call_contract(self) -> None:
        executor = ExecutorClient(
            backend=FakeTextBackend(
                stage_responses={
                    "executor": json.dumps(
                        {
                            "think": "Start from the current crop and run OCR directly.",
                            "tool_call": {
                                "name": "code_image_tool",
                                "arguments": {
                                    "code": 'ocr = _call_ocr_assist(image_obj=image)\nprint(ocr.get("text", ""))\nresult = image',
                                    "description": "Run OCR on the current crop and keep it active.",
                                },
                            },
                        }
                    )
                }
            )
        )
        output = executor.run(self._build_executor_request())
        self.assertEqual(output.cot, "Start from the current crop and run OCR directly.")
        self.assertIn("_call_ocr_assist", output.code)
        self.assertEqual(output.description, "Run OCR on the current crop and keep it active.")

    def test_complex_multiround_branching_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            scenario = build_three_round_demo_scenario(root_dir)
            sample = scenario.sample
            sample_id = sample.sample_id

            root_id = build_root_trajectory_id(sample_id)
            traj_s1 = build_child_trajectory_id(root_id, 0, "s1")
            traj_s2 = build_child_trajectory_id(root_id, 0, "s2")
            traj_s3 = build_child_trajectory_id(root_id, 0, "s3")
            traj_s21 = build_child_trajectory_id(traj_s2, 1, "s21")
            traj_s22 = build_child_trajectory_id(traj_s2, 1, "s22")
            traj_s31 = build_child_trajectory_id(traj_s3, 1, "s31")
            traj_s32 = build_child_trajectory_id(traj_s3, 1, "s32")
            traj_s221 = build_child_trajectory_id(traj_s22, 2, "s221")

            planner = scenario.planner_client
            executor = scenario.executor_client
            runtime = scenario.runtime
            judge_backend = scenario.judge_backend
            judge = JudgeClient(backend=judge_backend)
            store = OfflineTrajectoryStore(root_dir / "run_outputs")
            orchestrator = OrchestratorV01(
                store=store,
                planner_client=planner,
                executor_client=executor,
                judge_client=judge,
                runtime=runtime,
                config=scenario.config,
            )

            result = orchestrator.run(sample)
            trajectories = {
                item.trajectory_id: item
                for item in store.list_trajectories(sample_id=sample_id)
            }

            self.assertEqual(result.sample_id, sample_id)
            self.assertEqual(
                set(result.all_trajectory_ids),
                {
                    root_id,
                    traj_s1,
                    traj_s2,
                    traj_s3,
                    traj_s21,
                    traj_s22,
                    traj_s221,
                },
            )
            self.assertNotIn(traj_s31, trajectories)
            self.assertNotIn(traj_s32, trajectories)
            self.assertEqual(trajectories[root_id].status, "expanded")
            self.assertEqual(trajectories[traj_s1].status, "answered")
            self.assertEqual(trajectories[traj_s2].status, "expanded")
            self.assertEqual(trajectories[traj_s3].status, "stopped_early")
            self.assertEqual(trajectories[traj_s21].status, "answered")
            self.assertEqual(trajectories[traj_s22].status, "expanded")
            self.assertEqual(trajectories[traj_s221].status, "answered")
            self.assertEqual(trajectories[traj_s1].final_answer, "249")
            self.assertEqual(trajectories[traj_s21].final_answer, "249")
            self.assertEqual(trajectories[traj_s221].final_answer, "249")
            self.assertFalse(result.running_trajectory_ids)

            messages_s2 = store.load_messages(sample_id, traj_s2).root
            self.assertEqual(
                [message.metadata.get("message_kind") for message in messages_s2],
                ["system_instruction", "user_question", "executor_step", "tool_result"],
            )
            self.assertIn("_call_ground_box", messages_s2[2].content)
            self.assertIn("_call_dino_crop", messages_s2[2].content)
            self.assertIn('"name": "code_image_tool"', messages_s2[2].content)
            self.assertIn('"image_index": 0', messages_s2[2].content)
            self.assertEqual(messages_s2[3].content, "cropped serial region")

            messages_s21 = store.load_messages(sample_id, traj_s21).root
            self.assertEqual(
                [message.metadata.get("message_kind") for message in messages_s21],
                [
                    "system_instruction",
                    "user_question",
                    "executor_step",
                    "tool_result",
                    "executor_step",
                    "tool_result",
                    "final_answer",
                ],
            )
            self.assertIn("<answer>\n249\n</answer>", messages_s21[-1].content)

            messages_s221 = store.load_messages(sample_id, traj_s221).root
            self.assertEqual(
                [message.metadata.get("message_kind") for message in messages_s221],
                [
                    "system_instruction",
                    "user_question",
                    "executor_step",
                    "tool_result",
                    "executor_step",
                    "tool_result",
                    "executor_step",
                    "tool_result",
                    "final_answer",
                ],
            )
            self.assertIn("<answer>\n249\n</answer>", messages_s221[-1].content)

            planner_requests = {
                (request.trajectory_id, request.round_idx): request
                for request in planner.requests
            }
            self.assertEqual(
                [image.artifact_id for image in planner_requests[(root_id, 0)].visible_images],
                ["img_root_0"],
            )
            self.assertEqual(
                [image.artifact_id for image in planner_requests[(traj_s2, 1)].visible_images],
                ["img_root_0", "img_step_001_0"],
            )
            self.assertIsNotNone(planner_requests[(traj_s2, 1)].latest_runtime_result)
            self.assertEqual(
                planner_requests[(traj_s2, 1)].latest_runtime_result.text,
                "cropped serial region",
            )
            self.assertIn("_call_ground_box", planner_requests[(traj_s2, 1)].messages[2].content)
            self.assertIn("_call_dino_crop", planner_requests[(traj_s2, 1)].messages[2].content)
            self.assertEqual(
                [image.artifact_id for image in planner_requests[(traj_s21, 2)].visible_images],
                ["img_root_0", "img_step_001_0", "img_step_002_0"],
            )
            self.assertTrue(planner_requests[(traj_s221, 3)].must_answer_now)
            self.assertEqual(planner_requests[(traj_s221, 3)].requested_suggestion_count, 0)

            executor_requests = {
                (request.trajectory_id, request.step_idx): request
                for request in executor.requests
            }
            self.assertEqual(
                executor_requests[(traj_s21, 2)].planner_global_chain_cot,
                (
                    "Round 1 rethink on s2: the prior crop looks good. One branch should continue the "
                    "existing crop with OCR; another should re-ground and recrop in case the crop is biased."
                ),
            )
            self.assertEqual(
                executor_requests[(traj_s21, 2)].suggestion_cot,
                "Continue from the previous crop and run OCR directly because the prior tool call is still visible.",
            )
            self.assertEqual(
                executor_requests[(traj_s22, 2)].suggestion_cot,
                "Revisit the earlier crop by grounding again, then OCR the refreshed view.",
            )
            self.assertEqual(executor_requests[(traj_s21, 2)].step_spec.input_image_index, 1)
            self.assertEqual(executor_requests[(traj_s22, 2)].step_spec.input_image_index, 0)

            runtime_requests = {
                (request.trajectory_id, request.step_idx): request
                for request in runtime.requests
            }
            self.assertEqual(runtime_requests[(traj_s1, 1)].image_index, 0)
            self.assertEqual(runtime_requests[(traj_s21, 2)].image_index, 1)
            self.assertEqual(runtime_requests[(traj_s22, 2)].image_index, 0)

            self.assertEqual(trajectories[traj_s21].steps[-1].input_image_index, 1)
            self.assertEqual(trajectories[traj_s21].steps[-1].input_artifact_id, "img_step_001_0")
            self.assertTrue(trajectories[traj_s21].steps[-1].executor_description)
            self.assertEqual(trajectories[traj_s22].steps[-1].input_image_index, 0)
            self.assertEqual(trajectories[traj_s22].steps[-1].input_artifact_id, "img_root_0")

            self.assertFalse(
                store.planner_output_path(sample_id, traj_s3, 1).exists(),
                "A branch dropped after step-judge frontier selection should not enter another planner round.",
            )
            self.assertTrue(
                store.build_step_file_paths(sample_id, traj_s221, 3, create_dirs=False).runtime_result_path.exists()
            )

    def test_zero_exec_budget_still_allows_direct_answer_and_blocks_expansion(self) -> None:
        cases = [("single_round", Budget(remaining_exec_steps=1))]
        for case_name, budget in cases:
            with self.subTest(case=case_name):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root_dir = Path(tmpdir)
                    sample = build_demo_root_sample(root_dir, sample_id=f"demo__train__{case_name}")
                    sample_id = sample.sample_id
                    root_id = build_root_trajectory_id(sample_id)

                    planner = ScriptedPlannerClient(
                        {
                            (root_id, 0): make_planner_output(
                                sample_id=sample_id,
                                trajectory_id=root_id,
                                round_idx=0,
                                global_chain_cot="Budget exhausted for execution, but the current context is already enough to answer.",
                                direct_answer="249",
                            )
                        }
                    )
                    executor = ScriptedExecutorClient({})
                    runtime = ScriptedRuntime({})
                    judge = JudgeClient(backend=ScriptedJudgeBackend({(root_id, 0): 0.0}))
                    store = OfflineTrajectoryStore(root_dir / "run_outputs")
                    orchestrator = OrchestratorV01(
                        store=store,
                        planner_client=planner,
                        executor_client=executor,
                        judge_client=judge,
                        runtime=runtime,
                        config=OrchestratorConfig(default_budget=budget),
                    )

                    result = orchestrator.run(sample)
                    trajectories = store.list_trajectories(sample_id=sample_id)
                    self.assertEqual(len(trajectories), 1)
                    self.assertEqual(trajectories[0].status, "answered")
                    self.assertEqual(trajectories[0].final_answer, "249")
                    self.assertEqual(result.all_trajectory_ids, [root_id])
                    self.assertFalse(executor.requests)
                    self.assertFalse(runtime.requests)
                    self.assertEqual(planner.requests[0].requested_suggestion_count, 2)

    def test_must_answer_threshold_takes_priority_over_stop_policy_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            sample = build_demo_root_sample(root_dir, sample_id="chartqa__train__must_answer_priority")
            sample = sample.model_copy(
                update={
                    "question": "What value is shown?",
                    "answer": "75",
                    "metadata": {"source_dataset": "chartqa"},
                }
            )
            sample_id = sample.sample_id
            root_id = build_root_trajectory_id(sample_id)
            child_id = build_child_trajectory_id(root_id, planner_round_idx=0, suggestion_id="s1")

            planner = ScriptedPlannerClient(
                {
                    (root_id, 0): make_planner_output(
                        sample_id=sample_id,
                        trajectory_id=root_id,
                        round_idx=0,
                        global_chain_cot="Use one visual check before answering.",
                        suggestions=[
                            make_suggestion(
                                "s1",
                                "Inspect the chart value before finalizing.",
                                [make_step("step_1", "Inspect the chart value.", ["manual_crop"])],
                            )
                        ],
                    ),
                    (child_id, 1): make_planner_output(
                        sample_id=sample_id,
                        trajectory_id=child_id,
                        round_idx=1,
                        global_chain_cot="The judged evidence is strong enough to answer directly.",
                        direct_answer="75",
                    ),
                }
            )
            executor = ScriptedExecutorClient(
                {
                    (child_id, 1): make_executor_output(
                        "Crop the relevant chart region.",
                        "result = image",
                        description="Keep the chart evidence visible.",
                    )
                }
            )
            runtime = ScriptedRuntime(
                {
                    (child_id, 1): RuntimeSpec(
                        text="chart evidence supports 75",
                        helper_names=["manual_crop"],
                        image_label="chart evidence",
                    )
                }
            )
            judge = JudgeClient(backend=ScriptedJudgeBackend({(root_id, 0): 0.80, (child_id, 1): 0.75}))
            store = OfflineTrajectoryStore(root_dir / "run_outputs")
            orchestrator = OrchestratorV01(
                store=store,
                planner_client=planner,
                executor_client=executor,
                judge_client=judge,
                runtime=runtime,
                config=OrchestratorConfig(default_budget=Budget(remaining_exec_steps=3)),
            )

            orchestrator.run(sample)
            child = store.load_trajectory(sample_id, child_id)
            child_request = planner.requests[-1]
            child_judge = store.load_judge_record(sample_id, child_id, "cheap_filter", scope_step_idx=1)

            self.assertEqual(child.status, "answered")
            self.assertEqual(child.final_answer, "75")
            self.assertTrue(child_request.must_answer_now)
            self.assertEqual(child_request.requested_suggestion_count, 0)
            self.assertEqual(
                child_request.metadata.get("planning_policy_reason"),
                "score_at_or_above_must_answer_threshold",
            )
            self.assertEqual(
                child_judge.metadata.get("stop_policy", {}).get("stop_reason"),
                "regressed",
            )
            self.assertEqual(
                child_judge.metadata.get("stop_policy_must_answer_override", {}).get("suppressed_stop_reason"),
                "regressed",
            )

    def _build_planner_request(self) -> PlannerClientRequest:
        return PlannerClientRequest(
            sample_id="demo__train__planner_parse",
            trajectory_id="traj__demo__train__planner_parse__root",
            round_idx=0,
            question="What number is written on the hanging tag?",
            messages=[
                ConversationMessage(
                    message_id="m_system",
                    role="system",
                    content="You are a helpful planner.",
                ),
                ConversationMessage(
                    message_id="m_user",
                    role="user",
                    content="What number is written on the hanging tag?",
                ),
            ],
            visible_images=[
                ImageArtifactRef(
                    artifact_id="img_root_0",
                    path="/tmp/fake_root.png",
                    media_type="image/png",
                )
            ],
            budget=Budget(remaining_exec_steps=2),
        )

    def test_client_fake_backend_round_trips_through_real_clients(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            spec = build_three_round_demo_spec(root_dir)
            text_backend = ScriptedTextBackend(
                planner_outputs=spec.planner_outputs,
                executor_outputs=spec.executor_outputs,
            )
            planner = PlannerClient(backend=text_backend)
            executor = ExecutorClient(backend=text_backend)
            runtime = ScriptedRuntime(spec.runtime_specs)
            judge = JudgeClient(backend=ScriptedJudgeBackend(spec.judge_scores))
            store = OfflineTrajectoryStore(root_dir / "run_outputs")
            orchestrator = OrchestratorV01(
                store=store,
                planner_client=planner,
                executor_client=executor,
                judge_client=judge,
                runtime=runtime,
                config=spec.config,
            )

            result = orchestrator.run(spec.sample)
            trajectories = {
                item.trajectory_id: item
                for item in store.list_trajectories(sample_id=spec.sample.sample_id)
            }

            self.assertEqual(
                set(result.all_trajectory_ids),
                {
                    "traj__demo__train__0001__root",
                    "traj__demo__train__0001__root__r000_s1",
                    "traj__demo__train__0001__root__r000_s2",
                    "traj__demo__train__0001__root__r000_s3",
                    "traj__demo__train__0001__root__r000_s2__r001_s21",
                    "traj__demo__train__0001__root__r000_s2__r001_s22",
                    "traj__demo__train__0001__root__r000_s2__r001_s22__r002_s221",
                },
            )
            self.assertEqual(trajectories["traj__demo__train__0001__root__r000_s1"].status, "answered")
            self.assertEqual(trajectories["traj__demo__train__0001__root__r000_s2"].status, "expanded")
            self.assertEqual(trajectories["traj__demo__train__0001__root__r000_s3"].status, "stopped_early")
            self.assertEqual(trajectories["traj__demo__train__0001__root__r000_s2__r001_s21"].final_answer, "249")
            self.assertFalse(result.running_trajectory_ids)
            self.assertGreaterEqual(len(text_backend.requests), 10)
            self.assertEqual(text_backend.requests[0]["stage"], "planner")
            self.assertEqual(text_backend.requests[0]["trajectory_id"], "traj__demo__train__0001__root")
            self.assertEqual(text_backend.requests[0]["round_idx"], 0)
            self.assertIsNone(text_backend.requests[0]["step_idx"])
            self.assertEqual(
                [item["stage"] for item in text_backend.requests[:4]],
                ["planner", "executor", "executor", "executor"],
            )

    def _build_executor_request(self) -> ExecutorClientRequest:
        return ExecutorClientRequest(
            sample_id="demo__train__executor_parse",
            trajectory_id="traj__demo__train__executor_parse__r001_s1",
            round_idx=1,
            step_idx=2,
            question="What number is written on the hanging tag?",
            messages=[
                ConversationMessage(
                    message_id="m_user",
                    role="user",
                    content="What number is written on the hanging tag?",
                ),
                ConversationMessage(
                    message_id="m_tool",
                    role="tool",
                    content="cropped serial region",
                    image_artifact_ids=["img_step_001_0"],
                ),
            ],
            visible_images=[
                ImageArtifactRef(
                    artifact_id="img_root_0",
                    path="/tmp/fake_root.png",
                    media_type="image/png",
                ),
                ImageArtifactRef(
                    artifact_id="img_step_001_0",
                    path="/tmp/fake_step.png",
                    media_type="image/png",
                ),
            ],
            suggestion_id="s21",
            suggestion_step_index=0,
            step_spec=PlannerStepSpec(
                step_id="step_continue_ocr",
                step_goal="Continue with OCR on the latest crop.",
                input_image_index=1,
                capability_plan=[
                    CapabilityPlanItem(order=1, capability="ocr_assist", instruction="Read the text from the crop.")
                ],
                executor_instruction="Run OCR on the latest crop and keep the crop as result.",
            ),
            planner_global_chain_cot="The previous crop already isolates the right region.",
            suggestion_cot="Continue from the previous crop and OCR it directly.",
            tool_capabilities=[
                ToolCapability(name="ocr_assist", description="Read text from the current image.")
            ],
        )

    def test_forced_final_answer_keeps_candidate_answer_in_audit_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            sample = build_demo_root_sample(root_dir, sample_id="fsc147__train__demo")
            sample = sample.model_copy(
                update={
                    "question": "How many pigeons are there in the image?",
                    "answer": "21",
                    "metadata": {"source_dataset": "fsc147"},
                }
            )
            sample_id = sample.sample_id
            root_id = build_root_trajectory_id(sample_id)
            child_id = build_child_trajectory_id(root_id, planner_round_idx=0, suggestion_id="s1")

            planner = ScriptedPlannerClient(
                {
                    (root_id, 0): make_planner_output(
                        sample_id=sample_id,
                        trajectory_id=root_id,
                        round_idx=0,
                        global_chain_cot="Need one counting step before answering.",
                        suggestions=[
                            make_suggestion(
                                "s1",
                                "Use counting to estimate the bird count.",
                                [make_step("step_1", "Count the birds.", ["count_assist"])],
                            )
                        ],
                    ),
                    (child_id, 1): make_planner_output(
                        sample_id=sample_id,
                        trajectory_id=child_id,
                        round_idx=1,
                        global_chain_cot="The judge-backed count is strong, but I think the image looks like 22.",
                        direct_answer="22",
                    ),
                }
            )
            executor = ScriptedExecutorClient(
                {
                    (child_id, 1): make_executor_output(
                        "Use count assist to count the birds.",
                        'count = _call_count_assist("pigeon")\nresult = image',
                        description="Count the pigeons.",
                    )
                }
            )
            runtime = ScriptedRuntime(
                {
                    (child_id, 1): RuntimeSpec(
                        text="counted 21 pigeons",
                        helper_names=["count_assist"],
                        image_label="counted birds",
                    )
                }
            )
            judge = JudgeClient(backend=_ConsensusJudgeBackend({(root_id, 0): 0.8, (child_id, 1): 1.0}, candidate_answer="21"))
            store = OfflineTrajectoryStore(root_dir / "run_outputs")
            orchestrator = OrchestratorV01(
                store=store,
                planner_client=planner,
                executor_client=executor,
                judge_client=judge,
                runtime=runtime,
                config=OrchestratorConfig(default_budget=Budget(remaining_exec_steps=3)),
            )

            orchestrator.run(sample)
            child = store.load_trajectory(sample_id, child_id)
            planner_request = planner.requests[-1]
            planner_output = store.load_planner_output(sample_id, child_id, 1)

            self.assertEqual(child.status, "answered")
            self.assertEqual(child.final_answer, "22")
            self.assertTrue(planner_request.must_answer_now)
            self.assertEqual(planner_request.requested_suggestion_count, 0)
            self.assertEqual(
                planner_request.metadata.get("forced_final_answer_audit", {}).get("candidate_answer"),
                "21",
            )
            self.assertNotIn("forced_final_answer", planner_request.metadata)
            self.assertTrue(planner_output.can_answer_now)
            self.assertEqual(planner_output.direct_answer, "22")

    def test_gqa_forced_final_answer_uses_normalized_reference_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            sample = build_demo_root_sample(root_dir, sample_id="gqa__train__demo")
            sample = sample.model_copy(
                update={
                    "question": "What animal is on the sofa?",
                    "answer": "cat",
                    "metadata": {"source_dataset": "gqa"},
                }
            )
            sample_id = sample.sample_id
            root_id = build_root_trajectory_id(sample_id)
            child_id = build_child_trajectory_id(root_id, planner_round_idx=0, suggestion_id="s1")

            planner = ScriptedPlannerClient(
                {
                    (root_id, 0): make_planner_output(
                        sample_id=sample_id,
                        trajectory_id=root_id,
                        round_idx=0,
                        global_chain_cot="Need one localization step before answering.",
                        suggestions=[
                            make_suggestion(
                                "s1",
                                "Inspect the sofa region closely.",
                                [make_step("step_1", "Check the sofa area.", ["ground_box"])],
                            )
                        ],
                    ),
                    (child_id, 1): make_planner_output(
                        sample_id=sample_id,
                        trajectory_id=child_id,
                        round_idx=1,
                        global_chain_cot="The judged answer looks reliable now.",
                        direct_answer="cat",
                    ),
                }
            )
            executor = ScriptedExecutorClient(
                {
                    (child_id, 1): make_executor_output(
                        "Inspect the sofa region.",
                        'box = _call_ground_box("sofa")\nresult = image',
                        description="Inspect the sofa area.",
                    )
                }
            )
            runtime = ScriptedRuntime(
                {
                    (child_id, 1): RuntimeSpec(
                        text="observed a cat on the sofa",
                        helper_names=["ground_box"],
                        image_label="sofa region",
                    )
                }
            )
            judge = JudgeClient(
                backend=_ReferenceAwareJudgeBackend(
                    {(root_id, 0): 0.50, (child_id, 1): 0.9},
                    normalized_reference="cat",
                )
            )
            store = OfflineTrajectoryStore(root_dir / "run_outputs")
            orchestrator = OrchestratorV01(
                store=store,
                planner_client=planner,
                executor_client=executor,
                judge_client=judge,
                runtime=runtime,
                config=OrchestratorConfig(default_budget=Budget(remaining_exec_steps=4)),
            )

            orchestrator.run(sample)
            planner_request = planner.requests[-1]
            child = store.load_trajectory(sample_id, child_id)

            self.assertEqual(child.status, "answered")
            self.assertEqual(child.final_answer, "cat")
            self.assertTrue(planner_request.must_answer_now)
            self.assertEqual(planner_request.requested_suggestion_count, 0)
            self.assertEqual(
                planner_request.metadata.get("forced_final_answer_audit", {}).get("reason"),
                "exact_match_high_score_reference",
            )
            self.assertEqual(
                planner_request.metadata.get("forced_final_answer_audit", {}).get("candidate_answer"),
                "cat",
            )
            self.assertNotIn("forced_final_answer", planner_request.metadata)

    def test_gqa_first_round_must_suggest_takes_priority_over_forced_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            sample = build_demo_root_sample(root_dir, sample_id="gqa__train__first_round_priority")
            sample = sample.model_copy(
                update={
                    "question": "What animal is on the sofa?",
                    "answer": "cat",
                    "metadata": {"source_dataset": "gqa"},
                }
            )
            sample_id = sample.sample_id
            root_id = build_root_trajectory_id(sample_id)
            child_id = build_child_trajectory_id(root_id, planner_round_idx=0, suggestion_id="s1")

            planner = ScriptedPlannerClient(
                {
                    (root_id, 0): make_planner_output(
                        sample_id=sample_id,
                        trajectory_id=root_id,
                        round_idx=0,
                        global_chain_cot="Even with a strong root judge, inspect the sofa once before answering.",
                        suggestions=[
                            make_suggestion(
                                "s1",
                                "Inspect the sofa region closely.",
                                [make_step("step_1", "Check the sofa area.", ["ground_box"])],
                            )
                        ],
                    ),
                    (child_id, 1): make_planner_output(
                        sample_id=sample_id,
                        trajectory_id=child_id,
                        round_idx=1,
                        global_chain_cot="The judged answer is now reliable enough to finalize.",
                        direct_answer="cat",
                    ),
                }
            )
            executor = ScriptedExecutorClient(
                {
                    (child_id, 1): make_executor_output(
                        "Inspect the sofa region.",
                        'box = _call_ground_box("sofa")\nresult = image',
                        description="Inspect the sofa area.",
                    )
                }
            )
            runtime = ScriptedRuntime(
                {
                    (child_id, 1): RuntimeSpec(
                        text="observed a cat on the sofa",
                        helper_names=["ground_box"],
                        image_label="sofa region",
                    )
                }
            )
            judge = JudgeClient(
                backend=_ReferenceAwareJudgeBackend(
                    {(root_id, 0): 1.0, (child_id, 1): 1.0},
                    normalized_reference="cat",
                )
            )
            store = OfflineTrajectoryStore(root_dir / "run_outputs")
            orchestrator = OrchestratorV01(
                store=store,
                planner_client=planner,
                executor_client=executor,
                judge_client=judge,
                runtime=runtime,
                config=OrchestratorConfig(default_budget=Budget(remaining_exec_steps=4)),
            )

            orchestrator.run(sample)
            root_request = planner.requests[0]
            child_request = planner.requests[-1]
            child = store.load_trajectory(sample_id, child_id)

            self.assertFalse(root_request.must_answer_now)
            self.assertEqual(root_request.requested_suggestion_count, 2)
            self.assertTrue(child_request.must_answer_now)
            self.assertEqual(
                child_request.metadata.get("forced_final_answer_audit", {}).get("candidate_answer"),
                "cat",
            )
            self.assertNotIn("forced_final_answer", child_request.metadata)
            self.assertEqual(child.status, "answered")
            self.assertEqual(child.final_answer, "cat")

    def test_perfect_reference_score_short_circuits_remaining_rollout_for_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            sample = build_demo_root_sample(root_dir, sample_id="gqa__train__perfect_exit")
            sample = sample.model_copy(
                update={
                    "question": "What animal is on the sofa?",
                    "answer": "cat",
                    "metadata": {"source_dataset": "gqa"},
                }
            )
            sample_id = sample.sample_id
            root_id = build_root_trajectory_id(sample_id)
            child_id = build_child_trajectory_id(root_id, planner_round_idx=0, suggestion_id="s1")
            skipped_child_id = build_child_trajectory_id(root_id, planner_round_idx=0, suggestion_id="s2")

            planner = ScriptedPlannerClient(
                {
                    (root_id, 0): make_planner_output(
                        sample_id=sample_id,
                        trajectory_id=root_id,
                        round_idx=0,
                        global_chain_cot="Try two localization branches if needed.",
                        suggestions=[
                            make_suggestion(
                                "s1",
                                "Inspect the sofa region first.",
                                [make_step("step_1", "Check the sofa area.", ["ground_box"])],
                            ),
                            make_suggestion(
                                "s2",
                                "Inspect the floor area as a fallback.",
                                [make_step("step_1", "Check the floor area.", ["ground_box"])],
                            ),
                        ],
                    ),
                    (child_id, 1): make_planner_output(
                        sample_id=sample_id,
                        trajectory_id=child_id,
                        round_idx=1,
                        global_chain_cot="The judged answer is already reliable, so finalize now.",
                        direct_answer="cat",
                    ),
                }
            )
            executor = ScriptedExecutorClient(
                {
                    (child_id, 1): make_executor_output(
                        "Inspect the sofa region.",
                        'box = _call_ground_box("sofa")\nresult = image',
                        description="Inspect the sofa area.",
                    )
                }
            )
            runtime = ScriptedRuntime(
                {
                    (child_id, 1): RuntimeSpec(
                        text="observed a cat on the sofa",
                        helper_names=["ground_box"],
                        image_label="sofa region",
                    )
                }
            )
            judge = JudgeClient(
                backend=_ReferenceAwareJudgeBackend(
                    {(root_id, 0): 0.50, (child_id, 1): 1.0},
                    normalized_reference="cat",
                )
            )
            store = OfflineTrajectoryStore(root_dir / "run_outputs")
            orchestrator = OrchestratorV01(
                store=store,
                planner_client=planner,
                executor_client=executor,
                judge_client=judge,
                runtime=runtime,
                config=OrchestratorConfig(default_budget=Budget(remaining_exec_steps=3)),
            )

            result = orchestrator.run(sample)
            child = store.load_trajectory(sample_id, child_id)
            messages = store.load_messages(sample_id, child_id).root
            planner_request = planner.requests[-1]
            all_trajectory_ids = {item.trajectory_id for item in store.list_trajectories(sample_id=sample_id)}

            self.assertEqual(child.status, "answered")
            self.assertEqual(child.final_answer, "cat")
            self.assertEqual(result.running_trajectory_ids, [])
            self.assertEqual(len(planner.requests), 2)
            self.assertTrue(planner_request.must_answer_now)
            self.assertNotIn(skipped_child_id, all_trajectory_ids)
            self.assertEqual(messages[-1].metadata.get("final_answer_source"), "planner")
            self.assertEqual(
                planner_request.metadata.get("forced_final_answer_audit", {}).get("candidate_answer"),
                "cat",
            )
            self.assertNotIn("forced_final_answer", planner_request.metadata)

    def test_perfect_count_score_short_circuits_remaining_rollout_for_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            sample = build_demo_root_sample(root_dir, sample_id="fsc147__train__perfect_exit")
            sample = sample.model_copy(
                update={
                    "question": "How many beads are there in the image?",
                    "answer": "25",
                    "metadata": {"source_dataset": "fsc147"},
                }
            )
            sample_id = sample.sample_id
            root_id = build_root_trajectory_id(sample_id)
            child_id = build_child_trajectory_id(root_id, planner_round_idx=0, suggestion_id="s1")
            skipped_child_id = build_child_trajectory_id(root_id, planner_round_idx=0, suggestion_id="s2")

            planner = ScriptedPlannerClient(
                {
                    (root_id, 0): make_planner_output(
                        sample_id=sample_id,
                        trajectory_id=root_id,
                        round_idx=0,
                        global_chain_cot="Try the direct counting branch first, then a fallback transform if needed.",
                        suggestions=[
                            make_suggestion(
                                "s1",
                                "Count the beads directly.",
                                [make_step("step_1", "Count the beads.", ["count_assist"])],
                            ),
                            make_suggestion(
                                "s2",
                                "Increase contrast as a fallback before counting.",
                                [make_step("step_1", "Boost contrast first.", ["adjust_contrast"])],
                            ),
                        ],
                    ),
                    (child_id, 1): make_planner_output(
                        sample_id=sample_id,
                        trajectory_id=child_id,
                        round_idx=1,
                        global_chain_cot="The counted result is already fully reliable, so finalize now.",
                        direct_answer="25",
                    ),
                }
            )
            executor = ScriptedExecutorClient(
                {
                    (child_id, 1): make_executor_output(
                        "Count the beads directly.",
                        'count = _call_count_assist("beads")\nresult = image',
                        description="Count the beads.",
                    )
                }
            )
            runtime = ScriptedRuntime(
                {
                    (child_id, 1): RuntimeSpec(
                        text="counted 25 beads",
                        helper_names=["count_assist"],
                        image_label="counted beads",
                    )
                }
            )
            judge = JudgeClient(
                backend=_ConsensusJudgeBackend(
                    {(root_id, 0): 0.50, (child_id, 1): 1.0},
                    candidate_answer="25",
                )
            )
            store = OfflineTrajectoryStore(root_dir / "run_outputs")
            orchestrator = OrchestratorV01(
                store=store,
                planner_client=planner,
                executor_client=executor,
                judge_client=judge,
                runtime=runtime,
                config=OrchestratorConfig(default_budget=Budget(remaining_exec_steps=3)),
            )

            result = orchestrator.run(sample)
            child = store.load_trajectory(sample_id, child_id)
            messages = store.load_messages(sample_id, child_id).root
            planner_request = planner.requests[-1]
            all_trajectory_ids = {item.trajectory_id for item in store.list_trajectories(sample_id=sample_id)}

            self.assertEqual(child.status, "answered")
            self.assertEqual(child.final_answer, "25")
            self.assertEqual(result.running_trajectory_ids, [])
            self.assertEqual(len(planner.requests), 2)
            self.assertTrue(planner_request.must_answer_now)
            self.assertNotIn(skipped_child_id, all_trajectory_ids)
            self.assertEqual(messages[-1].metadata.get("final_answer_source"), "planner")
            self.assertEqual(
                planner_request.metadata.get("forced_final_answer_audit", {}).get("candidate_answer"),
                "25",
            )
            self.assertNotIn("forced_final_answer", planner_request.metadata)

    def test_textvqa_forced_final_answer_uses_majority_reference_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            sample = build_demo_root_sample(root_dir, sample_id="textvqa__train__demo")
            sample = sample.model_copy(
                update={
                    "question": "what does the ad say?",
                    "answer": [
                        "firekeepers",
                        "firekeepers",
                        "firekeepers",
                        "fire keepers",
                        "fire keepers",
                        "firekeepers casino",
                        "firekeepers casino",
                        "casino",
                        "firekeepers",
                        "firekeepers casino",
                    ],
                    "metadata": {"source_dataset": "textvqa"},
                }
            )
            sample_id = sample.sample_id
            root_id = build_root_trajectory_id(sample_id)
            child_id = build_child_trajectory_id(root_id, planner_round_idx=0, suggestion_id="s1")

            planner = ScriptedPlannerClient(
                {
                    (root_id, 0): make_planner_output(
                        sample_id=sample_id,
                        trajectory_id=root_id,
                        round_idx=0,
                        global_chain_cot="Need one OCR-oriented step before answering.",
                        suggestions=[
                            make_suggestion(
                                "s1",
                                "Inspect the ad text closely.",
                                [make_step("step_1", "Check the ad area.", ["ocr_assist"])],
                            )
                        ],
                    ),
                    (child_id, 1): make_planner_output(
                        sample_id=sample_id,
                        trajectory_id=child_id,
                        round_idx=1,
                        global_chain_cot="The judged answer looks reliable now.",
                        direct_answer="firekeepers",
                    ),
                }
            )
            executor = ScriptedExecutorClient(
                {
                    (child_id, 1): make_executor_output(
                        "Inspect the ad text.",
                        'ocr = _call_ocr_assist()\nresult = image',
                        description="Inspect the ad text.",
                    )
                }
            )
            runtime = ScriptedRuntime(
                {
                    (child_id, 1): RuntimeSpec(
                        text="observed ad text",
                        helper_names=["ocr_assist"],
                        image_label="ad text",
                    )
                }
            )
            judge = JudgeClient(
                backend=_ReferenceAwareJudgeBackend(
                    {(root_id, 0): 0.30, (child_id, 1): 0.9},
                    normalized_reference=[
                        "firekeepers",
                        "firekeepers",
                        "firekeepers",
                        "fire keepers",
                        "fire keepers",
                        "firekeepers casino",
                        "firekeepers casino",
                        "casino",
                        "firekeepers",
                        "firekeepers casino",
                    ],
                )
            )
            store = OfflineTrajectoryStore(root_dir / "run_outputs")
            orchestrator = OrchestratorV01(
                store=store,
                planner_client=planner,
                executor_client=executor,
                judge_client=judge,
                runtime=runtime,
                config=OrchestratorConfig(default_budget=Budget(remaining_exec_steps=3)),
            )

            orchestrator.run(sample)
            planner_request = planner.requests[-1]
            child = store.load_trajectory(sample_id, child_id)

            self.assertEqual(child.status, "answered")
            self.assertEqual(child.final_answer, "firekeepers")
            self.assertTrue(planner_request.must_answer_now)
            self.assertEqual(
                planner_request.metadata.get("forced_final_answer_audit", {}).get("reason"),
                "textvqa_high_score_reference",
            )
            self.assertEqual(
                planner_request.metadata.get("forced_final_answer_audit", {}).get("candidate_answer"),
                "firekeepers",
            )
            self.assertNotIn("forced_final_answer", planner_request.metadata)

    def test_zero_exec_budget_without_direct_answer_marks_max_step_reached(self) -> None:
        cases = [("zero_depth", Budget(remaining_exec_steps=0))]
        for case_name, budget in cases:
            with self.subTest(case=case_name):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root_dir = Path(tmpdir)
                    sample = build_demo_root_sample(root_dir, sample_id=f"demo__train__{case_name}_suggest")
                    sample_id = sample.sample_id
                    root_id = build_root_trajectory_id(sample_id)

                    planner = ScriptedPlannerClient(
                        {
                            (root_id, 0): make_planner_output(
                                sample_id=sample_id,
                                trajectory_id=root_id,
                                round_idx=0,
                                global_chain_cot="Forced answer round still refuses to answer.",
                                suggestions=[
                                    make_suggestion(
                                        "s1",
                                        "This invalid suggestion should be ignored in must-answer-now mode.",
                                        [
                                            make_step(
                                                "step_invalid",
                                                "Do not execute because tool budget is exhausted.",
                                                ["ground_box"],
                                                input_image_index=0,
                                            )
                                        ],
                                    )
                                ],
                            )
                        }
                    )
                    executor = ScriptedExecutorClient({})
                    runtime = ScriptedRuntime({})
                    judge = JudgeClient(backend=ScriptedJudgeBackend({(root_id, 0): 0.0}))
                    store = OfflineTrajectoryStore(root_dir / "run_outputs")
                    orchestrator = OrchestratorV01(
                        store=store,
                        planner_client=planner,
                        executor_client=executor,
                        judge_client=judge,
                        runtime=runtime,
                        config=OrchestratorConfig(default_budget=budget),
                    )

                    result = orchestrator.run(sample)
                    trajectories = store.list_trajectories(sample_id=sample_id)
                    self.assertEqual(len(trajectories), 1)
                    self.assertEqual(trajectories[0].status, "max_step_reached")
                    self.assertEqual(result.terminal_trajectory_ids, [root_id])
                    self.assertEqual(len(planner.requests), 1)
                    self.assertTrue(planner.requests[0].must_answer_now)
                    self.assertFalse(executor.requests)
                    self.assertFalse(runtime.requests)


if __name__ == "__main__":
    unittest.main()
