from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from offline_sft_pipeline.core.models import (
    Budget,
    build_child_trajectory_id,
    build_root_trajectory_id,
)
from offline_sft_pipeline.core.store import OfflineTrajectoryStore
from offline_sft_pipeline.pipelines.executor_client import ExecutorClient
from offline_sft_pipeline.pipelines.judge_client import JudgeClient
from offline_sft_pipeline.pipelines.orchestrator_v01 import OrchestratorConfig, OrchestratorV01
from offline_sft_pipeline.pipelines.planner_client import PlannerClient
from offline_sft_pipeline.pipelines.scripted_components import (
    ScriptedExecutorClient,
    ScriptedJudgeBackend,
    ScriptedPlannerClient,
    ScriptedRuntime,
    ScriptedTextBackend,
    build_demo_root_sample,
    build_three_round_demo_scenario,
    build_three_round_demo_spec,
    make_planner_output,
    make_step,
    make_suggestion,
)


class OrchestratorV01SmokeTest(unittest.TestCase):
    maxDiff = None

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
                    traj_s31,
                    traj_s221,
                },
            )
            self.assertNotIn(traj_s32, trajectories)
            self.assertEqual(trajectories[root_id].status, "expanded")
            self.assertEqual(trajectories[traj_s1].status, "answered")
            self.assertEqual(trajectories[traj_s2].status, "expanded")
            self.assertEqual(trajectories[traj_s3].status, "expanded")
            self.assertEqual(trajectories[traj_s21].status, "answered")
            self.assertEqual(trajectories[traj_s22].status, "expanded")
            self.assertEqual(trajectories[traj_s31].status, "failed")
            self.assertEqual(trajectories[traj_s221].status, "max_step_reached")
            self.assertEqual(trajectories[traj_s1].final_answer, "249")
            self.assertEqual(trajectories[traj_s21].final_answer, "249")
            self.assertFalse(result.running_trajectory_ids)

            messages_s2 = store.load_messages(sample_id, traj_s2).root
            self.assertEqual(
                [message.metadata.get("message_kind") for message in messages_s2],
                ["system_instruction", "user_question", "executor_step", "tool_result"],
            )
            self.assertIn("_call_ground_box", messages_s2[2].content)
            self.assertIn("_call_dino_crop", messages_s2[2].content)
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
                ],
            )

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
                ["img_root_0", "img_step_002_0"],
            )

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

            runtime_requests = {
                (request.trajectory_id, request.step_idx): request
                for request in runtime.requests
            }
            self.assertEqual(runtime_requests[(traj_s1, 1)].image_index, 0)
            self.assertEqual(runtime_requests[(traj_s21, 2)].image_index, 1)
            self.assertEqual(runtime_requests[(traj_s22, 2)].image_index, 1)

            runtime_result_s31 = store.load_runtime_result(sample_id, traj_s31, 2)
            self.assertFalse(runtime_result_s31.success)
            self.assertEqual(runtime_result_s31.error.message, "mock runtime failure on retry path")

            self.assertTrue(
                store.planner_output_path(sample_id, traj_s3, 1).exists(),
                "Second-round planner output should be saved for unselected suggestion inspection.",
            )
            self.assertTrue(
                store.build_step_file_paths(sample_id, traj_s221, 3, create_dirs=False).runtime_result_path.exists()
            )

    def test_zero_exec_budget_still_allows_direct_answer_and_blocks_expansion(self) -> None:
        cases = [
            ("no_child_budget", Budget(remaining_rounds=2, remaining_children=0, remaining_steps=2)),
            ("no_step_budget", Budget(remaining_rounds=2, remaining_children=2, remaining_steps=0)),
        ]
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
                    judge = JudgeClient(backend=ScriptedJudgeBackend({}))
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
                    self.assertIsNone(planner.requests[0].requested_suggestion_count)

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
                    "traj__demo__train__0001__root__r000_s2__r001_s21",
                    "traj__demo__train__0001__root__r000_s2__r001_s22",
                    "traj__demo__train__0001__root__r000_s2__r001_s22__r002_s221",
                    "traj__demo__train__0001__root__r000_s3",
                    "traj__demo__train__0001__root__r000_s3__r001_s31",
                },
            )
            self.assertEqual(trajectories["traj__demo__train__0001__root__r000_s1"].status, "answered")
            self.assertEqual(trajectories["traj__demo__train__0001__root__r000_s2"].status, "expanded")
            self.assertEqual(trajectories["traj__demo__train__0001__root__r000_s3__r001_s31"].status, "failed")
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

    def test_zero_exec_budget_without_direct_answer_marks_max_step_reached(self) -> None:
        cases = [
            ("no_child_budget", Budget(remaining_rounds=2, remaining_children=0, remaining_steps=2)),
            ("no_step_budget", Budget(remaining_rounds=2, remaining_children=2, remaining_steps=0)),
        ]
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
                                global_chain_cot="Execution budget is exhausted, so only a direct answer decision is allowed here.",
                                suggestions=[
                                    make_suggestion(
                                        "s1",
                                        "This suggestion should be recorded in planner history but never expanded.",
                                        [
                                            make_step(
                                                "step_blocked",
                                                "Attempt a blocked execution.",
                                                ["ocr_assist"],
                                            )
                                        ],
                                    )
                                ],
                            )
                        }
                    )
                    executor = ScriptedExecutorClient({})
                    runtime = ScriptedRuntime({})
                    judge = JudgeClient(backend=ScriptedJudgeBackend({}))
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
                    self.assertFalse(executor.requests)
                    self.assertFalse(runtime.requests)
                    self.assertIsNone(planner.requests[0].requested_suggestion_count)


if __name__ == "__main__":
    unittest.main()
