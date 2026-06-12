from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from offline_sft_pipeline.core.models import (
    Budget,
    ExecutorRuntimeResult,
    ForkProvenance,
    ImageArtifactRef,
    JudgeRecord,
    PendingExecution,
    PlannerOutput,
    RootSample,
    RuntimeCodeExecution,
    StepRecord,
    utc_now,
)
from offline_sft_pipeline.core.store import OfflineTrajectoryStore


class IncrementalForkStoreTests(unittest.TestCase):
    def test_child_trajectory_inherits_history_without_copying_parent_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_image = root / "source.png"
            Image.new("RGB", (8, 8), color=(255, 0, 0)).save(source_image)

            store = OfflineTrajectoryStore(root / "store")
            sample = RootSample(
                sample_id="demo__sample",
                question="What is shown?",
                images=[{"image_id": "img0", "path": str(source_image)}],
            )
            parent_init = store.init_root_trajectory(sample, budget=Budget(remaining_exec_steps=3))
            sample_id = sample.sample_id
            parent_id = parent_init.trajectory.trajectory_id

            planner_output = PlannerOutput.model_validate(
                {
                    "sample_id": sample_id,
                    "trajectory_id": parent_id,
                    "round_idx": 0,
                    "created_at": utc_now(),
                    "can_answer_now": False,
                    "global_chain_cot": "Inspect the object first.",
                    "suggestions": [
                        {
                            "suggestion_id": "s1",
                            "suggestion_cot": "Crop and inspect the image.",
                            "steps": [
                                {
                                    "step_id": "step_001",
                                    "step_goal": "Inspect the object.",
                                    "input_image_index": 0,
                                    "capability_plan": [
                                        {
                                            "order": 1,
                                            "capability": "manual_crop",
                                            "instruction": "Crop the interesting region.",
                                        }
                                    ],
                                    "executor_instruction": "Crop the region.",
                                }
                            ],
                        }
                    ],
                }
            )
            store.register_planner_round(planner_output, selected_for_expansion=True)

            step_paths = store.write_executor_step_files(
                sample_id,
                parent_id,
                1,
                executor_cot="Need one crop.",
                executor_code="result = image",
            )
            output_image = step_paths.step_dir / "output_0.png"
            Image.new("RGB", (6, 6), color=(0, 255, 0)).save(output_image)

            now = utc_now()
            runtime_result = ExecutorRuntimeResult(
                sample_id=sample_id,
                trajectory_id=parent_id,
                round_idx=0,
                step_idx=1,
                created_at=now,
                success=True,
                images=[
                    ImageArtifactRef(
                        artifact_id="img_step_001_0",
                        path=str(output_image),
                        media_type="image/png",
                        width=6,
                        height=6,
                    )
                ],
                text="cropped image",
                observed_helper_call_count=0,
                observed_helper_calls=[],
                code_execution=RuntimeCodeExecution(
                    code_path=str(step_paths.executor_code_path),
                    exit_code=0,
                    started_at=now,
                    finished_at=now,
                    elapsed_seconds=0.01,
                ),
                error=None,
            )
            runtime_result.to_json_file(step_paths.runtime_result_path)

            step_record = StepRecord(
                execution_trajectory_id=parent_id,
                step_idx=1,
                planner_round_idx=0,
                suggestion_id="s1",
                suggestion_step_index=0,
                step_id="step_001",
                step_goal="Inspect the object.",
                input_image_index=0,
                input_artifact_id="img_root_0",
                capability_plan=[{"order": 1, "capability": "manual_crop", "instruction": "Crop the region."}],
                executor_description="Crop the region.",
                executor_cot_path=str(step_paths.executor_cot_path),
                executor_code_path=str(step_paths.executor_code_path),
                runtime_result_path=str(step_paths.runtime_result_path),
                assistant_message_id="m_step_001_assistant",
            )
            store.register_step_record(sample_id, parent_id, step_record)

            judge_record = JudgeRecord(
                judge_record_id="judge__demo",
                sample_id=sample_id,
                trajectory_id=parent_id,
                scope_type="step",
                scope_step_idx=1,
                judge_stage="cheap_filter",
                created_at=utc_now(),
                keep_for_frontier=True,
                exportable=False,
                overall_score=0.75,
            )
            store.register_judge_record(judge_record)

            child_init = store.init_child_trajectory(
                store.load_trajectory(sample_id, parent_id),
                fork_provenance=ForkProvenance(
                    parent_trajectory_id=parent_id,
                    parent_planner_round_idx=0,
                    parent_suggestion_id="s1",
                ),
                pending_execution=PendingExecution(
                    planner_round_idx=0,
                    suggestion_id="s1",
                    suggestion_step_index=0,
                    step_id="step_001",
                ),
                budget=Budget(remaining_exec_steps=2),
            )
            child_id = child_init.trajectory.trajectory_id
            child_trajectory = store.load_trajectory(sample_id, child_id)

            self.assertFalse(store.planner_output_path(sample_id, child_id, 0).exists())
            self.assertFalse(store.build_step_file_paths(sample_id, child_id, 1, create_dirs=False).runtime_result_path.exists())
            self.assertFalse(store.judge_record_path(sample_id, child_id, "cheap_filter", scope_step_idx=1).exists())

            self.assertEqual(
                child_trajectory.planner_history[0].planner_output_path,
                "../traj__demo__sample__root/planner/round_000.json",
            )
            self.assertEqual(
                child_trajectory.steps[0].runtime_result_path,
                "../traj__demo__sample__root/steps/step_001/runtime_result.json",
            )
            self.assertEqual(
                child_trajectory.judge_records[0].judge_record_path,
                "../traj__demo__sample__root/judge/step_001_cheap_filter.json",
            )

            loaded_planner = store.load_planner_output(sample_id, child_id, 0)
            self.assertEqual(loaded_planner.trajectory_id, parent_id)

            loaded_runtime = store.load_runtime_result(sample_id, child_id, 1)
            self.assertEqual(loaded_runtime.images[0].artifact_id, "img_step_001_0")
            self.assertEqual(Path(loaded_runtime.images[0].path), output_image)

            loaded_judge = store.load_judge_record(
                sample_id,
                child_id,
                "cheap_filter",
                scope_step_idx=1,
            )
            self.assertEqual(loaded_judge.judge_record_id, "judge__demo")

            resolved_artifact = store.resolve_artifact_id(sample_id, child_id, "img_step_001_0")
            self.assertEqual(Path(resolved_artifact.path), output_image)


if __name__ == "__main__":
    unittest.main()
