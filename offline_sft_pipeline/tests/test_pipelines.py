"""Consolidated tests for ``pipelines`` (multimodal, planner parse, ApiTextBackend, optional live API)."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from offline_sft_pipeline.core.models import (
    Budget,
    ConversationMessage,
    ExecutorRuntimeResult,
    ForkProvenance,
    ImageArtifactRef,
    PendingExecution,
    RootSample,
    RuntimeCodeExecution,
    StepRecord,
    utc_now,
)
from offline_sft_pipeline.core.store import OfflineTrajectoryStore
from offline_sft_pipeline.pipelines.api_text_multimodal import (
    build_artifact_path_index,
    build_executor_control_user_text,
    build_planner_control_user_text,
    chat_completions_text,
    env_qwen_config,
    executor_to_openai_messages,
    file_to_data_url,
    judge_to_openai_messages,
    planner_to_openai_messages,
    summarize_openai_message_for_debug,
)
from offline_sft_pipeline.pipelines.backends import (
    DEFAULT_FAKE_EXECUTOR_TEXT,
    DEFAULT_FAKE_PLANNER_TEXT,
    ApiTextBackend,
    ApiTextBackendConfig,
    BackendResponse,
)
from offline_sft_pipeline.pipelines.executor_client import ExecutorClient
from offline_sft_pipeline.pipelines.planner_client import PlannerClient
from offline_sft_pipeline.pipelines.request_models import (
    ExecutorClientRequest,
    JudgeClientRequest,
    PlannerClientRequest,
    ToolCapability,
)
from offline_sft_pipeline.pipelines.tool_capabilities_io import load_tool_capabilities_from_file

_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example"


def _resolve_example_user_image() -> tuple[str, Path]:
    qpath = _EXAMPLE_DIR / "question.json"
    if not qpath.is_file():
        raise unittest.SkipTest(f"missing {qpath}")
    payload = json.loads(qpath.read_text(encoding="utf-8"))
    rel = payload.get("image")
    if not rel:
        raise unittest.SkipTest("question.json has no 'image' field")
    img_path = (_EXAMPLE_DIR / str(rel)).resolve()
    if not img_path.is_file():
        raise unittest.SkipTest(f"missing example image: {img_path}")
    return str(payload["question"]), img_path


def _write_sample_tree(root: Path) -> PlannerClientRequest:
    question, img_src = _resolve_example_user_image()
    art = root / "artifacts"
    art.mkdir(parents=True)
    png = art / "img_root_0.png"
    shutil.copy2(img_src, png)
    vis = [ImageArtifactRef(artifact_id="img_root_0", path=str(png), media_type="image/png")]
    return PlannerClientRequest(
        sample_id="s1",
        trajectory_id="t1",
        round_idx=0,
        sample_dir=str(root),
        trajectory_dir=str(root / "traj"),
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
        budget=Budget(remaining_exec_steps=3),
        tool_capabilities=load_tool_capabilities_from_file(),
        requested_suggestion_count=2,
    )


def _example_planner_request(*, trajectory_dir: str) -> PlannerClientRequest:
    question, img_path = _resolve_example_user_image()
    vis = [ImageArtifactRef(artifact_id="img_root_0", path=str(img_path), media_type="image/png")]
    return PlannerClientRequest(
        sample_id="example",
        trajectory_id="traj_example",
        round_idx=0,
        sample_dir=str(_EXAMPLE_DIR),
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
        budget=Budget(remaining_exec_steps=3),
        tool_capabilities=load_tool_capabilities_from_file(),
        requested_suggestion_count=2,
    )


def _example_executor_request(*, root: Path) -> ExecutorClientRequest:
    question, img_src = _resolve_example_user_image()
    art = root / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    root_png = art / "img_root_0.png"
    current_png = art / "img_step_001_0.png"
    shutil.copy2(img_src, root_png)
    shutil.copy2(img_src, current_png)
    return ExecutorClientRequest(
        sample_id="s1",
        trajectory_id="t1",
        round_idx=1,
        step_idx=1,
        sample_dir=str(root),
        trajectory_dir=str(root / "traj"),
        question=question,
        messages=[
            ConversationMessage(
                message_id="m_user",
                role="user",
                content=question,
                image_artifact_ids=["img_root_0"],
                metadata={},
            ),
            ConversationMessage(
                message_id="m_tool",
                role="tool",
                content="Here is the processed image for the previous step.",
                image_artifact_ids=["img_step_001_0"],
                metadata={},
            ),
        ],
        visible_images=[
            ImageArtifactRef(artifact_id="img_root_0", path=str(root_png), media_type="image/png"),
            ImageArtifactRef(artifact_id="img_step_001_0", path=str(current_png), media_type="image/png"),
        ],
        suggestion_id="s1",
        suggestion_step_index=0,
        step_spec={
            "step_id": "step_crop",
            "step_goal": "Tighten onto the most relevant text region in the current crop.",
            "input_image_index": 1,
            "capability_plan": [
                {"order": 1, "capability": "manual_crop", "instruction": "Crop the current text region tighter."}
            ],
            "executor_instruction": "Use a tighter crop on the current image and preserve the crop.",
        },
        planner_global_chain_cot=(
            "The answer depends on reading a small text region. The current image already narrows the search area, "
            "but the text likely still needs a tighter crop before OCR."
        ),
        suggestion_cot="Continue from the current crop, refine the region, then read the text.",
        tool_capabilities=load_tool_capabilities_from_file(),
    )


def _build_forked_lineage_fixture(root: Path) -> tuple[str, str, list[ConversationMessage], list[ImageArtifactRef], str, str]:
    store = OfflineTrajectoryStore(root / "store")
    source_image = root / "source.png"
    Image.new("RGB", (12, 12), color=(255, 255, 255)).save(source_image)

    sample = RootSample(
        sample_id="demo__sample",
        question="What number is shown?",
        images=[{"image_id": "img0", "path": str(source_image)}],
    )
    parent_init = store.init_root_trajectory(sample, budget=Budget(remaining_exec_steps=3))
    sample_id = sample.sample_id
    parent_id = parent_init.trajectory.trajectory_id

    def _register_step(trajectory_id: str, step_idx: int, artifact_id: str, color: tuple[int, int, int]) -> StepRecord:
        step_paths = store.write_executor_step_files(
            sample_id,
            trajectory_id,
            step_idx,
            executor_cot=f"cot {step_idx}",
            executor_code="result = image",
        )
        output_path = step_paths.step_dir / f"{artifact_id}.png"
        Image.new("RGB", (10 - step_idx, 10 - step_idx), color=color).save(output_path)
        now = utc_now()
        runtime_result = ExecutorRuntimeResult(
            sample_id=sample_id,
            trajectory_id=trajectory_id,
            round_idx=step_idx - 1,
            step_idx=step_idx,
            created_at=now,
            success=True,
            images=[
                ImageArtifactRef(
                    artifact_id=artifact_id,
                    path=str(output_path),
                    media_type="image/png",
                    width=10 - step_idx,
                    height=10 - step_idx,
                )
            ],
            text=f"tool output {step_idx}",
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
            execution_trajectory_id=trajectory_id,
            step_idx=step_idx,
            planner_round_idx=step_idx - 1,
            suggestion_id=f"s{step_idx}",
            suggestion_step_index=0,
            step_id=f"step_{step_idx:03d}",
            step_goal=f"goal {step_idx}",
            input_image_index=0 if step_idx == 1 else step_idx - 1,
            input_artifact_id="img_root_0" if step_idx == 1 else f"img_step_{step_idx - 1:03d}_0",
            capability_plan=[{"order": 1, "capability": "manual_crop", "instruction": f"do {step_idx}"}],
            executor_description=f"desc {step_idx}",
            executor_cot_path=str(step_paths.executor_cot_path),
            executor_code_path=str(step_paths.executor_code_path),
            runtime_result_path=str(step_paths.runtime_result_path),
            assistant_message_id=f"m_step_{step_idx:03d}_assistant",
            tool_message_id=f"m_step_{step_idx:03d}_tool",
        )
        store.register_step_record(sample_id, trajectory_id, step_record)
        store.append_messages(
            sample_id,
            trajectory_id,
            [
                ConversationMessage(
                    message_id=f"m_step_{step_idx:03d}_assistant",
                    role="assistant",
                    content=f"assistant trace {step_idx}",
                    image_artifact_ids=[],
                    metadata={"message_kind": "executor_step", "step_idx": step_idx},
                ),
                ConversationMessage(
                    message_id=f"m_step_{step_idx:03d}_tool",
                    role="tool",
                    content=f"tool trace {step_idx}",
                    image_artifact_ids=[artifact_id],
                    metadata={
                        "message_kind": "tool_result",
                        "step_idx": step_idx,
                        "runtime_result_path": str(step_paths.runtime_result_path),
                    },
                ),
            ],
        )
        return step_record

    _register_step(parent_id, 1, "img_step_001_0", (255, 0, 0))

    child_init = store.init_child_trajectory(
        store.load_trajectory(sample_id, parent_id),
        fork_provenance=ForkProvenance(
            parent_trajectory_id=parent_id,
            parent_planner_round_idx=0,
            parent_suggestion_id="s1",
        ),
        pending_execution=PendingExecution(
            planner_round_idx=1,
            suggestion_id="s2",
            suggestion_step_index=0,
            step_id="step_002",
        ),
        budget=Budget(remaining_exec_steps=2),
    )
    child_id = child_init.trajectory.trajectory_id
    _register_step(child_id, 2, "img_step_002_0", (0, 255, 0))

    messages = list(store.load_messages(sample_id, child_id).root)
    visible_images = [
        *[item.model_copy(deep=True) for item in store.load_root_artifacts(sample_id)],
        store.load_runtime_result(sample_id, child_id, 1).images[0].model_copy(deep=True),
        store.load_runtime_result(sample_id, child_id, 2).images[0].model_copy(deep=True),
    ]
    return sample_id, child_id, messages, visible_images, str(store.sample_dir(sample_id)), str(store.trajectory_dir(sample_id, child_id))


def _live_api_enabled() -> bool:
    return bool(os.environ.get("OFFLINE_SFT_QWEN_API_KEY")) and os.environ.get(
        "OFFLINE_SFT_API_DRY_RUN", ""
    ).strip().lower() not in {"1", "true", "yes"}


class TestMultimodalHelpers(unittest.TestCase):
    def test_build_artifact_path_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            req = _write_sample_tree(root)
            idx = build_artifact_path_index(
                sample_dir=req.sample_dir,
                trajectory_dir=req.trajectory_dir,
                visible_images=req.visible_images,
            )
            self.assertTrue(idx["img_root_0"].is_file())

    def test_build_artifact_path_index_reads_ancestor_runtime_results_from_trajectory_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (
                _sample_id,
                _child_id,
                _messages,
                visible_images,
                sample_dir,
                trajectory_dir,
            ) = _build_forked_lineage_fixture(Path(tmp))
            idx = build_artifact_path_index(
                sample_dir=sample_dir,
                trajectory_dir=trajectory_dir,
                visible_images=visible_images,
            )
            self.assertIn("img_step_001_0", idx)
            self.assertIn("img_step_002_0", idx)
            self.assertTrue(idx["img_step_001_0"].is_file())
            self.assertTrue(idx["img_step_002_0"].is_file())

    def test_planner_to_openai_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            req = _write_sample_tree(root)
            req = req.model_copy(
                update={
                    "messages": [
                        ConversationMessage(
                            message_id="m_sys",
                            role="system",
                            content="store system",
                            image_artifact_ids=[],
                            metadata={},
                        ),
                        ConversationMessage(
                            message_id="m_user",
                            role="user",
                            content=req.question,
                            image_artifact_ids=["img_root_0"],
                            metadata={},
                        ),
                    ],
                }
            )
            msgs, missing = planner_to_openai_messages(system_prompt="SYS", req=req)
            self.assertEqual(msgs[0], {"role": "system", "content": "SYS"})
            self.assertNotIn("system", [m["role"] for m in msgs[1:]])
            # Final user turn is build_planner_control_user_text.
            tail = str(msgs[-1]["content"])
            self.assertTrue(
                "exactly 2 branch objects" in tail,
                msg=tail[:200],
            )
            self.assertEqual(missing, [])

    def test_build_planner_control_user_text_hides_forced_final_answer_handoff_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            req = _write_sample_tree(root).model_copy(
                update={
                    "must_answer_now": True,
                    "requested_suggestion_count": 0,
                    "metadata": {
                        "forced_final_answer": {
                            "reason": "count_high_score_consensus",
                            "candidate_answer": "88",
                            "overall_score": 0.9887640449438203,
                            "successful_model_count": 3,
                        }
                    },
                }
            )
            text = build_planner_control_user_text(req)
            self.assertIn("This round is `MUST_ANSWER`.", text)
            self.assertIn("executed trajectory appears sufficient", text)
            self.assertNotIn("Latest judged candidate answer", text)
            self.assertNotIn("Latest judge overall_score", text)
            self.assertNotIn("successful_model_count", text)
            self.assertNotIn("must exactly equal", text)
            self.assertNotIn("88", text)

    def test_build_planner_control_user_text_includes_multiple_choice_json_reminder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            req = _write_sample_tree(root).model_copy(
                update={
                    "answer_instruction": "Answer with the option letter only.",
                }
            )
            text = build_planner_control_user_text(req)
            self.assertIn("Answer format constraint:", text)
            self.assertIn("the `answer` field must follow this instruction: Answer with the option letter only.", text)
            self.assertIn("If you return `mode=\"suggestions\"`, do not emit any `answer` field.", text)

    def test_planner_to_openai_messages_can_use_native_tool_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            req = _write_sample_tree(root)
            req = req.model_copy(
                update={
                    "messages": [
                        ConversationMessage(
                            message_id="m_user",
                            role="user",
                            content=req.question,
                            image_artifact_ids=["img_root_0"],
                            metadata={},
                        ),
                        ConversationMessage(
                            message_id="m_tool",
                            role="tool",
                            content="mock tool result",
                            image_artifact_ids=["img_root_0"],
                            metadata={},
                        ),
                    ],
                }
            )
            with patch.dict(os.environ, {"OFFLINE_SFT_PLANNER_USE_TOOL_ROLE": "1"}):
                msgs, missing = planner_to_openai_messages(system_prompt="SYS", req=req)
            self.assertEqual(msgs[2]["role"], "tool")
            self.assertNotEqual(msgs[2]["role"], "user")
            self.assertEqual(missing, [])

    def test_executor_to_openai_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            req = _example_executor_request(root=Path(tmp))
            msgs, missing = executor_to_openai_messages(system_prompt="SYS", req=req)
        self.assertEqual(msgs[0], {"role": "system", "content": "SYS"})
        tail = str(msgs[-1]["content"])
        self.assertIn("Global CoT", tail)
        self.assertIn("Current visual inputs available for this step", tail)
        self.assertIn("local image index 0: artifact_id=`img_root_0`", tail)
        self.assertIn("size=(", tail)
        self.assertIn("default bound image is local image index 1", tail)
        self.assertEqual(missing, [])

    def test_forked_lineage_history_images_render_for_planner_executor_and_judge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample_id, trajectory_id, messages, visible_images, sample_dir, trajectory_dir = _build_forked_lineage_fixture(Path(tmp))
            planner_req = PlannerClientRequest(
                sample_id=sample_id,
                trajectory_id=trajectory_id,
                round_idx=2,
                sample_dir=sample_dir,
                trajectory_dir=trajectory_dir,
                question="What number is shown?",
                messages=messages,
                visible_images=visible_images,
                budget=Budget(remaining_exec_steps=2),
                tool_capabilities=load_tool_capabilities_from_file(),
                requested_suggestion_count=1,
            )
            executor_req = ExecutorClientRequest(
                sample_id=sample_id,
                trajectory_id=trajectory_id,
                round_idx=2,
                step_idx=3,
                sample_dir=sample_dir,
                trajectory_dir=trajectory_dir,
                question="What number is shown?",
                messages=messages,
                visible_images=visible_images,
                suggestion_id="s3",
                suggestion_step_index=0,
                step_spec={
                    "step_id": "step_003",
                    "step_goal": "Read the current evidence.",
                    "input_image_index": 2,
                    "capability_plan": [
                        {"order": 1, "capability": "ocr_assist", "instruction": "Read the visible number."}
                    ],
                    "executor_instruction": "Read the current image.",
                },
                planner_global_chain_cot="Use the latest crop but keep the prior crop history available.",
                suggestion_cot="The prior crop is still relevant context.",
                tool_capabilities=load_tool_capabilities_from_file(),
            )
            judge_req = JudgeClientRequest(
                sample_id=sample_id,
                trajectory_id=trajectory_id,
                sample_dir=sample_dir,
                trajectory_dir=trajectory_dir,
                scope_type="trajectory",
                judge_stage="committee",
                question="What number is shown?",
                answer="42",
                messages=messages,
                visible_images=visible_images,
            )

            planner_msgs, planner_missing = planner_to_openai_messages(system_prompt="SYS", req=planner_req)
            executor_msgs, executor_missing = executor_to_openai_messages(system_prompt="SYS", req=executor_req)
            judge_msgs, judge_missing = judge_to_openai_messages(system_prompt="SYS", req=judge_req)

        self.assertEqual(planner_missing, [])
        self.assertEqual(executor_missing, [])
        self.assertEqual(judge_missing, [])
        self.assertTrue(any(part.get("type") == "image_url" for part in planner_msgs[3]["content"]))
        self.assertTrue(any(part.get("type") == "image_url" for part in executor_msgs[3]["content"]))
        self.assertTrue(any(part.get("type") == "image_url" for part in judge_msgs[3]["content"]))

    def test_build_executor_control_user_text_explains_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            req = _example_executor_request(root=Path(tmp))
            text = build_executor_control_user_text(req)
        self.assertIn("How to use the hidden context", text)
        self.assertIn("Your `think` must read as your own direct reasoning", text)
        self.assertIn("Do not use words such as `planner`, `suggestion`, `branch`", text)
        self.assertIn("If you use a manual coordinate tool", text)
        self.assertIn("Capability plan for this step", text)
        self.assertIn("capability=`manual_crop`", text)
        self.assertIn("Use only the real helper functions described below", text)
        self.assertIn("_call_manual_crop(x1, y1, x2, y2, padding=0, ...)", text)

    def test_build_planner_control_user_text_adds_cavqa_coordinate_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            req = _example_planner_request(trajectory_dir=str(Path(tmp) / "traj"))
            req.sample_id = "cavqa_multichoice__demo"
            text = build_planner_control_user_text(req)
        self.assertIn("Dataset-specific guidance", text)
        self.assertIn("coordinates written in the question are only hints", text)
        self.assertIn("prefer proposing both a coordinate-based branch", text)
        self.assertIn("explicitly rethink the localization", text)

    def test_summarize_openai_message_for_debug(self) -> None:
        m1 = {"role": "assistant", "content": "hello" * 20, "reasoning": "a"}
        s1 = summarize_openai_message_for_debug(m1)
        self.assertIn("len=", s1["content"])
        self.assertEqual(s1["reasoning"], "a")
        m2 = {"role": "assistant", "content": "x", "foo": "y" * 900}
        self.assertIn("truncated", summarize_openai_message_for_debug(m2)["foo"])


class TestToolCapabilities(unittest.TestCase):
    def test_default_json_loads(self) -> None:
        names = {c.name for c in load_tool_capabilities_from_file()}
        self.assertIn("ground_box", names)
        self.assertIn("ocr_assist", names)


class TestPlannerClientParsing(unittest.TestCase):
    def test_json_suggestions_and_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            req = _write_sample_tree(Path(tmp))

        class _Sug:
            def generate(self, **kwargs: object) -> BackendResponse:
                p = {
                    "mode": "suggestions",
                    "think": "t",
                    "suggestions": [
                        {
                            "suggestion_id": "s1",
                            "suggestion_cot": "c",
                            "steps": [
                                {
                                    "step_id": "a",
                                    "step_goal": "g",
                                    "input_image_index": 0,
                                    "capability_plan": [
                                        {"order": 1, "capability": "ground_box", "instruction": "i"}
                                    ],
                                    "executor_instruction": "e",
                                }
                            ],
                        }
                    ],
                }
                return BackendResponse(text=json.dumps(p), metadata={})

        class _Ans:
            def generate(self, **kwargs: object) -> BackendResponse:
                return BackendResponse(
                    text=json.dumps({"mode": "answer", "think": "t2", "answer": "42"}),
                    metadata={},
                )

        o1 = PlannerClient(backend=_Sug()).run(req)
        self.assertFalse(o1.can_answer_now)
        self.assertEqual(len(o1.suggestions), 1)
        o2 = PlannerClient(backend=_Ans()).run(req)
        self.assertTrue(o2.can_answer_now)
        self.assertEqual(o2.direct_answer, "42")

    def test_json_suggestions_with_inline_markdown_fence_is_salvaged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            req = _write_sample_tree(Path(tmp))

        class _B:
            def generate(self, **kwargs: object) -> BackendResponse:
                return BackendResponse(
                    text='```json {"mode":"suggestions","think":"t","suggestions":[{"suggestion_id":"s1","suggestion_cot":"c","steps":[{"step_id":"a","step_goal":"g","input_image_index":0,"capability_plan":[{"order":1,"capability":"ground_box","instruction":"i"}],"executor_instruction":"e"}]}]} ```',
                    metadata={},
                )

        out = PlannerClient(backend=_B()).run(req)
        self.assertFalse(out.can_answer_now)
        self.assertEqual(len(out.suggestions), 1)

    def test_json_suggestions_with_invalid_escaped_apostrophe_is_salvaged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            req = _write_sample_tree(Path(tmp))

        class _B:
            def generate(self, **kwargs: object) -> BackendResponse:
                return BackendResponse(
                    text='{"mode":"suggestions","think":"The graph\\\'s peak is clear.","suggestions":[{"suggestion_id":"s1","suggestion_cot":"c","steps":[{"step_id":"a","step_goal":"g","input_image_index":0,"capability_plan":[{"order":1,"capability":"ground_box","instruction":"i"}],"executor_instruction":"e"}]}]}',
                    metadata={},
                )

        out = PlannerClient(backend=_B()).run(req)
        self.assertFalse(out.can_answer_now)
        self.assertEqual(out.global_chain_cot, "The graph's peak is clear.")

    def test_xml_answer_with_think(self) -> None:
        class _B:
            def generate(self, **kwargs: object) -> BackendResponse:
                return BackendResponse(
                    text="<think>t</think>\n<answer>\nx\n</answer>",
                    metadata={},
                )

        with tempfile.TemporaryDirectory() as tmp:
            req = _write_sample_tree(Path(tmp))
            out = PlannerClient(backend=_B()).run(req)
        self.assertTrue(out.can_answer_now)
        self.assertEqual(out.direct_answer, "x")


class TestExecutorClientParsing(unittest.TestCase):
    def test_json_tool_call_contract(self) -> None:
        class _B:
            def generate(self, **kwargs: object) -> BackendResponse:
                return BackendResponse(
                    text=json.dumps(
                        {
                            "think": "Use the planner-selected current image and OCR it directly.",
                            "tool_call": {
                                "name": "code_image_tool",
                                "arguments": {
                                    "code": 'ocr = _call_ocr_assist(image_obj=image)\nprint(ocr.get("text", ""))\nresult = image',
                                    "description": "Run OCR on the current image and preserve it as the result.",
                                },
                            },
                        }
                    ),
                    metadata={},
                )

        req = ExecutorClientRequest(
            sample_id="s1",
            trajectory_id="t1",
            round_idx=1,
            step_idx=2,
            question="What number is written on the hanging tag?",
            messages=[
                ConversationMessage(message_id="m_user", role="user", content="What number is written on the hanging tag?"),
            ],
            visible_images=[
                ImageArtifactRef(artifact_id="img_root_0", path="/tmp/root.png", media_type="image/png"),
                ImageArtifactRef(artifact_id="img_step_001_0", path="/tmp/step.png", media_type="image/png"),
            ],
            suggestion_id="s1",
            suggestion_step_index=0,
            step_spec={
                "step_id": "step_ocr",
                "step_goal": "Run OCR on the current crop.",
                "input_image_index": 1,
                "capability_plan": [{"order": 1, "capability": "ocr_assist", "instruction": "Read the crop text."}],
                "executor_instruction": "Run OCR on the current crop.",
            },
            planner_global_chain_cot="Use the crop from the previous step.",
            suggestion_cot="Continue from the crop.",
            tool_capabilities=[ToolCapability(name="ocr_assist", description="Read text from the current image.")],
        )
        out = ExecutorClient(backend=_B()).run(req)
        self.assertEqual(out.description, "Run OCR on the current image and preserve it as the result.")
        self.assertIn("_call_ocr_assist", out.code)


class TestApiTextBackend(unittest.TestCase):
    def test_dry_run_executor(self) -> None:
        with patch.dict(os.environ, {"OFFLINE_SFT_API_DRY_RUN": "1"}):
            b = ApiTextBackend()
        r = b.generate(stage="executor", system_prompt="s", user_prompt="u", context=None)
        self.assertTrue(r.metadata.get("dry_run"))
        self.assertEqual(r.text, DEFAULT_FAKE_EXECUTOR_TEXT)

    def test_dry_run_planner(self) -> None:
        with patch.dict(os.environ, {"OFFLINE_SFT_API_DRY_RUN": "1"}):
            b = ApiTextBackend()
        r = b.generate(stage="planner", system_prompt="s", user_prompt="u", context=None)
        self.assertTrue(r.metadata.get("dry_run"))
        self.assertEqual(r.text, DEFAULT_FAKE_PLANNER_TEXT)

    def test_missing_key_raises(self) -> None:
        with patch.dict(os.environ, {"OFFLINE_SFT_QWEN_API_KEY": "", "OFFLINE_SFT_API_DRY_RUN": ""}, clear=False):
            b = ApiTextBackend()
        with self.assertRaises(RuntimeError):
            b.generate(stage="planner", system_prompt="s", user_prompt="u", context={"request": "invalid"})

    def test_planner_requires_request_in_context(self) -> None:
        with patch.dict(os.environ, {"OFFLINE_SFT_QWEN_API_KEY": "sk-real", "OFFLINE_SFT_API_DRY_RUN": ""}):
            b = ApiTextBackend(config=ApiTextBackendConfig(api_key="sk-real", dry_run=False))
        with self.assertRaises(ValueError):
            b.generate(stage="planner", system_prompt="s", user_prompt="u", context=None)

    def test_executor_requires_request_in_context(self) -> None:
        with patch.dict(os.environ, {"OFFLINE_SFT_QWEN_API_KEY": "sk-real", "OFFLINE_SFT_API_DRY_RUN": ""}):
            b = ApiTextBackend(config=ApiTextBackendConfig(api_key="sk-real", dry_run=False))
        with self.assertRaises(ValueError):
            b.generate(stage="executor", system_prompt="s", user_prompt="u", context=None)

    def test_extracts_token_usage_into_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            req = _write_sample_tree(Path(tmp))
            with patch.dict(os.environ, {"OFFLINE_SFT_QWEN_API_KEY": "sk-real", "OFFLINE_SFT_API_DRY_RUN": ""}):
                b = ApiTextBackend(config=ApiTextBackendConfig(api_key="sk-real", dry_run=False))
            fake_raw = {
                "choices": [{"message": {"content": "{\"mode\":\"answer\",\"think\":\"ok\",\"answer\":\"A\"}"}}],
                "usage": {
                    "prompt_tokens": 111,
                    "completion_tokens": 22,
                    "total_tokens": 133,
                },
            }
            with patch(
                "offline_sft_pipeline.pipelines.backends.chat_completions_text",
                return_value=("{}", fake_raw),
            ):
                result = b.generate(stage="planner", system_prompt="s", user_prompt="u", context={"request": req})
        self.assertEqual(
            result.metadata.get("token_usage"),
            {"prompt_tokens": 111, "completion_tokens": 22, "total_tokens": 133},
        )


@unittest.skipUnless(_live_api_enabled(), "Needs OFFLINE_SFT_QWEN_API_KEY; unset OFFLINE_SFT_API_DRY_RUN.")
class TestLiveQwenOptional(unittest.TestCase):
    def test_raw_chat_completions_example_image(self) -> None:
        qpath = _EXAMPLE_DIR / "question.json"
        payload = json.loads(qpath.read_text(encoding="utf-8"))
        img_path = _EXAMPLE_DIR / payload["image"]
        cfg = env_qwen_config()
        text, raw = chat_completions_text(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"] or "",
            model=cfg["model"],
            messages=[
                {"role": "system", "content": "You are a helpful vision assistant. Answer concisely."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": str(payload["question"])},
                        {"type": "image_url", "image_url": {"url": file_to_data_url(img_path)}},
                    ],
                },
            ],
            timeout_s=cfg["timeout_s"],
        )
        self.assertTrue(text.strip())
        self.assertIn("choices", raw)

    def test_planner_client_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            traj = Path(tmp) / "traj"
            traj.mkdir()
            out = PlannerClient(backend=ApiTextBackend()).run(_example_planner_request(trajectory_dir=str(traj)))
        out.validate_against_schema()
        self.assertTrue(out.can_answer_now or out.suggestions)


class TestPipelinesPackageExports(unittest.TestCase):
    """Sanity: package ``__init__`` re-exports match submodule definitions."""

    def test_planner_client_same_as_submodule(self) -> None:
        from offline_sft_pipeline.pipelines.planner_client import PlannerClient as PC_mod

        self.assertIs(PC_mod, PlannerClient)


if __name__ == "__main__":
    unittest.main()
