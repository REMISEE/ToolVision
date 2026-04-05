"""Consolidated tests for ``pipelines`` (multimodal, planner parse, ApiTextBackend, optional live API)."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from offline_sft_pipeline.core.models import Budget, ConversationMessage, ImageArtifactRef
from offline_sft_pipeline.pipelines.api_text_multimodal import (
    build_artifact_path_index,
    chat_completions_text,
    env_qwen_config,
    file_to_data_url,
    planner_to_openai_messages,
    summarize_openai_message_for_debug,
)
from offline_sft_pipeline.pipelines.backends import (
    DEFAULT_FAKE_PLANNER_TEXT,
    ApiTextBackend,
    ApiTextBackendConfig,
    BackendResponse,
)
from offline_sft_pipeline.pipelines.planner_client import PlannerClient
from offline_sft_pipeline.pipelines.request_models import PlannerClientRequest
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
        budget=Budget(remaining_rounds=3),
        tool_capabilities=load_tool_capabilities_from_file(),
        requested_suggestion_count=3,
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
        budget=Budget(remaining_rounds=3),
        tool_capabilities=load_tool_capabilities_from_file(),
        requested_suggestion_count=3,
    )


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
                "exactly 3 branch objects" in tail,
                msg=tail[:200],
            )
            self.assertEqual(missing, [])

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


class TestApiTextBackend(unittest.TestCase):
    def test_executor_not_implemented(self) -> None:
        with patch.dict(os.environ, {"OFFLINE_SFT_API_DRY_RUN": "1"}):
            b = ApiTextBackend()
        with self.assertRaises(NotImplementedError):
            b.generate(stage="executor", system_prompt="s", user_prompt="u", context=None)

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
