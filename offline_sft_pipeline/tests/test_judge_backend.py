from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from offline_sft_pipeline.core.models import ConversationMessage
from offline_sft_pipeline.eval.scorers import score_answer_for_dataset
from offline_sft_pipeline.pipelines.backends import CommitteeJudgeBackend
from offline_sft_pipeline.pipelines.request_models import JudgeClientRequest


class ScorerTests(unittest.TestCase):
    def test_multiple_choice_scoring_normalizes_option_letter(self) -> None:
        result = score_answer_for_dataset(
            source_dataset="cavqa_multichoice",
            pred_answer="Option B",
            answer="B",
            metadata={},
        )
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.normalized_prediction, "B")

    def test_multiple_choice_dataset_aliases_use_option_letter_scoring(self) -> None:
        arxiv_result = score_answer_for_dataset(
            source_dataset="arxivvqa",
            pred_answer="option c",
            answer="C",
            metadata={},
        )
        cavqa_result = score_answer_for_dataset(
            source_dataset="cavqa",
            pred_answer="b",
            answer="B",
            metadata={},
        )

        self.assertEqual(arxiv_result.score, 1.0)
        self.assertEqual(cavqa_result.score, 1.0)

    def test_gqa_scoring_matches_ignore_case_ignore_punctuation_exact_match(self) -> None:
        result = score_answer_for_dataset(
            source_dataset="gqa",
            pred_answer="Cat!",
            answer="cat",
            metadata={},
        )
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.matcher_name, "gqa_exact_match_ignore_case_punctuation")

    def test_gqa_scoring_does_not_strip_articles(self) -> None:
        result = score_answer_for_dataset(
            source_dataset="gqa",
            pred_answer="The cat",
            answer="cat",
            metadata={},
        )
        self.assertEqual(result.score, 0.0)

    def test_textvqa_list_answers_use_soft_scoring(self) -> None:
        result = score_answer_for_dataset(
            source_dataset="textvqa",
            pred_answer="red",
            answer=["red", "red", "red", "blue", "blue", "blue", "blue", "blue", "blue", "blue"],
            metadata={},
        )
        self.assertAlmostEqual(result.score, 0.9)
        self.assertEqual(result.matcher_name, "textvqa_evalai_soft_vqa")

    def test_fsc147_scoring_uses_relative_error(self) -> None:
        result = score_answer_for_dataset(
            source_dataset="fsc147",
            pred_answer="9",
            answer="10",
            metadata={},
        )
        self.assertAlmostEqual(result.score, 0.9)
        self.assertEqual(result.matcher_name, "fsc147_relative_error")


class CommitteeJudgeBackendTests(unittest.TestCase):
    def test_committee_backend_aggregates_model_scores_and_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "judge_models.json"
            config_path.write_text(
                json.dumps(
                    [
                        {
                            "name": "judge_a",
                            "model": "model-a",
                            "base_url": "https://example.com/v1",
                            "api_key_env": "TEST_JUDGE_KEY_A",
                            "timeout_s": 30,
                            "enabled": True,
                            "request_body": {
                                "extra_body": {
                                    "enable_thinking": False,
                                }
                            },
                        },
                        {
                            "name": "judge_b",
                            "model": "model-b",
                            "base_url": "https://example.com/v1",
                            "api_key_env": "TEST_JUDGE_KEY_B",
                            "timeout_s": 30,
                            "enabled": True,
                            "request_body": {
                                "reasoning_effort": "none",
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )

            request = JudgeClientRequest(
                sample_id="demo__sample",
                trajectory_id="traj__demo__sample__root",
                sample_dir=tmpdir,
                trajectory_dir=tmpdir,
                scope_type="step",
                scope_step_idx=1,
                judge_stage="committee",
                question="Which option is correct?",
                answer_instruction="Answer with the option letter only.",
                answer="B",
                messages=[
                    ConversationMessage(
                        message_id="m_user",
                        role="user",
                        content="Which option is correct?",
                        metadata={},
                    )
                ],
                metadata={"source_dataset": "cavqa_multichoice"},
            )

            observed_request_bodies: dict[str, object] = {}

            def _fake_chat_completion(
                *,
                model: str,
                request_body: object = None,
                **_: object,
            ) -> tuple[str, dict[str, object]]:
                observed_request_bodies[model] = request_body
                if model == "model-a":
                    return "B", {"usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
                if model == "model-b":
                    return "C", {"usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}}
                raise AssertionError(f"unexpected model {model}")

            with patch.dict(
                "os.environ",
                {
                    "TEST_JUDGE_KEY_A": "key-a",
                    "TEST_JUDGE_KEY_B": "key-b",
                },
                clear=False,
            ):
                with patch(
                    "offline_sft_pipeline.pipelines.backends.chat_completions_text",
                    side_effect=_fake_chat_completion,
                ):
                    backend = CommitteeJudgeBackend(config_path=config_path, max_concurrency=2, max_retries=0)
                    result = backend.score(request)

            self.assertAlmostEqual(result.overall_score, 0.5)
            self.assertEqual(result.per_model_scores["judge_a"], 1.0)
            self.assertEqual(result.per_model_scores["judge_b"], 0.0)
            self.assertEqual(result.metadata["token_usage"]["prompt_tokens"], 21)
            self.assertEqual(result.metadata["successful_model_count"], 2)
            self.assertEqual(
                observed_request_bodies["model-a"],
                {"extra_body": {"enable_thinking": False}},
            )
            self.assertEqual(
                observed_request_bodies["model-b"],
                {"reasoning_effort": "none"},
            )


if __name__ == "__main__":
    unittest.main()
