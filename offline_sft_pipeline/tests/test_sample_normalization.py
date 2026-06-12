from __future__ import annotations

import unittest

from offline_sft_pipeline.core.models import RootImage, RootSample
from offline_sft_pipeline.core.sample_normalization import normalize_root_sample


class SampleNormalizationTest(unittest.TestCase):
    def _build_sample(
        self,
        *,
        dataset_name: str,
        question: str,
        answer_instruction: str | None = None,
        answer: str | list[str] | None = None,
    ) -> RootSample:
        return RootSample(
            sample_id=f"{dataset_name}__train__demo",
            question=question,
            answer_instruction=answer_instruction,
            images=[RootImage(image_id="img0", path="/tmp/fake.png")],
            metadata={"source_dataset": dataset_name},
            answer=answer,
        )

    def test_cavqa_suffix_is_stripped_and_instruction_is_set(self) -> None:
        sample = self._build_sample(
            dataset_name="cavqa_multichoice",
            question=(
                "Is the shelf longer than the cabinet?\n"
                "A. yes\nB. no\n"
                "Answer with the option's letter from the given choices directly."
            ),
            answer="B",
        )

        normalized = normalize_root_sample(sample)

        self.assertEqual(
            normalized.question,
            "Is the shelf longer than the cabinet?\nA. yes\nB. no",
        )
        self.assertEqual(normalized.answer_instruction, "Answer with the option letter only.")
        self.assertEqual(normalized.answer, "B")

    def test_arxivqa_suffix_is_stripped_and_instruction_is_set(self) -> None:
        sample = self._build_sample(
            dataset_name="arxivqa",
            question=(
                "Which option is correct?\n"
                "A. left\nB. right\n"
                "Answer with the option's letter from the given choices directly."
            ),
            answer="A",
        )

        normalized = normalize_root_sample(sample)

        self.assertEqual(
            normalized.question,
            "Which option is correct?\nA. left\nB. right",
        )
        self.assertEqual(normalized.answer_instruction, "Answer with the option letter only.")
        self.assertEqual(normalized.answer, "A")

    def test_gqa_gets_default_instruction_without_question_rewrite(self) -> None:
        sample = self._build_sample(
            dataset_name="gqa",
            question=(
                "What color is the umbrella?\n"
                "Answer the question using a single word or phrase."
            ),
            answer="red",
        )

        normalized = normalize_root_sample(sample)

        self.assertEqual(normalized.question, "What color is the umbrella?")
        self.assertEqual(
            normalized.answer_instruction,
            "Answer the question using a single word or phrase.",
        )

    def test_arxivvqa_alias_gets_multiple_choice_instruction(self) -> None:
        sample = self._build_sample(
            dataset_name="arxivvqa",
            question="Which option is correct?\nA. left\nB. right",
            answer="A",
        )

        normalized = normalize_root_sample(sample)

        self.assertEqual(normalized.question, "Which option is correct?\nA. left\nB. right")
        self.assertEqual(normalized.answer_instruction, "Answer with the option letter only.")

    def test_explicit_instruction_and_answer_list_are_preserved(self) -> None:
        sample = self._build_sample(
            dataset_name="textvqa",
            question="What word is written on the sign?",
            answer_instruction="Answer with a short phrase only.",
            answer=["new york", "nyc"],
        )

        normalized = normalize_root_sample(sample)

        self.assertEqual(normalized.answer_instruction, "Answer with a short phrase only.")
        self.assertEqual(normalized.answer, ["new york", "nyc"])

    def test_textvqa_suffix_is_stripped_and_instruction_is_set(self) -> None:
        sample = self._build_sample(
            dataset_name="textvqa",
            question=(
                "what does the ad say?\n"
                "Answer the question using a single word or phrase."
            ),
            answer=["firekeepers"] * 10,
        )

        normalized = normalize_root_sample(sample)

        self.assertEqual(normalized.question, "what does the ad say?")
        self.assertEqual(
            normalized.answer_instruction,
            "Answer the question using a single word or phrase.",
        )

    def test_chartqa_suffix_is_stripped_and_instruction_is_set(self) -> None:
        sample = self._build_sample(
            dataset_name="chartqa",
            question=(
                "What is the value of the blue bar?\n"
                "Answer the question with a single word."
            ),
            answer="42",
        )

        normalized = normalize_root_sample(sample)

        self.assertEqual(normalized.question, "What is the value of the blue bar?")
        self.assertEqual(
            normalized.answer_instruction,
            "Answer the question with a single word, number, or concise value.",
        )


if __name__ == "__main__":
    unittest.main()
