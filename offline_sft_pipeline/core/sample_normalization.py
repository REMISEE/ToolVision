from __future__ import annotations

import re
from typing import Any

from offline_sft_pipeline.core.dataset_names import canonicalize_dataset_name
from offline_sft_pipeline.core.models import RootSample

_CAVQA_SUFFIX_RE = re.compile(
    r"\s*Answer with the option'?s letter from the given choices directly\.\s*$",
    flags=re.IGNORECASE,
)
_ARXIVQA_SUFFIX_RE = re.compile(
    r"\s*Answer with the option'?s letter from the given choices directly\.\s*$",
    flags=re.IGNORECASE,
)
_FSC147_SUFFIX_RE = re.compile(
    r"\s*Answer with only an integer\.\s*$",
    flags=re.IGNORECASE,
)
_TEXTVQA_SUFFIX_RE = re.compile(
    r"\s*Answer the question using a single word or phrase\.\s*$",
    flags=re.IGNORECASE,
)
_GQA_SUFFIX_RE = re.compile(
    r"\s*Answer the question using a single word or phrase\.\s*$",
    flags=re.IGNORECASE,
)
_CHARTQA_SUFFIX_RE = re.compile(
    r"\s*Answer the question with a single word\.\s*$",
    flags=re.IGNORECASE,
)

_DEFAULT_ANSWER_INSTRUCTIONS: dict[str, str] = {
    "arxivqa": "Answer with the option letter only.",
    "cavqa_multichoice": "Answer with the option letter only.",
    "chartqa": "Answer the question with a single word, number, or concise value.",
    "fsc147": "Answer with only an integer.",
    "gqa": "Answer the question using a single word.",
    "textvqa": "Answer the question using a single word or phrase.",
}


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_root_sample(sample: RootSample) -> RootSample:
    normalized = sample.model_copy(deep=True)
    dataset_name = canonicalize_dataset_name(normalized.metadata.get("source_dataset"))

    question = str(normalized.question or "").strip()
    answer_instruction = _normalize_optional_text(normalized.answer_instruction)

    if dataset_name == "arxivqa":
        question = _ARXIVQA_SUFFIX_RE.sub("", question).strip()
    elif dataset_name == "cavqa_multichoice":
        question = _CAVQA_SUFFIX_RE.sub("", question).strip()
    elif dataset_name == "fsc147":
        question = _FSC147_SUFFIX_RE.sub("", question).strip()
    elif dataset_name == "gqa":
        question = _GQA_SUFFIX_RE.sub("", question).strip()
    elif dataset_name == "textvqa":
        question = _TEXTVQA_SUFFIX_RE.sub("", question).strip()
    elif dataset_name == "chartqa":
        question = _CHARTQA_SUFFIX_RE.sub("", question).strip()

    default_instruction = _DEFAULT_ANSWER_INSTRUCTIONS.get(dataset_name)
    if answer_instruction is None and default_instruction is not None:
        answer_instruction = default_instruction

    normalized.question = question or str(sample.question or "").strip()
    normalized.answer_instruction = answer_instruction
    return normalized
