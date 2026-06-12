from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any

from offline_sft_pipeline.core.dataset_names import canonicalize_dataset_name
from offline_sft_pipeline.eval.vqa_eval_metric import EvalAIAnswerProcessor


_ARTICLES = {"a", "an", "the"}
_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_PUNCT_TRANSLATION = str.maketrans({ch: " " for ch in string.punctuation})
_TEXTVQA_ANSWER_PROCESSOR = EvalAIAnswerProcessor()


@dataclass(frozen=True, slots=True)
class ScoreResult:
    score: float
    normalized_prediction: str | None
    normalized_reference: str | list[str] | None
    matcher_name: str


def _normalize_space(text: str) -> str:
    return " ".join(str(text).strip().split())


def _normalize_basic_text(text: str) -> str:
    text = _normalize_space(text).lower()
    text = text.translate(_PUNCT_TRANSLATION)
    return _normalize_space(text)


def _normalize_vqa_text(text: str) -> str:
    normalized = _normalize_basic_text(text)
    tokens: list[str] = []
    for token in normalized.split():
        if token in _ARTICLES:
            continue
        tokens.append(_NUMBER_WORDS.get(token, token))
    return _normalize_space(" ".join(tokens))


def _normalize_textvqa_answer(text: str) -> str:
    return _normalize_space(_TEXTVQA_ANSWER_PROCESSOR(str(text or "")))


def _normalize_option_letter(text: str) -> str | None:
    raw = _normalize_space(text).upper()
    if not raw:
        return None
    for marker in ("OPTION", "CHOICE", "ANSWER", "IS"):
        raw = raw.replace(marker, " ")
    tokens = re.findall(r"\b[A-Z]\b", raw)
    if tokens:
        return tokens[0]
    compact = re.sub(r"[^A-Z]", "", raw)
    if len(compact) == 1:
        return compact
    return None


def _normalize_integer(text: str) -> str | None:
    match = re.search(r"-?\d+", str(text))
    if match is None:
        return None
    return str(int(match.group(0)))


def _coerce_reference_text(answer: Any) -> str | None:
    if answer is None:
        return None
    if isinstance(answer, list):
        if not answer:
            return None
        return _normalize_space(str(answer[0]))
    return _normalize_space(str(answer))


def _score_option_letter(pred_answer: str, answer: Any, *, matcher_name: str) -> ScoreResult:
    pred = _normalize_option_letter(pred_answer or "")
    ref = _normalize_option_letter(_coerce_reference_text(answer) or "")
    score = 1.0 if pred is not None and ref is not None and pred == ref else 0.0
    return ScoreResult(
        score=score,
        normalized_prediction=pred,
        normalized_reference=ref,
        matcher_name=matcher_name,
    )


def _score_exact_normalized(pred_answer: str, answer: Any, *, matcher_name: str) -> ScoreResult:
    pred = _normalize_basic_text(pred_answer or "")
    ref = _normalize_basic_text(_coerce_reference_text(answer) or "")
    score = 1.0 if pred and ref and pred == ref else 0.0
    return ScoreResult(
        score=score,
        normalized_prediction=pred or None,
        normalized_reference=ref or None,
        matcher_name=matcher_name,
    )


def score_arxivqa(pred_answer: str, answer: Any, metadata: dict[str, Any] | None = None) -> ScoreResult:
    return _score_option_letter(pred_answer, answer, matcher_name="arxivqa_option_letter")


def score_cavqa_multichoice(
    pred_answer: str,
    answer: Any,
    metadata: dict[str, Any] | None = None,
) -> ScoreResult:
    return _score_option_letter(pred_answer, answer, matcher_name="cavqa_option_letter")


def score_gqa(pred_answer: str, answer: Any, metadata: dict[str, Any] | None = None) -> ScoreResult:
    pred = _normalize_basic_text(pred_answer or "")
    ref = _normalize_basic_text(_coerce_reference_text(answer) or "")
    score = 1.0 if pred and ref and pred == ref else 0.0
    return ScoreResult(
        score=score,
        normalized_prediction=pred or None,
        normalized_reference=ref or None,
        matcher_name="gqa_exact_match_ignore_case_punctuation",
    )


def score_textvqa(pred_answer: str, answer: Any, metadata: dict[str, Any] | None = None) -> ScoreResult:
    pred = _normalize_textvqa_answer(pred_answer or "")
    if isinstance(answer, list):
        refs = [_normalize_textvqa_answer(item) for item in answer if _normalize_textvqa_answer(item)]
        if pred and refs:
            per_reference_scores: list[float] = []
            for idx in range(len(refs)):
                other_refs = [refs[j] for j in range(len(refs)) if j != idx]
                matches = sum(1 for item in other_refs if item == pred)
                per_reference_scores.append(min(1.0, matches / 3.0))
            score = sum(per_reference_scores) / len(per_reference_scores)
        else:
            score = 0.0
        return ScoreResult(
            score=score,
            normalized_prediction=pred or None,
            normalized_reference=refs,
            matcher_name="textvqa_evalai_soft_vqa",
        )

    ref = _normalize_textvqa_answer(_coerce_reference_text(answer) or "")
    score = 1.0 if pred and ref and pred == ref else 0.0
    return ScoreResult(
        score=score,
        normalized_prediction=pred or None,
        normalized_reference=ref or None,
        matcher_name="textvqa_evalai_single_ref_exact",
    )


def score_fsc147(pred_answer: str, answer: Any, metadata: dict[str, Any] | None = None) -> ScoreResult:
    pred = _normalize_integer(pred_answer or "")
    ref = _normalize_integer(_coerce_reference_text(answer) or "")
    if pred is None or ref is None:
        score = 0.0
    else:
        pred_value = int(pred)
        ref_value = int(ref)
        score = max(0.0, 1.0 - (abs(pred_value - ref_value) / max(abs(ref_value), 1)))
    return ScoreResult(
        score=score,
        normalized_prediction=pred,
        normalized_reference=ref,
        matcher_name="fsc147_relative_error",
    )


def score_we_math_pro(pred_answer: str, answer: Any, metadata: dict[str, Any] | None = None) -> ScoreResult:
    return _score_exact_normalized(pred_answer, answer, matcher_name="we_math_pro_fallback_exact")


def score_we_math_standard(
    pred_answer: str,
    answer: Any,
    metadata: dict[str, Any] | None = None,
) -> ScoreResult:
    return _score_exact_normalized(pred_answer, answer, matcher_name="we_math_standard_fallback_exact")


def score_unknown_dataset(pred_answer: str, answer: Any, metadata: dict[str, Any] | None = None) -> ScoreResult:
    return _score_exact_normalized(pred_answer, answer, matcher_name="unknown_dataset_fallback_exact")


_SCORER_DISPATCH = {
    "arxivqa": score_arxivqa,
    "cavqa_multichoice": score_cavqa_multichoice,
    "fsc147": score_fsc147,
    "gqa": score_gqa,
    "textvqa": score_textvqa,
    "we_math_pro": score_we_math_pro,
    "we_math_standard": score_we_math_standard,
}


def score_answer_for_dataset(
    source_dataset: str,
    pred_answer: str,
    answer: Any,
    metadata: dict[str, Any] | None = None,
) -> ScoreResult:
    dataset_name = canonicalize_dataset_name(source_dataset)
    scorer = _SCORER_DISPATCH.get(dataset_name, score_unknown_dataset)
    return scorer(pred_answer, answer, metadata or {})
