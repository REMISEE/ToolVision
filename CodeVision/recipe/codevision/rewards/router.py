from typing import Any, Callable

from .common import answer_candidates, choice_options_from_extra_info, extract_answer, tool_usage_info
from .judge import judge_score, should_use_judge
from .rule_scorers import (
    bbox_iou_score,
    boolean_score,
    chartqa_relaxed_score,
    exact_score,
    levenshtein_ratio_score,
    math_verify_score,
    multiple_choice_score,
    numeric_exact_score,
    ocr_inclusion_score,
    relative_count_score,
    vqa_soft_score,
)

ScoreFn = Callable[[Any, Any, dict[str, Any]], float]


SOURCE_DATASET_TO_FAMILY = {
    "chartqa": "chartqa_relaxed",
    "ocrbench": "ocr_inclusion",
    "pixmo_count": "fsc147_relative",
    "pixmo_count_lmms": "fsc147_relative",
    "countqa": "numeric_exact",
    "fsc147": "fsc147_relative",
    "gqa": "exact",
    "textvqa": "vqa_soft",
    "mmstar": "multiple_choice",
    "arxivqa": "multiple_choice",
    "ai2d": "multiple_choice",
    "docvqa": "ocr_levenshtein",
    "infographicvqa": "ocr_levenshtein",
    "refl4": "bbox_iou",
    "ref_l4": "bbox_iou",
    "sat2": "exact",
    "virgorlsa": "multiple_choice",
}

SOURCE_DATASET_FORCE_FAMILY = {
    "chartqa": "chartqa_relaxed",
    "ocrbench": "ocr_inclusion",
    "docvqa": "ocr_levenshtein",
    "infographicvqa": "ocr_levenshtein",
    "textvqa": "vqa_soft",
    "fsc147": "fsc147_relative",
    "pixmo_count": "fsc147_relative",
    "pixmo_count_lmms": "fsc147_relative",
}

ANSWER_TYPE_TO_FAMILY = {
    "number": "numeric_exact",
    "numeric": "numeric_exact",
    "integer": "numeric_exact",
    "float": "numeric_exact",
    "multiple_choice": "multiple_choice",
    "multiple-choice": "multiple_choice",
    "choice": "multiple_choice",
    "mcq": "multiple_choice",
    "math_expressions": "math_verify",
    "math-expressions": "math_verify",
    "boolean": "boolean",
    "bool": "boolean",
    "ocrtext": "ocr_levenshtein",
    "ocr_text": "ocr_levenshtein",
    "short_text": "short_text",
    "string": "exact",
    "any": "exact",
    "bbox": "bbox_iou",
    "box": "bbox_iou",
    "judge": "judge",
    "html_code": "html_code",
    "html-code": "html_code",
    "svg_code": "svg_code",
    "svg-code": "svg_code",
    "general_code": "general_code",
    "general-code": "general_code",
    "critic": "judge",
}

REWARD_FAMILY_ALIASES = {
    "multiple_choice": "multiple_choice",
    "multiple-choice": "multiple_choice",
    "mcq": "multiple_choice",
    "math_expressions": "math_verify",
    "math-expressions": "math_verify",
    "math": "math_verify",
    "math_verify": "math_verify",
    "ocrtext": "ocr_levenshtein",
    "ocr_text": "ocr_levenshtein",
    "ocr_levenshtein": "ocr_levenshtein",
    "ocrbench_inclusion": "ocr_inclusion",
    "relative_count": "fsc147_relative",
    "relative_count_score": "fsc147_relative",
    "relative_count_threshold": "fsc147_relative",
    "fsc147_relative": "fsc147_relative",
    "textvqa_soft": "vqa_soft",
    "vqa_soft": "vqa_soft",
    "bbox": "bbox_iou",
    "box": "bbox_iou",
    "grounding_bbox": "bbox_iou",
    "judge_optional": "judge",
    "html_code": "html_code",
    "html-code": "html_code",
    "svg_code": "svg_code",
    "svg-code": "svg_code",
    "general_code": "general_code",
    "general-code": "general_code",
}


def compute_toolvision_score(
    *,
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None,
    extracted_answer: str | None = None,
) -> dict[str, float | str] | None:
    extra_info = extra_info or {}
    if not _looks_like_toolvision_rl(extra_info):
        return None

    answer = extract_answer(solution_str) or extracted_answer
    family = _reward_family(data_source, extra_info)
    if family is None:
        return None

    score = _score_by_family(family, answer or "", ground_truth, extra_info)
    judge_used = 0
    judge_fallback_score = score

    canonical_family = _canonical_family(family)
    if score < 1.0 and _family_allows_judge(canonical_family) and (
        canonical_family in {"judge", "html_code", "svg_code", "general_code"} or should_use_judge(extra_info)
    ):
        judge_ground_truth = " | ".join(str(item) for item in answer_candidates(ground_truth, extra_info)) or str(ground_truth or "")
        judge_fallback_score, judge_used = judge_score(
            answer or "",
            judge_ground_truth,
            question=str(extra_info.get("question") or ""),
            task=str(extra_info.get("source_dataset") or data_source or "general"),
        )
        score = max(score, judge_fallback_score)

    result = {
        "score": float(score),
        "official_like_score": float(score),
        "llmjudge_score": float(judge_fallback_score) if judge_used else 0.0,
        "llmjudge_used": float(judge_used),
        "llmjudge_fallback_score": float(score),
        "final_answer": answer or "",
        "reward_family": family,
        "source_dataset": str(extra_info.get("source_dataset") or data_source or ""),
    }
    for key in (
        "mut_weight",
        "regular_tool_penalty",
        "train_group",
        "mut_v1_bucket",
        "mut_v1_BC",
        "mut_v2_class",
    ):
        if key in extra_info:
            result[key] = extra_info[key]
    result.update(tool_usage_info(solution_str))
    return result


def _looks_like_toolvision_rl(extra_info: dict[str, Any]) -> bool:
    return any(key in extra_info for key in ("source_dataset", "reward_family", "answer_type", "task_type"))


def _reward_family(data_source: str, extra_info: dict[str, Any]) -> str | None:
    source_dataset = extra_info.get("source_dataset") or data_source
    source_key = _normalize_key(source_dataset)

    # Source-specific metrics are the authority for known benchmarks.  Some
    # existing parquet files still carry older reward_family labels, e.g.
    # TextVQA as ocr_levenshtein before we switched it to VQA soft accuracy.
    # Force these mappings first so old data is rescored/trained with the
    # current standard without rewriting the parquet.
    if source_key in SOURCE_DATASET_FORCE_FAMILY:
        return SOURCE_DATASET_FORCE_FAMILY[source_key]
    for prefix, family in SOURCE_DATASET_FORCE_FAMILY.items():
        if source_key.startswith(prefix):
            return family

    if extra_info.get("reward_family"):
        return _canonical_family(extra_info["reward_family"])

    answer_type = extra_info.get("answer_type")
    if answer_type is not None:
        family = ANSWER_TYPE_TO_FAMILY.get(_normalize_key(answer_type))
        if family:
            return family

    if source_key in SOURCE_DATASET_TO_FAMILY:
        return SOURCE_DATASET_TO_FAMILY[source_key]

    for prefix, family in SOURCE_DATASET_TO_FAMILY.items():
        if source_key.startswith(prefix):
            return family
    return None


def _score_by_family(family: str, prediction: Any, ground_truth: Any, extra_info: dict[str, Any]) -> float:
    options = choice_options_from_extra_info(extra_info)
    scorers: dict[str, ScoreFn] = {
        "exact": lambda pred, gt, _extra: exact_score(pred, gt, _extra),
        "short_text": lambda pred, gt, _extra: exact_score(pred, gt, _extra),
        "numeric_exact": lambda pred, gt, _extra: numeric_exact_score(pred, gt, _extra),
        "fsc147_relative": lambda pred, gt, _extra: relative_count_score(pred, gt, _extra),
        "chartqa_relaxed": lambda pred, gt, _extra: chartqa_relaxed_score(pred, gt, _extra),
        "ocr_inclusion": lambda pred, gt, _extra: ocr_inclusion_score(pred, gt, _extra),
        "ocr_levenshtein": lambda pred, gt, _extra: levenshtein_ratio_score(pred, gt, _extra),
        "vqa_soft": lambda pred, gt, _extra: vqa_soft_score(pred, gt, _extra),
        "multiple_choice": lambda pred, gt, _extra: multiple_choice_score(pred, gt, options, _extra),
        "boolean": lambda pred, gt, _extra: boolean_score(pred, gt, _extra),
        "bbox_iou": lambda pred, gt, _extra: bbox_iou_score(pred, gt),
        "math_verify": lambda pred, gt, _extra: math_verify_score(pred, gt, _extra),
        "judge": lambda pred, gt, _extra: exact_score(pred, gt, _extra),
        "html_code": lambda pred, gt, _extra: exact_score(pred, gt, _extra),
        "svg_code": lambda pred, gt, _extra: exact_score(pred, gt, _extra),
        "general_code": lambda pred, gt, _extra: exact_score(pred, gt, _extra),
    }
    scorer = scorers.get(_canonical_family(family))
    return 0.0 if scorer is None else float(scorer(prediction, ground_truth, extra_info))


def _family_allows_judge(family: str) -> bool:
    return _canonical_family(family) in {"short_text", "exact", "judge", "ocr_levenshtein", "html_code", "svg_code", "general_code"}


def _normalize_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _canonical_family(value: Any) -> str:
    key = _normalize_key(value)
    return REWARD_FAMILY_ALIASES.get(key, key)
