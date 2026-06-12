from __future__ import annotations


_DATASET_ALIASES: dict[str, str] = {
    "arxivvqa": "arxivqa",
    "arxivqa": "arxivqa",
    "cavqa": "cavqa_multichoice",
    "cavqa_multichoice": "cavqa_multichoice",
    "chartqa": "chartqa",
}
_COUNT_DATASETS = frozenset({"fsc147"})
_HIGH_CONF_EXACT_MATCH_DATASETS = frozenset({"arxivqa", "cavqa_multichoice", "gqa"})
# ChartQA intentionally stays out of reference forced-answer datasets. Its gold
# answer must not be injected into planner prompts; judge scores may only control
# whether the next round is MUST_ANSWER.
_REFERENCE_FORCED_FINAL_ANSWER_DATASETS = frozenset({"arxivqa", "cavqa_multichoice", "gqa", "textvqa"})


def canonicalize_dataset_name(name: str | None) -> str:
    normalized = str(name or "").strip().lower()
    if not normalized:
        return ""
    return _DATASET_ALIASES.get(normalized, normalized)


def is_count_dataset(name: str | None) -> bool:
    return canonicalize_dataset_name(name) in _COUNT_DATASETS


def is_high_conf_exact_match_dataset(name: str | None) -> bool:
    return canonicalize_dataset_name(name) in _HIGH_CONF_EXACT_MATCH_DATASETS


def is_reference_forced_final_answer_dataset(name: str | None) -> bool:
    return canonicalize_dataset_name(name) in _REFERENCE_FORCED_FINAL_ANSWER_DATASETS
