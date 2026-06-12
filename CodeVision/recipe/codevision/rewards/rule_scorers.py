import re
from typing import Any

from .common import answer_candidates, as_list, answers_equivalent, extract_numeric_answer, normalize_text


def exact_score(prediction: Any, ground_truth: Any, extra_info: dict[str, Any] | None = None) -> float:
    return 1.0 if any(answers_equivalent(prediction, target) for target in answer_candidates(ground_truth, extra_info)) else 0.0


def numeric_exact_score(
    prediction: Any, ground_truth: Any, extra_info: dict[str, Any] | None = None, tolerance: float = 1e-4
) -> float:
    pred = extract_numeric_answer(prediction)
    if pred is None:
        return 0.0
    for target in answer_candidates(ground_truth, extra_info):
        gt = extract_numeric_answer(target)
        if gt is not None and abs(pred - gt) <= tolerance:
            return 1.0
    return 0.0


def relative_count_threshold_score(
    prediction: Any, ground_truth: Any, extra_info: dict[str, Any] | None = None
) -> float:
    score = relative_count_score(prediction, ground_truth, extra_info)
    extra_info = extra_info or {}
    try:
        threshold = float(extra_info.get("relative_count_threshold", 0.9))
    except Exception:
        threshold = 0.9
    return 1.0 if score >= threshold else 0.0


def relative_count_score(prediction: Any, ground_truth: Any, extra_info: dict[str, Any] | None = None) -> float:
    pred = extract_numeric_answer(prediction)
    if pred is None:
        return 0.0
    best = 0.0
    for target in answer_candidates(ground_truth, extra_info):
        gt = extract_numeric_answer(target)
        if gt is None:
            continue
        relative_score = max(0.0, 1.0 - (abs(pred - gt) / max(abs(gt), 1.0)))
        best = max(best, relative_score)
    return best


def vqa_soft_score(prediction: Any, ground_truth: Any, extra_info: dict[str, Any] | None = None) -> float:
    """EvalAI/VQA-style soft accuracy, used by TextVQA.

    With 10 annotator answers this is equivalent to min(match_count / 3, 1).
    If fewer answers are available, keep the same convention rather than
    converting to exact match, so partial agreement remains visible.
    """
    pred = _normalize_vqa_answer(prediction)
    if not pred:
        return 0.0
    raw_targets: list[Any] = []
    extra_info = extra_info or {}
    for key in ("answers", "acceptable_answers", "answer_aliases"):
        raw_targets.extend(as_list(extra_info.get(key)))
    raw_targets.extend(as_list(ground_truth))
    targets = [_normalize_vqa_answer(target) for target in raw_targets]
    targets = [target for target in targets if target]
    if not targets:
        return 0.0
    match_count = sum(1 for target in targets if target == pred)
    return min(1.0, float(match_count) / 3.0)


def _normalize_vqa_answer(text: Any) -> str:
    text = normalize_text(text)
    text = re.sub(r"([,.;:!?\"'`])", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    number_map = {
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
    return number_map.get(text, text)


def chartqa_relaxed_score(
    prediction: Any, ground_truth: Any, extra_info: dict[str, Any] | None = None, max_relative_change: float = 0.05
) -> float:
    prediction = str(prediction or "").strip()

    def to_float(text: str) -> float | None:
        try:
            value = text.strip()
            if value.endswith("%"):
                return float(value.rstrip("%")) / 100.0
            return float(value)
        except Exception:
            return None

    pred_float = to_float(prediction)
    for target in answer_candidates(ground_truth, extra_info):
        target = str(target or "").strip()
        target_float = to_float(target)
        if pred_float is not None and target_float:
            relative_change = abs(pred_float - target_float) / abs(target_float)
            if relative_change <= max_relative_change:
                return 1.0
        elif normalize_text(prediction) == normalize_text(target):
            return 1.0
    return 0.0


def ocr_inclusion_score(prediction: Any, ground_truth: Any, extra_info: dict[str, Any] | None = None) -> float:
    pred = normalize_text(prediction)
    return 1.0 if any(normalize_text(target) and normalize_text(target) in pred for target in answer_candidates(ground_truth, extra_info)) else 0.0


def levenshtein_ratio_score(
    prediction: Any, ground_truth: Any | list[Any], extra_info: dict[str, Any] | None = None, min_score: float = 0.5
) -> float:
    pred = normalize_text(prediction)
    if not pred:
        return 0.0

    def distance(s1: str, s2: str) -> int:
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    scores = []
    for target in answer_candidates(ground_truth, extra_info):
        target_norm = normalize_text(target)
        if not target_norm:
            continue
        denom = max(len(target_norm), len(pred))
        score = 0.0 if denom == 0 else 1.0 - float(distance(target_norm, pred)) / float(denom)
        scores.append(max(0.0, score))
    if not scores:
        return 0.0
    best = max(scores)
    return best if best >= min_score else 0.0


def multiple_choice_score(
    prediction: Any, ground_truth: Any, options: list[Any] | None = None, extra_info: dict[str, Any] | None = None
) -> float:
    pred = extract_choice(prediction, options)
    if pred is None:
        return 0.0
    for target in answer_candidates(ground_truth, extra_info):
        gt = extract_choice(target, options) or str(target or "").strip()
        if pred.upper() == gt.upper():
            return 1.0
    return 0.0


def extract_choice(prediction: Any, options: list[Any] | None = None) -> str | None:
    text = str(prediction or "").strip()
    if not text:
        return None

    options = options or []
    max_letter = chr(ord("A") + (len(options) - 1 if options else 4))
    letter_range = f"A-{max_letter}"
    upper = text.upper()

    answer_patterns = [
        rf"(?:FINAL\s+ANSWER|CORRECT\s+ANSWER|ANSWER|OPTION|CHOICE)\s*(?:IS|:|：)?\s*\(?([{letter_range}])\)?",
        rf"(?:THE\s+)?(?:FINAL\s+)?(?:ANSWER|CORRECT\s+ANSWER)\s*(?:IS|:|：)?\s*\(?([{letter_range}])\)?",
        rf"^\s*\(?([{letter_range}])\)?\s*[\).:：]?\s*$",
        rf"\(([{letter_range}])\)",
        rf"\b([{letter_range}])\s*[\).:：]",
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, upper)
        if matches:
            return matches[-1]

    if options:
        pred_norm = normalize_text(text)
        for idx, option in enumerate(options):
            if normalize_text(option) == pred_norm:
                return chr(ord("A") + idx)
        contained = [
            chr(ord("A") + idx)
            for idx, option in enumerate(options)
            if normalize_text(option) and normalize_text(option) in pred_norm
        ]
        if len(contained) == 1:
            return contained[0]

    generic = re.findall(rf"\b([{letter_range}])\b", upper)
    return generic[-1] if generic else None


def boolean_score(prediction: Any, ground_truth: Any, extra_info: dict[str, Any] | None = None) -> float:
    pred = parse_bool(prediction)
    if pred is None:
        return 0.0
    for target in answer_candidates(ground_truth, extra_info):
        gt = parse_bool(target)
        if gt is not None and pred == gt:
            return 1.0
    return 0.0


def parse_bool(text: Any) -> bool | None:
    norm = normalize_text(text)
    direct = re.search(r"^\W*(yes|no|true|false)\b", norm)
    if direct:
        return direct.group(1) in {"yes", "true"}
    answer = re.findall(r"(?:final answer|correct answer|answer)\s*(?:is|:|：)?\s*(yes|no|true|false)\b", norm)
    if answer:
        return answer[-1] in {"yes", "true"}
    matches = re.findall(r"\b(yes|no|true|false)\b", norm)
    if matches:
        value = matches[-1]
        if value in {"yes", "true"}:
            return True
        if value in {"no", "false"}:
            return False
    return None


def bbox_iou_score(prediction: Any, ground_truth: Any) -> float:
    pred = extract_bbox(prediction)
    if pred is None or len(pred) != 4:
        return 0.0
    best = 0.0
    targets = [ground_truth]
    targets.extend(answer_candidates(ground_truth))
    for target in targets:
        gt = extract_bbox(target)
        if gt is None or len(gt) != 4:
            continue
        x_a = max(pred[0], gt[0])
        y_a = max(pred[1], gt[1])
        x_b = min(pred[2], gt[2])
        y_b = min(pred[3], gt[3])
        inter = max(0.0, x_b - x_a) * max(0.0, y_b - y_a)
        pred_area = max(0.0, pred[2] - pred[0]) * max(0.0, pred[3] - pred[1])
        gt_area = max(0.0, gt[2] - gt[0]) * max(0.0, gt[3] - gt[1])
        union = pred_area + gt_area - inter
        best = max(best, 0.0 if union <= 0 else inter / union)
    return best


def extract_bbox(text: Any) -> list[float] | None:
    if isinstance(text, (list, tuple)) and len(text) == 4:
        try:
            return _normalize_bbox([float(x) for x in text])
        except Exception:
            return None
    match = re.search(r"\[\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:\s*,\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?){3}\s*\]", str(text or ""))
    if not match:
        return None
    try:
        return _normalize_bbox([float(x.strip()) for x in match.group(0)[1:-1].split(",")])
    except Exception:
        return None


def _normalize_bbox(box: list[float]) -> list[float]:
    if len(box) != 4:
        return box
    if max(abs(value) for value in box) > 2.0:
        box = [value / 1000.0 for value in box]
    x1, y1, x2, y2 = box
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [min(1.0, max(0.0, value)) for value in [x1, y1, x2, y2]]


def math_verify_score(prediction: Any, ground_truth: Any, extra_info: dict[str, Any] | None = None) -> float:
    try:
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify
    except Exception:
        return exact_score(prediction, ground_truth, extra_info)

    extraction_config = [
        StringExtractionConfig(strings=tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")),
        LatexExtractionConfig(),
        ExprExtractionConfig(),
    ]
    pred_text = str(prediction or "").strip()
    for target in answer_candidates(ground_truth, extra_info):
        target_text = str(target or "").strip()
        try:
            if verify(parse(pred_text, extraction_config=extraction_config), parse(target_text, extraction_config=extraction_config)):
                return 1.0
        except Exception:
            pass
        if answers_equivalent(pred_text, target_text):
            return 1.0
    return 0.0
