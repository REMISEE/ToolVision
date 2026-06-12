import re
import statistics
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor
except Exception:  # pragma: no cover
    EvalAIAnswerProcessor = None


def _normalize(text: Any) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"<answer>\s*(.*?)\s*</answer>", r"\1", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"[^\w\s.%-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _answers(doc) -> list[str]:
    answers = doc.get("answers")
    if answers is None:
        answers = [doc.get("answer", "")]
    if not isinstance(answers, list):
        answers = [answers]
    return [str(answer) for answer in answers]


def tv_doc_to_visual(doc):
    image_path = Path(str(doc.get("image_path", "")))
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image for lmms pass16 rerun: {image_path}")
    return [Image.open(image_path).convert("RGB")]


def tv_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    return f"{pre_prompt}{str(doc.get('prompt', '')).strip()}{post_prompt}"


def tv_doc_to_target(doc):
    answers = _answers(doc)
    return answers[0] if answers else ""


def _exact_score(prediction: str, answers: list[str]) -> float:
    pred = _normalize(prediction)
    return float(any(pred == _normalize(answer) for answer in answers))


def _textvqa_score(prediction: str, answers: list[str]) -> float:
    if EvalAIAnswerProcessor is None:
        return _exact_score(prediction, answers)

    processor = EvalAIAnswerProcessor()
    pred = processor(_normalize(prediction))
    processed_answers = [processor(answer) for answer in answers]
    if not processed_answers:
        return 0.0

    # EvalAI/VQA soft accuracy uses each annotator as holdout, preserving repeats.
    scores = []
    for i, _ in enumerate(processed_answers):
        other_answers = processed_answers[:i] + processed_answers[i + 1 :]
        scores.append(min(1.0, other_answers.count(pred) / 3.0))
    return float(statistics.mean(scores))


def _to_number(text: Any) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _fsc_relative_score(prediction: str, answers: list[str]) -> float:
    pred = _to_number(prediction)
    target = _to_number(answers[0] if answers else "")
    if pred is None or target is None:
        return 0.0
    denom = max(abs(target), 1.0)
    return max(0.0, 1.0 - abs(pred - target) / denom)


def tv_process_results(doc, results):
    source = str(doc.get("source", "")).lower()
    answers = _answers(doc)
    scores = []
    flat_results = []
    for result in results:
        if isinstance(result, list):
            flat_results.extend(result)
        else:
            flat_results.append(result)
    for result in flat_results:
        if source == "textvqa":
            scores.append(_textvqa_score(str(result), answers))
        elif source == "fsc147":
            scores.append(_fsc_relative_score(str(result), answers))
        else:
            scores.append(_exact_score(str(result), answers))
    return {"tv_pass16_score": float(sum(scores) / len(scores)) if scores else 0.0}


def tv_aggregate_scores(items):
    if not items:
        return 0.0
    return float(sum(float(item) for item in items) / len(items))
