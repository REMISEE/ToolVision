import re
import os
from pathlib import Path
from typing import Any

from PIL import Image


def _normalize(text: Any) -> str:
    text = str(text or "").strip().lower()
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


def doc_to_visual(doc):
    image_path = Path(str(doc.get("image_path", "")))
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image for all-source lmms pass16 rerun: {image_path}")
    image = Image.open(image_path).convert("RGB")
    max_raw_pixels = int(os.environ.get("TV_DROP_IMAGE_IF_RAW_PIXELS_GT", "0") or "0")
    if max_raw_pixels > 0 and image.width * image.height > max_raw_pixels:
        blank_size = int(os.environ.get("TV_BLANK_IMAGE_SIZE", "64") or "64")
        image.close()
        return [Image.new("RGB", (blank_size, blank_size), color=(255, 255, 255))]
    return [image]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    return f"{kwargs.get('pre_prompt', '')}{str(doc.get('prompt', '')).strip()}{kwargs.get('post_prompt', '')}"


def doc_to_target(doc):
    answers = _answers(doc)
    return answers[0] if answers else ""


def _to_number(text: Any) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", str(text or "").replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _relative_count_score(prediction: str, answers: list[str]) -> float:
    pred = _to_number(prediction)
    target = _to_number(answers[0] if answers else "")
    if pred is None or target is None:
        return 0.0
    return max(0.0, 1.0 - abs(pred - target) / max(abs(target), 1.0))


def _exact_score(prediction: str, answers: list[str]) -> float:
    pred = _normalize(prediction)
    return float(any(pred == _normalize(answer) for answer in answers))


def process_results(doc, results):
    source = str(doc.get("source", "")).lower()
    answers = _answers(doc)
    flat_results = []
    for result in results:
        if isinstance(result, list):
            flat_results.extend(result)
        else:
            flat_results.append(result)

    scores = []
    for result in flat_results:
        if source in {"fsc147"}:
            scores.append(_relative_count_score(str(result), answers))
        else:
            scores.append(_exact_score(str(result), answers))
    return {"tv_pass16_score": float(sum(scores) / len(scores)) if scores else 0.0}


def aggregate_scores(items):
    return float(sum(float(item) for item in items) / len(items)) if items else 0.0
