import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image


def _first(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    try:
        if hasattr(value, "tolist"):
            return _first(value.tolist())
    except Exception:
        pass
    return value


def _as_list(value):
    if value is None:
        return []
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None and str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _nested(doc, key, default=None):
    value = doc.get(key)
    return value if isinstance(value, dict) else default


def _image_path(doc):
    extra = _nested(doc, "extra_info", {})
    path = extra.get("image_path") if extra else None
    if path:
        return Path(str(path))
    image = _first(doc.get("images"))
    if isinstance(image, dict):
        uri = str(image.get("image", ""))
        parsed = urlparse(uri)
        if parsed.scheme == "file":
            return Path(unquote(parsed.path))
        if uri:
            return Path(uri)
    raise FileNotFoundError("No DocVQA/InfoVQA image path found in doc")


def _prompt_text(doc):
    extra = _nested(doc, "extra_info", {})
    question = str(extra.get("question") or "").strip() if extra else ""
    if not question:
        prompt = _first(doc.get("prompt"))
        if isinstance(prompt, dict):
            question = str(prompt.get("content") or "").strip()
        elif prompt is not None:
            question = str(prompt).strip()
    question = question.replace("<image>", "").strip()
    if "answer the question" not in question.lower():
        question = f"{question}\nAnswer the question using a single word or phrase."
    return question


def _answers(doc):
    extra = _nested(doc, "extra_info", {})
    answers = _as_list(extra.get("answers") if extra else None)
    if answers:
        return answers
    reward_model = _nested(doc, "reward_model", {})
    return _as_list(reward_model.get("ground_truth") if reward_model else None)


def _extract_answer(text):
    text = str(text or "").strip()
    if not text:
        return ""

    answer_matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if answer_matches:
        text = answer_matches[-1].strip()
    elif "</think>" in text.lower():
        parts = re.split(r"</think>", text, flags=re.IGNORECASE)
        text = parts[-1].strip()

    for pattern in (
        r"(?:final answer|answer)\s*[:：]\s*(.+)$",
        r"(?:the answer is|therefore,?\s+the answer is)\s+(.+)$",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()
            break

    text = re.sub(r"^['\"`]+|['\"`]+$", "", text.strip())
    return text.splitlines()[0].strip() if "\n" in text else text.strip()


def _levenshtein_distance(s1, s2):
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


def _anls_score(references, prediction, threshold=0.5):
    prediction = " ".join(str(prediction).strip().lower().split())
    if not references:
        return 0.0
    values = []
    for answer in references:
        answer = str(answer)
        gt = " ".join(answer.strip().lower().split())
        length = max(len(gt), len(prediction))
        if length == 0:
            values.append(0.0)
            continue
        values.append(float(_levenshtein_distance(gt, prediction)) / float(length))
    score = 1.0 - min(values)
    return score if score >= threshold else 0.0


def doc_to_visual(doc):
    path = _image_path(doc)
    if not path.exists():
        raise FileNotFoundError(f"Missing DocVQA/InfoVQA image: {path}")
    return [Image.open(path).convert("RGB")]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    return f"{kwargs.get('pre_prompt', '')}{_prompt_text(doc)}{kwargs.get('post_prompt', '')}"


def doc_to_target(doc):
    return _answers(doc)


def process_results(doc, results):
    pred = _extract_answer(results[0] if results else "")
    return {"anls": _anls_score(doc_to_target(doc), pred)}
