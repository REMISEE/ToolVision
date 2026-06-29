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


def _nested(doc, key, default=None):
    value = doc.get(key)
    if isinstance(value, dict):
        return value
    return default


def _image_path(doc):
    extra = _nested(doc, "extra_info", {})
    path = extra.get("image_path") if extra else None
    if path:
        return Path(str(path))
    images = _first(doc.get("images"))
    if isinstance(images, dict):
        uri = str(images.get("image", ""))
        parsed = urlparse(uri)
        if parsed.scheme == "file":
            return Path(unquote(parsed.path))
        if uri:
            return Path(uri)
    raise FileNotFoundError("No FSC147 image path found in doc")


_NUMBER_RE = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])")


def _number_to_int(text):
    cleaned = text.replace(",", "")
    if "." not in cleaned:
        return int(cleaned)
    return int(round(float(cleaned)))


def _numeric_matches(text):
    return [match.group(0) for match in _NUMBER_RE.finditer(str(text))]


def _reasonable_count_matches(text):
    items = []
    for match in _numeric_matches(text):
        cleaned = match.replace(",", "")
        digits = cleaned.lstrip("-").split(".", 1)[0]
        if len(digits) > 6:
            continue
        try:
            value = _number_to_int(match)
        except Exception:
            continue
        if abs(value) <= 1_000_000:
            items.append(match)
    return items


def _to_int(value):
    value = _first(value)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    matches = _numeric_matches(str(value).strip())
    if not matches:
        return None
    return _number_to_int(matches[0])


def _prediction_to_int(value):
    value = _first(value)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    text = str(value).strip()
    if not text:
        return None

    answer_matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if answer_matches:
        matches = _reasonable_count_matches(answer_matches[-1])
        if matches:
            return _number_to_int(matches[-1])

    tail = text[-800:]
    for pattern in (
        r"(?:final answer|answer|there are|there is|total(?: number)?(?: is)?)[^\d-]{0,80}(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)",
        r"(?:so|therefore)[^\d-]{0,80}(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)",
    ):
        matches = re.findall(pattern, tail, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            return _number_to_int(matches[-1])

    matches = _reasonable_count_matches(tail)
    if not matches:
        matches = _reasonable_count_matches(text)
    if not matches:
        return None
    return _number_to_int(matches[-1])


def doc_to_visual(doc):
    path = _image_path(doc)
    if not path.exists():
        raise FileNotFoundError(f"Missing FSC147 image: {path}")
    return [Image.open(path).convert("RGB")]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    extra = _nested(doc, "extra_info", {})
    question = ""
    if extra:
        question = str(extra.get("question") or "").strip()
    if not question:
        prompt = _first(doc.get("prompt"))
        if isinstance(prompt, dict):
            question = str(prompt.get("content") or "").replace("<image>", "").strip()
    return f"{pre_prompt}{question}{post_prompt}"


def doc_to_target(doc):
    reward_model = _nested(doc, "reward_model", {})
    target = _to_int(reward_model.get("ground_truth") if reward_model else None)
    if target is None:
        extra = _nested(doc, "extra_info", {})
        target = _to_int(extra.get("count") if extra else None)
    return "" if target is None else str(target)


def process_results(doc, results):
    target = _to_int(doc_to_target(doc))
    pred = _prediction_to_int(results[0] if results else "")
    if target is None:
        return {"exact_match": 0.0, "mae": 0.0}
    if pred is None:
        return {"exact_match": 0.0, "mae": float(abs(target))}
    return {"exact_match": float(pred == target), "mae": float(abs(pred - target))}


def aggregate_mae(items):
    if not items:
        return 0.0
    return sum(float(item) for item in items) / len(items)
