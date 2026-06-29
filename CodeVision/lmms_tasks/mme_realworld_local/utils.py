import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image


CHOICE_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)


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
    raise FileNotFoundError("No MME-RealWorld image path found in doc")


def _prompt_text(doc):
    prompt = _first(doc.get("prompt"))
    if isinstance(prompt, dict):
        return str(prompt.get("content") or "").replace("<image>", "").strip()
    if prompt is not None:
        return str(prompt).replace("<image>", "").strip()
    extra = _nested(doc, "extra_info", {})
    return str(extra.get("question") or "").replace("<image>", "").strip()


def _target(doc):
    reward_model = _nested(doc, "reward_model", {})
    target = reward_model.get("ground_truth") if reward_model else None
    if target is None:
        extra = _nested(doc, "extra_info", {})
        target = extra.get("raw_answer") if extra else None
    return _extract_choice(target)


def _extract_choice(text):
    if text is None:
        return ""
    text = str(text).strip()
    if not text:
        return ""
    answer_matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if answer_matches:
        text = answer_matches[-1]
    matches = CHOICE_RE.findall(text)
    if matches:
        return matches[-1].upper()
    stripped = text.strip().upper()
    return stripped[0] if stripped[:1] in {"A", "B", "C", "D", "E"} else ""


def doc_to_visual(doc):
    path = _image_path(doc)
    if not path.exists():
        raise FileNotFoundError(f"Missing MME-RealWorld image: {path}")
    return [Image.open(path).convert("RGB")]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    return f"{kwargs.get('pre_prompt', '')}{_prompt_text(doc)}{kwargs.get('post_prompt', '')}"


def doc_to_target(doc):
    return _target(doc)


def process_results(doc, results):
    target = doc_to_target(doc)
    pred = _extract_choice(results[0] if results else "")
    score = float(bool(target) and pred == target)
    return {"acc": score, "exact_match": score}
