from pathlib import Path

from PIL import Image


NUMBER_WORD_TO_NUMERAL = {
    "none": "0",
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
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def _normalize_count_answer(answer) -> str:
    normalized = str(answer).strip().lower()
    return NUMBER_WORD_TO_NUMERAL.get(normalized, normalized)


def doc_to_visual(doc):
    image_path = Path(str(doc["image_path"]))
    if not image_path.exists():
        raise FileNotFoundError(f"Missing CountBench image: {image_path}")
    return [Image.open(image_path).convert("RGB")]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    question = str(doc["question"]).strip()
    return f"{pre_prompt}{question}{post_prompt}"


def doc_to_target(doc):
    return _normalize_count_answer(doc["answer"])


def process_results(doc, results):
    prediction = _normalize_count_answer(results[0])
    target = doc_to_target(doc)
    score = float(prediction == target)
    return {"acc": score, "exact_match": score}
