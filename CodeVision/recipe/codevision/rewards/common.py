import json
import re
import ast
from fractions import Fraction
from typing import Any


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


def extract_answer(solution_str: str) -> str | None:
    if not solution_str:
        return None

    cleaned = re.sub(r"<tool_response>.*?</tool_response>", "", str(solution_str), flags=re.DOTALL)
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", cleaned, flags=re.DOTALL)
    matches = re.findall(r"<answer>(.*?)</answer>", cleaned, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()

    parts = cleaned.strip().split("<answer>")
    if len(parts) > 1:
        return parts[-1].strip()

    boxed = extract_boxed_answer(cleaned)
    if boxed:
        return boxed

    answer_phrase = re.findall(
        r"(?:final answer|answer|the answer is|correct answer is)\s*[:：]?\s*([^\n<]+)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if answer_phrase:
        return answer_phrase[-1].strip().rstrip(".")

    lines = cleaned.strip().splitlines()
    return lines[-1].strip() if lines and len(lines) < 500 else None


def extract_boxed_answer(text: Any) -> str:
    text = str(text or "")
    marker = "\\boxed{"
    starts = []
    pos = 0
    while True:
        pos = text.find(marker, pos)
        if pos == -1:
            break
        starts.append(pos)
        pos += len(marker)

    results = []
    for start in starts:
        depth = 0
        content_start = start + len(marker)
        pos = content_start - 1
        while pos < len(text):
            char = text[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    results.append(text[content_start:pos])
                    break
            pos += 1
    return results[-1].strip() if results else ""


def as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return as_list(value.tolist())
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except Exception:
            pass
    return [value]


def answer_candidates(ground_truth: Any, extra_info: dict[str, Any] | None = None) -> list[Any]:
    extra_info = extra_info or {}
    candidates = []
    for key in ("acceptable_answers", "answers", "answer_aliases"):
        candidates.extend(as_list(extra_info.get(key)))
    candidates.extend(as_list(ground_truth))
    deduped = []
    seen = set()
    for item in candidates:
        key = normalize_text(item)
        if key and key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).casefold()


def normalize_count_answer(answer: Any) -> str:
    text = str(answer or "").strip().lower()
    return NUMBER_WORD_TO_NUMERAL.get(text, text)


def extract_numeric_answer(text: Any) -> float | None:
    text = normalize_count_answer(text)
    try:
        return float(text)
    except Exception:
        pass

    matches = re.findall(r"[-+]?\d*\.?\d+", str(text))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


def answers_equivalent(prediction: Any, ground_truth: Any, tolerance: float = 1e-4) -> bool:
    if prediction is None or ground_truth is None:
        return False

    pred = str(prediction).strip()
    gt = str(ground_truth).strip()
    if not pred or not gt:
        return False

    if normalize_text(pred) == normalize_text(gt):
        return True

    try:
        return abs(float(Fraction(pred)) - float(Fraction(gt))) <= tolerance
    except Exception:
        pass

    try:
        return abs(float(pred) - float(gt)) <= tolerance
    except Exception:
        return False


def tool_usage_info(solution_str: str) -> dict[str, float | str]:
    text = str(solution_str or "")
    stripped_text = re.sub(r"<tool_response>.*?</tool_response>", "", text, flags=re.DOTALL)

    tool_call_starts = len(re.findall(r"<tool_call>", stripped_text))
    tool_call_ends = len(re.findall(r"</tool_call>", stripped_text))
    tool_call_blocks = re.findall(r"<tool_call>(.*?)</tool_call>", stripped_text, flags=re.DOTALL)
    regex_tool_names = re.findall(r'"name"\s*:\s*"([^"]+)"', stripped_text)

    valid_tool_names: list[str] = []
    malformed_tool_call_count = abs(tool_call_starts - tool_call_ends)
    for block in tool_call_blocks:
        try:
            payload = json.loads(block.strip())
            if not isinstance(payload, dict):
                malformed_tool_call_count += 1
                continue
            name = payload.get("name")
            arguments = payload.get("arguments")
            if not isinstance(name, str) or not name.strip() or not isinstance(arguments, dict):
                malformed_tool_call_count += 1
                continue
            valid_tool_names.append(name.strip())
        except Exception:
            malformed_tool_call_count += 1

    answer_then_tool_call = 1.0 if re.search(r"</answer>.*<tool_call>", stripped_text, flags=re.DOTALL) else 0.0
    think_open = len(re.findall(r"<think>", stripped_text))
    think_close = len(re.findall(r"</think>", stripped_text))
    answer_open = len(re.findall(r"<answer>", stripped_text))
    answer_close = len(re.findall(r"</answer>", stripped_text))
    format_tag_mismatch = 1.0 if think_open != think_close or answer_open != answer_close else 0.0
    repeated_format_label = 1.0 if answer_open > 1 or answer_close > 1 else 0.0

    tool_names = sorted(set(valid_tool_names + regex_tool_names))
    invalid_tool_call = 1.0 if (
        malformed_tool_call_count > 0
        or answer_then_tool_call
        or format_tag_mismatch
        or repeated_format_label
    ) else 0.0
    return {
        "used_tool": 1.0 if tool_call_starts or tool_names else 0.0,
        "tool_names": ",".join(tool_names),
        "invalid_tool_call": invalid_tool_call,
        "tool_call_count": float(tool_call_starts),
        "valid_tool_call_count": float(len(valid_tool_names)),
        "malformed_tool_call_count": float(malformed_tool_call_count),
        "answer_then_tool_call": answer_then_tool_call,
        "format_tag_mismatch": format_tag_mismatch,
        "repeated_format_label": repeated_format_label,
    }
