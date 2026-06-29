# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import string
import json
import sys
import types
import importlib.util
from typing import Union, List
from collections import Counter
try:
    import math_verify  # noqa: F401
except Exception:
    math_verify = None
from fractions import Fraction

from verl.utils.reward_score.llmjudge import LLMJudgeClient
from recipe.codevision.rewards import compute_toolvision_score
from recipe.codevision.rewards.common import (
    choice_options_from_extra_info,
    tool_usage_info as _strict_tool_usage_info,
)

LLM_JUDGE_CLIENT = LLMJudgeClient(
    base_url=os.getenv("LLM_JUDGE_BASE_URL", ""),
    model_name=os.getenv("LLM_JUDGE_MODEL_NAME", None),
    api_key=os.getenv("LLM_JUDGE_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY")),
    timeout=int(os.getenv("LLM_JUDGE_TIMEOUT", "100")),
    max_retries=int(os.getenv("LLM_JUDGE_MAX_RETRIES", "3")),
    temperature=0.0,
)

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

    cleaned = re.sub(r"<tool_response>.*?</tool_response>", "", solution_str, flags=re.DOTALL)
    matches = re.findall(r"<answer>(.*?)</answer>", cleaned, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()

    parts = cleaned.strip().split("<answer>")
    if len(parts) > 1:
        return parts[-1].strip()

    lines = cleaned.strip().splitlines()
    return lines[-1].strip() if lines and len(lines) < 500 else None


def verify_answer_rule(prediction_text: str, ground_truth: str) -> str | None:
    if not prediction_text:
        return None
    text = prediction_text.upper()
    is_choice_question = str(ground_truth).upper() in ["A", "B", "C", "D"]
    if is_choice_question:
        possible_options = re.findall(r"\b([A-D])\b", text)
        if possible_options:
            return possible_options[-1]
    else:
        try:
            ans = float(prediction_text)
            return str(ans)
        except:
            possible_numbers = re.findall(r"[-+]?\d*\.?\d+", text)
            if possible_numbers:
                return possible_numbers[-1]
    return None


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        keys = list(value.keys())
        if keys and all(str(key).strip().upper() in "ABCDEFGH" for key in keys):
            return [value[key] for key in sorted(keys, key=lambda item: str(item).strip().upper())]
        return list(value.values())
    if hasattr(value, "tolist"):
        return _as_list(value.tolist())
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
    return [value]


def _is_fsc147_source(data_source: str) -> bool:
    return str(data_source or "").lower().startswith("fsc147")


def _source_benchmark(data_source: str, extra_info: dict | None) -> str:
    if extra_info and extra_info.get("source_benchmark"):
        return str(extra_info["source_benchmark"]).lower()
    source = str(data_source or "").lower()
    if source.startswith("ocrbench_v2"):
        return "ocrbench_v2"
    if source.startswith("chartqa"):
        return "chartqa"
    if source.startswith("ocrbench"):
        return "ocrbench"
    if source.startswith("cvbench"):
        return "cvbench"
    if source.startswith("hrbench"):
        return "hrbench"
    if source.startswith("mvtoolbench"):
        return "mvtoolbench"
    if source in {"direct_attributes", "relative_position"}:
        return "vstar_bench"
    if source.startswith("fsc147"):
        return "fsc147"
    if source.startswith("countbench"):
        return "countbench"
    if source.startswith("pixmo_count"):
        return "pixmo_count"
    if source.startswith("countqa"):
        return "countqa"
    if source.startswith("docvqa"):
        return "docvqa"
    if source.startswith("infovqa") or source.startswith("infographicvqa"):
        return "infovqa"
    if source.startswith("mme_realworld") or source.startswith("mmerealworld"):
        return "mme_realworld"
    if source.startswith("realworldqa"):
        return "realworldqa"
    if source.startswith("mmstar"):
        return "mmstar"
    if source.startswith("mmvet_hard"):
        return "mmvet_hard"
    if source.startswith("mmvet"):
        return "mmvet"
    if source.startswith("spatialmqa"):
        return "spatialmqa"
    return source


def _extract_numeric_answer(prediction_text: str) -> float | None:
    if not prediction_text:
        return None
    try:
        return float(prediction_text)
    except Exception:
        pass
    matches = re.findall(r"[-+]?\d*\.?\d+", str(prediction_text))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


def _compute_fsc147_score(answer: str, ground_truth: str) -> dict[str, float | str]:
    pred = _extract_numeric_answer(answer)
    gt = _extract_numeric_answer(str(ground_truth))
    if gt is None:
        return {
            "score": 0.0,
            "pred": "",
            "abs_error": 0.0,
            "squared_error": 0.0,
            "relative_score": 0.0,
            "official_like_score": 0.0,
            "official_abs_error": 0.0,
            "official_squared_error": 0.0,
            "llmjudge_score": 0.0,
            "llmjudge_used": 0.0,
        }
    if pred is None:
        pred = 0.0
    abs_error = abs(pred - gt)
    squared_error = abs_error * abs_error
    relative_score = max(0.0, 1.0 - (abs_error / max(abs(gt), 1.0)))
    return {
        "score": relative_score,
        "pred": str(int(round(pred))),
        "abs_error": abs_error,
        "squared_error": squared_error,
        "relative_score": relative_score,
        "official_like_score": relative_score,
        "official_abs_error": abs_error,
        "official_squared_error": squared_error,
        "llmjudge_score": 0.0,
        "llmjudge_used": 0.0,
    }


def _to_float_for_relaxed_accuracy(text: str) -> float | None:
    try:
        value = str(text).strip()
        if value.endswith("%"):
            return float(value.rstrip("%")) / 100.0
        return float(value)
    except Exception:
        return None


def _chartqa_relaxed_score(prediction: str, ground_truth: str, max_relative_change: float = 0.05) -> float:
    """ChartQA relaxed accuracy, matching the common lmms-eval/EvalScope rule."""
    prediction = str(prediction or "").strip()
    target = str(ground_truth or "").strip()
    prediction_float = _to_float_for_relaxed_accuracy(prediction)
    target_float = _to_float_for_relaxed_accuracy(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return 1.0 if relative_change <= max_relative_change else 0.0
    return 1.0 if prediction.lower() == target.lower() else 0.0


def _ocrbench_inclusion_score(prediction: str, ground_truth: str, extra_info: dict | None) -> float:
    """OCRBench inclusion-based matching used by common evaluation harnesses."""
    prediction = str(prediction or "")
    target = str(ground_truth or "")
    dataset = str((extra_info or {}).get("dataset") or "")
    if dataset == "HME100k":
        pred_norm = prediction.strip().replace("\n", " ").replace(" ", "")
        target_norm = target.strip().replace("\n", " ").replace(" ", "")
    else:
        pred_norm = prediction.lower().strip().replace("\n", " ")
        target_norm = target.lower().strip().replace("\n", " ")
    return 1.0 if target_norm and target_norm in pred_norm else 0.0


def _choice_score(prediction: str, ground_truth: str) -> float:
    extracted_pred = verify_answer_rule(str(prediction or ""), str(ground_truth or ""))
    if extracted_pred is None:
        return 0.0
    return 1.0 if extracted_pred.upper() == str(ground_truth).strip().upper() else 0.0


def _normalize_choice_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).casefold()


def _extract_choice_answer(prediction: str, options: list | None = None) -> str | None:
    text = str(prediction or "").strip()
    if not text:
        return None
    max_letter = chr(ord("A") + max(len(options or []) - 1, 0))
    if options:
        letter_matches = re.findall(rf"\b([A-{max_letter}])\b", text.upper())
        if letter_matches:
            return letter_matches[-1]
        option_b_matches = re.findall(rf"\bOPTION\s+([A-{max_letter}])\b", text.upper())
        if option_b_matches:
            return option_b_matches[-1]
        normalized_text = _normalize_choice_text(text)
        for idx, option in enumerate(options):
            if _normalize_choice_text(option) == normalized_text:
                return chr(ord("A") + idx)
        contained = [
            chr(ord("A") + idx)
            for idx, option in enumerate(options)
            if _normalize_choice_text(option) and _normalize_choice_text(option) in normalized_text
        ]
        if len(contained) == 1:
            return contained[0]
    generic = re.findall(r"\b([A-Z])\b", text.upper())
    if generic:
        return generic[-1]
    return None


def _multiple_choice_score(prediction: str, ground_truth: str, extra_info: dict | None = None) -> float:
    extra_info = extra_info or {}
    options = choice_options_from_extra_info(extra_info)
    extracted_pred = _extract_choice_answer(prediction, options)
    if extracted_pred is None:
        return 0.0
    return 1.0 if extracted_pred.upper() == str(ground_truth).strip().upper() else 0.0


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (char_a != char_b)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def _anls_score(prediction: str, ground_truth: str, extra_info: dict | None = None) -> float:
    prediction_norm = re.sub(r"\s+", " ", str(prediction or "").strip().lower())
    candidates = []
    if extra_info:
        candidates.extend(_as_list(extra_info.get("answers")))
        candidates.extend(_as_list(extra_info.get("acceptable_answers")))
    candidates.extend(_as_list(ground_truth))
    best = 0.0
    for candidate in candidates:
        target = re.sub(r"\s+", " ", str(candidate or "").strip().lower())
        if not target and not prediction_norm:
            best = max(best, 1.0)
            continue
        denom = max(len(prediction_norm), len(target))
        if denom <= 0:
            continue
        nls = 1.0 - (_levenshtein_distance(prediction_norm, target) / denom)
        best = max(best, nls if nls >= 0.5 else 0.0)
    return best


def _normalize_count_answer(answer) -> str:
    normalized = str(answer).strip().lower()
    return NUMBER_WORD_TO_NUMERAL.get(normalized, normalized)


def _countbench_score(prediction: str, ground_truth: str) -> float:
    prediction_norm = _normalize_count_answer(prediction)
    target_norm = _normalize_count_answer(ground_truth)
    return 1.0 if are_answers_equivalent(prediction_norm, target_norm) else 0.0


def _numeric_count_metrics(prediction: str, ground_truth: str) -> dict[str, float | str]:
    pred = _extract_numeric_answer(_normalize_count_answer(prediction))
    gt = _extract_numeric_answer(_normalize_count_answer(ground_truth))
    if gt is None:
        return {
            "score": 0.0,
            "pred": "",
            "abs_error": 0.0,
            "mae": 0.0,
            "squared_error": 0.0,
            "official_like_score": 0.0,
            "llmjudge_score": 0.0,
            "llmjudge_used": 0.0,
        }
    if pred is None:
        abs_error = abs(gt)
        pred_text = ""
    else:
        abs_error = abs(pred - gt)
        pred_text = str(int(pred)) if float(pred).is_integer() else str(pred)
    score = 1.0 if pred is not None and abs_error <= 1e-4 else 0.0
    return {
        "score": score,
        "pred": pred_text,
        "abs_error": abs_error,
        "mae": abs_error,
        "squared_error": abs_error * abs_error,
        "official_like_score": score,
        "llmjudge_score": 0.0,
        "llmjudge_used": 0.0,
    }


def _hrbench_score(prediction: str, ground_truth: str, extra_info: dict | None) -> float:
    score = _choice_score(prediction, ground_truth)
    if score:
        return score

    answer_text = str(prediction or "").lower()
    options = {}
    if extra_info:
        for letter in "ABCD":
            value = extra_info.get(f"option_{letter}")
            if value is not None:
                options[letter] = str(value).lower()
        if not options:
            question = str(extra_info.get("question") or "")
            for letter in "ABCD":
                match = re.search(
                    rf"\({letter}\)\s*(.*?)(?=\n\([A-D]\)|\nAnswer with|$)",
                    question,
                    flags=re.DOTALL,
                )
                if match:
                    options[letter] = match.group(1).strip().lower()

    matched = [letter for letter, value in options.items() if value and value in answer_text]
    if len(matched) == 1 and matched[0] == str(ground_truth).strip().upper():
        return 1.0
    return 0.0


def _official_like_score(answer: str | None, ground_truth: str, data_source: str, extra_info: dict | None) -> float:
    if answer is None:
        answer = ""
    benchmark = _source_benchmark(data_source, extra_info)
    if benchmark == "ocrbench_v2":
        return _ocrbench_v2_metrics(answer or "", ground_truth, extra_info)["score"]
    if benchmark == "chartqa":
        return _chartqa_relaxed_score(answer, ground_truth)
    if benchmark == "ocrbench":
        return _ocrbench_inclusion_score(answer, ground_truth, extra_info)
    if benchmark == "vstar_bench":
        return _choice_score(answer, ground_truth)
    if benchmark == "hrbench":
        return _hrbench_score(answer, ground_truth, extra_info)
    if benchmark in {"cvbench", "spatialmqa", "arxivqa", "ai2d", "mmstar", "sat2", "virgorlsa"}:
        return _multiple_choice_score(answer, ground_truth, extra_info)
    if benchmark in {"mme_realworld", "mme_realworld_lite", "mme_realworld_cn", "mmerealworld", "mmerealworld_lite", "mmerealworld_cn"}:
        return _multiple_choice_score(answer, ground_truth, extra_info)
    if benchmark == "realworldqa":
        target = str(ground_truth or "").strip()
        if target.upper() in {"A", "B", "C", "D"}:
            return _multiple_choice_score(answer, ground_truth, extra_info)
        return 1.0 if str(answer or "").strip().lower().rstrip(".") == target.lower().rstrip(".") else 0.0
    if benchmark in {"docvqa", "infovqa", "infographicvqa"}:
        return _anls_score(answer, ground_truth, extra_info)
    if benchmark in {"mmvet", "mmvet_hard"}:
        # MMVet official scoring is LLM-judge based. Keep the deterministic
        # official-like field at 0.0 and let llmjudge_fallback_score carry the
        # actual score when judge is enabled.
        return 0.0
    if benchmark == "countbench":
        # Current local CountBenchQA-491 follows the common lmms numeric QA
        # protocol and matches the Qwen3-VL report baseline scale.
        return _countbench_score(answer, ground_truth)
    if benchmark in {"pixmo_count", "pixmo_count_lmms", "countqa"}:
        return _numeric_count_metrics(answer, ground_truth)["score"]
    if benchmark == "fsc147":
        return _compute_fsc147_score(answer, ground_truth)["relative_score"]
    return 1.0 if are_answers_equivalent(answer, ground_truth) else 0.0


def _tool_usage_info(solution_str: str) -> dict[str, float | str]:
    return _strict_tool_usage_info(solution_str)


def _extract_ocrbench_v2_answer(solution_str: str) -> str:
    """Keep the final assistant answer raw while excluding intermediate tool I/O."""
    text = str(solution_str or "")
    cleaned = re.sub(r"<tool_response>.*?</tool_response>", "", text, flags=re.DOTALL)
    matches = re.findall(r"<answer>(.*?)</answer>", cleaned, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    parts = cleaned.split("<answer>")
    if len(parts) > 1:
        return parts[-1].strip()
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _load_ocrbench_v2_utils():
    lmms_eval_path = os.environ.get("LMMS_EVAL_PATH", "/mnt/cpfs/delinmao/lmms-eval")
    if "loguru" not in sys.modules:
        logger = types.SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
            debug=lambda *args, **kwargs: None,
        )
        sys.modules["loguru"] = types.SimpleNamespace(logger=logger)
    task_dir = os.path.join(lmms_eval_path, "lmms_eval", "tasks", "ocrbench_v2")
    try:
        if "lmms_eval.tasks.ocrbench_v2.utils" in sys.modules:
            return sys.modules["lmms_eval.tasks.ocrbench_v2.utils"]
        if not os.path.isdir(task_dir):
            raise FileNotFoundError(task_dir)

        for pkg_name, pkg_path in [
            ("lmms_eval", os.path.join(lmms_eval_path, "lmms_eval")),
            ("lmms_eval.tasks", os.path.join(lmms_eval_path, "lmms_eval", "tasks")),
            ("lmms_eval.tasks.ocrbench_v2", task_dir),
            ("lmms_eval.tasks.ocrbench_v2.spotting_eval", os.path.join(task_dir, "spotting_eval")),
        ]:
            module = sys.modules.get(pkg_name)
            if module is None:
                module = types.ModuleType(pkg_name)
                module.__path__ = [pkg_path]
                sys.modules[pkg_name] = module

        file_utils = types.ModuleType("lmms_eval.tasks._task_utils.file_utils")
        file_utils.generate_submission_file = lambda filename, args=None, subpath=None: os.path.join(
            os.getcwd(), filename
        )
        task_utils = types.ModuleType("lmms_eval.tasks._task_utils")
        task_utils.__path__ = []
        sys.modules.setdefault("lmms_eval.tasks._task_utils", task_utils)
        sys.modules.setdefault("lmms_eval.tasks._task_utils.file_utils", file_utils)

        def load_module(module_name: str, relative_path: str):
            if module_name in sys.modules:
                return sys.modules[module_name]
            module_path = os.path.join(task_dir, relative_path)
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load {module_name} from {module_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        load_module("lmms_eval.tasks.ocrbench_v2.parallel", "parallel.py")
        load_module("lmms_eval.tasks.ocrbench_v2.vqa_metric", "vqa_metric.py")
        load_module("lmms_eval.tasks.ocrbench_v2.IoUscore_metric", "IoUscore_metric.py")
        load_module("lmms_eval.tasks.ocrbench_v2.page_ocr_metric", "page_ocr_metric.py")
        load_module(
            "lmms_eval.tasks.ocrbench_v2.spotting_eval.rrc_evaluation_funcs_1_1",
            os.path.join("spotting_eval", "rrc_evaluation_funcs_1_1.py"),
        )
        load_module("lmms_eval.tasks.ocrbench_v2.spotting_eval.script", os.path.join("spotting_eval", "script.py"))
        load_module("lmms_eval.tasks.ocrbench_v2.spotting_metric", "spotting_metric.py")
        load_module("lmms_eval.tasks.ocrbench_v2.TEDS_metric", "TEDS_metric.py")
        return load_module("lmms_eval.tasks.ocrbench_v2.utils", "utils.py")
    except Exception as exc:
        raise RuntimeError(
            "Failed to import lmms-eval OCRBench v2 scorer. "
            "Set LMMS_EVAL_PATH to a checkout containing lmms_eval/tasks/ocrbench_v2. "
            f"Current LMMS_EVAL_PATH={lmms_eval_path!r}. Original error: {exc}"
        ) from exc
    return ocrbench_v2_utils


def _ocrbench_v2_doc(ground_truth: str, extra_info: dict | None) -> dict:
    extra_info = extra_info or {}
    def plain(value):
        try:
            if hasattr(value, "tolist"):
                return plain(value.tolist())
        except Exception:
            pass
        if isinstance(value, tuple):
            return [plain(v) for v in value]
        if isinstance(value, list):
            return [plain(v) for v in value]
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        return value

    answers = extra_info.get("answers")
    if answers is None:
        answers = [ground_truth]
    answers = plain(answers)
    return {
        "id": extra_info.get("ocrbench_v2_id", extra_info.get("id", extra_info.get("index"))),
        "dataset_name": extra_info.get("dataset_name"),
        "question": extra_info.get("question", ""),
        "type": extra_info.get("type", ""),
        "answers": answers,
        "eval": extra_info.get("eval"),
        "bbox": plain(extra_info.get("bbox")),
        "bbox_list": plain(extra_info.get("bbox_list")),
        "content": plain(extra_info.get("content")),
    }


def _ocrbench_v2_metrics(prediction: str, ground_truth: str, extra_info: dict | None) -> dict[str, float | str]:
    utils = _load_ocrbench_v2_utils()
    doc = _ocrbench_v2_doc(ground_truth, extra_info)
    payloads = utils.ocrbench_v2_process_results(doc, [prediction])
    payload = payloads["ocrbench_v2_accuracy"]
    score = float(payload.get("score", 0.0) or 0.0)
    question_type = str(payload.get("question_type") or doc.get("type") or "")
    lang = "cn" if question_type.endswith(" cn") else "en"
    return {
        "score": score,
        "official_like_score": score,
        "llmjudge_score": 0.0,
        "llmjudge_used": 0.0,
        "llmjudge_fallback_score": score,
        "ocrbench_v2_score": score,
        "ocrbench_v2_question_type": question_type,
        "ocrbench_v2_language": lang,
        "ocrbench_v2_payload": json.dumps(payload, ensure_ascii=False),
    }


def are_answers_equivalent(
    prediction: str, ground_truth: str, tolerance: float = 1e-4
) -> bool:
    if not isinstance(prediction, str) or not isinstance(ground_truth, str):
        return False

    if prediction is None or ground_truth is None:
        return False
    try:
        if float(prediction) == float(ground_truth):
            return True
    except:
        if prediction == ground_truth:
            return True
    extracted_pred = verify_answer_rule(str(prediction), str(ground_truth))
    if extracted_pred is None:
        return False
    gt_str = str(ground_truth).strip()
    if extracted_pred.upper() == gt_str.upper():
        return True
    try:
        if abs(float(Fraction(extracted_pred)) - float(Fraction(gt_str))) <= tolerance:
            return True
    except:
        pass
    return False


def compute_score(
    solution_str, ground_truth, data_source, extra_info
) -> dict[str, float | str]:
    answer = extract_answer(solution_str)
    toolvision_result = compute_toolvision_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        extracted_answer=answer,
    )
    if toolvision_result is not None:
        return toolvision_result

    benchmark = _source_benchmark(data_source, extra_info)

    if benchmark == "ocrbench_v2":
        raw_answer = _extract_ocrbench_v2_answer(solution_str)
        result = _ocrbench_v2_metrics(raw_answer, ground_truth, extra_info)
        result["final_answer"] = raw_answer
        result.update(_tool_usage_info(solution_str))
        return result

    if benchmark in {"pixmo_count", "pixmo_count_lmms", "countqa"}:
        result = _numeric_count_metrics(answer or "", ground_truth)
        result["final_answer"] = answer or ""
        result.update(_tool_usage_info(solution_str))
        return result

    official_score = _official_like_score(answer, ground_truth, data_source, extra_info)

    if _is_fsc147_source(data_source):
        result = _compute_fsc147_score(answer or "", ground_truth)
        result["final_answer"] = answer or ""
        result.update(_tool_usage_info(solution_str))
        return result

    result = {
        "score": official_score,
        "official_like_score": official_score,
        "llmjudge_score": 0.0,
        "llmjudge_used": 0.0,
        "llmjudge_fallback_score": official_score,
        "final_answer": answer or "",
    }
    result.update(_tool_usage_info(solution_str))

    if answer is None or official_score >= 1.0:
        return result

    # Optional LLM-as-judge (requires LLM_JUDGE_BASE_URL). If unset, skip to avoid
    # spamming ERROR logs from llmjudge.verify() ("Client not initialized") on
    # every rule-mismatch sample while still scoring 0.0.
    judge_client = LLM_JUDGE_CLIENT
    if judge_client.client is None:
        return result

    acc_score, reason = judge_client.verify(
        answer=answer,
        ground_truth=ground_truth,
        question=extra_info["question"],
        task=data_source,
    )
    result["llmjudge_score"] = acc_score
    result["llmjudge_used"] = 1.0
    result["llmjudge_fallback_score"] = max(official_score, acc_score)
    return result
