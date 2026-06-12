#!/usr/bin/env python3
"""Rescore pass@16 parquet generations with ToolVision scorer v2.

Input parquet is expected to contain one row per question and a
`pred_texts_json` column containing a JSON list of raw generations.
The script writes a row-level parquet plus a by-source CSV summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from recipe.codevision.rewards import compute_toolvision_score


def parse_jsonish(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    if isinstance(value, (list, tuple, dict)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def row_extra_info(row: pd.Series) -> dict[str, Any]:
    extra = parse_jsonish(row.get("extra_info"), {})
    if not isinstance(extra, dict):
        extra = {}
    metadata = parse_jsonish(row.get("metadata_json"), {})
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            extra.setdefault(key, value)
    for key in [
        "source",
        "source_dataset",
        "source_benchmark",
        "answer_type",
        "reward_family",
        "question",
        "answers",
        "acceptable_answers",
        "answer_aliases",
        "options",
        "choices",
        "relative_count_threshold",
    ]:
        if key in row and not _is_missing(row.get(key)) and key not in extra:
            extra[key] = row.get(key)
    if "question" not in extra and "problem" in row and not _is_missing(row.get("problem")):
        extra["question"] = row.get("problem")
    source = str(
        extra.get("source_dataset")
        or extra.get("source")
        or row.get("source_dataset")
        or row.get("source")
        or row.get("data_source")
        or ""
    )
    if source and "source_dataset" not in extra:
        extra["source_dataset"] = source
    return extra


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def ground_truth(row: pd.Series) -> Any:
    for key in ["ground_truth", "answer", "answer_json", "answers", "gt", "label"]:
        if key in row and not _is_missing(row.get(key)):
            value = row.get(key)
            if key == "answer_json":
                return parse_jsonish(value, value)
            return value
    reward_model = parse_jsonish(row.get("reward_model"), {})
    if isinstance(reward_model, dict):
        return reward_model.get("ground_truth", "")
    return ""


def source_name(row: pd.Series, extra: dict[str, Any]) -> str:
    return str(
        extra.get("source_dataset")
        or extra.get("source")
        or extra.get("source_benchmark")
        or row.get("source_dataset")
        or row.get("source")
        or row.get("data_source")
        or "unknown"
    )


def score_prediction(prediction: str, row: pd.Series) -> tuple[float, str]:
    extra = row_extra_info(row)
    source = source_name(row, extra)
    result = compute_toolvision_score(
        data_source=source,
        solution_str=str(prediction or ""),
        ground_truth=ground_truth(row),
        extra_info=extra,
    )
    if result is None:
        return 0.0, "unknown"
    return float(result.get("score", 0.0)), str(result.get("reward_family", "unknown"))


def primary_success(score: float, family: str, source: str) -> bool:
    family = family.lower()
    source = source.lower()
    if family == "vqa_soft" or source.startswith("textvqa"):
        return score == 1.0
    if family == "ocr_levenshtein" or source.startswith(("docvqa", "infographicvqa")):
        return score >= 0.5
    if family == "fsc147_relative" or source.startswith("fsc147"):
        return score >= 0.9
    if family == "bbox_iou" or source.startswith(("refl4", "ref_l4")):
        return score >= 0.5
    return score == 1.0


def bucket(count: int, total: int) -> str:
    if count <= 0:
        return "0"
    if count >= total:
        return str(total)
    return f"{count}/{total}"


def rescore_frame(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        preds = parse_jsonish(row.get("pred_texts_json"), [])
        if not isinstance(preds, list):
            preds = []
        extra = row_extra_info(row)
        source = source_name(row, extra)
        scores = []
        families = []
        for pred in preds:
            score, family = score_prediction(str(pred or ""), row)
            scores.append(score)
            families.append(family)
        family = next((item for item in families if item and item != "unknown"), "unknown")
        strict = [score == 1.0 for score in scores]
        lenient = [score > 0.0 for score in scores]
        primary = [primary_success(score, family, source) for score in scores]
        record = row.to_dict()
        record.update(
            {
                "score_raw_16": json.dumps(scores, ensure_ascii=False),
                "success_strict_16": json.dumps(strict),
                "success_lenient_16": json.dumps(lenient),
                "success_primary_16": json.dumps(primary),
                "correct_count_strict": int(sum(strict)),
                "correct_count_lenient": int(sum(lenient)),
                "correct_count_primary": int(sum(primary)),
                "mean_score_v2": float(sum(scores) / len(scores)) if scores else 0.0,
                "bucket_v2": bucket(int(sum(primary)), len(scores) or 16),
                "metric_family_v2": family,
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def write_summary(df: pd.DataFrame, path: Path) -> None:
    if "source_dataset" in df.columns:
        source_col = "source_dataset"
    elif "source" in df.columns:
        source_col = "source"
    else:
        source_col = "data_source"
    if source_col not in df.columns:
        df = df.assign(source_dataset="unknown")
        source_col = "source_dataset"
    rows = []
    for source, group in df.groupby(source_col, dropna=False):
        rows.append(
            {
                "source_dataset": source,
                "samples": len(group),
                "mean_score_v2": group["mean_score_v2"].mean(),
                "mean_correct_primary": group["correct_count_primary"].mean(),
                "zero_primary_rate": (group["correct_count_primary"] == 0).mean(),
                "full_primary_rate": (group["correct_count_primary"] >= 16).mean(),
            }
        )
    pd.DataFrame(rows).sort_values("source_dataset").to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    scored = rescore_frame(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(args.output, index=False)
    write_summary(scored, args.summary)
    print(f"Wrote rescored parquet: {args.output}")
    print(f"Wrote summary csv: {args.summary}")


if __name__ == "__main__":
    main()
