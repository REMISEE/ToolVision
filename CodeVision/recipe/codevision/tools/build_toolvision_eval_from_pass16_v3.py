#!/usr/bin/env python3
"""Build ToolVision eval parquets from final pass@16 candidate pools."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


DEFAULT_INPUT = Path("/mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun_all/final_v3/all_valid_pass16_v3.parquet")
DEFAULT_OUTPUT_DIR = Path("/mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun_all/final_v3/toolvision_eval")


def parse_jsonish(value: Any, default: Any) -> Any:
    if value is None:
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


def strip_image_token(text: Any) -> str:
    value = str(text or "").strip()
    if value.startswith("<image>"):
        value = value[len("<image>") :].strip()
    return value


def answers_from_row(row: pd.Series) -> list[str]:
    answers = parse_jsonish(row.get("answer_json"), [])
    if not isinstance(answers, list):
        answers = [answers]
    return [str(answer) for answer in answers if str(answer).strip()]


def image_path_from_row(row: pd.Series) -> str:
    refs = parse_jsonish(row.get("image_refs_json"), [])
    if isinstance(refs, list):
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            path = str(ref.get("path") or "").strip()
            if path:
                return path
            uri = str(ref.get("uri") or "").strip()
            if uri.startswith("file://"):
                return uri[7:]
    return ""


@lru_cache(maxsize=None)
def image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_text(value: Any, default: Any) -> str:
    return json.dumps(parse_jsonish(value, default), ensure_ascii=False)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    return bool(value)


def reward_family(row: pd.Series) -> str:
    family = str(row.get("metric_family_v2") or "").strip()
    if family:
        return family
    source = str(row.get("source_view") or row.get("source") or "").lower()
    if source.startswith("ocrbench"):
        return "ocr_inclusion"
    if source in {"docvqa", "infographicvqa"}:
        return "ocr_levenshtein"
    if source in {"fsc147", "pixmo_count"}:
        return "fsc147_relative"
    if source in {"ai2d", "arxivqa", "mmstar", "sat2", "virgorlsa"}:
        return "multiple_choice"
    if source == "chartqa":
        return "chartqa_relaxed"
    if source == "textvqa":
        return "vqa_soft"
    if source == "refl4":
        return "bbox_iou"
    return "exact"


def build_eval_rows(
    df: pd.DataFrame, *, read_image_size: bool = False, check_images: bool = False
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    missing_images: list[str] = []
    for _, row in df.iterrows():
        source = str(row.get("source_view") or row.get("source") or "unknown")
        question = strip_image_token(row.get("problem"))
        answers = answers_from_row(row)
        ground_truth = answers[0] if answers else ""
        image_path = image_path_from_row(row)
        if not image_path or (check_images and not Path(image_path).exists()):
            missing_images.append(str(row.get("global_pool_id") or row.get("id")))
            continue
        width, height = 0, 0
        if read_image_size:
            width, height = image_size(image_path)
        metadata = parse_jsonish(row.get("metadata_json"), {})
        if not isinstance(metadata, dict):
            metadata = {}
        source_original = str(row.get("source") or source)
        correct_count = int(row.get("pass16_correct_count", row.get("correct_count_primary", 0)))
        extra_info = {
            "acceptable_answers": answers,
            "answers": answers,
            "answer_type": str(row.get("answer_type") or ""),
                "image_height": int(height),
                "image_width": int(width),
            "index": int(row.get("source_index", -1)),
            "origin": "toolvision_pass16_v3",
            "pool_origin": str(row.get("pool_origin") or ""),
            "problem_type": str(row.get("problem_type") or ""),
            "prompt_type": str(row.get("prompt_type") or ""),
            "qa_hash": sha256_text(source + "\n" + str(row.get("id")) + "\n" + question + "\n" + ground_truth),
            "question": question,
            "question_hash": sha256_text(question),
            "raw_file": str(row.get("raw_file") or ""),
            "raw_problem": str(row.get("problem") or ""),
            "raw_row": int(row.get("raw_row", -1)),
            "reward_family": reward_family(row),
            "source_benchmark": source_original,
            "source_dataset": source,
            "source_original": source_original,
            "source_original_id": str(row.get("id") or ""),
            "source_raw": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            "uid": str(row.get("global_pool_id") or row.get("id") or ""),
            "pass16_correct_count": correct_count,
            "pass16_rollout_n": int(row.get("rollout_n", 16)),
            "pass16_bucket": str(row.get("pass16_bucket") or f"{correct_count}/16"),
            "pass16_pred_texts_json": json_text(row.get("pred_texts_json"), []),
            "pass16_score_raw_16_json": json_text(row.get("score_raw_16"), []),
            "pass16_success_primary_16_json": json_text(row.get("success_primary_16"), []),
            "pass16_success_strict_16_json": json_text(row.get("success_strict_16"), []),
            "pass16_success_lenient_16_json": json_text(row.get("success_lenient_16"), []),
            "pass16_mean_score": float(row.get("mean_score_v2", 0.0) or 0.0),
            "mut_candidate_hard_0_4": as_bool(row.get("mut_candidate_hard_0_4", correct_count <= 4)),
            "mut_candidate_medium_0_8": as_bool(row.get("mut_candidate_medium_0_8", correct_count <= 8)),
            "rl_mixed_1_15": as_bool(row.get("rl_mixed_1_15", 1 <= correct_count <= 15)),
            "first8_any_correct": as_bool(row.get("first8_any_correct", False)),
        }
        if "choices" in metadata:
            extra_info["choices"] = metadata["choices"]
        elif "options" in metadata:
            extra_info["options"] = metadata["options"]
        if source in {"fsc147", "pixmo_count"}:
            extra_info["relative_count_threshold"] = 0.9

        if width > 0 and height > 0:
            content = f"<image>Image size = {width}x{height} pixels.\n\n{question}"
        else:
            content = f"<image>{question}"

        rows.append(
            {
                "data_source": source,
                "agent_name": "tool_agent",
                "ability": "math_qa" if source in {"fsc147", "pixmo_count"} else "mm_qa",
                "prompt": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "images": [{"image": "file://" + image_path}],
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": extra_info,
            }
        )

    if missing_images:
        raise RuntimeError(f"Missing {len(missing_images)} images, examples={missing_images[:5]}")
    return pd.DataFrame(rows)


def write_split(
    df: pd.DataFrame,
    mask: pd.Series,
    path: Path,
    *,
    read_image_size: bool = False,
    check_images: bool = False,
) -> dict[str, Any]:
    subset = df[mask].copy().reset_index(drop=True)
    eval_df = build_eval_rows(subset, read_image_size=read_image_size, check_images=check_images)
    path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_parquet(path, index=False)
    source_counts = eval_df["data_source"].value_counts().sort_index().to_dict()
    return {"path": str(path), "rows": int(len(eval_df)), "source_counts": source_counts}


def write_source_shards(eval_path: Path, split_name: str, output_dir: Path) -> dict[str, Any]:
    eval_df = pd.read_parquet(eval_path)
    shard_dir = output_dir / "by_source" / split_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = {}
    for source, group in eval_df.groupby("data_source", sort=True):
        safe_source = str(source).replace("/", "_")
        shard_path = shard_dir / f"{safe_source}.parquet"
        group.reset_index(drop=True).to_parquet(shard_path, index=False)
        shards[str(source)] = {"path": str(shard_path), "rows": int(len(group))}
    return shards


def write_source_summary(df: pd.DataFrame, output_dir: Path) -> Path:
    cc = df["pass16_correct_count"].astype(int)
    summary = (
        df.assign(
            mut_0_8=cc.between(0, 8),
            regular_9_15=cc.between(9, 15),
            easy_16=cc.eq(16),
        )
        .groupby("source_view", dropna=False)
        .agg(
            rows=("source_view", "size"),
            mut_0_8=("mut_0_8", "sum"),
            regular_9_15=("regular_9_15", "sum"),
            easy_16=("easy_16", "sum"),
            mean_pass16_correct=("pass16_correct_count", "mean"),
        )
        .reset_index()
        .sort_values("source_view")
    )
    path = output_dir / "source_split_summary.csv"
    summary.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-easy", action="store_true", help="Also write the 16/16 easy split.")
    parser.add_argument(
        "--read-image-size",
        action="store_true",
        help="Open each image and include exact WxH in the prompt. Slower on CPFS.",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Stat every image path before writing. Slower on CPFS; pass16 image paths are assumed valid by default.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    if "pass16_correct_count" not in df.columns:
        df["pass16_correct_count"] = df["correct_count_primary"].astype(int)

    cc = df["pass16_correct_count"].astype(int)
    split_specs = {
        "mut_0_8": (cc.between(0, 8), args.output_dir / "mut_candidates_0_8_toolvision_eval.parquet"),
        "regular_9_15": (cc.between(9, 15), args.output_dir / "regular_9_15_toolvision_eval.parquet"),
    }
    if args.include_easy:
        split_specs["easy_16"] = (cc.eq(16), args.output_dir / "easy_16_toolvision_eval.parquet")
    outputs = {}
    for split_name, (mask, path) in split_specs.items():
        outputs[split_name] = write_split(
            df,
            mask,
            path,
            read_image_size=args.read_image_size,
            check_images=args.check_images,
        )
        outputs[split_name]["source_shards"] = write_source_shards(path, split_name, args.output_dir)
    summary_path = write_source_summary(df, args.output_dir)
    manifest = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "source_split_summary": str(summary_path),
        "splits": outputs,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
