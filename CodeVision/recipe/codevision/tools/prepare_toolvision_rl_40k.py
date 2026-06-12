"""Prepare ToolVision/Innovator RL data for CodeVision training.

This script is intentionally non-destructive.  It reads raw Innovator-style
parquet files, writes audit reports/manifests under a separate output root, and
does not mutate or delete source data.

Stages:

```
inventory  raw source/type counts only
dedup      inventory + normalized candidate manifest + hard-dedup reports
sample     dedup + deterministic 40k target sample manifest
convert    convert sampled manifest into CodeVision parquet + image cache
```

The conversion stage consumes the sampled manifest and writes a small parquet
with file URI image references.  Images are materialized into a deterministic
cache under the output root.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


DEFAULT_NEW_ROOT = Path("/mnt/cpfs/delinmao/data/toolvision_innovator_scaled")
DEFAULT_CONTROL_ROOT = Path("/mnt/cpfs/delinmao/data/Innovator-VL-RL-172K")
DEFAULT_OUTPUT_ROOT = Path("/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k")
DEFAULT_SEED = 20260521
FINAL_TARGET_ROWS = 40000

NEW_TARGETS = {
    "chartqa": 4790,
    "refl4": 3000,
    "virgorlsa": 3000,
    "pixmo_count": 4000,
    "sat2": 2500,
    "arxivqa": 2500,
    "ocrbench": 1269,
    "docvqa": 1244,
    "infographicvqa": 1001,
    "ai2d": 200,
    "countqa": 170,
    "mmstar": 326,
}

CONTROL_TARGETS = {
    "virl39k": 5000,
    "WaltonFuture": 3000,
    "thinklite_vl_hard": 2000,
    "tqa": 2000,
    "mmk12": 1500,
    "wemath_standard": 1500,
    "puzzlevqa": 1000,
}

SOURCE_FORCE_FAMILY = {
    "chartqa": "chartqa_relaxed",
    "ocrbench": "ocr_inclusion",
    "docvqa": "ocr_levenshtein",
    "infographicvqa": "ocr_levenshtein",
    "textvqa": "vqa_soft",
    "fsc147": "fsc147_relative",
    "pixmo_count": "fsc147_relative",
    "pixmo_count_lmms": "fsc147_relative",
}

SOURCE_FALLBACK_FAMILY = {
    "pixmo_count": "fsc147_relative",
    "countqa": "numeric_exact",
    "mmstar": "multiple_choice",
    "arxivqa": "multiple_choice",
    "ai2d": "multiple_choice",
    "docvqa": "short_text",
    "infographicvqa": "short_text",
    "refl4": "bbox_iou",
    "ref_l4": "bbox_iou",
    "sat2": "exact",
    "virgorlsa": "multiple_choice",
}

ANSWER_TYPE_TO_FAMILY = {
    "number": "numeric_exact",
    "numeric": "numeric_exact",
    "integer": "numeric_exact",
    "float": "numeric_exact",
    "multiple_choice": "multiple_choice",
    "multiple-choice": "multiple_choice",
    "choice": "multiple_choice",
    "mcq": "multiple_choice",
    "math_expressions": "math_verify",
    "math-expressions": "math_verify",
    "boolean": "boolean",
    "bool": "boolean",
    "ocrtext": "ocr_levenshtein",
    "ocr_text": "ocr_levenshtein",
    "short_text": "short_text",
    "string": "exact",
    "any": "exact",
    "bbox": "bbox_iou",
    "box": "bbox_iou",
    "judge": "judge",
    "html_code": "html_code",
    "html-code": "html_code",
    "svg_code": "svg_code",
    "svg-code": "svg_code",
    "general_code": "general_code",
    "general-code": "general_code",
    "critic": "judge",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW_ROOT)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage", choices=["inventory", "dedup", "sample", "convert"], default="sample")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--manifest", type=Path, default=None, help="Sample manifest for convert stage.")
    parser.add_argument("--converted-out", type=Path, default=None, help="Output CodeVision train parquet path.")
    parser.add_argument("--image-cache", type=Path, default=None, help="Output image cache directory.")
    parser.add_argument(
        "--exclude-csv",
        type=Path,
        default=None,
        help="CSV containing raw_file/raw_row pairs to exclude before sampling, e.g. conversion_failures.csv.",
    )
    parser.add_argument(
        "--hash-mode",
        choices=["fast", "content"],
        default="fast",
        help="For path images, 'fast' hashes the path string; 'content' hashes file bytes. Byte images are always hashed by bytes.",
    )
    parser.add_argument(
        "--max-duplicate-report-rows",
        type=int,
        default=20000,
        help="Cap row-level duplicate/conflict CSVs so reports remain manageable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = args.output_root.expanduser().resolve()
    report_dir = out_root / "reports"
    manifest_dir = out_root / "manifests"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] stage={args.stage}")
    print(f"[info] new_root={args.new_root}")
    print(f"[info] control_root={args.control_root}")
    print(f"[info] output_root={out_root}")
    print("[info] source data is read-only; outputs are written only under output_root")

    if args.stage == "convert":
        manifest_path = args.manifest or manifest_dir / "sampled_40k_manifest.parquet"
        converted_out = args.converted_out or out_root / "parquet" / "train.parquet"
        image_cache = args.image_cache or out_root / "images_raw"
        convert_sample_manifest(manifest_path, converted_out, image_cache, report_dir)
        return 0

    raw = load_raw_records(args.new_root, args.control_root, args.hash_mode)
    if raw.empty:
        print("[error] no raw rows loaded", file=sys.stderr)
        return 1

    write_inventory_reports(raw, report_dir)
    if args.stage == "inventory":
        print(f"[ok] wrote inventory reports -> {report_dir}")
        return 0

    candidates = build_candidate_manifest(raw)
    if args.exclude_csv:
        candidates = apply_manual_exclusions(candidates, args.exclude_csv)
    candidates_path = manifest_dir / "candidate_manifest.parquet"
    candidates.to_parquet(candidates_path, engine="pyarrow", index=False)
    write_dedup_reports(candidates, report_dir, args.max_duplicate_report_rows)
    if args.stage == "dedup":
        print(f"[ok] wrote candidate manifest -> {candidates_path}")
        print(f"[ok] wrote dedup reports -> {report_dir}")
        return 0

    sampled = sample_targets(candidates, args.seed)
    sampled_path = manifest_dir / "sampled_40k_manifest.parquet"
    sampled.to_parquet(sampled_path, engine="pyarrow", index=False)
    write_sample_reports(candidates, sampled, report_dir)

    print(f"[ok] wrote candidate manifest -> {candidates_path}")
    print(f"[ok] wrote sampled manifest -> {sampled_path}")
    print(f"[ok] wrote reports -> {report_dir}")
    print(f"[ok] sampled rows={len(sampled)}")
    return 0


def load_raw_records(new_root: Path, control_root: Path, hash_mode: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in iter_new_parquets(new_root):
        records.extend(load_parquet_records(path, origin="new", hash_mode=hash_mode, source_from_path=path.parent.name))
    for path in iter_control_parquets(control_root):
        records.extend(load_parquet_records(path, origin="control", hash_mode=hash_mode, source_from_path=""))
    return pd.DataFrame(records)


def iter_new_parquets(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("*/*.parquet"))


def iter_control_parquets(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("RL_part*.parquet"))


def load_parquet_records(path: Path, *, origin: str, hash_mode: str, source_from_path: str) -> list[dict[str, Any]]:
    print(f"[read] {origin} {path}")
    df = pd.read_parquet(path)
    records: list[dict[str, Any]] = []
    for row_idx, row in df.iterrows():
        raw_source = scalar(row.get("source", ""))
        source_dataset = normalize_source(raw_source, source_from_path=source_from_path, origin=origin)
        answer_type = normalize_key(row.get("answer_type", ""))
        problem_type = scalar(row.get("problem_type", ""))
        prompt_type = scalar(row.get("prompt_type", ""))
        source_original_id = scalar(row.get("id", ""))
        raw_problem = scalar(row.get("problem", ""))
        question = clean_question(raw_problem)
        normalized_question = normalize_question(question)

        answers = normalize_answers(row.get("answer"))
        canonical_answer = canonicalize_answer(answers, answer_type)
        image_meta = image_identity(row.get("images"), hash_mode=hash_mode)
        reward_family = reward_family_for(source_dataset, answer_type)
        target_group = classify_target_group(origin, source_dataset)

        records.append(
            {
                "origin": origin,
                "raw_file": str(path),
                "raw_row": int(row_idx),
                "source_raw": raw_source,
                "source_from_path": source_from_path,
                "source_dataset": source_dataset,
                "target_group": target_group,
                "is_target_source": bool(target_group),
                "source_original_id": source_original_id,
                "problem_type": problem_type,
                "answer_type": scalar(row.get("answer_type", "")),
                "answer_type_norm": answer_type,
                "prompt_type": prompt_type,
                "reward_family": reward_family,
                "raw_problem": raw_problem,
                "question": question,
                "normalized_question": normalized_question,
                "question_hash": stable_hash(normalized_question),
                "answer_json": json.dumps(answers, ensure_ascii=False),
                "canonical_answer": canonical_answer,
                "qa_hash": stable_hash(normalized_question + "\n" + canonical_answer),
                "image_count": image_meta["image_count"],
                "image_hash": image_meta["image_hash"],
                "image_ref_kind": image_meta["image_ref_kind"],
                "image_ref": image_meta["image_ref"],
                "image_error": image_meta["image_error"],
                "valid_single_image": image_meta["image_count"] == 1 and not image_meta["image_error"],
            }
        )
    return records


def convert_sample_manifest(manifest_path: Path, converted_out: Path, image_cache: Path, report_dir: Path) -> None:
    manifest_path = manifest_path.expanduser().resolve()
    converted_out = converted_out.expanduser().resolve()
    image_cache = image_cache.expanduser().resolve()
    converted_out.parent.mkdir(parents=True, exist_ok=True)
    image_cache.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"sample manifest not found: {manifest_path}")

    manifest = pd.read_parquet(manifest_path)
    rows: list[dict[str, Any]] = []
    conversion_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for raw_file, group in manifest.sort_values(["raw_file", "raw_row"]).groupby("raw_file", sort=True):
        print(f"[read-image-source] {raw_file} rows={len(group)}")
        raw_df = pd.read_parquet(str(raw_file))
        for _, rec in group.iterrows():
            try:
                raw_row = int(rec["raw_row"])
                raw = raw_df.iloc[raw_row].to_dict()
                image_path, width, height, wrote_image = materialize_image(raw.get("images"), rec, image_cache)
                rows.append(build_codevision_row(rec, image_path=image_path, width=width, height=height))
                conversion_records.append(
                    {
                        "sample_index": int(rec["sample_index"]),
                        "source_dataset": rec["source_dataset"],
                        "image_path": str(image_path),
                        "width": width,
                        "height": height,
                        "wrote_image": wrote_image,
                        "status": "ok",
                    }
                )
            except Exception as exc:
                failure = {
                    "sample_index": int(rec.get("sample_index", -1)),
                    "source_dataset": rec.get("source_dataset", ""),
                    "raw_file": rec.get("raw_file", ""),
                    "raw_row": rec.get("raw_row", ""),
                    "error": repr(exc),
                }
                failures.append(failure)
                conversion_records.append({**failure, "status": "failed"})

    pd.DataFrame(conversion_records).to_csv(report_dir / "conversion_report.csv", index=False)
    failure_path = report_dir / "conversion_failures.csv"
    if failures:
        pd.DataFrame(failures).to_csv(failure_path, index=False)
        raise RuntimeError(f"conversion failed for {len(failures)} rows; see {failure_path}")
    if failure_path.exists():
        failure_path.unlink()

    rows.sort(key=lambda row: row["extra_info"]["index"])
    df = pd.DataFrame(rows)
    df.to_parquet(converted_out, engine="pyarrow", index=False)
    df.groupby(["data_source"], dropna=False).size().reset_index(name="rows").sort_values("data_source").to_csv(
        report_dir / "converted_by_source.csv", index=False
    )
    df["reward_family"] = df["extra_info"].map(lambda x: (x or {}).get("reward_family", ""))
    df.groupby(["reward_family"], dropna=False).size().reset_index(name="rows").sort_values("reward_family").to_csv(
        report_dir / "converted_by_reward_family.csv", index=False
    )
    print(f"[ok] wrote CodeVision train parquet -> {converted_out}")
    print(f"[ok] wrote image cache -> {image_cache}")
    print(f"[ok] converted rows={len(df)}")


def materialize_image(images_value: Any, rec: pd.Series, image_cache: Path) -> tuple[Path, int, int, bool]:
    images = to_builtin(images_value)
    if isinstance(images, dict):
        images = [images]
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise ValueError("expected exactly one image dict")

    image_entry = images[0]
    source = str(rec["source_dataset"])
    image_hash = str(rec["image_hash"])
    out_dir = image_cache / source
    out_dir.mkdir(parents=True, exist_ok=True)

    image_bytes = image_entry.get("bytes")
    image_path = image_entry.get("path") or image_entry.get("image")
    if image_bytes:
        if isinstance(image_bytes, memoryview):
            image_bytes = image_bytes.tobytes()
        image_bytes = bytes(image_bytes)
        with Image.open(BytesIO(image_bytes)) as img:
            if img.width <= 0 or img.height <= 0:
                raise ValueError(f"bad image size: {img.size}")
            width, height = img.size
            ext = extension_for_pil_format(img.format)
        out_path = out_dir / f"{image_hash}{ext}"
        if out_path.is_file():
            return out_path, width, height, False
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_path.write_bytes(image_bytes)
        tmp_path.replace(out_path)
        return out_path, width, height, True
    elif image_path:
        path = Path(strip_file_uri(str(image_path)))
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as img:
            if img.width <= 0 or img.height <= 0:
                raise ValueError(f"bad image size: {img.size}")
            width, height = img.size
            ext = normalized_image_suffix(path, img.format)
        out_path = out_dir / f"{image_hash}{ext}"
        if out_path.is_file():
            return out_path, width, height, False
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        shutil.copyfile(path, tmp_path)
        tmp_path.replace(out_path)
        return out_path, width, height, True
    else:
        raise ValueError("image entry has neither bytes nor path/image")

def extension_for_pil_format(fmt: str | None) -> str:
    mapping = {
        "JPEG": ".jpg",
        "JPG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "BMP": ".bmp",
        "GIF": ".gif",
        "TIFF": ".tiff",
    }
    return mapping.get(str(fmt or "").upper(), ".png")


def normalized_image_suffix(path: Path, fmt: str | None) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    return extension_for_pil_format(fmt)


def build_codevision_row(rec: pd.Series, *, image_path: Path, width: int, height: int) -> dict[str, Any]:
    question = str(rec["question"]).strip()
    source_dataset = str(rec["source_dataset"])
    answer, acceptable_answers = converted_answers(rec)
    choices = extract_choices(question)
    extra_info = {
        "index": int(rec["sample_index"]),
        "uid": str(rec.get("sample_uid") or f"{source_dataset}::{rec['source_original_id']}"),
        "source_dataset": source_dataset,
        "source_raw": str(rec.get("source_raw", "")),
        "source_original_id": str(rec.get("source_original_id", "")),
        "origin": str(rec.get("origin", "")),
        "sample_bucket": str(rec.get("sample_bucket", "")),
        "question": question,
        "raw_problem": str(rec.get("raw_problem", "")),
        "problem_type": str(rec.get("problem_type", "")),
        "answer_type": str(rec.get("answer_type", "")),
        "answer_type_norm": str(rec.get("answer_type_norm", "")),
        "prompt_type": str(rec.get("prompt_type", "")),
        "reward_family": str(rec.get("reward_family", "")),
        "acceptable_answers": acceptable_answers,
        "image_hash": str(rec.get("image_hash", "")),
        "question_hash": str(rec.get("question_hash", "")),
        "qa_hash": str(rec.get("qa_hash", "")),
        "raw_file": str(rec.get("raw_file", "")),
        "raw_row": int(rec.get("raw_row", -1)),
        "image_width": int(width),
        "image_height": int(height),
    }
    if choices:
        extra_info["choices"] = choices

    return {
        "data_source": source_dataset,
        "agent_name": "tool_agent",
        "ability": ability_for(str(rec.get("reward_family", ""))),
        "prompt": [{"role": "user", "content": f"<image>Image size = {width}x{height} pixels.\n\n{question}"}],
        "images": [{"image": f"file://{image_path}"}],
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": extra_info,
    }


def ability_for(reward_family: str) -> str:
    if reward_family == "bbox_iou":
        return "grounding"
    if reward_family in {"math_verify", "numeric_exact"}:
        return "math_qa"
    if reward_family in {"html_code", "svg_code", "general_code"}:
        return "code"
    return "mm_qa"


def converted_answers(rec: pd.Series) -> tuple[str, list[str]]:
    reward_family = str(rec.get("reward_family", ""))
    answer = str(rec["canonical_answer"])
    acceptable_answers = parse_json_list(rec.get("answer_json", "[]"))
    if reward_family == "bbox_iou":
        bbox = extract_bbox(acceptable_answers)
        if bbox is not None:
            answer = json.dumps(bbox)
            acceptable_answers = [answer]
    return answer, acceptable_answers


def parse_json_list(text: Any) -> list[str]:
    try:
        parsed = json.loads(str(text or "[]"))
    except Exception:
        return normalize_answers(text)
    return [str(item) for item in normalize_answers(parsed)]


def extract_choices(question: str) -> list[str]:
    choices: dict[str, str] = {}
    for match in re.finditer(r"(?:^|\n)\s*([A-H])[\.\)]\s*([^\n]+)", question):
        choices[match.group(1)] = match.group(2).strip()
    if not choices:
        inline = re.findall(r"\b([A-H])[\.\)]\s*([^A-H\n]+?)(?=\s+[A-H][\.\)]|\s*$)", question)
        for letter, text in inline:
            choices[letter] = text.strip()
    if not choices:
        return []
    return [choices[key] for key in sorted(choices)]


def build_candidate_manifest(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.insert(0, "candidate_row_id", range(len(df)))
    df["filter_reason"] = ""
    df.loc[~df["is_target_source"], "filter_reason"] = "not_in_target_mix"
    df.loc[df["filter_reason"].eq("") & ~df["valid_single_image"], "filter_reason"] = "invalid_or_multi_image"
    df.loc[df["filter_reason"].eq("") & df["question"].eq(""), "filter_reason"] = "empty_question"
    df.loc[df["filter_reason"].eq("") & df["canonical_answer"].eq(""), "filter_reason"] = "empty_answer"
    df.loc[df["filter_reason"].eq("") & df["reward_family"].eq("unknown"), "filter_reason"] = "unknown_reward_family"

    eligible = df["filter_reason"].eq("")
    source_id_key = df["source_dataset"].astype(str) + "::" + df["source_original_id"].astype(str)
    df["source_id_key"] = source_id_key
    df["image_question_key"] = df["image_hash"].astype(str) + "::" + df["question_hash"].astype(str)

    df["duplicate_source_id"] = False
    df["duplicate_image_question"] = False
    df["conflicting_image_question"] = False

    eligible_df = df[eligible].copy()
    if not eligible_df.empty:
        source_id_dups = eligible_df.duplicated("source_id_key", keep="first")
        df.loc[eligible_df.index[source_id_dups], "duplicate_source_id"] = True

        grouped_answers = eligible_df.groupby("image_question_key")["canonical_answer"].nunique(dropna=False)
        conflict_keys = set(grouped_answers[grouped_answers > 1].index)
        dup_iq = eligible_df.duplicated("image_question_key", keep="first")
        df.loc[eligible_df.index[dup_iq], "duplicate_image_question"] = True
        df.loc[eligible_df["image_question_key"].isin(conflict_keys).index, "conflicting_image_question"] = eligible_df[
            "image_question_key"
        ].isin(conflict_keys).values

    df["hard_dedup_reason"] = ""
    df.loc[eligible & df["conflicting_image_question"], "hard_dedup_reason"] = "conflicting_image_question"
    df.loc[eligible & df["hard_dedup_reason"].eq("") & df["duplicate_source_id"], "hard_dedup_reason"] = "duplicate_source_id"
    df.loc[eligible & df["hard_dedup_reason"].eq("") & df["duplicate_image_question"], "hard_dedup_reason"] = "duplicate_image_question"
    df["keep_after_hard_dedup"] = df["filter_reason"].eq("") & df["hard_dedup_reason"].eq("")
    df["image_group_size"] = df.groupby(["origin", "source_dataset", "image_hash"])["image_hash"].transform("size")
    df["image_group_rank"] = df.groupby(["origin", "source_dataset", "image_hash"]).cumcount() + 1
    return df


def apply_manual_exclusions(candidates: pd.DataFrame, exclude_csv: Path) -> pd.DataFrame:
    exclude_csv = exclude_csv.expanduser().resolve()
    if not exclude_csv.is_file():
        raise FileNotFoundError(f"exclude csv not found: {exclude_csv}")
    excludes = pd.read_csv(exclude_csv)
    if "raw_file" not in excludes.columns or "raw_row" not in excludes.columns:
        raise ValueError(f"exclude csv must contain raw_file and raw_row columns: {exclude_csv}")
    exclude_keys = set(zip(excludes["raw_file"].astype(str), excludes["raw_row"].astype(int)))
    df = candidates.copy()
    keys = list(zip(df["raw_file"].astype(str), df["raw_row"].astype(int)))
    mask = pd.Series([key in exclude_keys for key in keys], index=df.index)
    df.loc[mask, "hard_dedup_reason"] = "manual_exclude"
    df.loc[mask, "keep_after_hard_dedup"] = False
    print(f"[info] manual exclusions applied: {int(mask.sum())} rows from {exclude_csv}")
    return df


def sample_targets(candidates: pd.DataFrame, seed: int) -> pd.DataFrame:
    kept = candidates[candidates["keep_after_hard_dedup"]].copy()
    pieces: list[pd.DataFrame] = []
    for source, target in NEW_TARGETS.items():
        subset = kept[(kept["origin"] == "new") & (kept["source_dataset"] == source)]
        pieces.append(mark_sample_bucket(sample_one_source(subset, target, seed=seed, stratify=False), "target"))
    for source, target in CONTROL_TARGETS.items():
        subset = kept[(kept["origin"] == "control") & (kept["source_dataset"] == source)]
        pieces.append(mark_sample_bucket(sample_one_source(subset, target, seed=seed, stratify=(source == "virl39k")), "target"))

    sampled = pd.concat([p for p in pieces if not p.empty], ignore_index=True) if pieces else pd.DataFrame()
    if sampled.empty:
        return sampled

    shortfall = FINAL_TARGET_ROWS - len(sampled)
    if shortfall > 0:
        sampled_ids = set(sampled["candidate_row_id"].tolist())
        topup = sample_topup(kept[~kept["candidate_row_id"].isin(sampled_ids)], shortfall, seed=seed)
        if not topup.empty:
            sampled = pd.concat([sampled, mark_sample_bucket(topup, "topup")], ignore_index=True)

    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sampled.insert(0, "sample_index", range(len(sampled)))
    sampled["sample_uid"] = sampled["source_dataset"].astype(str) + "::" + sampled["source_original_id"].astype(str)
    return sampled


def mark_sample_bucket(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["sample_bucket"] = bucket
    return out


def sample_topup(remaining: pd.DataFrame, shortfall: int, *, seed: int) -> pd.DataFrame:
    if remaining.empty or shortfall <= 0:
        return remaining.head(0).copy()
    topup_order = [
        ("new", "chartqa"),
        ("new", "pixmo_count"),
        ("new", "virgorlsa"),
        ("control", "virl39k"),
        ("control", "WaltonFuture"),
    ]
    pieces: list[pd.DataFrame] = []
    used_ids: set[int] = set()
    needed = shortfall
    for origin, source in topup_order:
        if needed <= 0:
            break
        pool = remaining[
            (remaining["origin"] == origin)
            & (remaining["source_dataset"] == source)
            & (~remaining["candidate_row_id"].isin(used_ids))
        ]
        if pool.empty:
            continue
        take = min(needed, len(pool))
        piece = pool.sample(n=take, random_state=seed + len(pieces) + 1)
        pieces.append(piece)
        used_ids.update(piece["candidate_row_id"].tolist())
        needed -= take
    return pd.concat(pieces, ignore_index=False).copy() if pieces else remaining.head(0).copy()


def sample_one_source(df: pd.DataFrame, target: int, *, seed: int, stratify: bool) -> pd.DataFrame:
    if df.empty or target <= 0:
        return df.head(0).copy()
    if len(df) <= target:
        return df.copy()
    if not stratify:
        return df.sample(n=target, random_state=seed).copy()

    strata = df.groupby(["problem_type", "answer_type_norm"], dropna=False)
    allocations: dict[Any, int] = {}
    remainders: list[tuple[float, Any]] = []
    for key, group in strata:
        exact = target * len(group) / len(df)
        alloc = min(len(group), int(exact))
        allocations[key] = alloc
        remainders.append((exact - alloc, key))

    remaining = target - sum(allocations.values())
    for _rem, key in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        group_size = len(strata.get_group(key))
        if allocations[key] < group_size:
            allocations[key] += 1
            remaining -= 1

    sampled_parts = []
    for key, n in allocations.items():
        if n <= 0:
            continue
        sampled_parts.append(strata.get_group(key).sample(n=n, random_state=seed))
    return pd.concat(sampled_parts, ignore_index=False).copy() if sampled_parts else df.head(0).copy()


def write_inventory_reports(raw: pd.DataFrame, report_dir: Path) -> None:
    raw.groupby(["origin", "source_dataset"], dropna=False).size().reset_index(name="rows").sort_values(
        ["origin", "source_dataset"]
    ).to_csv(report_dir / "raw_inventory_by_source.csv", index=False)
    raw.groupby(["origin", "source_dataset", "problem_type", "answer_type_norm"], dropna=False).size().reset_index(
        name="rows"
    ).sort_values(["origin", "source_dataset", "problem_type", "answer_type_norm"]).to_csv(
        report_dir / "raw_inventory_by_type.csv", index=False
    )
    raw.groupby(["origin", "source_raw", "source_dataset"], dropna=False).size().reset_index(name="rows").sort_values(
        ["origin", "source_dataset", "source_raw"]
    ).to_csv(report_dir / "source_normalization_report.csv", index=False)


def write_dedup_reports(candidates: pd.DataFrame, report_dir: Path, max_rows: int) -> None:
    candidates.groupby(["origin", "source_dataset", "filter_reason"], dropna=False).size().reset_index(
        name="rows"
    ).sort_values(["origin", "source_dataset", "filter_reason"]).to_csv(report_dir / "filter_report.csv", index=False)

    candidates.groupby(["origin", "source_dataset", "hard_dedup_reason"], dropna=False).size().reset_index(
        name="rows"
    ).sort_values(["origin", "source_dataset", "hard_dedup_reason"]).to_csv(
        report_dir / "hard_dedup_report.csv", index=False
    )

    kept = candidates[candidates["keep_after_hard_dedup"]]
    kept.groupby(["origin", "source_dataset"], dropna=False).size().reset_index(name="rows").sort_values(
        ["origin", "source_dataset"]
    ).to_csv(report_dir / "candidate_pool_after_dedup_by_source.csv", index=False)
    kept.groupby(["origin", "source_dataset", "reward_family"], dropna=False).size().reset_index(name="rows").sort_values(
        ["origin", "source_dataset", "reward_family"]
    ).to_csv(report_dir / "candidate_pool_after_dedup_by_reward_family.csv", index=False)
    kept.groupby(["origin", "source_dataset", "problem_type", "answer_type_norm", "reward_family"], dropna=False).size().reset_index(
        name="rows"
    ).sort_values(["origin", "source_dataset", "problem_type", "answer_type_norm", "reward_family"]).to_csv(
        report_dir / "candidate_pool_after_dedup_by_type.csv", index=False
    )

    target_rows = []
    for source, target in NEW_TARGETS.items():
        available = int(len(kept[(kept["origin"] == "new") & (kept["source_dataset"] == source)]))
        target_rows.append({"origin": "new", "source_dataset": source, "target": target, "available": available, "shortfall": max(0, target - available)})
    for source, target in CONTROL_TARGETS.items():
        available = int(len(kept[(kept["origin"] == "control") & (kept["source_dataset"] == source)]))
        target_rows.append({"origin": "control", "source_dataset": source, "target": target, "available": available, "shortfall": max(0, target - available)})
    pd.DataFrame(target_rows).to_csv(report_dir / "target_feasibility.csv", index=False)

    duplicate_cols = [
        "origin",
        "source_dataset",
        "source_original_id",
        "raw_file",
        "raw_row",
        "problem_type",
        "answer_type",
        "canonical_answer",
        "question",
        "source_id_key",
        "image_question_key",
        "hard_dedup_reason",
    ]
    candidates[candidates["duplicate_source_id"]][duplicate_cols].head(max_rows).to_csv(
        report_dir / "duplicate_source_ids.csv", index=False
    )
    candidates[candidates["duplicate_image_question"]][duplicate_cols].head(max_rows).to_csv(
        report_dir / "duplicate_image_questions.csv", index=False
    )
    candidates[candidates["conflicting_image_question"]][duplicate_cols].head(max_rows).to_csv(
        report_dir / "conflicting_image_questions.csv", index=False
    )


def write_sample_reports(candidates: pd.DataFrame, sampled: pd.DataFrame, report_dir: Path) -> None:
    if sampled.empty:
        pd.DataFrame().to_csv(report_dir / "sampled_by_source.csv", index=False)
        return

    sampled.groupby(["origin", "source_dataset"], dropna=False).size().reset_index(name="rows").sort_values(
        ["origin", "source_dataset"]
    ).to_csv(report_dir / "sampled_by_source.csv", index=False)
    sampled.groupby(["origin", "source_dataset", "problem_type", "answer_type_norm", "reward_family"], dropna=False).size().reset_index(
        name="rows"
    ).sort_values(["origin", "source_dataset", "problem_type", "answer_type_norm", "reward_family"]).to_csv(
        report_dir / "sampled_by_type.csv", index=False
    )
    sampled.groupby(["reward_family"], dropna=False).size().reset_index(name="rows").sort_values("reward_family").to_csv(
        report_dir / "sampled_by_reward_family.csv", index=False
    )

    kept = candidates[candidates["keep_after_hard_dedup"]]
    rows = []
    for source, target in NEW_TARGETS.items():
        rows.append(sample_summary_row(sampled, kept, "new", source, target))
    for source, target in CONTROL_TARGETS.items():
        rows.append(sample_summary_row(sampled, kept, "control", source, target))
    pd.DataFrame(rows).to_csv(report_dir / "sample_target_report.csv", index=False)


def sample_summary_row(sampled: pd.DataFrame, kept: pd.DataFrame, origin: str, source: str, target: int) -> dict[str, Any]:
    available = int(len(kept[(kept["origin"] == origin) & (kept["source_dataset"] == source)]))
    selected = int(len(sampled[(sampled["origin"] == origin) & (sampled["source_dataset"] == source)]))
    return {
        "origin": origin,
        "source_dataset": source,
        "target": target,
        "available": available,
        "selected": selected,
        "shortfall": max(0, target - selected),
        "over_target": max(0, selected - target),
    }


def classify_target_group(origin: str, source_dataset: str) -> str:
    if origin == "new" and source_dataset in NEW_TARGETS:
        return "new"
    if origin == "control" and source_dataset in CONTROL_TARGETS:
        return "control"
    return ""


def normalize_source(raw_source: Any, *, source_from_path: str, origin: str) -> str:
    raw = str(raw_source or "").strip()
    path_source = str(source_from_path or "").strip()
    text = f"{raw} {path_source}".strip()
    key = normalize_key(text)

    exact = {
        "ref_l4": "refl4",
        "refl4": "refl4",
        "virgorlsa": "virgorlsa",
        "pixmo_count": "pixmo_count",
        "sat2": "sat2",
        "arxivqa": "arxivqa",
        "ocrbench": "ocrbench",
        "docvqa": "docvqa",
        "infographicvqa": "infographicvqa",
        "ai2d": "ai2d",
        "countqa": "countqa",
        "mmstar": "mmstar",
        "chartqa": "chartqa",
        "virl39k": "virl39k",
        "thinklite_vl_hard": "thinklite_vl_hard",
        "tqa": "tqa",
        "mmk12": "mmk12",
        "wemath_standard": "wemath_standard",
        "puzzlevqa": "puzzlevqa",
    }
    raw_key = normalize_key(raw)
    path_key = normalize_key(path_source)
    if path_key in exact:
        return exact[path_key]
    if raw_key in exact:
        return exact[raw_key]

    lower = text.lower()
    if "waltonfuture" in lower or "multimodal-rl-data" in lower or "multimodal_rl_data" in lower:
        return "WaltonFuture"
    if "thinklite" in lower:
        return "thinklite_vl_hard"
    if "wemath" in lower:
        return "wemath_standard"
    if "puzzlevqa" in lower:
        return "puzzlevqa"
    if "virl39k" in lower or "virl" in lower:
        return "virl39k"
    return path_source if origin == "new" and path_source else raw


def reward_family_for(source_dataset: str, answer_type: str) -> str:
    source_key = normalize_key(source_dataset)
    if source_key in SOURCE_FORCE_FAMILY:
        return SOURCE_FORCE_FAMILY[source_key]
    family = ANSWER_TYPE_TO_FAMILY.get(normalize_key(answer_type))
    if family:
        return family
    return SOURCE_FALLBACK_FAMILY.get(source_key, "unknown")


def clean_question(problem: Any) -> str:
    text = str(problem or "")
    text = re.sub(r"^\s*(?:<image>\s*)+", "", text)
    return text.strip()


def normalize_question(question: Any) -> str:
    return re.sub(r"\s+", " ", str(question or "").strip()).casefold()


def normalize_answers(value: Any) -> list[str]:
    value = to_builtin(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(normalize_answers(item))
        return dedupe_preserve_order([item for item in out if item != ""])
    if isinstance(value, dict):
        return [json.dumps(value, ensure_ascii=False, sort_keys=True)]

    text = str(value).strip()
    if not text:
        return []
    if looks_like_list(text):
        parsed = parse_listish(text)
        if parsed is not None and parsed != text:
            return normalize_answers(parsed)
    return [text]


def canonicalize_answer(answers: list[str], answer_type: str) -> str:
    if not answers:
        return ""
    atype = normalize_key(answer_type)
    first = answers[0].strip()
    if atype in {"multiple_choice", "multiple-choice", "choice", "mcq"}:
        choice = extract_choice(first)
        return choice or first
    if atype in {"boolean", "bool"}:
        parsed = parse_bool(first)
        return parsed if parsed is not None else first.casefold()
    if atype in {"bbox", "box"}:
        if len(answers) == 4:
            bbox = extract_bbox(answers)
            if bbox is not None:
                return json.dumps(bbox)
        bbox = extract_bbox(first)
        return json.dumps(bbox) if bbox is not None else first
    return first


def image_identity(images_value: Any, *, hash_mode: str) -> dict[str, Any]:
    images = to_builtin(images_value)
    if images is None:
        return empty_image_meta("missing_images")
    if isinstance(images, dict):
        images = [images]
    if not isinstance(images, list):
        return empty_image_meta("bad_images_type")
    if len(images) != 1:
        meta = empty_image_meta("multi_image" if len(images) > 1 else "missing_images")
        meta["image_count"] = len(images)
        return meta

    image = images[0]
    if not isinstance(image, dict):
        return empty_image_meta("bad_image_entry")
    image_bytes = image.get("bytes")
    image_path = image.get("path") or image.get("image")
    if image_bytes:
        if isinstance(image_bytes, memoryview):
            image_bytes = image_bytes.tobytes()
        if not isinstance(image_bytes, (bytes, bytearray)):
            return empty_image_meta("bad_image_bytes")
        digest = stable_hash_bytes(bytes(image_bytes))
        return {
            "image_count": 1,
            "image_hash": digest,
            "image_ref_kind": "bytes",
            "image_ref": "",
            "image_error": "",
        }
    if image_path:
        image_path = strip_file_uri(str(image_path))
        path = Path(image_path)
        if hash_mode == "content":
            if not path.is_file():
                return empty_image_meta("image_path_missing")
            digest = hash_file(path)
        else:
            digest = stable_hash(str(path))
        return {
            "image_count": 1,
            "image_hash": digest,
            "image_ref_kind": "path",
            "image_ref": str(path),
            "image_error": "",
        }
    return empty_image_meta("missing_image_payload")


def empty_image_meta(error: str) -> dict[str, Any]:
    return {
        "image_count": 0,
        "image_hash": "",
        "image_ref_kind": "",
        "image_ref": "",
        "image_error": error,
    }


def strip_file_uri(path: str) -> str:
    return path[7:] if path.startswith("file://") else path


def extract_choice(text: Any) -> str | None:
    matches = re.findall(r"\b([A-Z])\b", str(text or "").strip().upper())
    return matches[-1] if matches else None


def parse_bool(text: Any) -> str | None:
    norm = normalize_question(text)
    if norm in {"yes", "true", "1"} or re.search(r"\b(yes|true)\b", norm):
        return "true"
    if norm in {"no", "false", "0"} or re.search(r"\b(no|false)\b", norm):
        return "false"
    return None


def extract_bbox(text: Any) -> list[float] | None:
    if isinstance(text, (list, tuple)) and len(text) == 4:
        try:
            return [float(x) for x in text]
        except Exception:
            return None
    match = re.search(r"\[\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:\s*,\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?){3}\s*\]", str(text or ""))
    if not match:
        return None
    try:
        return [float(x.strip()) for x in match.group(0)[1:-1].split(",")]
    except Exception:
        return None


def to_builtin(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def scalar(value: Any) -> str:
    value = to_builtin(value)
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "" if not value else scalar(value[0])
    return str(value)


def looks_like_list(text: str) -> bool:
    text = text.strip()
    return len(text) >= 2 and text[0] in "[(" and text[-1] in "])"


def parse_listish(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def normalize_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
