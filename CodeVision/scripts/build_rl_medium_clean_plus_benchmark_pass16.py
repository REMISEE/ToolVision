#!/usr/bin/env python3
"""Build a medium-clean RL parquet plus partial-pass SFT-source benchmark rows."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from PIL import Image


DEFAULT_BASE_PARQUET = Path(
    "/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k_clean_no_extra_image/parquet/"
    "train_medium_clean_21k.parquet"
)
DEFAULT_PASS16_DIR = Path("/mnt/cpfs/delinmao/data/toolvision_pass16_full/benchmark_sources")
DEFAULT_OUT_PARQUET = Path(
    "/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k_clean_no_extra_image/parquet/"
    "train_medium_clean_21k_plus_benchmark_pass16_partial_26591_rewardfix_fsc09.parquet"
)
DEFAULT_IMAGE_DIR = Path(
    "/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k_clean_no_extra_image/"
    "images_raw/benchmark_pass16"
)

GQA_IMAGE_DIR = Path("/mnt/cpfs/delinmao/Benchmarks/GQA/val_balanced_images")
TEXTVQA_DATA_DIR = Path("/mnt/cpfs/delinmao/Benchmarks/TextVQA/data")


def load_pass16_source(pass16_dir: Path, source: str) -> pd.DataFrame:
    files = sorted((pass16_dir / source).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No pass16 parquet files for {source}: {pass16_dir / source}")
    return pd.concat([pq.read_table(path).to_pandas() for path in files], ignore_index=True)


def select_gqa(df: pd.DataFrame, seed: int, target: int) -> pd.DataFrame:
    partial = df[df["correct_count"].between(1, 15)].copy()
    quotas = [
        ("hard_1_3", partial["correct_count"].between(1, 3), 1500),
        ("mid_4_8", partial["correct_count"].between(4, 8), 1500),
        ("easy_9_15", partial["correct_count"].between(9, 15), 1000),
    ]
    selected = []
    for bucket, mask, quota in quotas:
        cand = partial[mask].copy()
        take = min(quota, len(cand))
        if take:
            part = cand.sample(n=take, random_state=seed + len(selected))
            part["pass16_select_bucket"] = bucket
            selected.append(part)

    out = pd.concat(selected, ignore_index=True) if selected else partial.iloc[:0].copy()
    if len(out) < target:
        remaining = partial[~partial["id"].isin(set(out["id"]))]
        take = min(target - len(out), len(remaining))
        if take:
            fill = remaining.sample(n=take, random_state=seed + 99).copy()
            fill["pass16_select_bucket"] = "fill_partial"
            out = pd.concat([out, fill], ignore_index=True)
    if len(out) > target:
        out = out.sample(n=target, random_state=seed + 199).reset_index(drop=True)
    return out.reset_index(drop=True)


def select_all_partial(df: pd.DataFrame, bucket_prefix: str) -> pd.DataFrame:
    out = df[df["correct_count"].between(1, 15)].copy()
    out["pass16_select_bucket"] = bucket_prefix + "_partial"
    return out.reset_index(drop=True)


def json_loads(text: Any, default: Any) -> Any:
    if text is None:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def normalize_question(problem: str) -> str:
    text = str(problem or "")
    if text.startswith("<image>"):
        text = text[len("<image>") :]
    return text.strip()


def choose_ground_truth(answers: list[Any]) -> str:
    cleaned = [str(a).strip() for a in answers if str(a).strip()]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


def image_suffix(fmt: str | None) -> str:
    fmt = (fmt or "").upper()
    if fmt in {"JPEG", "JPG"}:
        return ".jpg"
    if fmt == "PNG":
        return ".png"
    if fmt == "WEBP":
        return ".webp"
    return ".jpg"


def write_image_bytes(image_bytes: bytes, out_stem: Path) -> tuple[Path, int, int]:
    with Image.open(io.BytesIO(image_bytes)) as im:
        width, height = im.size
        suffix = image_suffix(im.format)
    out_path = out_stem.with_suffix(suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        out_path.write_bytes(image_bytes)
    return out_path, width, height


def file_image_info(path: Path) -> tuple[Path, int, int]:
    with Image.open(path) as im:
        width, height = im.size
    return path, width, height


def load_gqa_images(image_ids: set[str]) -> dict[str, bytes]:
    if not image_ids:
        return {}
    table = ds.dataset(str(GQA_IMAGE_DIR), format="parquet").to_table(
        columns=["id", "image"],
        filter=pc.field("id").isin(sorted(image_ids)),
    )
    records = table.to_pandas()
    out = {}
    for _, row in records.iterrows():
        image_obj = row["image"]
        out[str(row["id"])] = image_obj["bytes"]
    missing = image_ids - set(out)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} GQA images, examples={sorted(missing)[:5]}")
    return out


def load_textvqa_rows(question_ids: set[int]) -> dict[int, dict[str, Any]]:
    if not question_ids:
        return {}
    table = ds.dataset(str(TEXTVQA_DATA_DIR), format="parquet").to_table(
        columns=["question_id", "image_id", "image", "image_width", "image_height"],
        filter=pc.field("question_id").isin(sorted(question_ids)),
    )
    records = table.to_pandas()
    out = {}
    for _, row in records.iterrows():
        out[int(row["question_id"])] = {
            "image_id": str(row["image_id"]),
            "bytes": row["image"]["bytes"],
            "width": int(row["image_width"]),
            "height": int(row["image_height"]),
        }
    missing = question_ids - set(out)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} TextVQA rows, examples={sorted(missing)[:5]}")
    return out


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_rows(selected: pd.DataFrame, image_dir: Path) -> list[dict[str, Any]]:
    gqa_rows = selected[selected["source"] == "gqa"]
    textvqa_rows = selected[selected["source"] == "textvqa"]

    gqa_image_ids = {
        str(json_loads(meta, {}).get("image_id"))
        for meta in gqa_rows["metadata_json"].tolist()
        if json_loads(meta, {}).get("image_id") is not None
    }
    textvqa_question_ids = {int(x) for x in textvqa_rows["source_index"].tolist()}

    gqa_images = load_gqa_images(gqa_image_ids)
    textvqa_records = load_textvqa_rows(textvqa_question_ids)

    out_rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        source = str(row["source"])
        uid = str(row["id"])
        question = normalize_question(row["problem"])
        answers = json_loads(row["answer_json"], [])
        if not isinstance(answers, list):
            answers = [answers]
        ground_truth = choose_ground_truth(answers)
        metadata = json_loads(row["metadata_json"], {})
        if not isinstance(metadata, dict):
            metadata = {}

        if source == "gqa":
            image_id = str(metadata["image_id"])
            img_path, width, height = write_image_bytes(
                gqa_images[image_id],
                image_dir / "gqa" / uid,
            )
        elif source == "textvqa":
            qid = int(row["source_index"])
            rec = textvqa_records[qid]
            img_path, width, height = write_image_bytes(
                rec["bytes"],
                image_dir / "textvqa" / uid,
            )
        elif source == "fsc147":
            refs = json_loads(row["image_refs_json"], [])
            if not refs:
                raise RuntimeError(f"Missing FSC147 image ref for {uid}")
            ref_path = Path(refs[0].get("path") or refs[0].get("uri", "").replace("file://", ""))
            img_path, width, height = file_image_info(ref_path)
        else:
            raise ValueError(f"Unexpected source: {source}")

        content = f"<image>Image size = {width}x{height} pixels.\n\n{question}"
        q_hash = sha256_text(question)
        qa_hash = sha256_text(source + "\n" + uid + "\n" + question + "\n" + ground_truth)
        problem_type = str(row["problem_type"])
        answer_type = str(row["answer_type"])
        answer_type_norm = "number" if source == "fsc147" else answer_type
        ability = "math_qa" if source == "fsc147" or problem_type == "counting" else "mm_qa"
        reward_family = {
            "gqa": "exact",
            "textvqa": "ocr_levenshtein",
            "fsc147": "fsc147_relative",
        }[source]

        extra_info = {
            "acceptable_answers": [str(a) for a in answers],
            "answer_type": answer_type,
            "answer_type_norm": answer_type_norm,
            "choices": None,
            "image_hash": sha256_text(str(img_path)),
            "image_height": int(height),
            "image_width": int(width),
            "index": int(row["source_index"]),
            "origin": "benchmark_pass16_partial",
            "problem_type": problem_type,
            "prompt_type": str(row["prompt_type"]),
            "qa_hash": qa_hash,
            "question": question,
            "question_hash": q_hash,
            "raw_file": str(row["raw_file"]),
            "raw_problem": str(row["problem"]),
            "raw_row": int(row["raw_row"]),
            "reward_family": reward_family,
            "sample_bucket": str(row["pass16_select_bucket"]),
            "source_benchmark": source,
            "source_dataset": source,
            "source_original_id": uid,
            "source_raw": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            "uid": uid,
            "pass16_acc_at_k": float(row["acc_at_k"]),
            "pass16_all_correct": int(row["all_correct"]),
            "pass16_correct_count": int(row["correct_count"]),
            "pass16_correct_threshold": float(row["correct_threshold"]),
            "pass16_num_preds": int(row["num_preds"]),
            "pass16_pass_at_k": int(row["pass_at_k"]),
            "pass16_rollout_n": int(row["rollout_n"]),
        }
        if source == "fsc147":
            extra_info["relative_count_threshold"] = 0.9

        out_rows.append(
            {
                "data_source": source,
                "agent_name": "tool_agent",
                "ability": ability,
                "prompt": [{"role": "user", "content": content}],
                "images": [{"image": "file://" + str(img_path)}],
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": extra_info,
            }
        )
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-parquet", type=Path, default=DEFAULT_BASE_PARQUET)
    parser.add_argument("--pass16-dir", type=Path, default=DEFAULT_PASS16_DIR)
    parser.add_argument("--out-parquet", type=Path, default=DEFAULT_OUT_PARQUET)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--gqa-target", type=int, default=4000)
    args = parser.parse_args()

    base_df = pq.read_table(args.base_parquet).to_pandas()
    gqa = load_pass16_source(args.pass16_dir, "gqa")
    textvqa = load_pass16_source(args.pass16_dir, "textvqa")
    fsc147 = load_pass16_source(args.pass16_dir, "fsc147")

    selected = pd.concat(
        [
            select_gqa(gqa, seed=args.seed, target=args.gqa_target),
            select_all_partial(textvqa, "textvqa"),
            select_all_partial(fsc147, "fsc147"),
        ],
        ignore_index=True,
    )
    selected = selected.sample(frac=1.0, random_state=args.seed + 777).reset_index(drop=True)
    add_df = pd.DataFrame(build_rows(selected, args.image_dir), columns=list(base_df.columns))
    out_df = pd.concat([base_df, add_df], ignore_index=True)
    out_df = out_df.sample(frac=1.0, random_state=args.seed + 1001).reset_index(drop=True)

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out_parquet, index=False)

    selection_csv = args.out_parquet.with_suffix(".selection.csv")
    selected[["id", "source", "correct_count", "acc_at_k", "pass16_select_bucket"]].to_csv(
        selection_csv, index=False
    )

    manifest = {
        "base_parquet": str(args.base_parquet),
        "pass16_dir": str(args.pass16_dir),
        "out_parquet": str(args.out_parquet),
        "image_dir": str(args.image_dir),
        "seed": args.seed,
        "base_rows": int(len(base_df)),
        "added_rows": int(len(add_df)),
        "out_rows": int(len(out_df)),
        "added_source_counts": add_df["data_source"].value_counts().sort_index().to_dict(),
        "out_source_counts": out_df["data_source"].value_counts().sort_index().to_dict(),
        "selection_bucket_counts": selected["pass16_select_bucket"].value_counts().sort_index().to_dict(),
        "selection_csv": str(selection_csv),
    }
    manifest_path = args.out_parquet.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
