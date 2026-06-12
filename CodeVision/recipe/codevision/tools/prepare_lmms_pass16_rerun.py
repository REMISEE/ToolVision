#!/usr/bin/env python3
"""Prepare same-sample JSONL inputs for lmms-eval pass@16 reruns.

This script exports the 06-02 benchmark pass16 samples into local JSONL files
that lmms-eval can load with `dataset_path: json`. GQA/TextVQA images are
materialized from local benchmark parquet bytes; FSC147 keeps the existing file
paths.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


DEFAULT_PASS16_ROOT = Path("/mnt/cpfs/delinmao/data/toolvision_pass16_full/benchmark_sources")
DEFAULT_OUTPUT_ROOT = Path("/mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun")
DEFAULT_GQA_IMAGE_DIR = Path("/mnt/cpfs/delinmao/Benchmarks/GQA/val_balanced_images")
DEFAULT_TEXTVQA_DIR = Path("/mnt/cpfs/delinmao/Benchmarks/TextVQA/data")


def _json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def _strip_image_token(text: str) -> str:
    return re.sub(r"^\s*<image>\s*", "", str(text)).strip()


def _answers_from_row(row: pd.Series) -> list[str]:
    answers = _json_loads(row.get("answer_json"), [])
    if not isinstance(answers, list):
        answers = [answers]
    return [str(answer) for answer in answers]


def _metadata_from_row(row: pd.Series) -> dict[str, Any]:
    metadata = _json_loads(row.get("metadata_json"), {})
    return metadata if isinstance(metadata, dict) else {}


def _image_refs_from_row(row: pd.Series) -> list[dict[str, Any]]:
    refs = _json_loads(row.get("image_refs_json"), [])
    return refs if isinstance(refs, list) else []


def _read_pass16_source(pass16_root: Path, source: str, limit: int | None) -> pd.DataFrame:
    files = sorted((pass16_root / source).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No pass16 parquet files for source={source} under {pass16_root}")

    frames: list[pd.DataFrame] = []
    remaining = limit
    for file_path in files:
        frame = pd.read_parquet(file_path)
        if remaining is not None:
            if remaining <= 0:
                break
            frame = frame.head(remaining)
            remaining -= len(frame)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _image_bytes_to_file(image_value: Any, output_path: Path) -> None:
    if output_path.exists():
        return

    if isinstance(image_value, dict) and image_value.get("bytes") is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_value["bytes"])
        return

    if isinstance(image_value, Image.Image):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image_value.convert("RGB").save(output_path, format="JPEG", quality=95)
        return

    raise ValueError(f"Unsupported image value type for {output_path}: {type(image_value)!r}")


def _materialize_gqa_images(rows: pd.DataFrame, image_dir: Path, out_dir: Path) -> dict[str, str]:
    needed = {
        str(_metadata_from_row(row).get("image_id", "")).strip()
        for _, row in rows.iterrows()
    }
    needed.discard("")
    missing = set(needed)
    resolved: dict[str, str] = {}

    for parquet_path in sorted(image_dir.glob("*.parquet")):
        if not missing:
            break
        image_rows = pd.read_parquet(parquet_path)
        for _, image_row in image_rows.iterrows():
            image_id = str(image_row["id"])
            if image_id not in missing:
                continue
            output_path = out_dir / "gqa" / f"{image_id}.jpg"
            _image_bytes_to_file(image_row["image"], output_path)
            resolved[image_id] = str(output_path)
            missing.remove(image_id)

    if missing:
        examples = ", ".join(sorted(missing)[:10])
        raise RuntimeError(f"GQA image bytes missing for {len(missing)} image ids, examples: {examples}")

    return resolved


def _materialize_textvqa_images(rows: pd.DataFrame, textvqa_dir: Path, out_dir: Path) -> dict[str, str]:
    needed_qids = {
        str(_metadata_from_row(row).get("question_id", row.get("source_index", ""))).strip()
        for _, row in rows.iterrows()
    }
    needed_qids.discard("")
    missing = set(needed_qids)
    resolved: dict[str, str] = {}

    for parquet_path in sorted(textvqa_dir.glob("train-*.parquet")):
        if not missing:
            break
        textvqa_rows = pd.read_parquet(parquet_path)
        for _, textvqa_row in textvqa_rows.iterrows():
            question_id = str(textvqa_row["question_id"])
            if question_id not in missing:
                continue
            image_id = str(textvqa_row["image_id"])
            output_path = out_dir / "textvqa" / f"{question_id}_{image_id}.jpg"
            _image_bytes_to_file(textvqa_row["image"], output_path)
            resolved[question_id] = str(output_path)
            missing.remove(question_id)

    if missing:
        examples = ", ".join(sorted(missing)[:10])
        raise RuntimeError(f"TextVQA image bytes missing for {len(missing)} question ids, examples: {examples}")

    return resolved


def _blank_image_path(out_dir: Path) -> str:
    path = out_dir / "controls" / "blank_512.jpg"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (512, 512), (255, 255, 255)).save(path, format="JPEG", quality=95)
    return str(path)


def _prompt_for_row(source: str, row: pd.Series, metadata: dict[str, Any]) -> str:
    if source == "fsc147":
        class_name = str(metadata.get("class_name") or "objects").strip() or "objects"
        return f"How many {class_name} are there in the image?\nAnswer with only an integer."

    question = _strip_image_token(str(row.get("problem", "")))
    if source in {"gqa", "textvqa"}:
        return f"{question}\nAnswer the question using a single word or phrase."
    return question


def _real_image_path(source: str, row: pd.Series, metadata: dict[str, Any], maps: dict[str, dict[str, str]]) -> str:
    if source == "gqa":
        image_id = str(metadata.get("image_id", "")).strip()
        return maps["gqa"][image_id]
    if source == "textvqa":
        question_id = str(metadata.get("question_id", row.get("source_index", ""))).strip()
        return maps["textvqa"][question_id]
    if source == "fsc147":
        refs = _image_refs_from_row(row)
        if refs and refs[0].get("path"):
            return str(refs[0]["path"])
        image_path = metadata.get("image_path")
        if image_path:
            return str(image_path)
    raise RuntimeError(f"Cannot resolve image path for source={source}, id={row.get('id')}")


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_source(
    source: str,
    pass16_root: Path,
    output_root: Path,
    limit: int | None,
    mode: str,
    seed: int,
    gqa_image_dir: Path,
    textvqa_dir: Path,
) -> dict[str, Any]:
    df = _read_pass16_source(pass16_root, source, limit)
    image_out_dir = output_root / "images"
    maps: dict[str, dict[str, str]] = {}
    if source == "gqa":
        maps["gqa"] = _materialize_gqa_images(df, gqa_image_dir, image_out_dir)
    elif source == "textvqa":
        maps["textvqa"] = _materialize_textvqa_images(df, textvqa_dir, image_out_dir)

    records: list[dict[str, Any]] = []
    real_paths: list[str] = []
    for _, row in df.iterrows():
        metadata = _metadata_from_row(row)
        answers = _answers_from_row(row)
        image_path = _real_image_path(source, row, metadata, maps)
        real_paths.append(image_path)
        records.append(
            {
                "id": str(row.get("id")),
                "source": source,
                "task_id": int(row.get("task_id", -1)),
                "source_index": int(row.get("source_index", -1)),
                "raw_file": str(row.get("raw_file", "")),
                "raw_row": int(row.get("raw_row", -1)),
                "question": _strip_image_token(str(row.get("problem", ""))),
                "prompt": _prompt_for_row(source, row, metadata),
                "answer": answers[0] if answers else "",
                "answers": answers,
                "answer_type": str(row.get("answer_type", "")),
                "metadata": metadata,
                "image_path": image_path,
                "original_image_refs_json": row.get("image_refs_json", "[]"),
            }
        )

    if mode == "blank":
        blank_path = _blank_image_path(image_out_dir)
        for record in records:
            record["image_path"] = blank_path
            record["control_mode"] = "blank"
    elif mode == "shuffled":
        rng = random.Random(seed)
        shuffled = list(real_paths)
        rng.shuffle(shuffled)
        for record, image_path in zip(records, shuffled):
            record["image_path"] = image_path
            record["control_mode"] = "shuffled"
    else:
        for record in records:
            record["control_mode"] = "real"

    out_path = output_root / "inputs" / f"{source}_{mode}.jsonl"
    _write_jsonl(records, out_path)

    missing_files = [record["image_path"] for record in records if not Path(record["image_path"]).exists()]
    return {
        "source": source,
        "mode": mode,
        "rows": len(records),
        "jsonl": str(out_path),
        "missing_images": len(missing_files),
        "missing_image_examples": missing_files[:10],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass16-root", type=Path, default=DEFAULT_PASS16_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sources", default="gqa,textvqa,fsc147")
    parser.add_argument("--mode", choices=["real", "blank", "shuffled"], default="real")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--gqa-image-dir", type=Path, default=DEFAULT_GQA_IMAGE_DIR)
    parser.add_argument("--textvqa-dir", type=Path, default=DEFAULT_TEXTVQA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [source.strip().lower() for source in args.sources.split(",") if source.strip()]
    summaries = [
        export_source(
            source=source,
            pass16_root=args.pass16_root,
            output_root=args.output_root,
            limit=args.limit,
            mode=args.mode,
            seed=args.seed,
            gqa_image_dir=args.gqa_image_dir,
            textvqa_dir=args.textvqa_dir,
        )
        for source in sources
    ]

    manifest_path = args.output_root / "inputs" / f"manifest_{args.mode}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
