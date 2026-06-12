#!/usr/bin/env python3
"""Prepare same-sample all-source JSONL inputs for lmms-eval pass@16 reruns.

The old pass16 parquet is kept as the source of truth for sample identity,
question, answers, and metadata. This script only resolves images into local
file paths that lmms-eval can open.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import os
import random
import re
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


DEFAULT_PASS16_ROOT = Path("/mnt/cpfs/delinmao/data/toolvision_pass16_full")
DEFAULT_OUTPUT_ROOT = Path("/mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun_all")
DEFAULT_BENCHMARK_ROOT = Path("/mnt/cpfs/delinmao/Benchmarks")

DEFAULT_SOURCES = [
    "gqa",
    "textvqa",
    "fsc147",
    "ai2d",
    "arxivqa",
    "chartqa",
    "countqa",
    "docvqa",
    "infographicvqa",
    "mmstar",
    "ocrbench",
    "pixmo_count",
    "refl4",
    "sat2",
    "virgorlsa",
]

SOURCE_GROUP = {
    "gqa": "benchmark_sources",
    "textvqa": "benchmark_sources",
    "fsc147": "benchmark_sources",
}


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
    except Exception:
        return default


def _strip_image_token(text: Any) -> str:
    return re.sub(r"^\s*(?:<image>\s*)+", "", str(text or "")).strip()


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


def _read_source_rows(pass16_root: Path, source: str, limit: int | None) -> pd.DataFrame:
    group = SOURCE_GROUP.get(source, "new_sources")
    files = sorted((pass16_root / group / source).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No pass16 parquet files found for source={source}")
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
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _image_bytes_to_file(image_value: Any, output_path: Path) -> str:
    if output_path.exists():
        return str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image_value, dict) and image_value.get("bytes") is not None:
        output_path.write_bytes(image_value["bytes"])
        return str(output_path)
    if isinstance(image_value, bytes):
        output_path.write_bytes(image_value)
        return str(output_path)
    if isinstance(image_value, Image.Image):
        image_value.convert("RGB").save(output_path, format="JPEG", quality=95)
        return str(output_path)
    try:
        Image.open(io.BytesIO(image_value)).convert("RGB").save(output_path, format="JPEG", quality=95)
        return str(output_path)
    except Exception as exc:
        raise ValueError(f"Unsupported image value for {output_path}: {type(image_value)!r}") from exc


def _load_dataset(*args: Any, **kwargs: Any) -> Any:
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def _download_image_to_file(url: str, output_path: Path, timeout: int = 30) -> str | None:
    if output_path.exists():
        return str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            image_bytes = response.read()
        Image.open(io.BytesIO(image_bytes)).convert("RGB").save(output_path, format="JPEG", quality=95)
        return str(output_path)
    except Exception as exc:
        print(f"[prepare] warning: failed to download image url={url!r}: {exc}", flush=True)
        return None


def _first_file_ref(row: pd.Series) -> str | None:
    for ref in _image_refs_from_row(row):
        path = ref.get("path")
        if path and Path(str(path)).exists():
            return str(path)
        uri = str(ref.get("uri") or "")
        if uri.startswith("file://") and Path(uri[7:]).exists():
            return uri[7:]
    return None


def _source_indices(rows: pd.DataFrame) -> set[int]:
    values = set()
    for value in rows.get("source_index", []):
        try:
            values.add(int(value))
        except Exception:
            continue
    return values


def _materialize_global_index_images(
    *,
    source: str,
    rows: pd.DataFrame,
    parquet_files: list[Path],
    output_root: Path,
    image_column: str = "image",
) -> dict[int, str]:
    needed = _source_indices(rows)
    missing = set(needed)
    resolved: dict[int, str] = {}
    global_index = 0
    for parquet_file in parquet_files:
        if not missing:
            break
        frame = pd.read_parquet(parquet_file)
        for _, image_row in frame.iterrows():
            if global_index in missing:
                if image_column not in image_row:
                    raise KeyError(f"{parquet_file} does not contain image column {image_column!r}")
                out = output_root / "images" / source / f"{source}_{global_index:08d}.jpg"
                resolved[global_index] = _image_bytes_to_file(image_row[image_column], out)
                missing.remove(global_index)
            global_index += 1
    return resolved


def _materialize_gqa(rows: pd.DataFrame, output_root: Path, benchmark_root: Path) -> dict[int, str]:
    needed_by_image_id: dict[str, list[int]] = {}
    for _, row in rows.iterrows():
        meta = _metadata_from_row(row)
        image_id = str(meta.get("image_id", "")).strip()
        if not image_id:
            continue
        needed_by_image_id.setdefault(image_id, []).append(int(row.get("source_index", -1)))
    missing = set(needed_by_image_id)
    resolved_by_index: dict[int, str] = {}
    for parquet_file in sorted((benchmark_root / "GQA" / "val_balanced_images").glob("*.parquet")):
        if not missing:
            break
        frame = pd.read_parquet(parquet_file)
        for _, image_row in frame.iterrows():
            image_id = str(image_row["id"])
            if image_id not in missing:
                continue
            out = output_root / "images" / "gqa" / f"{image_id}.jpg"
            image_path = _image_bytes_to_file(image_row["image"], out)
            for source_index in needed_by_image_id[image_id]:
                resolved_by_index[source_index] = image_path
            missing.remove(image_id)
    return resolved_by_index


def _materialize_textvqa(rows: pd.DataFrame, output_root: Path, benchmark_root: Path) -> dict[int, str]:
    needed_qids: dict[str, list[int]] = {}
    for _, row in rows.iterrows():
        meta = _metadata_from_row(row)
        qid = str(meta.get("question_id", row.get("source_index", ""))).strip()
        if not qid:
            continue
        needed_qids.setdefault(qid, []).append(int(row.get("source_index", -1)))
    missing = set(needed_qids)
    resolved_by_index: dict[int, str] = {}
    for parquet_file in sorted((benchmark_root / "TextVQA" / "data").glob("train-*.parquet")):
        if not missing:
            break
        frame = pd.read_parquet(parquet_file)
        for _, image_row in frame.iterrows():
            qid = str(image_row["question_id"])
            if qid not in missing:
                continue
            image_id = str(image_row.get("image_id", ""))
            out = output_root / "images" / "textvqa" / f"{qid}_{image_id}.jpg"
            image_path = _image_bytes_to_file(image_row["image"], out)
            for source_index in needed_qids[qid]:
                resolved_by_index[source_index] = image_path
            missing.remove(qid)
    return resolved_by_index


def _eval_parquet_image_index(source: str, rows: pd.DataFrame, output_root: Path, benchmark_root: Path) -> dict[int, str]:
    paths = {
        "countqa": benchmark_root / "CountQA" / "countqa_codevision_eval.parquet",
        # The pass16 "ocrbench" source is the 10k OCRBench_v2-aligned set.
        "ocrbench": benchmark_root / "OCRBench_v2" / "ocrbench_v2_codevision_eval.parquet",
    }
    parquet = paths.get(source)
    if not parquet or not parquet.exists():
        return {}
    needed = _source_indices(rows)
    frame = pd.read_parquet(parquet)
    resolved: dict[int, str] = {}
    for idx in sorted(needed):
        if idx < 0 or idx >= len(frame):
            continue
        extra = frame.iloc[idx].get("extra_info")
        extra = extra if isinstance(extra, dict) else {}
        image_path = extra.get("image_path")
        if image_path and Path(str(image_path)).exists():
            resolved[idx] = str(image_path)
            continue
        images = frame.iloc[idx].get("images") or []
        if images and isinstance(images, list):
            uri = str(images[0].get("image") or "")
            if uri.startswith("file://") and Path(uri[7:]).exists():
                resolved[idx] = uri[7:]
    return resolved


def _materialize_hf_index_images(
    *,
    source: str,
    rows: pd.DataFrame,
    output_root: Path,
    dataset_args: tuple[Any, ...],
    split: str,
    image_getter: Any,
) -> dict[int, str]:
    needed = _source_indices(rows)
    if not needed:
        return {}
    dataset = _load_dataset(*dataset_args, split=split)
    resolved: dict[int, str] = {}
    for n, idx in enumerate(sorted(needed), start=1):
        if idx < 0 or idx >= len(dataset):
            continue
        row = dataset[int(idx)]
        image_value = image_getter(row)
        if image_value is None:
            continue
        out = output_root / "images" / source / f"{source}_{idx:08d}.jpg"
        resolved[idx] = _image_bytes_to_file(image_value, out)
        if n % 1000 == 0:
            print(f"[prepare] source={source} materialized={n}/{len(needed)}", flush=True)
    return resolved


def _materialize_pixmo_count(rows: pd.DataFrame, output_root: Path) -> dict[int, str]:
    needed = _source_indices(rows)
    if not needed:
        return {}
    dataset = _load_dataset("allenai/pixmo-count", split="train")
    workers = int(os.environ.get("PIXMO_DOWNLOAD_WORKERS", "32"))
    resolved: dict[int, str] = {}
    jobs: list[tuple[int, str, Path]] = []
    for idx in sorted(needed):
        if idx < 0 or idx >= len(dataset):
            continue
        row = dataset[int(idx)]
        url = str(row.get("image_url") or "").strip()
        if not url:
            continue
        out = output_root / "images" / "pixmo_count" / f"pixmo_count_{idx:08d}.jpg"
        jobs.append((idx, url, out))

    def download_one(job: tuple[int, str, Path]) -> tuple[int, str | None]:
        idx, url, out = job
        return idx, _download_image_to_file(url, out)

    completed = 0
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_one, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            idx, image_path = future.result()
            completed += 1
            if image_path:
                resolved[idx] = image_path
            else:
                failed += 1
            if completed % 250 == 0:
                print(
                    "[prepare] source=pixmo_count "
                    f"downloaded_or_cached={completed}/{len(jobs)} failed={failed} workers={workers}",
                    flush=True,
                )
    return resolved


def _build_image_map(source: str, rows: pd.DataFrame, output_root: Path, benchmark_root: Path) -> dict[int, str]:
    if source == "gqa":
        return _materialize_gqa(rows, output_root, benchmark_root)
    if source == "textvqa":
        return _materialize_textvqa(rows, output_root, benchmark_root)
    if source == "chartqa":
        return _materialize_hf_index_images(
            source=source,
            rows=rows,
            output_root=output_root,
            dataset_args=("HuggingFaceM4/chart_qa",),
            split="train",
            image_getter=lambda row: row.get("image"),
        )
    if source == "ai2d":
        return _materialize_hf_index_images(
            source=source,
            rows=rows,
            output_root=output_root,
            dataset_args=("HuggingFaceM4/the_cauldron", "ai2d"),
            split="train",
            image_getter=lambda row: (row.get("images") or [None])[0],
        )
    if source == "docvqa":
        return _materialize_hf_index_images(
            source=source,
            rows=rows,
            output_root=output_root,
            dataset_args=("lmms-lab/DocVQA", "DocVQA"),
            split="validation",
            image_getter=lambda row: row.get("image"),
        )
    if source == "infographicvqa":
        return _materialize_hf_index_images(
            source=source,
            rows=rows,
            output_root=output_root,
            dataset_args=("lmms-lab/DocVQA", "InfographicVQA"),
            split="validation",
            image_getter=lambda row: row.get("image"),
        )
    if source == "mmstar":
        return _materialize_hf_index_images(
            source=source,
            rows=rows,
            output_root=output_root,
            dataset_args=("Lin-Chen/MMStar",),
            split="val",
            image_getter=lambda row: row.get("image"),
        )
    if source == "pixmo_count":
        return _materialize_pixmo_count(rows, output_root)
    if source in {"ocrbench", "countqa"}:
        return _eval_parquet_image_index(source, rows, output_root, benchmark_root)
    return {}


def _blank_image_path(output_root: Path) -> str:
    path = output_root / "images" / "controls" / "blank_512.jpg"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (512, 512), (255, 255, 255)).save(path, format="JPEG", quality=95)
    return str(path)


def _prompt_for_row(source: str, row: pd.Series) -> str:
    question = _strip_image_token(row.get("problem", ""))
    if source == "fsc147":
        meta = _metadata_from_row(row)
        class_name = str(meta.get("class_name") or "objects").strip() or "objects"
        return f"How many {class_name} are there in the image?\nAnswer with only an integer."
    if source in {"gqa", "textvqa"}:
        return f"{question}\nAnswer the question using a single word or phrase."
    if source in {"countqa", "pixmo_count"} and "single number" not in question.lower():
        return f"{question}\nAnswer with a single number."
    if source == "refl4" and "format" not in question.lower():
        return f"{question}\nThe bounding box coordinates should be in the format [x_min, y_min, x_max, y_max]."
    return question


def _record_from_row(source: str, row: pd.Series, image_path: str, mode: str) -> dict[str, Any]:
    answers = _answers_from_row(row)
    return {
        "id": str(row.get("id")),
        "source": source,
        "task_id": int(row.get("task_id", -1)),
        "source_index": int(row.get("source_index", -1)),
        "raw_file": str(row.get("raw_file", "")),
        "raw_row": int(row.get("raw_row", -1)),
        "question": _strip_image_token(row.get("problem", "")),
        "prompt": _prompt_for_row(source, row),
        "answer": answers[0] if answers else "",
        "answers": answers,
        "answer_type": str(row.get("answer_type", "")),
        "problem_type": str(row.get("problem_type", "")),
        "prompt_type": str(row.get("prompt_type", "normal")),
        "metadata": _metadata_from_row(row),
        "image_path": image_path,
        "control_mode": mode,
        "original_image_refs_json": row.get("image_refs_json", "[]"),
    }


def export_source(
    *,
    source: str,
    pass16_root: Path,
    output_root: Path,
    benchmark_root: Path,
    mode: str,
    limit: int | None,
    seed: int,
    allow_missing: bool,
) -> dict[str, Any]:
    rows = _read_source_rows(pass16_root, source, limit)
    image_map = _build_image_map(source, rows, output_root, benchmark_root)
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    real_paths: list[str] = []
    for _, row in rows.iterrows():
        source_index = int(row.get("source_index", -1))
        image_path = _first_file_ref(row) or image_map.get(source_index)
        if not image_path or not Path(str(image_path)).exists():
            missing.append({"id": str(row.get("id")), "source_index": source_index})
            if not allow_missing:
                continue
            image_path = _blank_image_path(output_root)
        real_paths.append(str(image_path))
        records.append(_record_from_row(source, row, str(image_path), "real"))

    if mode == "blank":
        blank = _blank_image_path(output_root)
        for record in records:
            record["image_path"] = blank
            record["control_mode"] = "blank"
    elif mode == "shuffled":
        shuffled = list(real_paths)
        random.Random(seed).shuffle(shuffled)
        for record, image_path in zip(records, shuffled):
            record["image_path"] = image_path
            record["control_mode"] = "shuffled"

    out_path = output_root / "inputs" / f"{source}_{mode}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "source": source,
        "mode": mode,
        "input_rows": int(len(rows)),
        "exported_rows": int(len(records)),
        "missing_images": int(len(missing)),
        "missing_examples": missing[:10],
        "jsonl": str(out_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass16-root", type=Path, default=DEFAULT_PASS16_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--mode", choices=["real", "blank", "shuffled"], default="real")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [source.strip().lower() for source in args.sources.split(",") if source.strip()]
    summaries = []
    for source in sources:
        print(f"[prepare] source={source} mode={args.mode} limit={args.limit or 'all'}", flush=True)
        summary = export_source(
            source=source,
            pass16_root=args.pass16_root,
            output_root=args.output_root,
            benchmark_root=args.benchmark_root,
            mode=args.mode,
            limit=args.limit,
            seed=args.seed,
            allow_missing=args.allow_missing,
        )
        summaries.append(summary)
        print(
            "[prepare] done "
            f"source={source} exported={summary['exported_rows']} missing_images={summary['missing_images']} "
            f"jsonl={summary['jsonl']}",
            flush=True,
        )
    manifest_path = args.output_root / "inputs" / f"manifest_{args.mode}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
