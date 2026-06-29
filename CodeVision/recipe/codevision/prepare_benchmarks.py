#!/usr/bin/env python3
"""Convert local benchmark files into CodeVision eval parquet files."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import html
import io
import json
import os
import re
import shutil
import ssl
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from PIL import Image


WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", "/mnt/cpfs/delinmao"))
BENCHMARK_ROOT = Path(os.getenv("BENCHMARK_ROOT", WORKSPACE_ROOT / "Benchmarks"))


def _sanitize(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "").strip().lower())
    return value.strip("_") or "unknown"


def _first_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    try:
        # numpy arrays from pandas parquet reads behave enough like a sequence.
        if hasattr(value, "tolist"):
            return _first_answer(value.tolist())
    except Exception:
        pass
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Image.Image):
        return f"<PIL.Image size={value.size}>"
    if isinstance(value, (bytes, bytearray)):
        return f"<bytes len={len(value)}>"
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        if hasattr(value, "tolist"):
            return _jsonable(value.tolist())
    except Exception:
        pass
    return value


def _as_plain_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    try:
        if hasattr(value, "tolist"):
            return _as_plain_list(value.tolist())
    except Exception:
        pass
    return [_jsonable(value)]


def _print_inspect(
    *,
    dataset_name: str,
    splits: list[str],
    raw_count: int,
    converted_count: int,
    raw_examples: list[Any],
    converted_examples: list[dict[str, Any]],
) -> None:
    from collections import Counter

    print(f"dataset={dataset_name}")
    print(f"splits={splits}")
    print(f"raw_row_count={raw_count}")
    print(f"converted_sample_count={converted_count}")
    print("raw_examples=")
    print(json.dumps(_jsonable(raw_examples), ensure_ascii=False, indent=2)[:8000])
    print("converted_examples=")
    print(json.dumps(_jsonable(converted_examples), ensure_ascii=False, indent=2)[:8000])
    source_counts = Counter(str(r.get("data_source")) for r in converted_examples)
    benchmark_counts = Counter(str((r.get("extra_info") or {}).get("source_benchmark")) for r in converted_examples)
    image_exists = [
        Path((r.get("extra_info") or {}).get("image_path", "")).exists()
        for r in converted_examples
    ]
    print(f"data_source_distribution={dict(source_counts)}")
    print(f"source_benchmark_distribution={dict(benchmark_counts)}")
    print(f"image_path_exists_first_examples={image_exists}")


def _image_bytes(image_value: Any) -> bytes:
    if isinstance(image_value, Image.Image):
        image = image_value.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format=image_value.format or "PNG")
        return buffer.getvalue()
    if isinstance(image_value, dict):
        if image_value.get("bytes") is not None:
            return image_value["bytes"]
        path = image_value.get("path")
        if path:
            return Path(path).read_bytes()
    if isinstance(image_value, (bytes, bytearray)):
        return bytes(image_value)
    if isinstance(image_value, str):
        value = image_value.strip()
        if value.startswith("data:image/") and "," in value:
            value = value.split(",", 1)[1].strip()
        try:
            # HR-Bench stores images as raw base64 strings in parquet.
            return base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError):
            pass
        try:
            path = Path(value)
            if path.exists():
                return path.read_bytes()
        except OSError:
            pass
    raise ValueError(f"Unsupported image value: {type(image_value)!r}")


def _save_image(image_value: Any, output_dir: Path, stem: str) -> tuple[Path, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = _image_bytes(image_value)
    return _save_image_bytes(raw, output_dir, stem)


def _save_image_bytes(raw: bytes, output_dir: Path, stem: str) -> tuple[Path, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        width, height = image.size
        ext = (image.format or "PNG").lower()
        if ext == "jpeg":
            ext = "jpg"
        out_path = output_dir / f"{stem}.{ext}"
    out_path.write_bytes(raw)
    return out_path.resolve(), width, height


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_existing_image(output_dir: Path, stem: str) -> tuple[Path, int, int] | None:
    matches = sorted(output_dir.glob(f"{stem}.*"))
    if not matches:
        return None
    image_path = matches[0]
    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
    return image_path.resolve(), width, height


def _existing_image_path(output_dir: Path, stem: str) -> Path | None:
    matches = sorted(output_dir.glob(f"{stem}.*"))
    return matches[0].resolve() if matches else None


def _save_image_url(url: str, output_dir: Path, stem: str, timeout: int) -> tuple[Path, int, int]:
    url = html.unescape(url)
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "KHTML, like Gecko Chrome/120.0 Safari/537.36"
            )
        },
    )
    context = ssl._create_unverified_context()
    with urlopen(request, timeout=timeout, context=context) as response:
        raw = response.read()
    return _save_image_bytes(raw, output_dir, stem)


def _image_uri(image_path: Path) -> str:
    return f"file://{quote(str(image_path))}"


def _sanitize_question_text(question: str) -> str:
    return str(question).replace("<image>", "").strip()


def _build_prompt(question: str, width: int, height: int) -> str:
    if width > 0 and height > 0:
        return f"<image>Image size = {width}x{height} pixels.\n\n{_sanitize_question_text(question)}"
    return f"<image>\n\n{_sanitize_question_text(question)}"


def _record(
    *,
    data_source: str,
    question: str,
    answer: Any,
    image_path: Path,
    width: int,
    height: int,
    index: int,
    source_benchmark: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    question = _sanitize_question_text(question)
    extra_info = {
        "index": index,
        "question": question,
        "image_path": str(image_path),
        "source_benchmark": source_benchmark,
    }
    if extra:
        extra_info.update(extra)

    return {
        "data_source": data_source,
        "ability": "mm_qa",
        "prompt": [{"role": "user", "content": _build_prompt(question, width, height)}],
        "images": [{"image": _image_uri(image_path)}],
        "reward_model": {
            "style": "rule",
            "ground_truth": _first_answer(answer),
        },
        "extra_info": extra_info,
    }


def _load_fsc147_class_map(path: Path) -> dict[str, str]:
    class_map: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                filename, class_name = text.split("\t", 1)
            except ValueError:
                continue
            class_map[filename.strip()] = class_name.strip()
    return class_map


def _write_parquet(records: list[dict[str, Any]], output_path: Path) -> None:
    import pandas as pd

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output_path, index=False)
    print(f"Wrote {len(records)} records to {output_path}")


def _hf_cache_dir() -> str:
    return os.getenv("HF_HUB_CACHE", str(WORKSPACE_ROOT / "cache" / "hf" / "hub"))


def _hf_endpoint() -> str | None:
    return os.getenv("HF_ENDPOINT", None)


def _hf_dataset_file(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        endpoint=_hf_endpoint(),
        cache_dir=_hf_cache_dir(),
    )


def _hf_dataset_snapshot(repo_id: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            endpoint=_hf_endpoint(),
            cache_dir=_hf_cache_dir(),
        )
    )


def _load_hf_dataset(repo_id: str, *, split: str, name: str | None = None):
    import datasets

    kwargs = {"split": split, "cache_dir": _hf_cache_dir()}
    token_kwargs_options = []
    if os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"):
        token_kwargs_options.extend([{"token": True}, {"use_auth_token": True}])
    token_kwargs_options.append({})
    for token_kwargs in token_kwargs_options:
        try:
            return datasets.load_dataset(repo_id, name, **kwargs, **token_kwargs)
        except TypeError:
            continue
        except Exception as exc:
            if token_kwargs and "Token is required" in str(exc):
                continue
            raise


def _read_hf_parquets(repo_id: str, filenames: list[str]):
    import pandas as pd

    paths = [_hf_dataset_file(repo_id, filename) for filename in filenames]
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _letter_for_index(index: int) -> str:
    return chr(ord("A") + index)


def _normalize_choice_answer(raw_answer: Any, choices: list[Any]) -> tuple[str, str]:
    raw = _first_answer(raw_answer).strip()
    raw_clean = raw.strip().strip("()").strip()
    if re.fullmatch(r"[A-Za-z]", raw_clean):
        return raw_clean.upper(), raw
    if re.fullmatch(r"\d+", raw_clean):
        value = int(raw_clean)
        if 0 <= value < len(choices):
            return _letter_for_index(value), raw
        if 1 <= value <= len(choices):
            return _letter_for_index(value - 1), raw
    raw_norm = re.sub(r"\s+", " ", raw_clean).casefold()
    for idx, choice in enumerate(choices):
        choice_norm = re.sub(r"\s+", " ", str(choice).strip()).casefold()
        if raw_norm == choice_norm:
            return _letter_for_index(idx), raw
    raise ValueError(f"Cannot normalize answer {raw_answer!r} against choices={choices!r}")


def _format_options(choices: list[Any]) -> str:
    return "\n".join(f"({_letter_for_index(idx)}) {choice}" for idx, choice in enumerate(choices))


def _cvbench_bucket(source: Any, task: Any, type_value: Any) -> str:
    text = " ".join(str(v or "") for v in [source, task, type_value]).lower()
    if "ade" in text:
        return "cvbench_2d_ade20k"
    if "coco" in text:
        return "cvbench_2d_coco"
    if "omni" in text or "3d" in str(type_value or "").lower():
        return "cvbench_3d_omni3d"
    raise ValueError(f"Unknown CV-Bench bucket: source={source!r} task={task!r} type={type_value!r}")


def convert_cvbench(root: Path, *, inspect: bool = False, limit: int | None = None, inspect_limit: int = 3) -> Path:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    benchmark_dir = root / "CV-Bench"
    output_path = benchmark_dir / "cvbench_codevision_eval.parquet"
    image_dir = benchmark_dir / "codevision_images"
    endpoint = os.getenv("HF_ENDPOINT", None)
    cache_dir = os.getenv("HF_HUB_CACHE", str(WORKSPACE_ROOT / "cache" / "hf" / "hub"))
    paths = [
        hf_hub_download(
            repo_id="nyu-visionx/CV-Bench",
            repo_type="dataset",
            filename=filename,
            endpoint=endpoint,
            cache_dir=cache_dir,
        )
        for filename in ["test_2d.parquet", "test_3d.parquet"]
    ]
    df = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    records = []
    raw_examples = []
    converted_examples = []
    max_rows = len(df) if limit is None else min(limit, len(df))
    for idx in range(max_rows):
        row = df.iloc[idx].to_dict()
        if len(raw_examples) < inspect_limit:
            raw_examples.append(row)
        choices = _as_plain_list(row["choices"])
        answer, raw_answer = _normalize_choice_answer(row["answer"], choices)
        bucket = _cvbench_bucket(row.get("source"), row.get("task"), row.get("type"))
        image_path, width, height = _save_image(row["image"], image_dir, f"cvbench_test_{idx:05d}")
        prompt = str(row.get("prompt") or "").strip()
        if not prompt:
            prompt = f"{row['question']}\n{_format_options(choices)}\nAnswer with the option letter only."
        record = _record(
            data_source=bucket,
            question=prompt,
            answer=answer,
            image_path=image_path,
            width=width,
            height=height,
            index=idx,
            source_benchmark="cvbench",
            extra={
                "dataset_path": "nyu-visionx/CV-Bench",
                "source_split": "test",
                "eval_protocol": "official_aggregation",
                "scorer_status": "implemented",
                "raw_answer": raw_answer,
                "choices": choices,
                "source": str(row.get("source") or ""),
                "task": str(row.get("task") or ""),
                "type": str(row.get("type") or ""),
            },
        )
        records.append(record)
        if len(converted_examples) < inspect_limit:
            converted_examples.append(record)
    if inspect:
        _print_inspect(
            dataset_name="nyu-visionx/CV-Bench",
            splits=["test"],
            raw_count=len(df),
            converted_count=len(records),
            raw_examples=raw_examples,
            converted_examples=converted_examples,
        )
        return output_path
    _write_parquet(records, output_path)
    return output_path


def convert_pixmo_count(
    root: Path,
    *,
    inspect: bool = False,
    limit: int | None = None,
    inspect_limit: int = 3,
    download_timeout: int = 30,
) -> Path:
    import datasets

    benchmark_dir = root / "Pixmo-Count"
    output_path = benchmark_dir / "pixmo_count_codevision_eval.parquet"
    image_dir = benchmark_dir / "codevision_images"
    failed_path = benchmark_dir / "failed_downloads.jsonl"
    import pandas as pd
    from huggingface_hub import hf_hub_download

    endpoint = os.getenv("HF_ENDPOINT", None)
    cache_dir = os.getenv("HF_HUB_CACHE", str(WORKSPACE_ROOT / "cache" / "hf" / "hub"))
    input_path = hf_hub_download(
        repo_id="allenai/pixmo-count",
        repo_type="dataset",
        filename="data/test-00000-of-00001.parquet",
        endpoint=endpoint,
        cache_dir=cache_dir,
    )
    df = pd.read_parquet(input_path)
    records = []
    failures = []
    sha_mismatches = []
    raw_examples = []
    converted_examples = []
    max_rows = len(df) if limit is None else min(limit, len(df))
    for idx in range(max_rows):
        row = df.iloc[idx].to_dict()
        if len(raw_examples) < inspect_limit:
            raw_examples.append(row)
        stem = f"pixmo_count_test_{idx:05d}"
        expected_sha = str(row.get("image_sha256") or "").strip().lower()
        actual_sha = ""
        sha_mismatch = False
        try:
            cached = _load_existing_image(image_dir, stem)
            if cached is None:
                image_path, width, height = _save_image_url(str(row["image_url"]), image_dir, stem, download_timeout)
            else:
                image_path, width, height = cached
            actual_sha = _sha256_file(image_path).lower()
            if expected_sha and actual_sha != expected_sha:
                sha_mismatch = True
                sha_mismatches.append(
                    {
                        "index": idx,
                        "image_url": row.get("image_url"),
                        "expected_sha256": expected_sha,
                        "actual_sha256": actual_sha,
                        "image_path": str(image_path),
                    }
                )
        except Exception as exc:
            failures.append({"index": idx, "image_url": row.get("image_url"), "error": str(exc)})
            continue

        label = str(row.get("label") or "objects").strip()
        question = f"How many {label} are there in the image? Answer with a single number."
        record = _record(
            data_source="pixmo_count",
            question=question,
            answer=row["count"],
            image_path=image_path,
            width=width,
            height=height,
            index=idx,
            source_benchmark="pixmo_count",
            extra={
                "dataset_path": "allenai/pixmo-count",
                "source_split": "test",
                "eval_protocol": "official_data_aligned",
                "scorer_status": "implemented",
                "label": label,
                "count": row["count"],
                "image_url": row.get("image_url"),
                "image_sha256": row.get("image_sha256"),
                "downloaded_image_sha256": actual_sha,
                "sha_mismatch": sha_mismatch,
                "points": _jsonable(row.get("points")),
            },
        )
        records.append(record)
        if len(converted_examples) < inspect_limit:
            converted_examples.append(record)

    if failures:
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        with failed_path.open("w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
        print(f"Logged {len(failures)} Pixmo-Count image failures to {failed_path}")
    if sha_mismatches:
        mismatch_path = benchmark_dir / "sha_mismatches.jsonl"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        with mismatch_path.open("w", encoding="utf-8") as handle:
            for item in sha_mismatches:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Logged {len(sha_mismatches)} Pixmo-Count sha mismatches to {mismatch_path}")
    print(f"Pixmo-Count final evaluated N={len(records)} of requested N={max_rows}")

    if inspect:
        _print_inspect(
            dataset_name="allenai/pixmo-count",
            splits=["test"],
            raw_count=len(df),
            converted_count=len(records),
            raw_examples=raw_examples,
            converted_examples=converted_examples,
        )
        return output_path
    _write_parquet(records, output_path)
    return output_path


def convert_pixmo_count_lmms(
    root: Path,
    *,
    inspect: bool = False,
    limit: int | None = None,
    inspect_limit: int = 3,
) -> Path:
    benchmark_dir = root / "Pixmo-Count-LMMS"
    output_path = benchmark_dir / "pixmo_count_lmms_codevision_eval.parquet"
    image_dir = benchmark_dir / "codevision_images"
    df = _read_hf_parquets("kcz358/pixmo-count", ["data/test-00000-of-00001.parquet"])
    records = []
    raw_examples = []
    converted_examples = []
    max_rows = len(df) if limit is None else min(limit, len(df))
    for idx in range(max_rows):
        row = df.iloc[idx].to_dict()
        if len(raw_examples) < inspect_limit:
            raw_examples.append(row)
        image_path, width, height = _save_image(row["image"], image_dir, f"pixmo_count_lmms_test_{idx:05d}")
        question = str(row["question"]).strip()
        if "single number" not in question.lower():
            question = f"{question}\nAnswer with a single number."
        record = _record(
            data_source="pixmo_count_lmms",
            question=question,
            answer=row["answer"],
            image_path=image_path,
            width=width,
            height=height,
            index=idx,
            source_benchmark="pixmo_count_lmms",
            extra={
                "dataset_path": "kcz358/pixmo-count",
                "canonical_dataset_path": "allenai/pixmo-count",
                "source_split": "test",
                "eval_protocol": "lmms_mirror_subset",
                "scorer_status": "implemented",
                "mirror_note": "QA mirror/subset used by lmms-style evaluation; 534 rows vs 540 official allenai/pixmo-count test rows.",
                "id": row.get("id"),
                "raw_answer": row.get("answer"),
            },
        )
        records.append(record)
        if len(converted_examples) < inspect_limit:
            converted_examples.append(record)
    if inspect:
        _print_inspect(
            dataset_name="kcz358/pixmo-count",
            splits=["test"],
            raw_count=len(df),
            converted_count=len(records),
            raw_examples=raw_examples,
            converted_examples=converted_examples,
        )
        return output_path
    _write_parquet(records, output_path)
    return output_path


def convert_ocrbench_v2(root: Path, *, inspect: bool = False, limit: int | None = None, inspect_limit: int = 3) -> Path:
    benchmark_dir = root / "OCRBench_v2"
    output_path = benchmark_dir / "ocrbench_v2_codevision_eval.parquet"
    image_dir = benchmark_dir / "codevision_images"
    df = _read_hf_parquets(
        "ling99/OCRBench_v2",
        [
            "data/test-00000-of-00003.parquet",
            "data/test-00001-of-00003.parquet",
            "data/test-00002-of-00003.parquet",
        ],
    )
    records = []
    raw_examples = []
    converted_examples = []
    max_rows = len(df) if limit is None else min(limit, len(df))
    for idx in range(max_rows):
        row = df.iloc[idx].to_dict()
        if len(raw_examples) < inspect_limit:
            raw_examples.append(row)
        image_path, width, height = _save_image(row["image"], image_dir, f"ocrbench_v2_test_{idx:05d}")
        data_type = str(row.get("type") or "unknown")
        lang = "cn" if data_type.endswith(" cn") or " cn" in data_type.lower() else "en"
        record = _record(
            data_source=f"ocrbench_v2_{lang}",
            question=str(row["question"]),
            answer=row.get("answers"),
            image_path=image_path,
            width=width,
            height=height,
            index=idx,
            source_benchmark="ocrbench_v2",
            extra={
                "dataset_path": "ling99/OCRBench_v2",
                "source_split": "test",
                "eval_protocol": "official_scorer",
                "scorer_status": "implemented_lmms_aligned",
                "ocrbench_v2_id": row.get("id"),
                "dataset_name": row.get("dataset_name"),
                "type": data_type,
                "eval": row.get("eval"),
                "answers": row.get("answers"),
                "bbox": _jsonable(row.get("bbox")),
                "bbox_list": _jsonable(row.get("bbox_list")),
                "content": _jsonable(row.get("content")),
            },
        )
        records.append(record)
        if len(converted_examples) < inspect_limit:
            converted_examples.append(record)
    if inspect:
        _print_inspect(
            dataset_name="ling99/OCRBench_v2",
            splits=["test"],
            raw_count=len(df),
            converted_count=len(records),
            raw_examples=raw_examples,
            converted_examples=converted_examples,
        )
        return output_path
    _write_parquet(records, output_path)
    return output_path


def convert_spatialmqa(
    root: Path,
    *,
    inspect: bool = False,
    limit: int | None = None,
    inspect_limit: int = 3,
    coco2017_test_image_dir: Path | None = None,
) -> Path:
    image_root_value = coco2017_test_image_dir or os.getenv("COCO2017_TEST_IMAGE_DIR", "")
    benchmark_dir = root / "SpatialMQA"
    output_path = benchmark_dir / "spatialmqa_codevision_eval.parquet"
    output_image_dir = benchmark_dir / "codevision_images"
    if image_root_value:
        image_root = Path(image_root_value)
        if not image_root.is_dir():
            raise FileNotFoundError(f"COCO2017 test image directory not found: {image_root}")
        dataset_path = _hf_dataset_file("liuziyan/SpatialMQA", "test.jsonl")
    else:
        snapshot_dir = _hf_dataset_snapshot("liuziyan/SpatialMQA")
        image_root = snapshot_dir / "images"
        dataset_path = snapshot_dir / "test.jsonl"
        if not image_root.is_dir():
            raise FileNotFoundError(f"SpatialMQA HF snapshot has no images directory: {image_root}")
    with Path(dataset_path).open("r", encoding="utf-8") as handle:
        dataset = [json.loads(line) for line in handle if line.strip()]
    records = []
    raw_examples = []
    converted_examples = []
    missing_images = []
    max_rows = len(dataset) if limit is None else min(limit, len(dataset))
    for idx in range(max_rows):
        row = dataset[idx]
        if len(raw_examples) < inspect_limit:
            raw_examples.append(row)
        image_name = str(row.get("image") or row.get("image_name") or "")
        image_path = (image_root / image_name).resolve()
        if not image_path.exists() and image_name and not image_name.startswith("COCO_"):
            coco_name = f"COCO_test2017_{image_name}"
            alt_path = (image_root / coco_name).resolve()
            if alt_path.exists():
                image_path = alt_path
        if not image_path.exists():
            missing_images.append(image_name)
            continue
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            ext = (image.format or image_path.suffix.lstrip(".") or "jpg").lower()
        if ext == "jpeg":
            ext = "jpg"
        output_image_dir.mkdir(parents=True, exist_ok=True)
        local_image_path = (output_image_dir / f"spatialmqa_test_{idx:05d}.{ext}").resolve()
        if not local_image_path.exists():
            shutil.copy2(image_path, local_image_path)
        options = list(row["options"])
        answer, raw_answer = _normalize_choice_answer(row["answer"], options)
        question = f"{row['question']}\n{_format_options(options)}\nAnswer with the option letter only."
        record = _record(
            data_source="spatialmqa",
            question=question,
            answer=answer,
            image_path=local_image_path,
            width=width,
            height=height,
            index=idx,
            source_benchmark="spatialmqa",
            extra={
                "dataset_path": "liuziyan/SpatialMQA",
                "source_split": "test",
                "eval_protocol": "official_data_aligned",
                "scorer_status": "implemented",
                "image_name": image_name,
                "source_image_path": str(image_path),
                "options": options,
                "answer_text": raw_answer,
            },
        )
        records.append(record)
        if len(converted_examples) < inspect_limit:
            converted_examples.append(record)
    if missing_images:
        preview = ", ".join(missing_images[:10])
        raise FileNotFoundError(f"SpatialMQA missing {len(missing_images)} COCO images. First missing: {preview}")
    if inspect:
        _print_inspect(
            dataset_name="liuziyan/SpatialMQA",
            splits=["test"],
            raw_count=len(dataset),
            converted_count=len(records),
            raw_examples=raw_examples,
            converted_examples=converted_examples,
        )
        return output_path
    _write_parquet(records, output_path)
    return output_path


def _iter_countqa_pairs(row: dict[str, Any]) -> list[tuple[str, Any]]:
    if "QA" in row and row["QA"] is not None:
        pairs = []
        for qa in row["QA"]:
            pairs.append((str(qa["question"]), qa["answer"]))
        return pairs
    if "questions" in row and "answers" in row:
        return [(str(q), a) for q, a in zip(row["questions"], row["answers"])]
    if "question" in row and "answer" in row:
        return [(str(row["question"]), row["answer"])]
    raise KeyError(f"Unsupported CountQA row fields: {sorted(row.keys())}")


def convert_countqa(root: Path, *, inspect: bool = False, limit: int | None = None, inspect_limit: int = 3) -> Path:
    benchmark_dir = root / "CountQA"
    output_path = benchmark_dir / "countqa_codevision_eval.parquet"
    image_dir = benchmark_dir / "codevision_images"
    df = _read_hf_parquets(
        "Jayant-Sravan/CountQA",
        [f"data/test-{idx:05d}-of-00013.parquet" for idx in range(13)],
    )
    records = []
    raw_examples = []
    converted_examples = []
    max_rows = len(df) if limit is None else min(limit, len(df))
    for image_idx in range(max_rows):
        row = df.iloc[image_idx].to_dict()
        if len(raw_examples) < inspect_limit:
            raw_examples.append(row)
        image_value = row.get("image", None)
        if image_value is None:
            image_value = row.get("images", None)
        if image_value is None:
            raise KeyError(f"CountQA row {image_idx} has no image field. Fields: {sorted(row.keys())}")
        image_path, width, height = _save_image(image_value, image_dir, f"countqa_test_{image_idx:05d}")
        pairs = _iter_countqa_pairs(row)
        for qa_idx, (question_text, answer) in enumerate(pairs):
            question = f"{question_text.strip()}\nAnswer with a single number."
            record = _record(
                data_source="countqa",
                question=question,
                answer=answer,
                image_path=image_path,
                width=width,
                height=height,
                index=len(records),
                source_benchmark="countqa",
                extra={
                    "dataset_path": "Jayant-Sravan/CountQA",
                    "source_split": "test",
                    "eval_protocol": "official_data_aligned",
                    "scorer_status": "implemented",
                    "image_index": image_idx,
                    "qa_index": qa_idx,
                    "objects": row.get("objects"),
                    "categories": row.get("categories"),
                    "focused": row.get("focused", row.get("is_focused")),
                    "available_columns": sorted(row.keys()),
                },
            )
            records.append(record)
            if len(converted_examples) < inspect_limit:
                converted_examples.append(record)
    if inspect:
        _print_inspect(
            dataset_name="Jayant-Sravan/CountQA",
            splits=["test"],
            raw_count=len(df),
            converted_count=len(records),
            raw_examples=raw_examples,
            converted_examples=converted_examples,
        )
        return output_path
    _write_parquet(records, output_path)
    return output_path


def convert_chartqa(root: Path) -> Path:
    import pandas as pd

    input_path = next((root / "ChartQA" / "data").glob("test-*.parquet"))
    output_path = root / "ChartQA" / "chartqa_codevision_eval.parquet"
    image_dir = root / "ChartQA" / "codevision_images"
    df = pd.read_parquet(input_path)
    records = []
    for idx, row in df.iterrows():
        image_path, width, height = _save_image(row["image"], image_dir, f"chartqa_test_{idx:05d}")
        split = "human" if int(row.get("human_or_machine", 0)) == 0 else "machine"
        records.append(
            _record(
                data_source=f"chartqa_{split}",
                question=str(row["query"]),
                answer=row["label"],
                image_path=image_path,
                width=width,
                height=height,
                index=idx,
                source_benchmark="chartqa",
                extra={"human_or_machine": split},
            )
        )
    _write_parquet(records, output_path)
    return output_path


def convert_mvtoolbench(root: Path, *, inspect: bool = False, limit: int | None = None, inspect_limit: int = 3) -> Path:
    import datasets

    benchmark_dir = root / "MVToolBench"
    output_path = benchmark_dir / "mvtoolbench_codevision_eval.parquet"
    image_dir = benchmark_dir / "codevision_images"

    dataset = datasets.load_dataset("kkwok/MVToolBench", split="train", cache_dir=_hf_cache_dir())
    if limit is not None:
        dataset = dataset.select(range(min(int(limit), len(dataset))))

    records: list[dict[str, Any]] = []
    raw_examples: list[Any] = []
    for idx, row in enumerate(dataset):
        if idx < inspect_limit:
            raw_examples.append(row)

        images = row.get("images") or []
        if not images:
            raise ValueError(f"MVToolBench row {idx} has no images")
        image_path, width, height = _save_image(images[0], image_dir, f"mvtoolbench_train_{idx:05d}")

        extra_info = row.get("extra_info") or {}
        prompt_items = row.get("prompt") or []
        question = extra_info.get("question")
        if not question and prompt_items:
            question = prompt_items[0].get("content", "")

        reward_model = row.get("reward_model") or {}
        answer = reward_model.get("ground_truth") or extra_info.get("answer", "")
        source_name = str(row.get("data_source") or extra_info.get("dataset_name") or "unknown")

        records.append(
            _record(
                data_source=f"mvtoolbench_{_sanitize(source_name)}",
                question=str(question or ""),
                answer=answer,
                image_path=image_path,
                width=width,
                height=height,
                index=idx,
                source_benchmark="mvtoolbench",
                extra={
                    "dataset_path": "kkwok/MVToolBench",
                    "split": str(extra_info.get("split") or "train"),
                    "source_data_source": source_name,
                    "dataset_name": str(extra_info.get("dataset_name") or ""),
                    "source_index": int(extra_info.get("index", idx)),
                    "answer": str(extra_info.get("answer") or answer or ""),
                    "bbox": _as_plain_list(extra_info.get("bbox")),
                    "transform": _as_plain_list(extra_info.get("transform")),
                    "eval_protocol": "official_data_aligned",
                    "scorer_status": "implemented",
                    "metric_note": "Exact answer matching through CodeVision reward; required transforms are preserved for tool-process diagnostics.",
                },
            )
        )

    if inspect:
        _print_inspect(
            dataset_name="kkwok/MVToolBench",
            splits=["train"],
            raw_count=len(dataset),
            converted_count=len(records),
            raw_examples=raw_examples,
            converted_examples=records[:inspect_limit],
        )
        return output_path

    _write_parquet(records, output_path)
    return output_path


def convert_ocrbench(root: Path) -> Path:
    import pandas as pd

    input_path = root / "OCRBench" / "data" / "test-00000-of-00001.parquet"
    output_path = root / "OCRBench" / "ocrbench_codevision_eval.parquet"
    image_dir = root / "OCRBench" / "codevision_images"
    df = pd.read_parquet(input_path)
    records = []
    for idx, row in df.iterrows():
        image_path, width, height = _save_image(row["image"], image_dir, f"ocrbench_test_{idx:05d}")
        question_type = str(row.get("question_type") or "ocr")
        records.append(
            _record(
                data_source=f"ocrbench_{_sanitize(question_type)}",
                question=str(row["question"]),
                answer=row["answer"],
                image_path=image_path,
                width=width,
                height=height,
                index=idx,
                source_benchmark="ocrbench",
                extra={
                    "dataset": str(row.get("dataset") or ""),
                    "question_type": question_type,
                },
            )
        )
    _write_parquet(records, output_path)
    return output_path


def convert_countbench(root: Path, *, download_missing_images: bool = True, download_timeout: int = 30) -> Path:
    import datasets

    benchmark_dir = root / "countbench"
    output_path = root / "countbench" / "countbench_codevision_eval.parquet"
    image_dir = root / "countbench" / "codevision_images"
    dataset = datasets.load_dataset("vikhyatk/CountBenchQA", split="test")
    records = []
    for idx, row in enumerate(dataset):
        image_path, width, height = _save_image(row["image"], image_dir, f"countbenchqa_test_{idx:05d}")
        question = (
            "Look at the image carefully and count the objects. "
            "Answer with just a number, without any additional text. "
            f"{str(row['question']).strip()}"
        )
        text = str(row.get("text") or "").strip()
        records.append(
            _record(
                data_source="countbench",
                question=question,
                answer=row["number"],
                image_path=image_path,
                width=width,
                height=height,
                index=idx,
                source_benchmark="countbench",
                extra={
                    "caption": text,
                    "dataset_path": "vikhyatk/CountBenchQA",
                    "source_split": "test",
                },
            )
        )
    _write_parquet(records, output_path)
    return output_path


def convert_fsc147(root: Path) -> list[Path]:
    import json

    benchmark_dir = root / "FSC147"
    split_path = benchmark_dir / "Train_Test_Val_FSC_147.json"
    annotation_path = benchmark_dir / "annotation_FSC147_384.json"
    class_path = benchmark_dir / "ImageClasses_FSC147.txt"
    image_dir = benchmark_dir / "images_384_VarV2"
    if not split_path.exists() or not annotation_path.exists() or not class_path.exists():
        raise FileNotFoundError(
            "FSC147 requires Train_Test_Val_FSC_147.json, annotation_FSC147_384.json, "
            f"and ImageClasses_FSC147.txt under {benchmark_dir}"
        )
    if not image_dir.is_dir():
        raise FileNotFoundError(f"FSC147 image directory not found: {image_dir}")

    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    class_map = _load_fsc147_class_map(class_path)

    outputs = []
    for split_name in ["val", "test"]:
        records = []
        missing_images: list[str] = []
        filenames = split_data.get(split_name) or []
        for idx, filename in enumerate(filenames):
            filename = str(filename)
            image_path = (image_dir / filename).resolve()
            if not image_path.exists():
                missing_images.append(filename)
                continue
            ann = annotations.get(filename)
            if ann is None:
                raise KeyError(f"Missing FSC147 annotation for {filename}")
            with Image.open(image_path) as image:
                width, height = image.size
            class_name = class_map.get(filename, "objects")
            count = len(ann.get("points") or [])
            question = f"How many {class_name} are there in the image?\nAnswer with only an integer."
            records.append(
                _record(
                    data_source=f"fsc147_{split_name}",
                    question=question,
                    answer=count,
                    image_path=image_path,
                    width=width,
                    height=height,
                    index=idx,
                    source_benchmark="fsc147",
                    extra={
                        "source_split": split_name,
                        "source_filename": filename,
                        "class_name": class_name,
                        "count": count,
                    },
                )
            )
        if missing_images:
            preview = ", ".join(missing_images[:10])
            raise FileNotFoundError(
                f"FSC147 {split_name} is missing {len(missing_images)} images under {image_dir}. "
                f"First missing: {preview}"
            )
        output_path = benchmark_dir / f"fsc147_{split_name}_codevision_eval.parquet"
        _write_parquet(records, output_path)
        outputs.append(output_path)
    return outputs


def convert_hrbench(root: Path) -> list[Path]:
    import pandas as pd

    benchmark_dir = root / "HR-Bench"
    input_paths = sorted(benchmark_dir.glob("hr_bench_*.parquet"))
    if not input_paths:
        print(f"Skipping HR-Bench: no hr_bench_*.parquet files under {benchmark_dir}")
        return []

    outputs = []
    image_dir = benchmark_dir / "codevision_images"
    for input_path in input_paths:
        split_name = input_path.stem
        output_path = benchmark_dir / f"{split_name}_codevision_eval.parquet"
        df = pd.read_parquet(input_path)
        records = []
        for idx, row in df.iterrows():
            image_path, width, height = _save_image(row["image"], image_dir, f"{split_name}_{idx:05d}")
            choices = "\n".join(f"({letter}) {row[letter]}" for letter in ["A", "B", "C", "D"])
            question = f"{row['question']}\n{choices}\nAnswer with the option's letter from the given choices."
            records.append(
                _record(
                    data_source=f"hrbench_{_sanitize(row.get('category') or split_name)}",
                    question=question,
                    answer=row["answer"],
                    image_path=image_path,
                    width=width,
                    height=height,
                    index=idx,
                    source_benchmark="hrbench",
                    extra={
                        "category": str(row.get("category") or ""),
                        "cycle_category": str(row.get("cycle_category") or ""),
                        "source_index": str(row.get("index") or idx),
                        "split": split_name,
                        **{f"option_{letter}": str(row[letter]) for letter in ["A", "B", "C", "D"]},
                    },
                )
            )
        _write_parquet(records, output_path)
        outputs.append(output_path)
    return outputs


def _limit_dataset(dataset: Any, limit: int | None):
    if limit is None:
        return dataset
    return dataset.select(range(min(int(limit), len(dataset))))


def _mme_realworld_prompt(row: dict[str, Any], *, cn: bool) -> str:
    question = str(row["question"]).strip()
    choices = [str(choice) for choice in _as_plain_list(row.get("multi-choice options"))]
    if cn:
        return (
            f"{question} 选项如下所示:\n"
            + "\n".join(choices)
            + "\n根据图像选择上述多项选择题的最佳答案。只需回答正确选项的字母（A, B, C, D 或 E）。\n最佳答案为："
        )
    return (
        f"{question} The choices are listed below:\n"
        + "\n".join(choices)
        + "\nSelect the best answer to the above multiple-choice question based on the image. "
        "Respond with only the letter (A, B, C, D, or E) of the correct option.\nThe best answer is:"
    )


def _convert_mme_realworld_variant(
    root: Path,
    *,
    repo_id: str,
    benchmark_name: str,
    directory_name: str,
    cn: bool = False,
    inspect: bool = False,
    limit: int | None = None,
    inspect_limit: int = 3,
) -> Path:
    benchmark_dir = root / directory_name
    output_path = benchmark_dir / f"{benchmark_name}_codevision_eval.parquet"
    image_dir = benchmark_dir / "codevision_images"
    dataset = _limit_dataset(_load_hf_dataset(repo_id, split="train"), limit)
    records = []
    raw_examples = []
    converted_examples = []
    for idx, row in enumerate(dataset):
        row = dict(row)
        if idx < inspect_limit:
            raw_examples.append(row)
        image_path, width, height = _save_image(row["bytes"], image_dir, f"{benchmark_name}_train_{idx:05d}")
        options = [str(choice) for choice in _as_plain_list(row.get("multi-choice options"))]
        category = str(row.get("category") or "")
        task_category = str(row.get("l2-category") or "")
        high_level = "perception" if "perception" in category.lower() else "reasoning"
        record = _record(
            data_source=f"{benchmark_name}_{_sanitize(high_level)}",
            question=_mme_realworld_prompt(row, cn=cn),
            answer=row["answer"],
            image_path=image_path,
            width=width,
            height=height,
            index=idx,
            source_benchmark=benchmark_name,
            extra={
                "dataset_path": repo_id,
                "source_split": "train",
                "eval_protocol": "lmms_eval_aligned_sample_accuracy",
                "scorer_status": "implemented",
                "mme_realworld_index": row.get("index", idx),
                "category": category,
                "sub_category": category.split("/")[-1] if category else "",
                "task_category": task_category,
                "options": options,
                "raw_answer": row.get("answer"),
                "aggregation_note": "Official lmms-eval aggregation is sample-weighted overall accuracy after A-E extraction.",
            },
        )
        records.append(record)
        if len(converted_examples) < inspect_limit:
            converted_examples.append(record)
    if inspect:
        _print_inspect(
            dataset_name=repo_id,
            splits=["train"],
            raw_count=len(dataset),
            converted_count=len(records),
            raw_examples=raw_examples,
            converted_examples=converted_examples,
        )
        return output_path
    _write_parquet(records, output_path)
    return output_path


def convert_mme_realworld(root: Path, *, inspect: bool = False, limit: int | None = None, inspect_limit: int = 3) -> Path:
    return _convert_mme_realworld_variant(
        root,
        repo_id="yifanzhang114/MME-RealWorld-Lmms-eval",
        benchmark_name="mme_realworld",
        directory_name="MME-RealWorld",
        inspect=inspect,
        limit=limit,
        inspect_limit=inspect_limit,
    )


def convert_mme_realworld_lite(root: Path, *, inspect: bool = False, limit: int | None = None, inspect_limit: int = 3) -> Path:
    return _convert_mme_realworld_variant(
        root,
        repo_id="yifanzhang114/MME-RealWorld-lite-lmms-eval",
        benchmark_name="mme_realworld_lite",
        directory_name="MME-RealWorld-Lite",
        inspect=inspect,
        limit=limit,
        inspect_limit=inspect_limit,
    )


def convert_mme_realworld_cn(root: Path, *, inspect: bool = False, limit: int | None = None, inspect_limit: int = 3) -> Path:
    return _convert_mme_realworld_variant(
        root,
        repo_id="yifanzhang114/MME-RealWorld-CN-Lmms-eval",
        benchmark_name="mme_realworld_cn",
        directory_name="MME-RealWorld-CN",
        cn=True,
        inspect=inspect,
        limit=limit,
        inspect_limit=inspect_limit,
    )


def convert_realworldqa(root: Path, *, inspect: bool = False, limit: int | None = None, inspect_limit: int = 3) -> Path:
    benchmark_dir = root / "RealWorldQA"
    output_path = benchmark_dir / "realworldqa_codevision_eval.parquet"
    image_dir = benchmark_dir / "codevision_images"
    dataset = _limit_dataset(_load_hf_dataset("lmms-lab/RealWorldQA", split="test"), limit)
    records = []
    raw_examples = []
    converted_examples = []
    for idx, row in enumerate(dataset):
        row = dict(row)
        if idx < inspect_limit:
            raw_examples.append(row)
        image_path, width, height = _save_image(row["image"], image_dir, f"realworldqa_test_{idx:05d}")
        record = _record(
            data_source="realworldqa",
            question=str(row["question"]).strip(),
            answer=row["answer"],
            image_path=image_path,
            width=width,
            height=height,
            index=idx,
            source_benchmark="realworldqa",
            extra={
                "dataset_path": "lmms-lab/RealWorldQA",
                "source_split": "test",
                "eval_protocol": "lmms_eval_exact_match",
                "scorer_status": "implemented",
                "raw_answer": row.get("answer"),
            },
        )
        records.append(record)
        if len(converted_examples) < inspect_limit:
            converted_examples.append(record)
    if inspect:
        _print_inspect(
            dataset_name="lmms-lab/RealWorldQA",
            splits=["test"],
            raw_count=len(dataset),
            converted_count=len(records),
            raw_examples=raw_examples,
            converted_examples=converted_examples,
        )
        return output_path
    _write_parquet(records, output_path)
    return output_path


def convert_mmstar(root: Path, *, inspect: bool = False, limit: int | None = None, inspect_limit: int = 3) -> Path:
    benchmark_dir = root / "MMStar"
    output_path = benchmark_dir / "mmstar_codevision_eval.parquet"
    image_dir = benchmark_dir / "codevision_images"
    dataset = _limit_dataset(_load_hf_dataset("Lin-Chen/MMStar", split="val"), limit)
    records = []
    raw_examples = []
    converted_examples = []
    for idx, row in enumerate(dataset):
        row = dict(row)
        if idx < inspect_limit:
            raw_examples.append(row)
        image_path, width, height = _save_image(row["image"], image_dir, f"mmstar_val_{idx:05d}")
        question = str(row["question"]).strip()
        if "answer with" not in question.lower():
            question = f"{question}\nAnswer with the option's letter from the given choices directly."
        l2_category = str(row.get("l2_category") or "unknown")
        record = _record(
            data_source=f"mmstar_{_sanitize(l2_category)}",
            question=question,
            answer=row["answer"],
            image_path=image_path,
            width=width,
            height=height,
            index=idx,
            source_benchmark="mmstar",
            extra={
                "dataset_path": "Lin-Chen/MMStar",
                "source_split": "val",
                "eval_protocol": "lmms_eval_l2_macro",
                "scorer_status": "implemented",
                "category": str(row.get("category") or ""),
                "l2_category": l2_category,
                "raw_answer": row.get("answer"),
                "aggregation_note": "Official lmms-eval aggregate is macro average over l2_category.",
            },
        )
        records.append(record)
        if len(converted_examples) < inspect_limit:
            converted_examples.append(record)
    if inspect:
        _print_inspect(
            dataset_name="Lin-Chen/MMStar",
            splits=["val"],
            raw_count=len(dataset),
            converted_count=len(records),
            raw_examples=raw_examples,
            converted_examples=converted_examples,
        )
        return output_path
    _write_parquet(records, output_path)
    return output_path


def _convert_docvqa_family(
    root: Path,
    *,
    dataset_name: str,
    benchmark_name: str,
    directory_name: str,
    split: str = "validation",
    inspect: bool = False,
    limit: int | None = None,
    inspect_limit: int = 3,
) -> Path:
    benchmark_dir = root / directory_name
    split_label = "val" if split == "validation" else _sanitize(split)
    output_path = benchmark_dir / f"{benchmark_name}_{split_label}_codevision_eval.parquet"
    image_dir = benchmark_dir / "codevision_images"
    dataset = _load_hf_dataset("lmms-lab/DocVQA", name=dataset_name, split=split)
    try:
        from datasets import Image as HFImage

        dataset = dataset.cast_column("image", HFImage(decode=False))
    except Exception:
        pass
    dataset = _limit_dataset(dataset, limit)
    records = []
    raw_examples = []
    converted_examples = []
    for idx, row in enumerate(dataset):
        row = dict(row)
        if idx < inspect_limit:
            raw_examples.append(row)
        image_stem = f"{benchmark_name}_{split}_{idx:05d}"
        existing_image_path = _existing_image_path(image_dir, image_stem)
        if existing_image_path is not None:
            image_path = existing_image_path
            width, height = 0, 0
        else:
            image_path, width, height = _save_image(row["image"], image_dir, image_stem)
        answers = _as_plain_list(row.get("answers"))
        has_public_answers = bool(answers)
        question = f"{str(row['question']).strip()}\nAnswer the question using a single word or phrase."
        record = _record(
            data_source=benchmark_name if split == "validation" else f"{benchmark_name}_{split_label}",
            question=question,
            answer=answers,
            image_path=image_path,
            width=width,
            height=height,
            index=idx,
            source_benchmark=benchmark_name,
            extra={
                "dataset_path": "lmms-lab/DocVQA",
                "dataset_name": dataset_name,
                "source_split": split,
                "eval_protocol": "ANLS",
                "scorer_status": "implemented" if has_public_answers else "hidden_labels_submission_only",
                "has_public_answers": has_public_answers,
                "questionId": row.get("questionId"),
                "answers": answers,
            },
        )
        records.append(record)
        if len(converted_examples) < inspect_limit:
            converted_examples.append(record)
    if inspect:
        _print_inspect(
            dataset_name=f"lmms-lab/DocVQA/{dataset_name}",
            splits=[split],
            raw_count=len(dataset),
            converted_count=len(records),
            raw_examples=raw_examples,
            converted_examples=converted_examples,
        )
        return output_path
    _write_parquet(records, output_path)
    return output_path


def convert_docvqa_val(root: Path, *, inspect: bool = False, limit: int | None = None, inspect_limit: int = 3) -> Path:
    return _convert_docvqa_family(
        root,
        dataset_name="DocVQA",
        benchmark_name="docvqa",
        directory_name="DocVQA",
        inspect=inspect,
        limit=limit,
        inspect_limit=inspect_limit,
    )


def convert_docvqa_test(root: Path, *, inspect: bool = False, limit: int | None = None, inspect_limit: int = 3) -> Path:
    return _convert_docvqa_family(
        root,
        dataset_name="DocVQA",
        benchmark_name="docvqa",
        directory_name="DocVQA",
        split="test",
        inspect=inspect,
        limit=limit,
        inspect_limit=inspect_limit,
    )


def convert_infovqa_val(root: Path, *, inspect: bool = False, limit: int | None = None, inspect_limit: int = 3) -> Path:
    return _convert_docvqa_family(
        root,
        dataset_name="InfographicVQA",
        benchmark_name="infovqa",
        directory_name="InfoVQA",
        inspect=inspect,
        limit=limit,
        inspect_limit=inspect_limit,
    )


def convert_infovqa_test(root: Path, *, inspect: bool = False, limit: int | None = None, inspect_limit: int = 3) -> Path:
    return _convert_docvqa_family(
        root,
        dataset_name="InfographicVQA",
        benchmark_name="infovqa",
        directory_name="InfoVQA",
        split="test",
        inspect=inspect,
        limit=limit,
        inspect_limit=inspect_limit,
    )


def convert_mmvet(root: Path, *, inspect: bool = False, limit: int | None = None, inspect_limit: int = 3) -> Path:
    benchmark_dir = root / "MMVet"
    output_path = benchmark_dir / "mmvet_codevision_eval.parquet"
    image_dir = benchmark_dir / "codevision_images"
    dataset = _limit_dataset(_load_hf_dataset("lmms-lab/MMVet", split="test"), limit)
    records = []
    raw_examples = []
    converted_examples = []
    for idx, row in enumerate(dataset):
        row = dict(row)
        if idx < inspect_limit:
            raw_examples.append(row)
        image_path, width, height = _save_image(row["image"], image_dir, f"mmvet_test_{idx:05d}")
        question = (
            "First please perform reasoning, and think step by step to provide best answer to the following question:\n\n"
            f"{str(row['question']).strip()}"
        )
        record = _record(
            data_source="mmvet",
            question=question,
            answer=row["answer"],
            image_path=image_path,
            width=width,
            height=height,
            index=idx,
            source_benchmark="mmvet",
            extra={
                "dataset_path": "lmms-lab/MMVet",
                "source_split": "test",
                "eval_protocol": "llm_judge_score",
                "scorer_status": "llm_judge_required",
                "question_id": row.get("question_id", row.get("id", idx)),
                "capability": row.get("capability", ""),
                "answers": [row.get("answer", "")],
            },
        )
        records.append(record)
        if len(converted_examples) < inspect_limit:
            converted_examples.append(record)
    if inspect:
        _print_inspect(
            dataset_name="lmms-lab/MMVet",
            splits=["test"],
            raw_count=len(dataset),
            converted_count=len(records),
            raw_examples=raw_examples,
            converted_examples=converted_examples,
        )
        return output_path
    _write_parquet(records, output_path)
    return output_path


def convert_mmvet_hard(root: Path, *, inspect: bool = False, limit: int | None = None, inspect_limit: int = 3) -> Path:
    data_path = os.getenv("MMVET_HARD_DATA_PATH", "").strip()
    if not data_path:
        raise FileNotFoundError(
            "MMVet-Hard has no standard local lmms-eval task in this checkout. "
            "Set MMVET_HARD_DATA_PATH to a json/jsonl/parquet file with image/question/answer fields."
        )
    import pandas as pd

    path = Path(data_path)
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else list(payload.values())
    else:
        rows = pd.read_parquet(path).to_dict(orient="records")
    if limit is not None:
        rows = rows[: int(limit)]

    benchmark_dir = root / "MMVet-Hard"
    output_path = benchmark_dir / "mmvet_hard_codevision_eval.parquet"
    image_dir = benchmark_dir / "codevision_images"
    records = []
    converted_examples = []
    for idx, row in enumerate(rows):
        image_value = row.get("image", row.get("image_path", row.get("image_file")))
        if isinstance(image_value, str) and not Path(image_value).exists() and row.get("image_root"):
            image_value = str(Path(row["image_root"]) / image_value)
        image_path, width, height = _save_image(image_value, image_dir, f"mmvet_hard_{idx:05d}")
        question = (
            "First please perform reasoning, and think step by step to provide best answer to the following question:\n\n"
            f"{str(row['question']).strip()}"
        )
        record = _record(
            data_source="mmvet_hard",
            question=question,
            answer=row.get("answer", row.get("ground_truth", "")),
            image_path=image_path,
            width=width,
            height=height,
            index=idx,
            source_benchmark="mmvet_hard",
            extra={
                "dataset_path": str(path),
                "source_split": str(row.get("split", "test")),
                "eval_protocol": "llm_judge_score",
                "scorer_status": "llm_judge_required",
                "question_id": row.get("question_id", row.get("id", idx)),
                "capability": row.get("capability", row.get("capabilities", "")),
                "answers": [row.get("answer", row.get("ground_truth", ""))],
            },
        )
        records.append(record)
        if len(converted_examples) < inspect_limit:
            converted_examples.append(record)
    if inspect:
        _print_inspect(
            dataset_name="MMVet-Hard local",
            splits=["custom"],
            raw_count=len(rows),
            converted_count=len(records),
            raw_examples=rows[:inspect_limit],
            converted_examples=converted_examples,
        )
        return output_path
    _write_parquet(records, output_path)
    return output_path


CONVERTERS = {
    "chartqa": convert_chartqa,
    "countqa": convert_countqa,
    "docvqa": convert_docvqa_val,
    "docvqa_val": convert_docvqa_val,
    "docvqa_test": convert_docvqa_test,
    "ocrbench": convert_ocrbench,
    "ocrbench_v2": convert_ocrbench_v2,
    "countbench": convert_countbench,
    "cvbench": convert_cvbench,
    "fsc147": convert_fsc147,
    "hrbench": convert_hrbench,
    "infovqa": convert_infovqa_val,
    "infovqa_val": convert_infovqa_val,
    "infovqa_test": convert_infovqa_test,
    "mme_realworld": convert_mme_realworld,
    "mme_realworld_cn": convert_mme_realworld_cn,
    "mme_realworld_lite": convert_mme_realworld_lite,
    "mmstar": convert_mmstar,
    "mmvet": convert_mmvet,
    "mmvet_hard": convert_mmvet_hard,
    "mvtoolbench": convert_mvtoolbench,
    "pixmo_count": convert_pixmo_count,
    "pixmo_count_lmms": convert_pixmo_count_lmms,
    "realworldqa": convert_realworldqa,
    "spatialmqa": convert_spatialmqa,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local benchmarks for CodeVision eval.")
    parser.add_argument("--benchmark-root", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["chartqa", "ocrbench", "countbench", "hrbench"],
        choices=sorted(CONVERTERS),
        help="Datasets to convert.",
    )
    parser.add_argument(
        "--no-download-missing-images",
        action="store_true",
        help="For countbench, skip rows with missing embedded images instead of downloading image_url.",
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=30,
        help="Timeout in seconds for downloading missing image_url entries.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print dataset/schema/conversion samples without writing parquet.",
    )
    parser.add_argument(
        "--inspect-limit",
        type=int,
        default=3,
        help="Number of raw and converted examples to print in --inspect mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke conversion.",
    )
    parser.add_argument(
        "--coco2017-test-image-dir",
        type=Path,
        default=None,
        help="COCO2017 test image directory for SpatialMQA.",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        print(f"=== Converting {dataset} ===")
        if dataset == "countbench":
            convert_countbench(
                args.benchmark_root,
                download_missing_images=not args.no_download_missing_images,
                download_timeout=args.download_timeout,
            )
        elif dataset == "cvbench":
            convert_cvbench(
                args.benchmark_root,
                inspect=args.inspect,
                limit=args.limit,
                inspect_limit=args.inspect_limit,
            )
        elif dataset == "pixmo_count":
            convert_pixmo_count(
                args.benchmark_root,
                inspect=args.inspect,
                limit=args.limit,
                inspect_limit=args.inspect_limit,
                download_timeout=args.download_timeout,
            )
        elif dataset == "pixmo_count_lmms":
            convert_pixmo_count_lmms(
                args.benchmark_root,
                inspect=args.inspect,
                limit=args.limit,
                inspect_limit=args.inspect_limit,
            )
        elif dataset == "ocrbench_v2":
            convert_ocrbench_v2(
                args.benchmark_root,
                inspect=args.inspect,
                limit=args.limit,
                inspect_limit=args.inspect_limit,
            )
        elif dataset == "spatialmqa":
            convert_spatialmqa(
                args.benchmark_root,
                inspect=args.inspect,
                limit=args.limit,
                inspect_limit=args.inspect_limit,
                coco2017_test_image_dir=args.coco2017_test_image_dir,
            )
        elif dataset == "countqa":
            convert_countqa(
                args.benchmark_root,
                inspect=args.inspect,
                limit=args.limit,
                inspect_limit=args.inspect_limit,
            )
        elif dataset == "mvtoolbench":
            convert_mvtoolbench(
                args.benchmark_root,
                inspect=args.inspect,
                limit=args.limit,
                inspect_limit=args.inspect_limit,
            )
        elif dataset in {
            "docvqa",
            "docvqa_val",
            "docvqa_test",
            "infovqa",
            "infovqa_val",
            "infovqa_test",
            "mme_realworld",
            "mme_realworld_cn",
            "mme_realworld_lite",
            "mmstar",
            "mmvet",
            "mmvet_hard",
            "realworldqa",
        }:
            CONVERTERS[dataset](
                args.benchmark_root,
                inspect=args.inspect,
                limit=args.limit,
                inspect_limit=args.inspect_limit,
            )
        else:
            CONVERTERS[dataset](args.benchmark_root)


if __name__ == "__main__":
    main()
