#!/usr/bin/env python3
"""Merge CodeVision-style datasets, sample easy rows, and shuffle the result."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_DEFAULT_BASE_DIR = Path("/data/home/suchenghao/ToolVision/CodeVision-SFT")
_DEFAULT_EXPORT_ROOT = Path("/data/home/suchenghao/ToolVision/offline_sft_pipeline/outputs/sft_prep/codevision_exports")
_DEFAULT_COMPLEX_DIRS = ["textvqa", "fsc147", "gqa", "cavqa"]
_DEFAULT_EASY_QUOTAS = {
    "textvqa_easy": 250,
    "fsc147_easy": 250,
    "gqa_easy": 450,
    "cavqa_easy": 250,
}


@dataclass
class RowSource:
    source_name: str
    root_dir: Path
    row_index: int
    row: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=_DEFAULT_BASE_DIR,
        help="Existing CodeVision-SFT directory to include in full.",
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        default=_DEFAULT_EXPORT_ROOT,
        help="Root containing exported CodeVision-style folders.",
    )
    parser.add_argument(
        "--complex-dir",
        action="append",
        default=[],
        help="Complex export folder name under export-root. Can be provided multiple times.",
    )
    parser.add_argument(
        "--easy-quota",
        action="append",
        default=[],
        help="Easy quota in the form name=count, for example gqa_easy=450.",
    )
    parser.add_argument(
        "--easy-total",
        type=int,
        default=1200,
        help="Expected total sampled easy count. Used for validation only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Merged output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for easy sampling and final shuffle.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Actually copy images and write codevision_sft.json. Without this flag, only print plan.",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=0,
        help="Optional cap on total merged rows after shuffle, for smoke tests.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_easy_quotas(raw_values: list[str]) -> dict[str, int]:
    if not raw_values:
        return dict(_DEFAULT_EASY_QUOTAS)
    quotas: dict[str, int] = {}
    for item in raw_values:
        if "=" not in item:
            raise ValueError(f"Invalid --easy-quota value: {item!r}")
        name, count = item.split("=", 1)
        quotas[name.strip()] = int(count.strip())
    return quotas


def resolve_complex_dirs(args: argparse.Namespace) -> list[str]:
    if args.complex_dir:
        return list(args.complex_dir)
    return list(_DEFAULT_COMPLEX_DIRS)


def load_dataset_rows(root_dir: Path) -> list[dict[str, Any]]:
    data_path = root_dir / "codevision_sft.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing codevision_sft.json in {root_dir}")
    data = load_json(data_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {data_path}")
    return data


def sample_easy_rows(
    *,
    export_root: Path,
    easy_quotas: dict[str, int],
    rng: random.Random,
) -> list[RowSource]:
    sampled: list[RowSource] = []
    for name, quota in easy_quotas.items():
        root_dir = export_root / name
        rows = load_dataset_rows(root_dir)
        if quota > len(rows):
            raise ValueError(f"Requested {quota} rows from {name}, but only {len(rows)} available.")
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        for row_index in sorted(indices[:quota]):
            sampled.append(RowSource(source_name=name, root_dir=root_dir, row_index=row_index, row=rows[row_index]))
    return sampled


def collect_all_rows(
    *,
    base_dir: Path,
    export_root: Path,
    complex_dirs: list[str],
    easy_quotas: dict[str, int],
    seed: int,
) -> tuple[list[RowSource], dict[str, Any]]:
    base_rows = [
        RowSource(source_name="codevision_base", root_dir=base_dir, row_index=i, row=row)
        for i, row in enumerate(load_dataset_rows(base_dir))
    ]
    complex_rows: list[RowSource] = []
    for name in complex_dirs:
        root_dir = export_root / name
        rows = load_dataset_rows(root_dir)
        complex_rows.extend(RowSource(source_name=name, root_dir=root_dir, row_index=i, row=row) for i, row in enumerate(rows))

    rng = random.Random(seed)
    easy_rows = sample_easy_rows(export_root=export_root, easy_quotas=easy_quotas, rng=rng)
    all_rows = base_rows + complex_rows + easy_rows
    rng.shuffle(all_rows)
    summary = {
        "base_count": len(base_rows),
        "complex_counts": {name: sum(1 for row in complex_rows if row.source_name == name) for name in complex_dirs},
        "easy_sample_counts": {name: sum(1 for row in easy_rows if row.source_name == name) for name in easy_quotas},
        "total_before_shuffle": len(base_rows) + len(complex_rows) + len(easy_rows),
        "seed": seed,
    }
    return all_rows, summary


def copy_images_for_row(*, row: dict[str, Any], source_root: Path, dest_image_dir: Path, new_sample_id: int) -> list[str]:
    exported_images: list[str] = []
    for image_index, rel_path in enumerate(row.get("images") or []):
        source_path = source_root / str(rel_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Missing image {source_path}")
        suffix = source_path.suffix.lower() or ".png"
        dest_name = f"sample{new_sample_id}_{image_index}{suffix}"
        dest_path = dest_image_dir / dest_name
        if not dest_path.exists():
            shutil.copy2(source_path, dest_path)
        exported_images.append(f"codevision_images/{dest_name}")
    return exported_images


def rewrite_metadata_sample_id(metadata_text: str, new_sample_id: int) -> str:
    try:
        payload = json.loads(metadata_text)
    except Exception:
        return metadata_text
    if isinstance(payload, dict):
        payload["sample_id"] = new_sample_id
        return json.dumps(payload, ensure_ascii=False)
    return metadata_text


def materialize_merged_dataset(
    *,
    rows: list[RowSource],
    output_dir: Path,
    max_total: int,
) -> dict[str, Any]:
    if output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "codevision_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    row_iter = rows[:max_total] if max_total > 0 else rows

    for new_sample_id, row_source in enumerate(row_iter):
        new_row = dict(row_source.row)
        new_row["images"] = copy_images_for_row(
            row=row_source.row,
            source_root=row_source.root_dir,
            dest_image_dir=image_dir,
            new_sample_id=new_sample_id,
        )
        if "metadata" in new_row:
            new_row["metadata"] = rewrite_metadata_sample_id(str(new_row["metadata"]), new_sample_id)
        merged_rows.append(new_row)
        manifest.append(
            {
                "new_sample_id": new_sample_id,
                "source_name": row_source.source_name,
                "source_row_index": row_source.row_index,
                "source_root_dir": str(row_source.root_dir),
            }
        )

    write_json(output_dir / "codevision_sft.json", merged_rows)
    write_json(output_dir / "merge_manifest.json", manifest)
    write_json(
        output_dir / "dataset_info.snippet.json",
        {
            output_dir.name: {
                "file_name": "codevision_sft.json",
                "formatting": "sharegpt",
                "columns": {
                    "messages": "conversations",
                    "images": "images",
                    "system": "system",
                },
                "tags": {
                    "role_tag": "from",
                    "content_tag": "value",
                    "user_tag": "human",
                    "assistant_tag": "gpt",
                    "observation_tag": "tool",
                },
            }
        },
    )
    return {"merged_count": len(merged_rows)}


def main() -> None:
    args = parse_args()
    easy_quotas = parse_easy_quotas(args.easy_quota)
    if sum(easy_quotas.values()) != int(args.easy_total):
        raise ValueError(
            f"Easy quota sum {sum(easy_quotas.values())} does not match --easy-total {args.easy_total}."
        )

    rows, summary = collect_all_rows(
        base_dir=args.base_dir.resolve(),
        export_root=args.export_root.resolve(),
        complex_dirs=resolve_complex_dirs(args),
        easy_quotas=easy_quotas,
        seed=int(args.seed),
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "merge_plan.json",
        {
            "base_dir": str(args.base_dir.resolve()),
            "export_root": str(args.export_root.resolve()),
            "complex_dirs": resolve_complex_dirs(args),
            "easy_quotas": easy_quotas,
            "summary": summary,
            "max_total": args.max_total,
            "copy_images": bool(args.copy_images),
        },
    )

    result = {"planned_total": len(rows)}
    if args.copy_images:
        result.update(
            materialize_merged_dataset(
                rows=rows,
                output_dir=output_dir,
                max_total=int(args.max_total),
            )
        )

    print(json.dumps({**summary, **result, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
