#!/usr/bin/env python3
"""Build balanced sample selections from <samples-root>/*/samples.jsonl."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read <samples-root>/*/samples.jsonl and write a sample_ids file with up to N items per folder."
        )
    )
    parser.add_argument(
        "--samples-root",
        type=str,
        default="export_images/unified_train_samples_50",
        help="Directory containing one subdirectory per dataset, each with samples.jsonl.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to the newline-delimited sample_ids.txt to write.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="",
        help="Optional merged RootSample JSONL containing the selected rows.",
    )
    parser.add_argument(
        "--absolute-image-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When writing --output-jsonl, rewrite each images[].path to an absolute path under samples-root.",
    )
    parser.add_argument(
        "--per-dataset",
        type=int,
        default=2,
        help="Maximum number of sample_ids to take from each dataset folder.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Optional allowlist of dataset folder names. Default: include every folder under samples-root.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle sample order within each dataset before taking the first N.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used together with --shuffle.",
    )
    return parser.parse_args()


def load_samples(path: Path) -> list[dict]:
    samples: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue
            payload = json.loads(text)
            sample_id = str(payload.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"Missing sample_id in {path} line {line_no}")
            samples.append(payload)
    return samples


def absolutize_sample_image_paths(sample: dict, *, samples_root: Path) -> dict:
    row = json.loads(json.dumps(sample, ensure_ascii=False))
    images = row.get("images")
    if not isinstance(images, list):
        return row
    for image in images:
        if not isinstance(image, dict):
            continue
        raw_path = str(image.get("path", "")).strip()
        if not raw_path:
            continue
        path_obj = Path(raw_path).expanduser()
        if not path_obj.is_absolute():
            path_obj = (samples_root / raw_path).resolve()
        else:
            path_obj = path_obj.resolve()
        image["path"] = str(path_obj)
    return row


def main() -> None:
    args = parse_args()
    if args.per_dataset <= 0:
        raise ValueError("--per-dataset must be > 0")

    samples_root = Path(args.samples_root).expanduser().resolve()
    if not samples_root.is_dir():
        raise FileNotFoundError(f"samples root not found: {samples_root}")

    requested = {name.strip() for name in args.datasets if str(name).strip()}
    dataset_dirs = [p for p in sorted(samples_root.iterdir()) if p.is_dir()]
    if requested:
        dataset_dirs = [p for p in dataset_dirs if p.name in requested]
        missing = sorted(requested - {p.name for p in dataset_dirs})
        if missing:
            raise FileNotFoundError(f"Requested dataset folders not found under {samples_root}: {missing}")

    if not dataset_dirs:
        raise RuntimeError(f"No dataset folders found under {samples_root}")

    rng = random.Random(args.seed)
    chosen: list[str] = []
    chosen_rows: list[dict] = []
    summary_lines: list[str] = []

    for dataset_dir in dataset_dirs:
        samples_path = dataset_dir / "samples.jsonl"
        if not samples_path.is_file():
            raise FileNotFoundError(f"Missing samples.jsonl: {samples_path}")
        sample_rows = load_samples(samples_path)
        ordered_rows = list(sample_rows)
        if args.shuffle:
            rng.shuffle(ordered_rows)
        picked_rows = ordered_rows[: args.per_dataset]
        chosen.extend(str(row["sample_id"]).strip() for row in picked_rows)
        if args.absolute_image_paths:
            chosen_rows.extend(
                absolutize_sample_image_paths(row, samples_root=samples_root) for row in picked_rows
            )
        else:
            chosen_rows.extend(picked_rows)
        summary_lines.append(
            f"{dataset_dir.name}: picked {len(picked_rows)} / {len(sample_rows)}"
        )

    output_file = Path(args.output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(chosen) + "\n", encoding="utf-8")

    if args.output_jsonl:
        output_jsonl = Path(args.output_jsonl).expanduser().resolve()
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as handle:
            for row in chosen_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(chosen_rows)} merged rows to {output_jsonl}")

    print(f"Wrote {len(chosen)} sample_ids to {output_file}")
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
