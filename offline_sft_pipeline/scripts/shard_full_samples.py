#!/usr/bin/env python3
"""Split a full RootSample JSONL into N shards from scratch.

This version does NOT subtract finished samples from a previous run.
It simply reads the full input JSONL, validates sample_id uniqueness,
and writes:
  - all_sample_ids.txt
  - shard_00.txt, shard_01.txt, ...   (sample_id lists)
  - shard_00.jsonl, shard_01.jsonl, ... (full JSONL rows; optional, enabled by default)
  - summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a full input samples.jsonl and split all samples evenly into shard files. "
            "No previous-run filtering is applied."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        required=True,
        help="Input RootSample JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write shard outputs into.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=4,
        help="Number of shards to generate. Default: 4",
    )
    parser.add_argument(
        "--write-shard-jsonl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to also write shard_XX.jsonl files. Default: true",
    )
    parser.add_argument(
        "--write-full-jsonl-copy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to also write full_samples.jsonl as a clean copy in output-dir.",
    )
    return parser.parse_args()


def _write_text_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = "\n" if lines else ""
    path.write_text("\n".join(lines) + suffix, encoding="utf-8")


def _load_all_rows(input_jsonl: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with input_jsonl.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue

            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_no} in {input_jsonl} is not a JSON object")

            sample_id = str(payload.get("sample_id") or "").strip()
            if not sample_id:
                raise ValueError(f"Missing sample_id in {input_jsonl} line {line_no}")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicated sample_id in input JSONL: {sample_id}")

            seen_ids.add(sample_id)
            rows.append(payload)

    return rows


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be > 0")

    input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_jsonl.is_file():
        raise FileNotFoundError(f"input JSONL not found: {input_jsonl}")

    rows = _load_all_rows(input_jsonl)
    sample_ids = [str(row["sample_id"]).strip() for row in rows]

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_text_lines(output_dir / "all_sample_ids.txt", sample_ids)

    shard_summaries: list[dict[str, Any]] = []
    for shard_idx in range(args.num_shards):
        shard_rows = rows[shard_idx :: args.num_shards]
        shard_ids = [str(row["sample_id"]).strip() for row in shard_rows]

        shard_txt_path = output_dir / f"shard_{shard_idx:02d}.txt"
        _write_text_lines(shard_txt_path, shard_ids)

        shard_jsonl_path = output_dir / f"shard_{shard_idx:02d}.jsonl"
        if args.write_shard_jsonl:
            with shard_jsonl_path.open("w", encoding="utf-8") as handle:
                for row in shard_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        shard_summaries.append(
            {
                "index": shard_idx,
                "txt_path": str(shard_txt_path.resolve()),
                "jsonl_path": str(shard_jsonl_path.resolve()) if args.write_shard_jsonl else None,
                "count": len(shard_rows),
                "first_sample_id": shard_ids[0] if shard_ids else None,
                "last_sample_id": shard_ids[-1] if shard_ids else None,
            }
        )

    if args.write_full_jsonl_copy:
        full_copy_path = output_dir / "full_samples.jsonl"
        with full_copy_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_jsonl": str(input_jsonl),
        "total_count": len(rows),
        "num_shards": int(args.num_shards),
        "output_dir": str(output_dir),
        "write_shard_jsonl": bool(args.write_shard_jsonl),
        "write_full_jsonl_copy": bool(args.write_full_jsonl_copy),
        "shards": shard_summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
