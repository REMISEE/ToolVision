#!/usr/bin/env python3
"""Build remaining sample-id shards from a previous dataset pipeline run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a previous run's sample_results.jsonl, subtract completed sample_ids "
            "from the input samples.jsonl, and write shard text files."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Previous run directory under offline_sft_pipeline/outputs/dataset_pipeline_runs/...",
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        required=True,
        help="Original input RootSample JSONL used by the pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write remaining_all.txt and shard_XX.txt files into.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=4,
        help="Number of shard text files to generate.",
    )
    parser.add_argument(
        "--done-statuses",
        nargs="*",
        default=["ok", "no_answer"],
        help=(
            "Statuses in sample_results.jsonl that should be treated as already done. "
            "Default: ok no_answer"
        ),
    )
    parser.add_argument(
        "--write-remaining-jsonl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also write remaining_samples.jsonl containing only unfinished rows.",
    )
    return parser.parse_args()


def _load_done_ids(sample_results_path: Path, *, done_statuses: set[str]) -> set[str]:
    done_ids: set[str] = set()
    with sample_results_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue
            payload = json.loads(text)
            status = str(payload.get("status") or "").strip()
            if status not in done_statuses:
                continue
            sample_id = str(payload.get("sample_id") or "").strip()
            if not sample_id:
                raise ValueError(f"Missing sample_id in {sample_results_path} line {line_no}")
            done_ids.add(sample_id)
    return done_ids


def _load_remaining_rows(input_jsonl: Path, *, done_ids: set[str]) -> list[dict]:
    remaining_rows: list[dict] = []
    seen_ids: set[str] = set()
    with input_jsonl.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue
            payload = json.loads(text)
            sample_id = str(payload.get("sample_id") or "").strip()
            if not sample_id:
                raise ValueError(f"Missing sample_id in {input_jsonl} line {line_no}")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicated sample_id in input JSONL: {sample_id}")
            seen_ids.add(sample_id)
            if sample_id in done_ids:
                continue
            remaining_rows.append(payload)
    return remaining_rows


def _write_text_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = "\n" if lines else ""
    path.write_text("\n".join(lines) + suffix, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be > 0")

    run_dir = Path(args.run_dir).expanduser().resolve()
    input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    sample_results_path = run_dir / "sample_results.jsonl"

    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir not found: {run_dir}")
    if not input_jsonl.is_file():
        raise FileNotFoundError(f"input JSONL not found: {input_jsonl}")
    if not sample_results_path.is_file():
        raise FileNotFoundError(f"sample_results.jsonl not found: {sample_results_path}")

    done_statuses = {str(item).strip() for item in args.done_statuses if str(item).strip()}
    if not done_statuses:
        raise ValueError("--done-statuses must not be empty")

    done_ids = _load_done_ids(sample_results_path, done_statuses=done_statuses)
    remaining_rows = _load_remaining_rows(input_jsonl, done_ids=done_ids)
    remaining_ids = [str(row["sample_id"]).strip() for row in remaining_rows]

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_text_lines(output_dir / "remaining_all.txt", remaining_ids)

    for shard_idx in range(args.num_shards):
        shard_ids = remaining_ids[shard_idx :: args.num_shards]
        _write_text_lines(output_dir / f"shard_{shard_idx:02d}.txt", shard_ids)

    if args.write_remaining_jsonl:
        remaining_jsonl_path = output_dir / "remaining_samples.jsonl"
        with remaining_jsonl_path.open("w", encoding="utf-8") as handle:
            for row in remaining_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "run_dir": str(run_dir),
        "input_jsonl": str(input_jsonl),
        "sample_results_jsonl": str(sample_results_path),
        "done_statuses": sorted(done_statuses),
        "done_count": len(done_ids),
        "remaining_count": len(remaining_ids),
        "num_shards": int(args.num_shards),
        "output_dir": str(output_dir),
        "shards": [
            {
                "index": shard_idx,
                "path": str((output_dir / f"shard_{shard_idx:02d}.txt").resolve()),
                "count": len(remaining_ids[shard_idx :: args.num_shards]),
            }
            for shard_idx in range(args.num_shards)
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
