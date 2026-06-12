#!/usr/bin/env python3
"""Collapse append-only sample_results.jsonl into one record per sample_id."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STATUS_PRIORITY = {
    "ok": 3,
    "no_answer": 2,
    "error": 1,
    "skipped_resume": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collapse sample_results.jsonl for one run into one best record per sample_id."
    )
    parser.add_argument(
        "--run-root",
        type=str,
        required=True,
        help="Run root such as offline_sft_pipeline/outputs/dataset_pipeline_runs/<run_id>.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="",
        help="Optional output JSONL path. Defaults to <run-root>/sample_results_collapsed.jsonl.",
    )
    parser.add_argument(
        "--output-summary-json",
        type=str,
        default="",
        help="Optional output summary JSON path. Defaults to <run-root>/sample_results_collapsed_summary.json.",
    )
    parser.add_argument(
        "--sample-ids-file",
        type=str,
        default="",
        help="Optional newline-delimited sample_id file. When provided, only those sample_ids are kept.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def status_priority(status: str) -> int:
    return STATUS_PRIORITY.get(str(status or "").strip(), -1)


def pick_best_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    best = records[0]
    for item in records[1:]:
        item_status = str(item.get("status") or "")
        best_status = str(best.get("status") or "")
        if status_priority(item_status) > status_priority(best_status):
            best = item
            continue
        if status_priority(item_status) == status_priority(best_status):
            item_started = str(item.get("started_at") or "")
            best_started = str(best.get("started_at") or "")
            if item_started > best_started:
                best = item
    return dict(best)


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    sample_results_path = run_root / "sample_results.jsonl"
    if not sample_results_path.is_file():
        raise FileNotFoundError(f"sample_results.jsonl not found: {sample_results_path}")

    output_jsonl = (
        Path(args.output_jsonl).expanduser().resolve()
        if args.output_jsonl
        else run_root / "sample_results_collapsed.jsonl"
    )
    output_summary_json = (
        Path(args.output_summary_json).expanduser().resolve()
        if args.output_summary_json
        else run_root / "sample_results_collapsed_summary.json"
    )

    rows = load_jsonl(sample_results_path)
    allowed_sample_ids: set[str] | None = None
    if args.sample_ids_file:
        sample_ids_path = Path(args.sample_ids_file).expanduser().resolve()
        if not sample_ids_path.is_file():
            raise FileNotFoundError(f"sample_ids_file not found: {sample_ids_path}")
        allowed_sample_ids = {
            line.strip()
            for line in sample_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in rows:
        sample_id = str(item.get("sample_id") or "").strip()
        if not sample_id:
            continue
        if allowed_sample_ids is not None and sample_id not in allowed_sample_ids:
            continue
        grouped[sample_id].append(item)

    collapsed_rows: list[dict[str, Any]] = []
    history_status_counts = Counter()
    final_status_counts = Counter()
    duplicate_sample_ids = 0
    for sample_id in sorted(grouped):
        records = grouped[sample_id]
        if len(records) > 1:
            duplicate_sample_ids += 1
        best = pick_best_record(records)
        status_history = [str(item.get("status") or "") for item in records]
        for status in status_history:
            history_status_counts[status] += 1
        best["status_history"] = status_history
        best["attempt_count"] = len(records)
        best["latest_status"] = str(records[-1].get("status") or "")
        final_status_counts[str(best.get("status") or "")] += 1
        collapsed_rows.append(best)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for item in collapsed_rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "run_root": str(run_root),
        "input_path": str(sample_results_path),
        "total_rows": len(rows),
        "sample_ids_filter_path": (
            str(Path(args.sample_ids_file).expanduser().resolve()) if args.sample_ids_file else None
        ),
        "filtered_unique_sample_ids": len(grouped),
        "unique_sample_ids": len(collapsed_rows),
        "duplicate_sample_ids": duplicate_sample_ids,
        "history_status_counts": dict(history_status_counts),
        "collapsed_status_counts": dict(final_status_counts),
        "output_jsonl": str(output_jsonl),
    }
    output_summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
