#!/usr/bin/env python3
"""Collect correct answered trajectories from dataset_pipeline_runs.

This is the first-pass filter only:
1. Collapse append-only sample_results.jsonl into one effective record per sample_id.
2. Read answered trajectory ids from that effective record.
3. Rebuild trajectory-level correctness from store/ instead of trusting raw
   answered_results.jsonl, which may contain duplicated rows after resume.
4. Record concise per-trajectory fields for downstream inspection.

Outputs:
- <output_prefix>.jsonl: one correct trajectory per line
- <output_prefix>.summary.json: concise aggregate summary
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from offline_sft_pipeline.eval.scorers import score_answer_for_dataset

STATUS_PRIORITY = {
    "ok": 3,
    "no_answer": 2,
    "error": 1,
    "skipped_resume": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        required=True,
        help="One or more run roots under outputs/dataset_pipeline_runs/",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output prefix; writes .jsonl and .summary.json",
    )
    parser.add_argument(
        "--textvqa-min-score",
        type=float,
        default=0.9,
        help="Minimum official TextVQA soft score to keep as a first-pass correct trajectory.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dataset_name_from_sample_id(sample_id: str) -> str:
    return sample_id.split("__", 1)[0] if "__" in sample_id else sample_id


def coerce_exact_answer(answer: Any) -> str:
    if answer is None:
        return ""
    if isinstance(answer, list):
        if not answer:
            return ""
        return str(answer[0]).strip()
    return str(answer).strip()


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
    output_rows: list[dict[str, Any]] = []
    summary_runs: list[dict[str, Any]] = []

    for run_root in [path.resolve() for path in args.run_root]:
        sample_results_path = run_root / "sample_results.jsonl"
        sample_rows = load_jsonl(sample_results_path)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in sample_rows:
            sample_id = str(row.get("sample_id") or "").strip()
            if sample_id:
                grouped[sample_id].append(row)
        effective_rows = {sample_id: pick_best_record(records) for sample_id, records in grouped.items()}

        run_output_rows: list[dict[str, Any]] = []
        total_answered_trajectories = 0
        duplicate_sample_ids = sum(1 for records in grouped.values() if len(records) > 1)

        for sample_id, row in sorted(effective_rows.items()):
            root_sample_path = run_root / "store" / "samples" / sample_id / "root_sample.json"
            if not root_sample_path.is_file():
                continue
            root_sample = json.loads(root_sample_path.read_text(encoding="utf-8"))
            answer = root_sample.get("answer")
            root_meta = root_sample.get("metadata") or {}
            source_dataset = str(root_meta.get("source_dataset") or dataset_name_from_sample_id(sample_id)).strip()
            answered_trajectory_ids = row.get("answered_trajectory_ids") or []

            for trajectory_id in answered_trajectory_ids:
                total_answered_trajectories += 1
                trajectory_path = (
                    run_root
                    / "store"
                    / "samples"
                    / sample_id
                    / "trajectories"
                    / trajectory_id
                    / "trajectory.json"
                )
                if not trajectory_path.is_file():
                    continue
                trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
                pred = str(trajectory.get("final_answer") or "").strip()

                score_result = score_answer_for_dataset(
                    source_dataset,
                    pred,
                    answer,
                    root_meta,
                )
                gt_score = float(score_result.score)

                keep = False
                if source_dataset == "textvqa":
                    keep = gt_score >= float(args.textvqa_min_score)
                else:
                    keep = pred == coerce_exact_answer(answer)
                if not keep:
                    continue

                step_count = len(trajectory.get("steps") or [])
                judge_score = None
                for judge_ref in reversed(trajectory.get("judge_records") or []):
                    judge_rel = str(judge_ref.get("judge_record_path") or "").strip()
                    if not judge_rel:
                        continue
                    judge_path = (trajectory_path.parent / judge_rel).resolve()
                    if judge_path.is_file():
                        judge_record = json.loads(judge_path.read_text(encoding="utf-8"))
                        if judge_record.get("overall_score") is not None:
                            judge_score = judge_record.get("overall_score")
                            break

                record = {
                    "run_root": str(run_root),
                    "run_name": run_root.name,
                    "dataset": source_dataset,
                    "sample_id": sample_id,
                    "trajectory_id": trajectory_id,
                    "pred": pred,
                    "answer": answer,
                    "gt_score": gt_score,
                    "matcher_name": score_result.matcher_name,
                    "judge_score": judge_score,
                    "step_count": step_count,
                    "is_root_trajectory": trajectory_id.endswith("__root"),
                }
                run_output_rows.append(record)
                output_rows.append(record)

        step_dist: dict[str, int] = {}
        for row in run_output_rows:
            key = str(row["step_count"])
            step_dist[key] = step_dist.get(key, 0) + 1

        summary_runs.append(
            {
                "run_root": str(run_root),
                "run_name": run_root.name,
                "sample_results_rows": len(sample_rows),
                "unique_sample_ids": len(grouped),
                "duplicate_sample_ids": duplicate_sample_ids,
                "answered_trajectory_count": total_answered_trajectories,
                "correct_trajectory_count": len(run_output_rows),
                "step_dist": step_dist,
            }
        )

    output_prefix = args.output_prefix.resolve()
    write_jsonl(output_prefix.with_suffix(".jsonl"), output_rows)

    summary = {
        "run_count": len(summary_runs),
        "correct_trajectory_count": len(output_rows),
        "runs": summary_runs,
    }
    output_prefix.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
