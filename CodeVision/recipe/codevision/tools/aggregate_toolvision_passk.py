"""Aggregate dumped validation generations into per-question pass@k reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Validation JSONL file or a directory containing JSONL dumps.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument(
        "--score-key",
        default="official_like_score",
        help="Generation-level score field to aggregate; falls back to 'score'.",
    )
    parser.add_argument(
        "--correct-threshold",
        type=float,
        default=1.0,
        help="Score threshold counted as a correct completion.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(iter_rows(args.input.expanduser()))
    if not rows:
        raise RuntimeError(f"no rows found under {args.input}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row_uid(row)].append(row)

    sample_rows = []
    for uid, group in grouped.items():
        group = sorted(group, key=lambda r: len(group))
        first = group[0]
        scores = [score_value(r, args.score_key) for r in group]
        correct = [1 if s >= args.correct_threshold else 0 for s in scores]
        extra = as_dict(first.get("extra_info"))
        source = str(first.get("data_source") or extra.get("source_dataset") or "")
        index = extra.get("index", "")
        sample_rows.append(
            {
                "uid": uid,
                "index": index,
                "data_source": source,
                "origin": extra.get("origin", ""),
                "reward_family": extra.get("reward_family", ""),
                "n": len(group),
                "expected_k": args.k,
                "complete_k": int(len(group) >= args.k),
                "correct_count": sum(correct),
                "acc_at_k": sum(correct) / len(group),
                "score_mean": mean(scores),
                "score_max": max(scores),
                "pass_at_k": int(any(correct)),
                "all_correct": int(all(correct)),
                "bucket": bucket(sum(correct), len(group)),
                "question": extra.get("question", ""),
                "ground_truth": first.get("gts", ""),
                "raw_file": extra.get("raw_file", ""),
                "raw_row": extra.get("raw_row", ""),
            }
        )

    sample_rows.sort(key=lambda r: (str(r["data_source"]), int_or_large(r["index"]), str(r["uid"])))

    write_csv(out_dir / "passk_by_sample.csv", sample_rows)
    write_jsonl(out_dir / "passk_by_sample.jsonl", sample_rows)
    write_source_summary(out_dir / "passk_by_source.csv", sample_rows)
    write_bucket_summary(out_dir / "passk_bucket_summary.csv", sample_rows)
    write_selection_files(out_dir, sample_rows)

    manifest = {
        "input": str(args.input),
        "output_dir": str(out_dir),
        "generation_rows": len(rows),
        "samples": len(sample_rows),
        "k": args.k,
        "score_key": args.score_key,
        "correct_threshold": args.correct_threshold,
        "complete_k_samples": sum(int(r["complete_k"]) for r in sample_rows),
        "mean_acc_at_k": mean([float(r["acc_at_k"]) for r in sample_rows]),
        "pass_at_k_rate": mean([float(r["pass_at_k"]) for r in sample_rows]),
    }
    (out_dir / "passk_summary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[ok] wrote reports -> {out_dir}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def iter_rows(path: Path):
    files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def row_uid(row: dict[str, Any]) -> str:
    extra = as_dict(row.get("extra_info"))
    for key in ("uid", "sample_uid"):
        if row.get(key):
            return str(row[key])
        if extra.get(key):
            return str(extra[key])
    if extra.get("index") is not None:
        return f"index::{extra['index']}"
    return str(row.get("input", ""))


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def score_value(row: dict[str, Any], score_key: str) -> float:
    value = row.get(score_key, row.get("score", 0.0))
    try:
        return float(value)
    except Exception:
        return 0.0


def bucket(correct_count: int, n: int) -> str:
    if correct_count <= 0:
        return "00_all_wrong"
    if correct_count == n:
        return "05_all_correct"
    rate = correct_count / max(n, 1)
    if rate <= 0.25:
        return "01_hard_0_25"
    if rate <= 0.50:
        return "02_mid_25_50"
    if rate <= 0.75:
        return "03_mid_50_75"
    return "04_easy_75_100"


def int_or_large(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 10**18


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_source_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["data_source"])].append(row)
    summary = []
    for source, group in sorted(by_source.items()):
        summary.append(
            {
                "data_source": source,
                "samples": len(group),
                "complete_k_samples": sum(int(r["complete_k"]) for r in group),
                "mean_acc_at_k": mean([float(r["acc_at_k"]) for r in group]),
                "pass_at_k_rate": mean([float(r["pass_at_k"]) for r in group]),
                "all_wrong": sum(1 for r in group if int(r["correct_count"]) == 0),
                "all_correct": sum(1 for r in group if int(r["correct_count"]) == int(r["n"])),
                "rl_variable": sum(1 for r in group if 0 < int(r["correct_count"]) < int(r["n"])),
            }
        )
    write_csv(path, summary)


def write_bucket_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    counts = Counter(str(row["bucket"]) for row in rows)
    write_csv(path, [{"bucket": key, "samples": counts[key]} for key in sorted(counts)])


def write_selection_files(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    selections = {
        "selection_rl_variable.csv": [r for r in rows if 0 < int(r["correct_count"]) < int(r["n"])],
        "selection_sft_has_success.csv": [r for r in rows if int(r["correct_count"]) > 0],
        "selection_all_wrong.csv": [r for r in rows if int(r["correct_count"]) == 0],
        "selection_all_correct.csv": [r for r in rows if int(r["correct_count"]) == int(r["n"])],
    }
    for filename, selected in selections.items():
        write_csv(out_dir / filename, selected)


if __name__ == "__main__":
    raise SystemExit(main())
