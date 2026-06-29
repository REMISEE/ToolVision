#!/usr/bin/env python3
"""Build clean pending shards after a partial ToolVision rollout8 run.

The first rollout8 shard can stop after streaming most generations. This script
uses the streamed generation file to identify original eval rows that already
have all responses, then appends the true remainder to the untouched shards.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-shard-dir",
        default="/mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun_all/final_v3/toolvision_eval/shards_8way_safe",
    )
    parser.add_argument(
        "--generation-jsonl",
        default="/mnt/cpfs/delinmao/ToolVision/CodeVision/saves/CodeVision/mut_rollout8_stream_shard00of08_t0p7/generations/0.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun_all/final_v3/toolvision_eval/shards_7way_pending_after_shard00_mns32",
    )
    parser.add_argument("--expected-responses", type=int, default=8)
    parser.add_argument("--smoke-rows", type=int, default=128)
    return parser.parse_args()


def extra_uid(value: Any) -> str | None:
    if isinstance(value, dict):
        uid = value.get("uid")
        return str(uid) if uid is not None else None
    if isinstance(value, str):
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            return None
        uid = data.get("uid") if isinstance(data, dict) else None
        return str(uid) if uid is not None else None
    return None


def count_generated_uids(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            uid = extra_uid(obj.get("extra_info"))
            if uid:
                counts[uid] += 1
    return counts


def source_benchmark(row: pd.Series) -> str:
    info = row.get("extra_info")
    if isinstance(info, dict):
        value = info.get("source_benchmark") or info.get("source_dataset") or row.get("data_source")
        return str(value)
    return str(row.get("data_source"))


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_shard_dir)
    generation_path = Path(args.generation_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_shards = [
        source_dir / f"mut_candidates_0_8_toolvision_eval_shard{i:02d}of08.parquet"
        for i in range(8)
    ]
    missing = [str(path) for path in source_shards if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source shards: {missing}")
    if not generation_path.exists():
        raise FileNotFoundError(f"Missing generation file: {generation_path}")

    shard00 = pd.read_parquet(source_shards[0])
    shard00 = shard00.copy()
    shard00["_original_uid"] = shard00["extra_info"].map(extra_uid)
    if shard00["_original_uid"].isna().any():
        bad = int(shard00["_original_uid"].isna().sum())
        raise ValueError(f"shard00 has {bad} rows without extra_info.uid")

    generated_counts = count_generated_uids(generation_path)
    completed = {uid for uid, count in generated_counts.items() if count >= args.expected_responses}
    true_remaining = shard00[~shard00["_original_uid"].isin(completed)].drop(columns=["_original_uid"])

    # Keep this file as the audit trail. The runnable shards below spread these
    # rows across the untouched shards so shard00 itself does not need a special
    # rerun.
    true_remaining_path = output_dir / "shard00_true_remaining_170.parquet"
    true_remaining.to_parquet(true_remaining_path, index=False)

    remainder_parts = []
    for idx in range(7):
        start = round(idx * len(true_remaining) / 7)
        end = round((idx + 1) * len(true_remaining) / 7)
        remainder_parts.append(true_remaining.iloc[start:end])

    plan_rows = []
    output_paths = []
    for pending_idx, original_idx in enumerate(range(1, 8)):
        base = pd.read_parquet(source_shards[original_idx])
        add = remainder_parts[pending_idx]
        merged = pd.concat([base, add], ignore_index=True)
        out_path = output_dir / f"mut_candidates_0_8_toolvision_eval_pending_mns32_shard{pending_idx:02d}of07.parquet"
        merged.to_parquet(out_path, index=False)
        output_paths.append(out_path)

        plan_rows.append(
            {
                "pending_shard": f"{pending_idx:02d}/07",
                "path": str(out_path),
                "base_source_shard": f"{original_idx:02d}/08",
                "base_rows": len(base),
                "added_shard00_remaining_rows": len(add),
                "total_rows": len(merged),
            }
        )

    smoke_rows_per_shard = max(1, math.ceil(args.smoke_rows / len(output_paths)))
    smoke = pd.concat(
        [pd.read_parquet(path).head(smoke_rows_per_shard) for path in output_paths],
        ignore_index=True,
    ).head(args.smoke_rows)
    smoke_path = output_dir / "smoke_mns32_128.parquet"
    smoke.to_parquet(smoke_path, index=False)

    benchmark_counts = true_remaining.apply(source_benchmark, axis=1).value_counts().sort_index()
    manifest = {
        "purpose": "Clean pending ToolVision rollout8 shards after partial shard00 run.",
        "source_shard_dir": str(source_dir),
        "generation_jsonl": str(generation_path),
        "expected_responses_per_uid": args.expected_responses,
        "source_shard00_rows": int(len(shard00)),
        "generated_uid_count": int(len(generated_counts)),
        "completed_uid_count": int(len(completed)),
        "true_remaining_rows": int(len(true_remaining)),
        "true_remaining_by_benchmark": {str(k): int(v) for k, v in benchmark_counts.items()},
        "true_remaining_path": str(true_remaining_path),
        "pending_shards": plan_rows,
        "smoke_path": str(smoke_path),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pd.DataFrame(plan_rows).to_csv(output_dir / "shard_plan.tsv", sep="\t", index=False)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
