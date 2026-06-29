#!/usr/bin/env python3
"""Build a remaining eval parquet from a streaming rollout metadata file.

Rows whose UID has fewer than --expected-rollouts entries in metadata are kept.
This is intended for interrupted ToolVision rollout jobs with stream dump enabled.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def read_completed_counts(stream_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not stream_path.exists():
        return counts
    with stream_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            extra_info = item.get("extra_info") if isinstance(item.get("extra_info"), dict) else {}
            uid = extra_info.get("uid") or item.get("uid")
            if uid:
                counts[str(uid)] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--stream-jsonl")
    parser.add_argument("--metadata-jsonl")
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--expected-rollouts", type=int, default=8)
    args = parser.parse_args()

    input_path = Path(args.input_parquet)
    stream_arg = args.stream_jsonl or args.metadata_jsonl
    if not stream_arg:
        raise ValueError("--stream-jsonl is required")
    stream_path = Path(stream_arg)
    output_path = Path(args.output_parquet)

    table = pq.read_table(input_path)
    if "extra_info" not in table.column_names:
        raise ValueError("input parquet has no extra_info column")

    counts = read_completed_counts(stream_path)
    extra_infos = table.column("extra_info").to_pylist()
    keep_mask = []
    source_counts: Counter[str] = Counter()
    for extra in extra_infos:
        extra = extra or {}
        uid = str(extra.get("uid") or "")
        source = str(extra.get("source_dataset") or extra.get("source_benchmark") or "")
        keep = counts.get(uid, 0) < args.expected_rollouts
        keep_mask.append(keep)
        if keep:
            source_counts[source] += 1

    remaining = table.filter(pa.array(keep_mask))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(remaining, output_path)

    print(f"input_rows={table.num_rows}")
    print(f"metadata_rollout_rows={sum(counts.values())}")
    print(f"completed_uids={sum(1 for value in counts.values() if value >= args.expected_rollouts)}")
    print(f"remaining_rows={remaining.num_rows}")
    print(f"output={output_path}")
    for source, count in source_counts.most_common():
        print(f"remaining_source {source} {count}")


if __name__ == "__main__":
    main()
