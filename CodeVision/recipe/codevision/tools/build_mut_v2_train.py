#!/usr/bin/env python3
"""Build MUT v2 balanced-order RL training parquet.

Input is the MUT v1 train parquet.  Output keeps the original RL schema but
rewrites only extra_info labels used by the reward:

  mut            -> mut_weight=0.5, regular_tool_penalty=0.0
  weak_clean     -> mut_weight=0.2, regular_tool_penalty=0.0
  hard_regular   -> mut_weight=0.0, regular_tool_penalty=0.0
  regular_9_15   -> mut_weight=0.0, regular_tool_penalty=0.05

The output rows are ordered in fixed 64-row blocks:

  regular_9_15=28, hard_regular=10, mut=20, weak_clean=6

Use data.shuffle=False for this parquet if strict per-batch composition is
desired.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = (
    "/mnt/cpfs/delinmao/ToolVision/CodeVision/outputs/analysis/"
    "mut_v1_20260616/mut_v1_train.parquet"
)
DEFAULT_OUTPUT = (
    "/mnt/cpfs/delinmao/ToolVision/CodeVision/outputs/analysis/"
    "mut_v2_20260617/mut_v2_train_balanced.parquet"
)

BATCH_QUOTAS = {
    "regular_9_15": 28,
    "hard_regular": 10,
    "mut": 20,
    "weak_clean": 6,
}

MUT_WEIGHTS = {
    "mut": 0.5,
    "weak_clean": 0.2,
    "hard_regular": 0.0,
    "regular_9_15": 0.0,
}

REGULAR_TOOL_PENALTIES = {
    "mut": 0.0,
    "weak_clean": 0.0,
    "hard_regular": 0.0,
    "regular_9_15": 0.05,
}


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def classify(extra: dict[str, Any]) -> str:
    group = str(extra.get("train_group") or "")
    if group == "mut":
        return "mut"
    if group == "regular":
        return "regular_9_15"
    if group == "weak":
        ntc = _float_or_none(extra.get("mut_v1_NTC"))
        if ntc is not None and ntc == 0:
            return "weak_clean"
        return "hard_regular"
    raise ValueError(f"Unexpected train_group={group!r} for uid={extra.get('uid')!r}")


def _cycle_take(pool: list[int], state: dict[str, Any], n: int, rng: random.Random) -> list[int]:
    out: list[int] = []
    while len(out) < n:
        cursor = state["cursor"]
        order = state["order"]
        if cursor >= len(order):
            order = list(pool)
            rng.shuffle(order)
            state["order"] = order
            state["cursor"] = 0
            cursor = 0
        take = min(n - len(out), len(order) - cursor)
        out.extend(order[cursor : cursor + take])
        state["cursor"] = cursor + take
    return out


def build(input_path: Path, output_path: Path, summary_path: Path, seed: int) -> None:
    rng = random.Random(seed)
    df = pd.read_parquet(input_path)
    rows = df.to_dict("records")

    pools: dict[str, list[int]] = defaultdict(list)
    extras: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        extra = _as_dict(row.get("extra_info"))
        klass = classify(extra)
        pools[klass].append(idx)
        extras.append(extra)

    missing = sorted(set(BATCH_QUOTAS) - set(pools))
    if missing:
        raise RuntimeError(f"Missing v2 classes in input: {missing}")

    blocks = max(math.ceil(len(pools[name]) / quota) for name, quota in BATCH_QUOTAS.items())
    states = {}
    for name, pool in pools.items():
        order = list(pool)
        rng.shuffle(order)
        states[name] = {"order": order, "cursor": 0}

    selected: list[tuple[int, str, int, int]] = []
    for block_idx in range(blocks):
        block_items: list[tuple[int, str, int, int]] = []
        for klass, quota in BATCH_QUOTAS.items():
            taken = _cycle_take(pools[klass], states[klass], quota, rng)
            for local_slot, row_idx in enumerate(taken):
                block_items.append((row_idx, klass, block_idx, local_slot))
        rng.shuffle(block_items)
        selected.extend(block_items)

    per_original_uid_counter: Counter[str] = Counter()
    output_rows: list[dict[str, Any]] = []
    for global_idx, (row_idx, klass, block_idx, local_slot) in enumerate(selected):
        row = copy.deepcopy(rows[row_idx])
        extra = copy.deepcopy(extras[row_idx])
        original_uid = str(extra.get("uid") or f"row_{row_idx}")
        dup_index = per_original_uid_counter[original_uid]
        per_original_uid_counter[original_uid] += 1

        extra["mut_v2_original_uid"] = original_uid
        extra["uid"] = f"mutv2_{global_idx:08d}_{original_uid}"
        extra["train_group"] = klass
        extra["mut_v2_class"] = klass
        extra["mut_weight"] = MUT_WEIGHTS[klass]
        extra["regular_tool_penalty"] = REGULAR_TOOL_PENALTIES[klass]
        extra["mut_v2_batch_block"] = block_idx
        extra["mut_v2_batch_slot"] = local_slot
        extra["mut_v2_duplicate_index"] = dup_index
        extra["mut_v2_batch_ratio"] = dict(BATCH_QUOTAS)
        extra["mut_v2_data_shuffle_required"] = False
        row["extra_info"] = extra
        output_rows.append(row)

    out_df = pd.DataFrame(output_rows, columns=df.columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    class_counts = Counter(row["extra_info"]["mut_v2_class"] for row in output_rows)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in output_rows:
        extra = row["extra_info"]
        source_counts[str(extra.get("source_dataset") or row.get("data_source") or "unknown")][
            extra["mut_v2_class"]
        ] += 1

    duplicate_hist = Counter(per_original_uid_counter.values())
    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows_in": int(len(df)),
        "rows_out": int(len(out_df)),
        "seed": seed,
        "batch_size": int(sum(BATCH_QUOTAS.values())),
        "blocks": int(blocks),
        "batch_quotas": dict(BATCH_QUOTAS),
        "class_counts_input": {k: len(v) for k, v in sorted(pools.items())},
        "class_counts_output": dict(sorted(class_counts.items())),
        "mut_weights": dict(MUT_WEIGHTS),
        "regular_tool_penalties": dict(REGULAR_TOOL_PENALTIES),
        "unique_original_uid": int(len(per_original_uid_counter)),
        "unique_output_uid": int(out_df["extra_info"].map(lambda x: x["uid"]).nunique()),
        "duplicate_count_histogram": {str(k): int(v) for k, v in sorted(duplicate_hist.items())},
        "source_by_class": {
            source: dict(sorted(counter.items())) for source, counter in sorted(source_counts.items())
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    summary_path = Path(args.summary) if args.summary else output_path.with_name("mut_v2_train_summary.json")
    build(Path(args.input), output_path, summary_path, args.seed)


if __name__ == "__main__":
    main()
