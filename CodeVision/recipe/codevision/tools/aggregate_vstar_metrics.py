"""Aggregate per-category V*Bench metrics from an ``only_test`` run.

Reads the JSON file that ``verl/trainer/ppo/ray_trainer.py`` writes at the end
of ``only_test`` mode (``--val_metrics_output``), prints per-category
accuracy and computes an overall accuracy weighted by sample counts (115 for
``direct_attributes`` and 76 for ``relative_position`` in the stock V*Bench
test set, totalling 191).

Usage::

    python recipe/codevision/tools/aggregate_vstar_metrics.py \\
        ./saves/CodeVision/vstar_base/metrics.json

Optional: pass ``--counts direct_attributes=115,relative_position=76`` to use
custom weights, or ``--counts-from-jsonl <jsonl>`` to derive counts from the
V*Bench jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


DEFAULT_COUNTS = {
    "direct_attributes": 115,
    "relative_position": 76,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metrics_json", type=str, help="Path to metrics.json written by only_test run.")
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy/mean",
        help="Metric suffix under val-<data_source>/ to aggregate. Default: accuracy/mean.",
    )
    parser.add_argument(
        "--counts",
        type=str,
        default=None,
        help='Per-category sample counts, e.g. "direct_attributes=115,relative_position=76".',
    )
    parser.add_argument(
        "--counts-from-jsonl",
        type=str,
        default=None,
        help="Derive category counts by parsing a V*Bench jsonl (reads 'category' field).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Where to write aggregated summary JSON. Defaults to <metrics_json dir>/summary.json.",
    )
    return parser.parse_args()


def parse_counts_arg(text: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"bad --counts token: {part!r}")
        k, v = part.split("=", 1)
        result[k.strip()] = int(v.strip())
    return result


def counts_from_jsonl(path: Path) -> dict[str, int]:
    counter: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            counter[rec["category"]] += 1
    return dict(counter)


def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # ray_trainer.py saves `{"global_steps": ..., "metrics": {...}}`.
    if isinstance(data, dict) and "metrics" in data and isinstance(data["metrics"], dict):
        return data["metrics"]
    return data


def pick_metric(metrics: dict, category: str, metric_suffix: str) -> float | None:
    key = f"val-{category}/{metric_suffix}"
    if key in metrics:
        return float(metrics[key])
    return None


def collect_extra(metrics: dict, category: str, keys: list[str]) -> dict[str, float]:
    extra: dict[str, float] = {}
    for suffix in keys:
        key = f"val-{category}/{suffix}"
        if key in metrics:
            extra[suffix] = float(metrics[key])
    return extra


def main() -> int:
    args = parse_args()
    metrics_path = Path(args.metrics_json).expanduser().resolve()
    if not metrics_path.is_file():
        print(f"[error] metrics file not found: {metrics_path}", file=sys.stderr)
        return 1

    if args.counts and args.counts_from_jsonl:
        print("[error] use only one of --counts or --counts-from-jsonl", file=sys.stderr)
        return 2

    if args.counts:
        counts = parse_counts_arg(args.counts)
    elif args.counts_from_jsonl:
        counts = counts_from_jsonl(Path(args.counts_from_jsonl).expanduser().resolve())
    else:
        counts = dict(DEFAULT_COUNTS)

    metrics = load_metrics(metrics_path)

    extra_metric_suffixes = [
        "format_reward/mean",
        "num_turns/mean",
        "tool_call_counts/mean",
    ]

    per_cat: dict[str, dict] = {}
    numer = 0.0
    denom = 0
    missing: list[str] = []
    for category, n in counts.items():
        acc = pick_metric(metrics, category, args.metric)
        if acc is None:
            missing.append(category)
            per_cat[category] = {"n": n, args.metric: None}
            continue
        per_cat[category] = {
            "n": n,
            args.metric: acc,
            "extra": collect_extra(metrics, category, extra_metric_suffixes),
        }
        numer += acc * n
        denom += n

    overall = (numer / denom) if denom > 0 else None

    summary = {
        "metrics_file": str(metrics_path),
        "metric": args.metric,
        "per_category": per_cat,
        "overall_weighted": overall,
        "total_samples": denom,
        "missing_categories": missing,
    }

    print("=" * 60)
    print(f"metrics file : {metrics_path}")
    print(f"metric       : val-<category>/{args.metric}")
    print("-" * 60)
    for cat, info in per_cat.items():
        acc = info[args.metric]
        acc_str = f"{acc:.4f}" if acc is not None else "MISSING"
        print(f"  {cat:<24s} n={info['n']:<4d} acc={acc_str}")
        extra = info.get("extra", {})
        for k, v in extra.items():
            print(f"      {k:<24s} {v:.4f}")
    print("-" * 60)
    if overall is not None:
        print(f"overall (weighted by n, total={denom}): {overall:.4f}")
    else:
        print("overall: N/A (no category metrics found)")
    if missing:
        print(f"[warn] missing categories in metrics.json: {missing}")
    print("=" * 60)

    out_path = Path(args.out).expanduser().resolve() if args.out else metrics_path.parent / "summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[ok] wrote summary -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
