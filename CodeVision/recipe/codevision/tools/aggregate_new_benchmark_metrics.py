#!/usr/bin/env python3
"""Print official/data-aligned metrics for new CodeVision benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", payload)
    if not isinstance(metrics, dict):
        raise ValueError(f"Expected metrics object in {path}")
    return metrics


def _metric(metrics: dict[str, float], data_source: str, name: str) -> float | None:
    value = metrics.get(f"val-{data_source}/{name}")
    return None if value is None else float(value)


def _print_cvbench(metrics: dict[str, float]) -> bool:
    ade = _metric(metrics, "cvbench_2d_ade20k", "score")
    coco = _metric(metrics, "cvbench_2d_coco", "score")
    omni = _metric(metrics, "cvbench_3d_omni3d", "score")
    if ade is None or coco is None or omni is None:
        return False
    combined = 0.5 * (((ade + coco) / 2.0) + omni)
    print(f"CV-Bench official_combined={combined:.6f} ade20k_2d={ade:.6f} coco_2d={coco:.6f} omni3d={omni:.6f}")
    return True


def _print_counting(metrics: dict[str, float], data_source: str, label: str) -> bool:
    acc = _metric(metrics, data_source, "score")
    mae = _metric(metrics, data_source, "mae")
    if mae is None:
        mae = _metric(metrics, data_source, "abs_error")
    mse = _metric(metrics, data_source, "squared_error")
    if acc is None:
        return False
    parts = [f"{label} accuracy={acc:.6f}"]
    if mae is not None:
        parts.append(f"MAE={mae:.6f}")
    if mse is not None:
        parts.append(f"MSE={mse:.6f}")
    print(" ".join(parts))
    return True


def _load_ocrbench_v2_utils():
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from recipe.codevision.reward import _load_ocrbench_v2_utils as load_utils

    return load_utils()


def _metadata_path(metrics_json: Path) -> Path:
    candidates = [
        metrics_json.parent / "diagnostics" / "metadata.jsonl",
        metrics_json.parent / "metadata.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _print_ocrbench_v2(metadata_path: Path) -> bool:
    if not metadata_path.exists():
        return False
    payloads = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            payload = row.get("ocrbench_v2_payload")
            if not payload:
                continue
            if isinstance(payload, str):
                payload = json.loads(payload)
            payloads.append(payload)
    if not payloads:
        return False

    utils = _load_ocrbench_v2_utils()
    _, score_buckets = utils._fill_score_buckets(payloads)
    en = utils.calculate_average_score(utils.ENGLISH_TASKS, score_buckets)
    cn = utils.calculate_average_score(utils.CHINESE_TASKS, score_buckets)
    en_count = sum(len(score_buckets[t]) for t in utils.ENGLISH_TASKS)
    cn_count = sum(len(score_buckets[t]) for t in utils.CHINESE_TASKS)
    total = en_count + cn_count
    overall = (en * en_count + cn * cn_count) / total if total > 0 else 0.0
    print(f"OCRBench-v2 lmms_official overall={overall:.6f} en={en:.6f} cn={cn:.6f} n={total}")
    return True


def _print_diagnostics(path: Path) -> None:
    summary_path = path / "diagnostics" / "bucket_summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"diagnostics total={summary.get('total')} sampled_trace_count={summary.get('sampled_trace_count')}")
    quadrants = summary.get("quadrants") or {}
    if quadrants:
        print("quadrants=" + json.dumps(quadrants, ensure_ascii=False, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_json", type=Path, help="Path to saves/CodeVision/<exp>/metrics.json")
    args = parser.parse_args()

    metrics = _load_metrics(args.metrics_json)
    found = False
    found = _print_cvbench(metrics) or found
    found = _print_counting(metrics, "pixmo_count", "Pixmo-Count") or found
    found = _print_counting(metrics, "pixmo_count_lmms", "Pixmo-Count-LMMS") or found
    found = _print_counting(metrics, "countqa", "CountQA") or found
    found = _print_ocrbench_v2(_metadata_path(args.metrics_json)) or found
    _print_diagnostics(args.metrics_json.parent)
    if not found:
        available = "\n".join(sorted(k for k in metrics if any(x in k for x in ["cvbench", "pixmo", "countqa"])))
        raise SystemExit(f"No new benchmark metrics found. Available relevant keys:\n{available}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
