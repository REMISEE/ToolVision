#!/usr/bin/env python3
"""Print FSC147 official counting metrics from a CodeVision metrics.json file."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", payload)
    if not isinstance(metrics, dict):
        raise ValueError(f"Expected metrics object in {path}")
    return metrics


def _metric(metrics: dict[str, float], split: str, name: str) -> float | None:
    key = f"val-fsc147_{split}/{name}"
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_json", type=Path, help="Path to saves/.../metrics.json")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    args = parser.parse_args()

    metrics = _load_metrics(args.metrics_json)
    found = False
    for split in args.splits:
        mae = _metric(metrics, split, "abs_error")
        mse = _metric(metrics, split, "squared_error")
        if mae is None or mse is None:
            continue
        found = True
        rmse = math.sqrt(max(mse, 0.0))
        rel = _metric(metrics, split, "relative_score")
        print(f"FSC147 {split}: MAE={mae:.4f} RMSE={rmse:.4f}", end="")
        if rel is not None:
            print(f" relative_score={rel:.4f}")
        else:
            print()

    if not found:
        available = "\n".join(sorted(k for k in metrics if "fsc147" in k.lower()))
        raise SystemExit(f"No FSC147 abs_error/squared_error metrics found. Available FSC147 keys:\n{available}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
