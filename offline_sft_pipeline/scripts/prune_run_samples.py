#!/usr/bin/env python3
"""Prune bad samples from an existing dataset pipeline run so RESUME can re-run them.

Typical use:
  python offline_sft_pipeline/scripts/prune_run_samples.py \
    --run-root /path/to/run_root \
    --criterion root_zero_step_answered \
    --apply

This script assumes the target run is NOT actively writing.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove selected samples from store/logs/results within one run_root so "
            "RESUME=1 will re-run them."
        )
    )
    parser.add_argument(
        "--run-root",
        type=str,
        required=True,
        help="Run directory under offline_sft_pipeline/outputs/dataset_pipeline_runs/.",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        choices=("root_zero_step_answered",),
        default="root_zero_step_answered",
        help=(
            "Built-in pruning rule. "
            "'root_zero_step_answered' removes samples whose root trajectory answered with zero tool steps."
        ),
    )
    parser.add_argument(
        "--sample-ids-file",
        type=str,
        default="",
        help="Optional explicit sample_id list. If provided, it overrides --criterion.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete and rewrite files. Default is dry-run.",
    )
    return parser.parse_args()


def build_root_trajectory_id(sample_id: str) -> str:
    return f"traj__{sample_id}__root"


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_sample_ids_from_file(path: Path) -> set[str]:
    sample_ids: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if text:
            sample_ids.add(text)
    return sample_ids


def should_prune_root_zero_step_answered(sample_dir: Path) -> bool:
    sample_id = sample_dir.name
    root_traj_path = (
        sample_dir
        / "trajectories"
        / build_root_trajectory_id(sample_id)
        / "trajectory.json"
    )
    if not root_traj_path.is_file():
        return False
    trajectory = json.loads(root_traj_path.read_text(encoding="utf-8"))
    if trajectory.get("status") != "answered":
        return False
    return len(trajectory.get("steps") or []) == 0


def collect_target_sample_ids(run_root: Path, *, criterion: str, sample_ids_file: Path | None) -> set[str]:
    if sample_ids_file is not None:
        return load_sample_ids_from_file(sample_ids_file)

    store_samples_dir = run_root / "store" / "samples"
    if not store_samples_dir.is_dir():
        return set()

    sample_ids: set[str] = set()
    for sample_dir in sorted(store_samples_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        if criterion == "root_zero_step_answered" and should_prune_root_zero_step_answered(sample_dir):
            sample_ids.add(sample_dir.name)
    return sample_ids


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"run_root not found: {run_root}")

    sample_ids_file = Path(args.sample_ids_file).expanduser().resolve() if args.sample_ids_file else None
    if sample_ids_file is not None and not sample_ids_file.is_file():
        raise FileNotFoundError(f"sample_ids_file not found: {sample_ids_file}")

    target_sample_ids = collect_target_sample_ids(
        run_root,
        criterion=args.criterion,
        sample_ids_file=sample_ids_file,
    )
    store_samples_dir = run_root / "store" / "samples"
    logs_dir = run_root / "logs"
    sample_results_path = run_root / "sample_results.jsonl"
    answered_results_path = run_root / "answered_results.jsonl"

    sample_rows = load_jsonl(sample_results_path)
    answered_rows = load_jsonl(answered_results_path)

    kept_sample_rows = [row for row in sample_rows if str(row.get("sample_id") or "").strip() not in target_sample_ids]
    kept_answered_rows = [
        row for row in answered_rows if str(row.get("sample_id") or "").strip() not in target_sample_ids
    ]

    summary = {
        "run_root": str(run_root),
        "criterion": args.criterion if sample_ids_file is None else "explicit_sample_ids_file",
        "target_sample_count": len(target_sample_ids),
        "target_sample_ids_preview": sorted(target_sample_ids)[:20],
        "sample_results_removed": len(sample_rows) - len(kept_sample_rows),
        "answered_results_removed": len(answered_rows) - len(kept_answered_rows),
        "store_dirs_to_remove": [
            str((store_samples_dir / sample_id).resolve()) for sample_id in sorted(target_sample_ids)
        ][:20],
        "log_files_to_remove": [
            str((logs_dir / f"{sample_id}.log").resolve()) for sample_id in sorted(target_sample_ids)
        ][:20],
        "apply": bool(args.apply),
    }

    if args.apply:
        for sample_id in sorted(target_sample_ids):
            sample_dir = store_samples_dir / sample_id
            if sample_dir.exists():
                shutil.rmtree(sample_dir)
            log_path = logs_dir / f"{sample_id}.log"
            if log_path.exists():
                log_path.unlink()
        write_jsonl(sample_results_path, kept_sample_rows)
        write_jsonl(answered_results_path, kept_answered_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
