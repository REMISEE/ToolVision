"""Create an incremental repaired ToolVision RL manifest.

This does not overwrite the original sampled manifest or train parquet.  It
removes sampled rows whose question still contains image placeholders, then
fills the same sample slots from the unused candidate pool.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_ROOT = Path("/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k")
DEFAULT_OUTPUT_ROOT = Path("/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k_clean_no_extra_image")
DEFAULT_SEED = 20260521
IMAGE_TAG_RE = re.compile(r"<\s*image\s*>", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--sampled-manifest",
        type=Path,
        default=None,
        help="Defaults to INPUT_ROOT/manifests/sampled_40k_manifest.parquet.",
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=None,
        help="Defaults to INPUT_ROOT/manifests/candidate_manifest.parquet.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    manifest_dir = output_root / "manifests"
    report_dir = output_root / "reports"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    sampled_path = args.sampled_manifest or input_root / "manifests" / "sampled_40k_manifest.parquet"
    candidate_path = args.candidate_manifest or input_root / "manifests" / "candidate_manifest.parquet"
    sampled = pd.read_parquet(sampled_path)
    candidates = pd.read_parquet(candidate_path)

    bad_mask = contains_image_tag(sampled["question"])
    removed = sampled[bad_mask].copy()
    kept = sampled[~bad_mask].copy()

    used_candidate_ids = set(kept["candidate_row_id"].astype(int).tolist())
    replacement_pool = candidates[
        candidates["keep_after_hard_dedup"].astype(bool)
        & candidates["valid_single_image"].astype(bool)
        & ~candidates["candidate_row_id"].astype(int).isin(used_candidate_ids)
        & ~contains_image_tag(candidates["question"])
    ].copy()

    replacements = choose_replacements(removed, replacement_pool, seed=args.seed)
    if len(replacements) != len(removed):
        raise RuntimeError(f"needed {len(removed)} replacements, found {len(replacements)}")

    repaired = build_repaired_manifest(sampled, kept, removed, replacements)
    repaired_path = manifest_dir / "sampled_40k_manifest.parquet"
    repaired.to_parquet(repaired_path, engine="pyarrow", index=False)

    removed.to_csv(report_dir / "removed_extra_image_tag_rows.csv", index=False)
    replacements.to_csv(report_dir / "replacement_rows.csv", index=False)
    write_reports(sampled, repaired, removed, replacements, report_dir, sampled_path, candidate_path)

    print(f"[ok] wrote repaired manifest -> {repaired_path}")
    print(f"[ok] original sampled manifest left unchanged -> {sampled_path}")
    print(f"[ok] removed rows={len(removed)} replacements={len(replacements)} final rows={len(repaired)}")
    return 0


def contains_image_tag(series: pd.Series) -> pd.Series:
    return series.astype(str).map(lambda text: bool(IMAGE_TAG_RE.search(text)))


def choose_replacements(removed: pd.DataFrame, pool: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    if removed.empty:
        return pool.head(0).copy()

    pieces: list[pd.DataFrame] = []
    remaining_pool = pool.copy()
    for idx, ((origin, source), group) in enumerate(removed.groupby(["origin", "source_dataset"], sort=True)):
        source_pool = remaining_pool[
            (remaining_pool["origin"] == origin) & (remaining_pool["source_dataset"] == source)
        ]
        take = min(len(group), len(source_pool))
        if take:
            piece = source_pool.sample(n=take, random_state=seed + idx)
            pieces.append(piece)
            remaining_pool = remaining_pool[~remaining_pool["candidate_row_id"].isin(piece["candidate_row_id"])]

    selected = pd.concat(pieces, ignore_index=False) if pieces else pool.head(0).copy()
    shortfall = len(removed) - len(selected)
    if shortfall > 0:
        extra = remaining_pool.sample(n=shortfall, random_state=seed + 1000)
        selected = pd.concat([selected, extra], ignore_index=False)
    return selected.copy()


def build_repaired_manifest(
    sampled: pd.DataFrame, kept: pd.DataFrame, removed: pd.DataFrame, replacements: pd.DataFrame
) -> pd.DataFrame:
    replacement_rows = replacements.copy()
    replacement_rows["sample_index"] = sorted(removed["sample_index"].astype(int).tolist())
    replacement_rows["sample_bucket"] = "repair_topup"
    replacement_rows["sample_uid"] = (
        replacement_rows["source_dataset"].astype(str) + "::" + replacement_rows["source_original_id"].astype(str)
    )

    for col in sampled.columns:
        if col not in replacement_rows.columns:
            replacement_rows[col] = default_value_for_column(col)
    replacement_rows = replacement_rows[sampled.columns]

    repaired = pd.concat([kept[sampled.columns], replacement_rows], ignore_index=True)
    repaired = repaired.sort_values("sample_index").reset_index(drop=True)

    if len(repaired) != len(sampled):
        raise RuntimeError(f"row count changed: {len(sampled)} -> {len(repaired)}")
    if contains_image_tag(repaired["question"]).any():
        raise RuntimeError("repaired manifest still contains residual <image> tags")
    if repaired["sample_index"].astype(int).tolist() != list(range(len(repaired))):
        raise RuntimeError("sample_index is not contiguous after repair")
    if repaired["candidate_row_id"].duplicated().any():
        raise RuntimeError("duplicate candidate_row_id after repair")
    return repaired


def default_value_for_column(col: str) -> Any:
    if col in {"sample_index", "candidate_row_id", "raw_row", "image_count", "image_group_size", "image_group_rank"}:
        return -1
    if col.startswith("is_") or col.startswith("valid_") or col.startswith("duplicate_") or col.startswith("keep_"):
        return False
    return ""


def write_reports(
    sampled: pd.DataFrame,
    repaired: pd.DataFrame,
    removed: pd.DataFrame,
    replacements: pd.DataFrame,
    report_dir: Path,
    sampled_path: Path,
    candidate_path: Path,
) -> None:
    summary = {
        "sampled_manifest": str(sampled_path),
        "candidate_manifest": str(candidate_path),
        "original_rows": int(len(sampled)),
        "removed_rows": int(len(removed)),
        "replacement_rows": int(len(replacements)),
        "final_rows": int(len(repaired)),
        "residual_image_tag_rows": int(contains_image_tag(repaired["question"]).sum()),
    }
    (report_dir / "repair_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    removed.groupby(["origin", "source_dataset"], dropna=False).size().reset_index(name="rows").to_csv(
        report_dir / "removed_by_source.csv", index=False
    )
    replacements.groupby(["origin", "source_dataset"], dropna=False).size().reset_index(name="rows").to_csv(
        report_dir / "replacements_by_source.csv", index=False
    )
    repaired.groupby(["origin", "source_dataset"], dropna=False).size().reset_index(name="rows").sort_values(
        ["origin", "source_dataset"]
    ).to_csv(report_dir / "repaired_by_source.csv", index=False)


if __name__ == "__main__":
    raise SystemExit(main())
