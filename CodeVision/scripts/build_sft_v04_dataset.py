#!/usr/bin/env python3
"""Build the v04 CodeVision SFT mixture with a capped orientation prior.

The intended v04 recipe is:
  1. keep v03 offline-pipeline rows whose metadata has source_dataset;
  2. drop retained rows that already call rotate/flip, so orientation examples
     are controlled by the explicit quotas below;
  3. add a small quota from the original CodeVision-SFT base:
       rotate=100, flip=50, crop=1000;
  4. rewrite every row to the RL-aligned sp3 + sftclean tool schema system.

The original 6k CodeVision-SFT JSON found at ToolVision/codevision_sft.json is
not image-aligned with the current LLaMA-Factory data/codevision_images folder.
When adding rows from that JSON, pass --base-image-root pointing to the real
dataset root that contains codevision_images/.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


WORKSPACE = Path("/mnt/cpfs/delinmao")
LF_DATA_DIR = WORKSPACE / "CodeVision/LLaMA-Factory/data"
DEFAULT_V03 = LF_DATA_DIR / "codevision_sft_mix200_simple_notool_sp3_v03.json"
DEFAULT_OUTPUT = LF_DATA_DIR / "codevision_sft_mix200_simple_notool_sp3_v04.json"
DEFAULT_BASE = WORKSPACE / "ToolVision/codevision_sft.json"
DEFAULT_SYSTEM_PROMPT = WORKSPACE / "ToolVision/CodeVision/recipe/codevision/config/sp3.txt"
DEFAULT_TOOL_CONFIG = WORKSPACE / "ToolVision/CodeVision/recipe/codevision/config/code_image_tool_config_v03_sftclean.yaml"
DEFAULT_STATS = WORKSPACE / "ToolVision/CodeVision/outputs/analysis/sft_v04_dataset_stats.json"

ROTATE_RE = re.compile(r"(?:\b|[.])rotate\s*\(", re.IGNORECASE)
FLIP_RE = re.compile(
    r"ImageOps\.(?:flip|mirror)\s*\(|transpose\s*\(\s*Image\.(?:FLIP|Transpose\.FLIP)",
    re.IGNORECASE,
)
CROP_RE = re.compile(r"_call_(?:manual|dino)_crop\s*\(|[.]crop\s*\(", re.IGNORECASE)

TOOL_WRAPPER = """# Tool
You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_schema}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v03-input", type=Path, default=DEFAULT_V03)
    parser.add_argument("--base-input", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--base-image-root",
        type=Path,
        default=None,
        help="Root containing original base images. It may be the dataset root or the codevision_images directory.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats-output", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--tool-config", type=Path, default=DEFAULT_TOOL_CONFIG)
    parser.add_argument("--rotate-quota", type=int, default=100)
    parser.add_argument("--flip-quota", type=int, default=50)
    parser.add_argument("--crop-quota", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument(
        "--keep-retained-orientation",
        action="store_true",
        help="Keep non-old v03 rows that already call rotate/flip. Default drops them.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only print candidate/retention stats. Does not write the output dataset.",
    )
    parser.add_argument(
        "--allow-missing-base-images",
        action="store_true",
        help="Allow plan-only stats when --base-image-root is missing. Never used for materialization.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata")
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {"_raw_metadata": raw}
    return {}


def set_metadata(row: dict[str, Any], payload: dict[str, Any]) -> None:
    row["metadata"] = json.dumps(payload, ensure_ascii=False)


def assistant_text(row: dict[str, Any]) -> str:
    chunks: list[str] = []
    for msg in row.get("conversations") or []:
        if msg.get("from") == "gpt":
            chunks.append(str(msg.get("value") or ""))
    return "\n".join(chunks)


def tool_ops(row: dict[str, Any]) -> set[str]:
    text = assistant_text(row)
    ops: set[str] = set()
    if ROTATE_RE.search(text):
        ops.add("rotate")
    if FLIP_RE.search(text):
        ops.add("flip")
    if CROP_RE.search(text):
        ops.add("crop")
    return ops


def row_key(row: dict[str, Any]) -> str:
    payload = {
        "conversations": row.get("conversations"),
        "images": row.get("images"),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def build_system_prompt(system_prompt_path: Path, tool_config_path: Path) -> str:
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    config = yaml.safe_load(tool_config_path.read_text(encoding="utf-8"))
    tool_schema = config["tools"][0]["tool_schema"]
    code_schema = tool_schema["function"]["parameters"]["properties"].get("code")
    if isinstance(code_schema, dict) and isinstance(code_schema.get("description"), str):
        code_schema["description"] = code_schema["description"].replace("\n", "\\n")
    schema_text = json.dumps(tool_schema, ensure_ascii=False)
    return f"{system_prompt}\n{TOOL_WRAPPER.format(tool_schema=schema_text)}"


def source_dataset(row: dict[str, Any]) -> str | None:
    src = metadata(row).get("source_dataset")
    return str(src) if src else None


def retain_v03_rows(rows: list[dict[str, Any]], *, keep_orientation: bool) -> tuple[list[dict[str, Any]], Counter]:
    kept: list[dict[str, Any]] = []
    stats: Counter = Counter()
    for row in rows:
        src = source_dataset(row)
        ops = tool_ops(row)
        if src is None:
            stats["drop_old_missing_source"] += 1
            continue
        if not keep_orientation and ("rotate" in ops or "flip" in ops):
            stats["drop_retained_orientation"] += 1
            stats[f"drop_retained_orientation_source::{src}"] += 1
            continue
        kept.append(copy.deepcopy(row))
        stats["retained"] += 1
        stats[f"retained_source::{src}"] += 1
    return kept, stats


def bucket_base_rows(rows: list[dict[str, Any]]) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    buckets: dict[str, list[tuple[int, dict[str, Any]]]] = {"rotate": [], "flip": [], "crop": []}
    for idx, row in enumerate(rows):
        ops = tool_ops(row)
        if "rotate" in ops and "flip" not in ops:
            buckets["rotate"].append((idx, row))
        elif "flip" in ops and "rotate" not in ops:
            buckets["flip"].append((idx, row))
        elif "crop" in ops and "rotate" not in ops and "flip" not in ops:
            buckets["crop"].append((idx, row))
    return buckets


def resolve_image_path(root: Path, rel_path: str) -> Path:
    rel = Path(rel_path)
    candidate = root / rel
    if candidate.exists():
        return candidate
    if root.name == "codevision_images" and len(rel.parts) >= 2 and rel.parts[0] == "codevision_images":
        return root / Path(*rel.parts[1:])
    return candidate


def missing_images_for_row(row: dict[str, Any], image_root: Path) -> list[str]:
    missing: list[str] = []
    for rel in row.get("images") or []:
        if not resolve_image_path(image_root, str(rel)).exists():
            missing.append(str(rel))
    return missing


def filter_image_available(
    candidates: list[tuple[int, dict[str, Any]]],
    *,
    image_root: Path | None,
) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    if image_root is None:
        return [], len(candidates)
    available: list[tuple[int, dict[str, Any]]] = []
    missing = 0
    for idx, row in candidates:
        if missing_images_for_row(row, image_root):
            missing += 1
        else:
            available.append((idx, row))
    return available, missing


def select_bucket(
    candidates: list[tuple[int, dict[str, Any]]],
    quota: int,
    *,
    rng: random.Random,
    bucket: str,
) -> list[tuple[int, dict[str, Any]]]:
    if quota <= 0:
        return []
    if len(candidates) < quota:
        raise RuntimeError(f"Not enough {bucket} candidates: need {quota}, have {len(candidates)}")
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return shuffled[:quota]


def copy_base_row(
    row: dict[str, Any],
    *,
    source_index: int,
    bucket: str,
    image_root: Path,
    output_image_dir: Path,
    new_system: str,
) -> dict[str, Any]:
    new_row = copy.deepcopy(row)
    new_images: list[str] = []
    for image_idx, rel in enumerate(row.get("images") or []):
        src = resolve_image_path(image_root, str(rel))
        if not src.exists():
            raise FileNotFoundError(f"Missing source image for base row {source_index}: {rel}")
        suffix = src.suffix or ".png"
        dest_name = f"v04_base_{bucket}_{source_index}_{image_idx}{suffix}"
        dest = output_image_dir / dest_name
        if not dest.exists():
            shutil.copy2(src, dest)
        new_images.append(f"codevision_images/{dest_name}")

    old_meta = metadata(row)
    new_meta = dict(old_meta)
    original_sample_id = old_meta.get("sample_id", source_index)
    new_meta.update(
        {
            "source_dataset": "codevision_sft_base_v04",
            "source_sample_id": f"codevision_sft_base__{original_sample_id}",
            "source_row_index": source_index,
            "v04_bucket": bucket,
            "v04_origin": "base_quota",
            "v04_tool_ops": sorted(tool_ops(row)),
        }
    )
    new_row["images"] = new_images
    new_row["system"] = new_system
    set_metadata(new_row, new_meta)
    return new_row


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts: Counter = Counter()
    op_counts: Counter = Counter()
    transform_counts: Counter = Counter()
    for row in rows:
        meta = metadata(row)
        source_counts[meta.get("source_dataset", "<none>")] += 1
        transform_counts[str(meta.get("transform", "<none>"))] += 1
        ops = tool_ops(row)
        if not ops:
            op_counts["no_detected_tool_op"] += 1
        for op in sorted(ops):
            op_counts[op] += 1
        if "crop" in ops and "rotate" not in ops and "flip" not in ops:
            op_counts["safe_crop"] += 1
    return {
        "total": len(rows),
        "source_counts": dict(source_counts.most_common()),
        "op_counts": dict(op_counts.most_common()),
        "transform_counts": dict(transform_counts.most_common()),
    }


def cleanup_stale_v04_images(output_image_dir: Path, rows: list[dict[str, Any]]) -> int:
    referenced = {
        Path(str(image)).name
        for row in rows
        for image in (row.get("images") or [])
        if Path(str(image)).name.startswith("v04_base_")
    }
    removed = 0
    for path in output_image_dir.glob("v04_base_*"):
        if path.name not in referenced:
            path.unlink()
            removed += 1
    return removed


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    if args.base_image_root is not None and args.base_input.resolve() == DEFAULT_BASE.resolve():
        unsafe_roots = {LF_DATA_DIR.resolve(), (LF_DATA_DIR / "codevision_images").resolve()}
        if args.base_image_root.resolve() in unsafe_roots:
            raise RuntimeError(
                "Do not use the current LLaMA-Factory data/codevision_images as --base-image-root for "
                "ToolVision/codevision_sft.json. That JSON is not image-aligned with the current LF image directory."
            )

    v03_rows = load_json(args.v03_input)
    base_rows = load_json(args.base_input)
    new_system = build_system_prompt(args.system_prompt, args.tool_config)

    retained, retain_stats = retain_v03_rows(v03_rows, keep_orientation=bool(args.keep_retained_orientation))
    base_buckets = bucket_base_rows(base_rows)
    quota_by_bucket = {
        "rotate": int(args.rotate_quota),
        "flip": int(args.flip_quota),
        "crop": int(args.crop_quota),
    }

    candidate_stats: dict[str, Any] = {}
    selected_by_bucket: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for bucket, quota in quota_by_bucket.items():
        candidates = base_buckets[bucket]
        available, missing_count = filter_image_available(candidates, image_root=args.base_image_root)
        candidate_stats[bucket] = {
            "quota": quota,
            "raw_candidates": len(candidates),
            "image_available_candidates": len(available),
            "missing_image_candidates": missing_count,
        }
        if args.plan_only:
            selected_by_bucket[bucket] = []
            continue
        if args.base_image_root is None:
            raise RuntimeError(
                "Materializing base quota rows requires --base-image-root. "
                "The ToolVision/codevision_sft.json image paths are not aligned with current LLaMA-Factory data."
            )
        selected_by_bucket[bucket] = select_bucket(available, quota, rng=rng, bucket=bucket)

    plan = {
        "seed": args.seed,
        "v03_input": str(args.v03_input),
        "base_input": str(args.base_input),
        "base_image_root": str(args.base_image_root) if args.base_image_root else None,
        "output": str(args.output),
        "system_prompt": str(args.system_prompt),
        "tool_config": str(args.tool_config),
        "quotas": quota_by_bucket,
        "keep_retained_orientation": bool(args.keep_retained_orientation),
        "retain_stats": dict(retain_stats.most_common()),
        "base_candidate_stats": candidate_stats,
        "retained_summary": summarize_rows(retained),
        "plan_only": bool(args.plan_only),
    }

    if args.plan_only:
        write_json(args.stats_output, plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    output_image_dir = args.output.parent / "codevision_images"
    output_image_dir.mkdir(parents=True, exist_ok=True)

    added_rows: list[dict[str, Any]] = []
    added_manifest: list[dict[str, Any]] = []
    assert args.base_image_root is not None
    for bucket in ("rotate", "flip", "crop"):
        for source_index, row in selected_by_bucket[bucket]:
            added_rows.append(
                copy_base_row(
                    row,
                    source_index=source_index,
                    bucket=bucket,
                    image_root=args.base_image_root,
                    output_image_dir=output_image_dir,
                    new_system=new_system,
                )
            )
            added_manifest.append({"bucket": bucket, "source_row_index": source_index, "source_images": row.get("images")})

    for row in retained:
        row["system"] = new_system

    combined = retained + added_rows
    rng.shuffle(combined)

    keys = [row_key(row) for row in combined]
    duplicate_count = len(keys) - len(set(keys))
    if duplicate_count:
        raise RuntimeError(f"Duplicate rows detected after v04 build: {duplicate_count}")

    output_summary = summarize_rows(combined)
    removed_stale_images = cleanup_stale_v04_images(output_image_dir, combined)
    plan.update(
        {
            "added_manifest": added_manifest,
            "added_counts": dict(Counter(item["bucket"] for item in added_manifest)),
            "removed_stale_v04_base_images": removed_stale_images,
            "output_summary": output_summary,
        }
    )
    write_json(args.output, combined)
    write_json(args.stats_output, plan)
    compact_plan = dict(plan)
    compact_plan["added_manifest_count"] = len(added_manifest)
    compact_plan["stats_output"] = str(args.stats_output)
    compact_plan.pop("added_manifest", None)
    print(json.dumps(compact_plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
