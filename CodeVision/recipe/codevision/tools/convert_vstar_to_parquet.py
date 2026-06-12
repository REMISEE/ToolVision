"""Convert V*Bench ``test_questions.jsonl`` into a Parquet file that is
compatible with the CodeVision evaluation pipeline
(``recipe/codevision/uvtr.py`` + ``verl/workers/reward_manager/uvtr.py``
+ ``recipe/codevision/reward.py``).

Each Parquet row will contain:

- ``data_source``: category name (``direct_attributes`` / ``relative_position``)
  so that ``verl/trainer/ppo/metric_utils.py::process_validation_metrics``
  automatically buckets per-category validation metrics.
- ``ability``: ``"mm_qa"`` (log-only tag).
- ``prompt``: OpenAI-style messages. ``<image>`` placeholder will be replaced
  by ``CustomRLHFDataset._build_messages``.
- ``images``: list of dicts consumable by ``qwen_vl_utils.fetch_image``.
- ``reward_model``: ``{"style": "rule", "ground_truth": "A"}``.
- ``extra_info``: ``{"question": ..., "index": ..., "category": ...}``.

Usage::

    python recipe/codevision/tools/convert_vstar_to_parquet.py \\
        --root /mnt/cpfs/delinmao/Benchmarks/vstar-bench \\
        --out  /mnt/cpfs/delinmao/Benchmarks/vstar-bench/vstar_eval.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="V*Bench root directory containing test_questions.jsonl and category image folders.",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help="Path to test_questions.jsonl. Defaults to <root>/test_questions.jsonl.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output Parquet file path.",
    )
    parser.add_argument(
        "--image-uri-scheme",
        type=str,
        default="file",
        choices=["file", "path"],
        help="How to encode image paths. 'file' uses 'file://' URIs (recommended for qwen_vl_utils); "
        "'path' uses plain absolute paths.",
    )
    return parser.parse_args()


def build_row(rec: dict, root: Path, uri_scheme: str) -> dict:
    image_rel = rec["image"]
    img_path = (root / image_rel).resolve()
    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found: {img_path}")

    if uri_scheme == "file":
        image_ref = f"file://{img_path}"
    else:
        image_ref = str(img_path)

    text = rec["text"]
    category = rec["category"]
    question_id = rec.get("question_id", "")
    label = rec["label"]

    return {
        "data_source": category,
        "ability": "mm_qa",
        "prompt": [
            {
                "role": "user",
                "content": f"<image>\n{text}",
            }
        ],
        "images": [{"image": image_ref}],
        "reward_model": {
            "style": "rule",
            "ground_truth": str(label),
        },
        "extra_info": {
            "question": text,
            "index": str(question_id),
            "category": category,
        },
    }


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"[error] root does not exist or is not a directory: {root}", file=sys.stderr)
        return 1

    jsonl_path = Path(args.jsonl).expanduser().resolve() if args.jsonl else root / "test_questions.jsonl"
    if not jsonl_path.is_file():
        print(f"[error] jsonl not found: {jsonl_path}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    cat_counter: Counter[str] = Counter()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[error] bad json at line {line_no}: {e}", file=sys.stderr)
                return 1
            row = build_row(rec, root=root, uri_scheme=args.image_uri_scheme)
            rows.append(row)
            cat_counter[row["data_source"]] += 1

    if not rows:
        print("[error] no samples parsed from jsonl", file=sys.stderr)
        return 1

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, engine="pyarrow", index=False)

    print(f"[ok] wrote {len(rows)} rows -> {out_path}")
    print("[ok] per-category counts:")
    for cat, n in sorted(cat_counter.items()):
        print(f"    {cat}: {n}")
    print(f"[ok] total: {sum(cat_counter.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
