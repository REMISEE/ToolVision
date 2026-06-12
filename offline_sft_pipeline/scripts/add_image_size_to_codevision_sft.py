#!/usr/bin/env python3
"""Append image size text after <image> tokens in exported CodeVision-SFT json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-json",
        type=Path,
        required=True,
        help="Path to codevision_sft.json.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional base directory for relative image paths. Defaults to dataset json parent.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path. Defaults to overwrite input file.",
    )
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}.")
    return data


def resolve_image_path(image_root: Path, rel_path: str) -> Path:
    path = (image_root / rel_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return path


def image_size_line(image_path: Path) -> str:
    with Image.open(image_path) as img:
        width, height = img.size
    return f"Image size = {int(width)}x{int(height)} pixels."


def inject_prefix(value: str, prefix: str) -> str:
    if not value.startswith("<image>"):
        return value
    body = value[len("<image>") :]
    if body.lstrip().startswith("Image size ="):
        return value
    if body:
        return f"<image>{prefix}\n\n{body}"
    return f"<image>{prefix}"


def main() -> None:
    args = parse_args()
    dataset_json = args.dataset_json.resolve()
    image_root = (args.image_root.resolve() if args.image_root else dataset_json.parent.resolve())
    output_json = args.output_json.resolve() if args.output_json else dataset_json

    rows = load_json(dataset_json)
    updated_rows: list[dict[str, Any]] = []
    updated_messages = 0

    for row in rows:
        row_copy = dict(row)
        images = [str(item) for item in row.get("images") or []]
        image_idx = 0
        conversations: list[dict[str, Any]] = []
        for message in row.get("conversations") or []:
            message_copy = dict(message)
            value = str(message.get("value") or "")
            if value.startswith("<image>"):
                if image_idx >= len(images):
                    raise ValueError(
                        f"Not enough images for row metadata={row.get('metadata')} at conversation index {len(conversations)}."
                    )
                size_text = image_size_line(resolve_image_path(image_root, images[image_idx]))
                updated_value = inject_prefix(value, size_text)
                if updated_value != value:
                    updated_messages += 1
                message_copy["value"] = updated_value
                image_idx += 1
            conversations.append(message_copy)
        row_copy["conversations"] = conversations
        updated_rows.append(row_copy)

    output_json.write_text(json.dumps(updated_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset_json": str(dataset_json),
                "output_json": str(output_json),
                "row_count": len(updated_rows),
                "updated_messages": updated_messages,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
