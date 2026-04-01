#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.external_services.paddleocr.client import PaddleOCRHTTPClient


def _sanitize_json(value: Any) -> Any:
    image_keys = {"ocrImage", "docPreprocessingImage", "inputImage"}
    if isinstance(value, dict):
        return {k: _sanitize_json(v) for k, v in value.items() if k not in image_keys}
    if isinstance(value, list):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, str) and len(value) > 512:
        return f"<omitted string len={len(value)}>"
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact PaddleOCR service probe.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="PaddleOCR service base URL.")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write sanitized OCR JSON.",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    image = Image.open(image_path).convert("RGB")
    client = PaddleOCRHTTPClient(
        base_url=args.base_url,
        request_timeout=args.timeout,
    )
    result = client.infer(image, {})
    meta = result.get("meta", {}) or {}
    pages = meta.get("ocr_pages", []) or []

    print(f"status=ok base_url={args.base_url}")
    print(f"image={image_path}")
    print(f"num_pages={len(pages)} num_ocr_items={meta.get('num_ocr_items', 0)}")
    print("text:")
    print(result.get("text", "").strip() or "<empty>")

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        payload = {
            "text": result.get("text", ""),
            "meta": _sanitize_json(meta),
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_json={output_path}")


if __name__ == "__main__":
    main()
