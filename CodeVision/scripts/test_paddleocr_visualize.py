#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Any

import requests
from PIL import Image


def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_base64_image(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def save_if_present(page: dict[str, Any], key: str, save_path: Path) -> tuple[bool, tuple[int, int] | None]:
    value = page.get(key)
    if not isinstance(value, str) or not value:
        return False, None
    image = decode_base64_image(value)
    image.save(save_path)
    return True, image.size


def main() -> None:
    parser = argparse.ArgumentParser(description="Test local PaddleOCR service visualize output.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="PaddleOCR base URL.")
    parser.add_argument("--out-dir", default="outputs/test_paddleocr_visualize", help="Output directory.")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    payload = {
        "file": image_to_base64(image),
        "fileType": 1,
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useTextlineOrientation": True,
        "textDetLimitSideLen": 960,
        "textDetLimitType": "max",
        "textDetThresh": 0.4,
        "textDetBoxThresh": 0.7,
        "textRecScoreThresh": 0.6,
        "visualize": True,
    }

    session = requests.Session()
    session.trust_env = False
    response = session.post(f"{args.base_url.rstrip('/')}/ocr", json=payload, timeout=args.timeout)
    response.raise_for_status()
    data = response.json()
    if int(data.get("errorCode", -1)) != 0:
        raise RuntimeError(json.dumps(data, ensure_ascii=False, indent=2)[:4000])

    result = data.get("result") or {}
    pages = result.get("ocrResults") or []
    if not pages:
        raise RuntimeError("OCR response has no pages.")
    page = pages[0]
    pruned = page.get("prunedResult") or {}

    with (out_dir / "response.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    saved: dict[str, Any] = {}
    for key, filename in (
        ("inputImage", "input_image.png"),
        ("docPreprocessingImage", "doc_preprocessing_image.png"),
        ("ocrImage", "ocr_image.png"),
    ):
        ok, size = save_if_present(page, key, out_dir / filename)
        saved[key] = {"present": ok, "size": size}

    model_settings = pruned.get("model_settings") or {}
    doc_preprocessor_res = pruned.get("doc_preprocessor_res") or {}
    summary = {
        "input_image": str(image_path),
        "orig_size": image.size,
        "saved_images": saved,
        "model_settings": model_settings,
        "doc_preprocessor_model_settings": doc_preprocessor_res.get("model_settings"),
        "doc_preprocessor_angle": doc_preprocessor_res.get("angle"),
        "num_texts": len((pruned.get("rec_texts") or [])),
        "first_texts": (pruned.get("rec_texts") or [])[:10],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
