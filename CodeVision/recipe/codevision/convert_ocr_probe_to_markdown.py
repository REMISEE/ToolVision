import argparse
import base64
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OCR probe JSON to Markdown + image assets."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to ocr_probe.json (or compatible OCR JSON).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/ocr_probe_markdown",
        help="Directory to save markdown and decoded images.",
    )
    parser.add_argument(
        "--merge-pages",
        action="store_true",
        help="Merge all pages into one doc.md (default: split + also write doc.md).",
    )
    return parser.parse_args()


def _strip_data_url_prefix(raw: str) -> str:
    if raw.startswith("data:") and "," in raw:
        return raw.split(",", 1)[1]
    return raw


def _decode_b64_to_file(b64_text: str, out_path: Path):
    payload = _strip_data_url_prefix(b64_text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(payload))


def _extract_layout_parsing_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    # demo ocr_probe.json
    if isinstance(data.get("meta"), dict):
        layout_result = data["meta"].get("layout_result")
        if isinstance(layout_result, dict):
            pages = layout_result.get("layoutParsingResults")
            if isinstance(pages, list):
                return pages

    # direct wrapped layout_result
    if isinstance(data.get("layout_result"), dict):
        pages = data["layout_result"].get("layoutParsingResults")
        if isinstance(pages, list):
            return pages

    # direct HTTP response body (result)
    if isinstance(data.get("result"), dict):
        pages = data["result"].get("layoutParsingResults")
        if isinstance(pages, list):
            return pages

    # already narrowed payload
    pages = data.get("layoutParsingResults")
    if isinstance(pages, list):
        return pages

    raise RuntimeError(
        "Cannot find layoutParsingResults. Expected one of: "
        "meta.layout_result.layoutParsingResults / layout_result.layoutParsingResults / "
        "result.layoutParsingResults / layoutParsingResults."
    )


def _extract_markdown_text(page_obj: dict[str, Any]) -> str:
    markdown = page_obj.get("markdown")
    if isinstance(markdown, dict):
        text = markdown.get("text")
        if isinstance(text, str):
            return text

    # Fallback: save prunedResult as JSON text
    pruned = page_obj.get("prunedResult")
    if pruned is not None:
        return "```json\n" + json.dumps(pruned, ensure_ascii=False, indent=2) + "\n```"
    return ""


def _extract_markdown_images(page_obj: dict[str, Any]) -> dict[str, str]:
    markdown = page_obj.get("markdown")
    if not isinstance(markdown, dict):
        return {}
    images = markdown.get("images")
    if not isinstance(images, dict):
        return {}
    return {str(k): v for k, v in images.items() if isinstance(v, str)}


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    pages = _extract_layout_parsing_results(data)

    combined_parts: list[str] = []
    total_images = 0

    for idx, page in enumerate(pages, start=1):
        page_title = f"# Page {idx}"
        page_text = _extract_markdown_text(page)
        page_images = _extract_markdown_images(page)

        # Decode images referenced by markdown text
        for rel_path, b64_img in page_images.items():
            save_path = out_dir / rel_path
            _decode_b64_to_file(b64_img, save_path)
            total_images += 1

        # Save per-page markdown
        per_page_md = f"{page_title}\n\n{page_text}\n"
        (out_dir / f"page_{idx:03d}.md").write_text(per_page_md, encoding="utf-8")
        combined_parts.append(per_page_md)

    # Always save merged markdown for convenience
    if args.merge_pages:
        merged = "\n\n---\n\n".join(combined_parts)
    else:
        merged = "\n\n".join(combined_parts)
    (out_dir / "doc.md").write_text(merged, encoding="utf-8")

    summary = {
        "input": str(input_path),
        "output_dir": str(out_dir),
        "pages": len(pages),
        "decoded_images": total_images,
        "merged_markdown": str(out_dir / "doc.md"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

