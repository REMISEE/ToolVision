import json
from typing import Any, Optional


def box_from_any(box: Any) -> Optional[list[int]]:
    if box is None:
        return None
    if hasattr(box, "tolist"):
        box = box.tolist()
    if not isinstance(box, list):
        return None
    if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
        return [int(round(v)) for v in box]

    xs: list[float] = []
    ys: list[float] = []
    for point in box:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        x, y = point
        xs.append(float(x))
        ys.append(float(y))
    if not xs or not ys:
        return None
    return [int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys)))]


def sorted_ocr_items(texts: list[Any], scores: list[Any], boxes: list[Any]) -> list[dict[str, Any]]:
    if not scores or len(scores) != len(texts):
        scores = [1.0] * len(texts)

    items: list[dict[str, Any]] = []
    for text, score, box in zip(texts, scores, boxes):
        normalized = " ".join(str(text).split()).strip()
        if not normalized:
            continue
        rect = box_from_any(box)
        if rect is None:
            continue
        items.append({"text": normalized, "score": float(score), "box": rect})
    items.sort(key=lambda item: (item["box"][1], item["box"][0], item["box"][3], item["box"][2]))
    return items


def group_items_into_lines(items: list[dict[str, Any]], line_y_threshold: float) -> list[str]:
    if not items:
        return []

    line_groups: list[list[dict[str, Any]]] = []
    for item in items:
        _, y1, _, y2 = item["box"]
        h = max(1, y2 - y1)
        center_y = (y1 + y2) / 2.0
        attached = False
        for group in line_groups:
            gy1 = min(member["box"][1] for member in group)
            gy2 = max(member["box"][3] for member in group)
            gh = max(1, gy2 - gy1)
            group_center_y = (gy1 + gy2) / 2.0
            tol = max(h, gh) * line_y_threshold
            if abs(center_y - group_center_y) <= tol:
                group.append(item)
                attached = True
                break
        if not attached:
            line_groups.append([item])

    lines: list[str] = []
    line_groups.sort(key=lambda group: min(member["box"][1] for member in group))
    for group in line_groups:
        group.sort(key=lambda member: (member["box"][0], member["box"][1]))
        line = " ".join(member["text"] for member in group).strip()
        if line:
            lines.append(line)
    return lines


def parse_ocr_page(page: dict[str, Any], line_y_threshold: float) -> dict[str, Any]:
    raw_result = page.get("prunedResult")
    if isinstance(raw_result, dict) and isinstance(raw_result.get("res"), dict):
        raw_result = raw_result["res"]
    if not isinstance(raw_result, dict):
        raw_result = {}

    rec_texts = list(raw_result.get("rec_texts") or [])
    rec_scores = list(raw_result.get("rec_scores") or [])
    rec_boxes = list(
        raw_result.get("rec_boxes")
        or raw_result.get("rec_polys")
        or raw_result.get("dt_polys")
        or []
    )
    ocr_items = sorted_ocr_items(rec_texts, rec_scores, rec_boxes)
    ocr_lines = group_items_into_lines(ocr_items, line_y_threshold)
    return {
        "raw_result": raw_result,
        "ocr_items": ocr_items,
        "ocr_lines": ocr_lines,
        "ocr_text": "\n".join(ocr_lines),
        "num_ocr_items": len(ocr_items),
    }


def parse_ocr_result(ocr_result: dict[str, Any], line_y_threshold: float) -> dict[str, Any]:
    raw_pages = ocr_result.get("ocrResults") or []
    if not isinstance(raw_pages, list):
        raw_pages = []
    parsed_pages = [parse_ocr_page(page, line_y_threshold) for page in raw_pages if isinstance(page, dict)]
    text_chunks = [page["ocr_text"].strip() for page in parsed_pages if page["ocr_text"].strip()]
    text = "\n\n".join(text_chunks)[:8000] or "OCR completed but no text extracted."
    return {
        "raw_pages": raw_pages,
        "ocr_pages": parsed_pages,
        "text": text,
        "num_ocr_items": sum(page["num_ocr_items"] for page in parsed_pages),
        "raw_json": json.dumps(ocr_result, ensure_ascii=False),
    }
