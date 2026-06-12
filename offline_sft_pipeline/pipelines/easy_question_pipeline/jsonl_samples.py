"""Load rows from export ``samples.jsonl`` and resolve image path + teacher reference answer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl_row(
    jsonl_path: Path,
    *,
    sample_id: str | None,
    line_index: int | None,
) -> dict[str, Any]:
    lines: list[str] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    if not lines:
        raise ValueError(f"Empty jsonl: {jsonl_path}")

    if sample_id is not None:
        for raw in lines:
            row = json.loads(raw)
            if row.get("sample_id") == sample_id:
                return row
        raise ValueError(f"sample_id not found in {jsonl_path}: {sample_id!r}")

    idx = 0 if line_index is None else int(line_index)
    if idx < 0 or idx >= len(lines):
        raise ValueError(f"line_index out of range: {idx} (file has {len(lines)} lines)")
    return json.loads(lines[idx])


def reference_answer_from_row(row: dict[str, Any], *, dataset_dir_name: str | None) -> str:
    """Teacher string for easy planner. TextVQA uses ``metadata.model_filtered_resps`` when log_exact_match pipeline applies."""
    meta = row.get("metadata") or {}
    src = str(meta.get("source_dataset") or "").strip().lower()
    folder = (dataset_dir_name or "").strip().lower()

    if folder == "textvqa" or src == "textvqa":
        v = meta.get("model_filtered_resps")
        if v is None or not str(v).strip():
            raise ValueError("textvqa row: missing or empty metadata.model_filtered_resps")
        return str(v).strip()

    ans = row.get("answer")
    if ans is None:
        raise ValueError("Row has no answer field")
    if isinstance(ans, list):
        if not ans:
            raise ValueError("Row has empty answer list")
        return str(ans[0]).strip()
    s = str(ans).strip()
    if not s:
        raise ValueError("Row has empty answer")
    return s


def resolve_image_and_question(
    export_root: Path,
    row: dict[str, Any],
) -> tuple[Path, str, str]:
    """Absolute image path, question text, sample_id string."""
    images = row.get("images") or []
    if not images:
        raise ValueError("Row has no images[]")
    rel = images[0].get("path") or images[0].get("relpath")
    if not rel:
        raise ValueError("Row images[0] has no path")
    rel_path = Path(str(rel))
    if rel_path.is_absolute():
        image_path = rel_path
    else:
        image_path = (export_root / rel_path).resolve()
    question = str(row.get("question") or "").strip()
    if not question:
        raise ValueError("Row has empty question")
    sid = row.get("sample_id")
    sample_id_str = str(sid).strip() if sid else "easy__from_jsonl"
    return image_path, question, sample_id_str


def resolve_row_for_easy_planner(
    export_root: Path,
    row: dict[str, Any],
    *,
    dataset_dir_name: str | None,
) -> tuple[Path, str, str, str]:
    """image_path, question, reference_answer, sample_id."""
    image_path, question, sid = resolve_image_and_question(export_root, row)
    ref = reference_answer_from_row(row, dataset_dir_name=dataset_dir_name)
    return image_path, question, ref, sid


__all__ = [
    "load_jsonl_row",
    "reference_answer_from_row",
    "resolve_image_and_question",
    "resolve_row_for_easy_planner",
]
