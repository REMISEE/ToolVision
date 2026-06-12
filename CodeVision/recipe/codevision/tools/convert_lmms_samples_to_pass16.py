#!/usr/bin/env python3
"""Convert lmms-eval log_samples output into pass16-like parquet."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _find_sample_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    patterns = ["**/samples*.jsonl", "**/*samples*.jsonl", "**/samples*.json", "**/*samples*.json"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(input_path.glob(pattern)))
    deduped = []
    seen = set()
    for path in files:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def _unwrap_response(value: Any) -> str:
    if isinstance(value, list) and len(value) == 1:
        return _unwrap_response(value[0])
    if value is None:
        return ""
    return str(value)


def _response_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_unwrap_response(item) for item in value]
    return [_unwrap_response(value)]


def _source_from_sample_file(path: Path) -> str | None:
    match = re.search(r"tv_pass16_([a-z0-9_]+)", path.name)
    if match:
        source = match.group(1).lower()
        for suffix in ("_blank", "_shuffled"):
            if source.endswith(suffix):
                source = source[: -len(suffix)]
        return source
    return None


def _mode_from_sample_file(path: Path) -> str:
    for mode in ("blank", "shuffled"):
        if mode in path.name:
            return mode
    return "real"


def _load_reference_inputs(reference_input_dir: Path, source: str, mode: str) -> list[dict[str, Any]]:
    path = reference_input_dir / f"{source}_{mode}.jsonl"
    if not path.exists() and mode != "real":
        path = reference_input_dir / f"{source}_real.jsonl"
    if not path.exists():
        return []
    return _read_json_records(path)


def _extract_source(sample: dict[str, Any], default_source: str | None) -> str:
    doc = sample.get("doc") or {}
    return str(doc.get("source") or default_source or sample.get("task_name") or "unknown").lower()


def _sample_to_row(
    sample: dict[str, Any],
    default_source: str | None,
    reference_doc: dict[str, Any] | None = None,
    expected_repeats: int | None = None,
) -> dict[str, Any]:
    doc = sample.get("doc") or reference_doc or {}
    source = _extract_source(sample, default_source)
    responses = sample.get("filtered_resps")
    if responses is None:
        responses = sample.get("resps", [])
    pred_texts = _response_list(responses)
    if expected_repeats is not None and expected_repeats > 0:
        pred_texts = pred_texts[:expected_repeats]
    answers = doc.get("answers")
    if answers is None:
        answers = [doc.get("answer", sample.get("target", ""))]
    if not isinstance(answers, list):
        answers = [answers]

    image_path = doc.get("image_path", "")
    image_refs = []
    if image_path:
        image_refs = [{"kind": "file", "path": image_path, "uri": f"file://{image_path}", "image_index": 0}]

    return {
        "id": str(doc.get("id", sample.get("doc_id", ""))),
        "source": source,
        "task_id": int(doc.get("task_id", -1) if doc.get("task_id", -1) is not None else -1),
        "source_index": int(doc.get("source_index", -1) if doc.get("source_index", -1) is not None else -1),
        "raw_file": str(doc.get("raw_file", "")),
        "raw_row": int(doc.get("raw_row", -1) if doc.get("raw_row", -1) is not None else -1),
        "problem": f"<image> {doc.get('question', doc.get('prompt', ''))}".strip(),
        "answer_json": json.dumps([str(answer) for answer in answers], ensure_ascii=False),
        "problem_type": "ocr" if source == "textvqa" else ("counting" if source == "fsc147" else "general"),
        "answer_type": str(doc.get("answer_type", "ocrtext" if source == "textvqa" else ("number" if source == "fsc147" else "any"))),
        "prompt_type": "lmms_default",
        "control_mode": str(doc.get("control_mode", "real")),
        "image_refs_json": json.dumps(image_refs, ensure_ascii=False),
        "pred_texts_json": json.dumps(pred_texts, ensure_ascii=False),
        "rollout_n": len(pred_texts),
        "num_preds": len(pred_texts),
        "metadata_json": json.dumps(doc.get("metadata", {}), ensure_ascii=False),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="lmms-eval output directory or samples json/jsonl file")
    parser.add_argument("--output", type=Path, required=True, help="Output parquet path")
    parser.add_argument("--source", default=None, help="Optional source override")
    parser.add_argument("--reference-input-dir", type=Path, default=Path("/mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun/inputs"))
    parser.add_argument("--expected-repeats", type=int, default=None, help="Trim each sample to this many generations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = _find_sample_files(args.input)
    if not files:
        raise FileNotFoundError(f"No lmms sample files found under {args.input}")

    rows = []
    for path in files:
        file_source = args.source or _source_from_sample_file(path)
        mode = _mode_from_sample_file(path)
        refs = _load_reference_inputs(args.reference_input_dir, file_source, mode) if file_source else []
        for sample in _read_json_records(path):
            doc_id = int(sample.get("doc_id", -1))
            ref = refs[doc_id] if 0 <= doc_id < len(refs) else None
            rows.append(_sample_to_row(sample, file_source or args.source, ref, args.expected_repeats))

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(json.dumps({"rows": len(df), "files": [str(path) for path in files], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
