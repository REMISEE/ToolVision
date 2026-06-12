#!/usr/bin/env python3
"""Export easy-pipeline kept samples to CodeVision-SFT style folders."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


_DATASET_TO_EXPORT_DIR = {
    "textvqa": "textvqa_easy",
    "fsc147": "fsc147_easy",
    "gqa_002": "gqa_easy",
    "cavqa_multichoice": "cavqa_easy",
}
_CODE_FENCE_JSON_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", flags=re.DOTALL | re.IGNORECASE)
_LAST_JSON_OBJECT_RE = re.compile(r"(\{[\s\S]*\})\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kept-jsonl",
        type=Path,
        default=Path("/data/home/suchenghao/ToolVision/offline_sft_pipeline/outputs/sft_prep/easy/easy.kept.jsonl"),
        help="Path to easy.kept.jsonl.",
    )
    parser.add_argument(
        "--batch-log-root",
        type=Path,
        default=Path("/data/home/suchenghao/ToolVision/offline_sft_pipeline/outputs/easy_pipeline"),
        help="Root containing one or more batch_run_summary.jsonl files.",
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        default=Path("/data/home/suchenghao/ToolVision/export_images/output_easy"),
        help="Root containing output_easy/<dataset>/samples.jsonl.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/home/suchenghao/ToolVision/offline_sft_pipeline/outputs/sft_prep/codevision_exports"),
        help="Root directory to create xxx_easy export folders under.",
    )
    parser.add_argument(
        "--system-template-json",
        type=Path,
        default=Path("/data/home/suchenghao/ToolVision/offline_sft_pipeline/prompts/codevision_sft_system_v02.txt"),
        help="Path to a system prompt .txt file, or to a CodeVision-style json whose first row contains 'system'.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_system_prompt(path: Path) -> str:
    if path.suffix.lower() != ".json":
        system_prompt = path.read_text(encoding="utf-8").strip()
        if not system_prompt:
            raise ValueError(f"Could not find non-empty system prompt in {path}.")
        return system_prompt
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected non-empty list in {path}.")
    system_prompt = data[0].get("system")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError(f"Could not find non-empty 'system' field in {path}.")
    return system_prompt


def build_kept_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(r["dataset"]), str(r["sample_id"])): r for r in rows}


def build_batch_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        dataset = row.get("dataset")
        sample_id = row.get("sample_id")
        summary = row.get("summary")
        if dataset is None or sample_id is None or not isinstance(summary, dict):
            continue
        index[(str(dataset), str(sample_id))] = row
    return index


def load_all_batch_rows(batch_log_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(batch_log_root.rglob("batch_run_summary.jsonl")):
        rows.extend(load_jsonl(path))
    return rows


def build_source_index(export_root: Path, dataset: str) -> dict[str, dict[str, Any]]:
    path = export_root / dataset / "samples.jsonl"
    return {str(r["sample_id"]): r for r in load_jsonl(path)}


def convert_easy_assistant_text(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    match = _CODE_FENCE_JSON_RE.match(text)
    if match:
        text = match.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = _LAST_JSON_OBJECT_RE.search(text)
        if not match:
            raise
        payload = json.loads(match.group(1))
    think = str(payload.get("think") or "").strip()
    answer = payload.get("answer")
    answer_text = "" if answer is None else str(answer).strip()
    return f"<think>\n{think}\n</think>\n<answer>\n{answer_text}\n</answer>"


def build_easy_assistant_value(batch_row: dict[str, Any]) -> str:
    artifact_files = (batch_row.get("summary") or {}).get("artifact_files") or {}
    planner_text_path = Path(str(artifact_files.get("planner_assistant_text") or ""))
    if planner_text_path.exists():
        try:
            return convert_easy_assistant_text(planner_text_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    planner_output_path = Path(str(((batch_row.get("summary") or {}).get("planner_output_path")) or ""))
    if planner_output_path.exists():
        payload = json.loads(planner_output_path.read_text(encoding="utf-8"))
        think = str(payload.get("global_chain_cot") or "").strip()
        answer = str(payload.get("direct_answer") or "").strip()
        return f"<think>\n{think}\n</think>\n<answer>\n{answer}\n</answer>"

    raise FileNotFoundError(
        f"Could not build assistant text for {batch_row.get('dataset')} / {batch_row.get('sample_id')}"
    )


def copy_image(source_path: Path, dest_dir: Path, sample_export_id: int, image_index: int) -> str:
    dest_name = f"sample{sample_export_id}_{image_index}{source_path.suffix.lower() or '.png'}"
    dest_path = dest_dir / dest_name
    if not dest_path.exists():
        shutil.copy2(source_path, dest_path)
    return f"codevision_images/{dest_name}"


def export_dataset(
    *,
    dataset: str,
    kept_rows: list[dict[str, Any]],
    batch_index: dict[tuple[str, str], dict[str, Any]],
    source_index: dict[str, dict[str, Any]],
    export_root: Path,
    output_dir: Path,
    system_prompt: str,
) -> dict[str, Any]:
    image_dir = output_dir / "codevision_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    exported_rows: list[dict[str, Any]] = []
    export_report: list[dict[str, Any]] = []

    for export_id, kept_row in enumerate(kept_rows):
        sample_id = str(kept_row["sample_id"])
        batch_row = batch_index[(dataset, sample_id)]
        source_row = source_index[sample_id]

        assistant_value = build_easy_assistant_value(batch_row)

        exported_images: list[str] = []
        for image_index, image_item in enumerate(source_row.get("images") or []):
            src = image_item.get("path")
            if not src:
                continue
            source_path = export_root / str(src)
            exported_images.append(copy_image(source_path, image_dir, export_id, image_index))

        metadata_payload = {
            "sample_id": export_id,
            "transform": None,
            "question": source_row.get("question"),
            "answer": kept_row.get("teacher_answer"),
            "source_dataset": dataset,
            "source_sample_id": sample_id,
        }

        exported_rows.append(
            {
                "conversations": [
                    {"from": "human", "value": f"<image>{source_row.get('question', '')}"},
                    {"from": "gpt", "value": assistant_value},
                ],
                "images": exported_images,
                "metadata": json.dumps(metadata_payload, ensure_ascii=False),
                "system": system_prompt,
            }
        )
        export_report.append(
            {
                "sample_id": sample_id,
                "export_sample_id": export_id,
                "image_count": len(exported_images),
            }
        )

    write_json(output_dir / "codevision_sft.json", exported_rows)
    write_json(output_dir / "export_report.json", export_report)
    write_json(
        output_dir / "export_summary.json",
        {
            "dataset": dataset,
            "exported_count": len(exported_rows),
            "output_dir": str(output_dir),
        },
    )
    write_json(
        output_dir / "dataset_info.snippet.json",
        {
            f"{output_dir.name}": {
                "file_name": "codevision_sft.json",
                "formatting": "sharegpt",
                "columns": {
                    "messages": "conversations",
                    "images": "images",
                    "system": "system",
                },
                "tags": {
                    "role_tag": "from",
                    "content_tag": "value",
                    "user_tag": "human",
                    "assistant_tag": "gpt",
                    "observation_tag": "tool",
                },
            }
        },
    )
    return {"dataset": dataset, "output_dir": str(output_dir), "exported_count": len(exported_rows)}


def main() -> None:
    args = parse_args()
    system_prompt = load_system_prompt(args.system_template_json.resolve())
    kept_index = build_kept_index(load_jsonl(args.kept_jsonl.resolve()))
    batch_index = build_batch_index(load_all_batch_rows(args.batch_log_root.resolve()))

    summaries: list[dict[str, Any]] = []
    for dataset, folder_name in _DATASET_TO_EXPORT_DIR.items():
        kept_rows = [
            row
            for key, row in kept_index.items()
            if key[0] == dataset
        ]
        source_index = build_source_index(args.export_root.resolve(), dataset)
        output_dir = args.output_root.resolve() / folder_name
        summaries.append(
            export_dataset(
                dataset=dataset,
                kept_rows=kept_rows,
                batch_index=batch_index,
                source_index=source_index,
                export_root=args.export_root.resolve(),
                output_dir=output_dir,
                system_prompt=system_prompt,
            )
        )

    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
