#!/usr/bin/env python3
"""Export cleaned ToolVision trajectories to a CodeVision-SFT style folder.

Input:
- one or more *_annotated.jsonl files from annotate_complex_leak_signals.py
- zero or more rewrite job directories from run_cot_leak_rewrite.py

Behavior:
- read answered trajectories from annotated rows
- apply available rewrite_text.md replacements to matching assistant messages
- copy referenced images into <output-dir>/codevision_images/
- write <output-dir>/codevision_sft.json

Notes:
- this script never modifies the original store
- rows with leak targets but missing rewrites are skipped by default
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


_THINK_BLOCK_RE = re.compile(r"(<think>\s*)(.*?)(\s*</think>)", flags=re.IGNORECASE | re.DOTALL)
_CODEVISION_TOOL_FOLLOWUP = (
    "Here is the processed image. Now, analyze the returned results. "
    "Please keep thinking step-by-step inside the <think></think> tags to determine the next action. "
    "If additional tools are required, call them inside the <tool_calls></tool_calls> tags. "
    "Otherwise, provide your final answer within the <answer></answer> tags."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        action="append",
        required=True,
        help="Annotated jsonl file. Can be provided multiple times.",
    )
    parser.add_argument(
        "--rewrite-dir",
        type=Path,
        action="append",
        default=[],
        help="Rewrite job directory produced by run_cot_leak_rewrite.py. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Target directory. Writes codevision_sft.json and codevision_images/ here.",
    )
    parser.add_argument(
        "--sample-id-start",
        type=int,
        default=0,
        help="Starting numeric sample id for exported rows.",
    )
    parser.add_argument(
        "--system-template-json",
        type=Path,
        default=Path("/data/home/suchenghao/ToolVision/offline_sft_pipeline/prompts/codevision_sft_system_v02.txt"),
        help="Path to a system prompt .txt file, or to a CodeVision-style json whose first row contains 'system'.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional limit on exported sample count.",
    )
    parser.add_argument(
        "--allow-missing-rewrite",
        action="store_true",
        help="Export rows even if they have edit_targets without a generated rewrite.",
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


def messages_path_for_row(row: dict[str, Any]) -> Path:
    return (
        Path(str(row["run_root"]))
        / "store"
        / "samples"
        / str(row["sample_id"])
        / "trajectories"
        / str(row["trajectory_id"])
        / "messages.json"
    )


def trajectory_path_for_row(row: dict[str, Any]) -> Path:
    return (
        Path(str(row["run_root"]))
        / "store"
        / "samples"
        / str(row["sample_id"])
        / "trajectories"
        / str(row["trajectory_id"])
        / "trajectory.json"
    )


def root_sample_path_for_row(row: dict[str, Any]) -> Path:
    return (
        Path(str(row["run_root"]))
        / "store"
        / "samples"
        / str(row["sample_id"])
        / "root_sample.json"
    )


def load_rewrite_index(rewrite_dirs: list[Path]) -> dict[tuple[str, str, str], Path]:
    index: dict[tuple[str, str, str], Path] = {}
    for rewrite_dir in rewrite_dirs:
        results_path = rewrite_dir.resolve() / "results.jsonl"
        if not results_path.exists():
            continue
        for row in load_jsonl(results_path):
            if not row.get("generated"):
                continue
            rewrite_path = row.get("rewrite_path")
            if not rewrite_path:
                continue
            key = (
                str(row.get("sample_id") or ""),
                str(row.get("trajectory_id") or ""),
                str(row.get("message_id") or ""),
            )
            index[key] = Path(str(rewrite_path)).resolve()
    return index


def replace_think_block(content: str, rewrite_text: str) -> str:
    text = str(content or "")
    replacement_text = str(rewrite_text or "").strip()
    if not replacement_text:
        return text

    def _repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}{replacement_text}{match.group(3)}"

    updated, count = _THINK_BLOCK_RE.subn(_repl, text, count=1)
    if count == 0:
        return replacement_text
    return updated


def build_runtime_artifact_map(trajectory_path: Path) -> dict[str, Path]:
    artifact_map: dict[str, Path] = {}
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    trajectory_dir = trajectory_path.parent
    for step in trajectory.get("steps") or []:
        runtime_rel = str(step.get("runtime_result_path") or "").strip()
        if not runtime_rel:
            continue
        runtime_path = trajectory_dir / runtime_rel
        if not runtime_path.exists():
            continue
        runtime_result = json.loads(runtime_path.read_text(encoding="utf-8"))
        for image_item in runtime_result.get("images") or []:
            artifact_id = str(image_item.get("artifact_id") or "").strip()
            image_path = str(image_item.get("path") or "").strip()
            if artifact_id and image_path:
                artifact_map[artifact_id] = Path(image_path).resolve()
    return artifact_map


def build_root_artifact_map(row: dict[str, Any]) -> dict[str, Path]:
    sample_dir = root_sample_path_for_row(row).parent
    artifact_dir = sample_dir / "artifacts"
    artifact_map: dict[str, Path] = {}
    if artifact_dir.exists():
        for path in artifact_dir.iterdir():
            if path.is_file():
                artifact_map[path.stem] = path.resolve()
    return artifact_map


def build_artifact_map(row: dict[str, Any]) -> dict[str, Path]:
    artifact_map = build_root_artifact_map(row)
    artifact_map.update(build_runtime_artifact_map(trajectory_path_for_row(row)))
    return artifact_map


def convert_role(role: str) -> str:
    mapping = {
        "user": "human",
        "assistant": "gpt",
        "tool": "tool",
    }
    if role not in mapping:
        raise ValueError(f"Unsupported role: {role!r}")
    return mapping[role]


def normalize_message_value(*, role: str, content: str, image_count: int) -> str:
    text = str(content or "")
    if role == "tool":
        if image_count > 0 and text:
            return f"<image>{text}\n\n{_CODEVISION_TOOL_FOLLOWUP}"
        if image_count > 0:
            return f"<image>{_CODEVISION_TOOL_FOLLOWUP}"
        if text:
            return f"{text}\n\n{_CODEVISION_TOOL_FOLLOWUP}"
        return _CODEVISION_TOOL_FOLLOWUP

    if image_count > 0:
        return f"<image>{text}"
    return text


def extract_transform_label(trajectory_path: Path) -> str:
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    trajectory_dir = trajectory_path.parent
    for step in trajectory.get("steps") or []:
        runtime_rel = str(step.get("runtime_result_path") or "").strip()
        if not runtime_rel:
            continue
        runtime_path = trajectory_dir / runtime_rel
        if not runtime_path.exists():
            continue
        runtime_result = json.loads(runtime_path.read_text(encoding="utf-8"))
        op = str(((runtime_result.get("meta") or {}).get("operation")) or "").strip()
        if op:
            return op
    return "tool_use"


def apply_rewrites_to_messages(
    *,
    row: dict[str, Any],
    messages: list[dict[str, Any]],
    rewrite_index: dict[tuple[str, str, str], Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    edits_required = {
        str(target.get("message_id") or "")
        for target in row.get("edit_targets") or []
        if str(target.get("message_id") or "").strip()
    }
    rewrites_applied: list[dict[str, Any]] = []
    missing_rewrites: list[str] = []
    updated_messages: list[dict[str, Any]] = []

    for message in messages:
        updated_message = dict(message)
        message_id = str(message.get("message_id") or "")
        key = (str(row["sample_id"]), str(row["trajectory_id"]), message_id)
        rewrite_path = rewrite_index.get(key)
        if rewrite_path is not None and rewrite_path.exists():
            rewrite_text = rewrite_path.read_text(encoding="utf-8").strip()
            updated_message["content"] = replace_think_block(str(message.get("content") or ""), rewrite_text)
            rewrites_applied.append(
                {
                    "message_id": message_id,
                    "rewrite_path": str(rewrite_path),
                }
            )
        elif message_id in edits_required:
            missing_rewrites.append(message_id)

        updated_messages.append(updated_message)

    return updated_messages, rewrites_applied, missing_rewrites


def copy_image(
    *,
    source_path: Path,
    sample_export_id: int,
    image_index: int,
    image_dir: Path,
) -> str:
    dest_name = f"sample{sample_export_id}_{image_index}{source_path.suffix.lower() or '.png'}"
    dest_path = image_dir / dest_name
    if not dest_path.exists():
        shutil.copy2(source_path, dest_path)
    return f"codevision_images/{dest_name}"


def build_export_row(
    *,
    row: dict[str, Any],
    export_id: int,
    system_prompt: str,
    rewrite_index: dict[tuple[str, str, str], Path],
    image_dir: Path,
    allow_missing_rewrite: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    messages = json.loads(messages_path_for_row(row).read_text(encoding="utf-8"))
    root_sample = json.loads(root_sample_path_for_row(row).read_text(encoding="utf-8"))
    artifact_map = build_artifact_map(row)
    updated_messages, rewrites_applied, missing_rewrites = apply_rewrites_to_messages(
        row=row,
        messages=messages,
        rewrite_index=rewrite_index,
    )
    transform_label = extract_transform_label(trajectory_path_for_row(row))

    if missing_rewrites and not allow_missing_rewrite:
        return None, {
            "sample_id": str(row["sample_id"]),
            "trajectory_id": str(row["trajectory_id"]),
            "reason": "missing_rewrite",
            "missing_message_ids": missing_rewrites,
        }

    conversations: list[dict[str, str]] = []
    exported_images: list[str] = []
    artifact_to_export_path: dict[str, str] = {}

    for message in updated_messages:
        role = str(message.get("role") or "")
        if role == "system":
            continue
        if role not in {"user", "assistant", "tool"}:
            continue

        image_artifact_ids = [str(item) for item in message.get("image_artifact_ids") or []]
        for artifact_id in image_artifact_ids:
            if artifact_id in artifact_to_export_path:
                continue
            source_path = artifact_map.get(artifact_id)
            if source_path is None or not source_path.exists():
                raise FileNotFoundError(
                    f"Could not resolve artifact_id={artifact_id!r} for "
                    f"{row['sample_id']} / {row['trajectory_id']}."
                )
            rel_path = copy_image(
                source_path=source_path,
                sample_export_id=export_id,
                image_index=len(exported_images),
                image_dir=image_dir,
            )
            artifact_to_export_path[artifact_id] = rel_path
            exported_images.append(rel_path)

        conversations.append(
            {
                "from": convert_role(role),
                "value": normalize_message_value(
                    role=role,
                    content=str(message.get("content") or ""),
                    image_count=len(image_artifact_ids),
                ),
            }
        )

    metadata_payload = {
        "sample_id": export_id,
        "transform": transform_label,
        "question": root_sample.get("question"),
        "answer": row.get("pred"),
        "source_dataset": row.get("dataset"),
        "source_sample_id": row.get("sample_id"),
    }
    if metadata_payload["answer"] is None:
        metadata_payload["answer"] = row.get("answer")

    export_row = {
        "conversations": conversations,
        "images": exported_images,
        "metadata": json.dumps(metadata_payload, ensure_ascii=False),
        "system": system_prompt,
    }

    report_row = {
        "sample_id": str(row["sample_id"]),
        "trajectory_id": str(row["trajectory_id"]),
        "export_sample_id": export_id,
        "rewrite_applied": bool(rewrites_applied),
        "missing_message_ids": missing_rewrites,
        "image_count": len(exported_images),
        "turn_count": len(conversations),
        "transform": transform_label,
    }
    if rewrites_applied:
        report_row["rewrites_applied"] = rewrites_applied

    return export_row, report_row


def main() -> None:
    args = parse_args()
    system_prompt = load_system_prompt(args.system_template_json.resolve())
    rewrite_index = load_rewrite_index([path.resolve() for path in args.rewrite_dir])

    rows: list[dict[str, Any]] = []
    for input_path in args.input_jsonl:
        rows.extend(load_jsonl(input_path.resolve()))

    output_dir = args.output_dir.resolve()
    image_dir = output_dir / "codevision_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    exported_rows: list[dict[str, Any]] = []
    export_report: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    next_export_id = int(args.sample_id_start)

    for row in rows:
        if args.max_samples and len(exported_rows) >= int(args.max_samples):
            break
        export_row, report_row = build_export_row(
            row=row,
            export_id=next_export_id,
            system_prompt=system_prompt,
            rewrite_index=rewrite_index,
            image_dir=image_dir,
            allow_missing_rewrite=args.allow_missing_rewrite,
        )
        if export_row is None:
            skipped_rows.append(report_row)
            continue

        exported_rows.append(export_row)
        export_report.append(report_row)
        next_export_id += 1

    write_json(output_dir / "codevision_sft.json", exported_rows)
    write_json(output_dir / "export_report.json", export_report)
    write_json(
        output_dir / "export_summary.json",
        {
            "input_files": [str(path.resolve()) for path in args.input_jsonl],
            "rewrite_dirs": [str(path.resolve()) for path in args.rewrite_dir],
            "output_dir": str(output_dir),
            "exported_count": len(exported_rows),
            "skipped_count": len(skipped_rows),
            "sample_id_start": args.sample_id_start,
            "sample_id_end_exclusive": next_export_id,
            "allow_missing_rewrite": bool(args.allow_missing_rewrite),
        },
    )
    write_json(output_dir / "skipped_rows.json", skipped_rows)
    write_json(
        output_dir / "dataset_info.snippet.json",
        {
            "toolvision_codevision_sft": {
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

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "exported_count": len(exported_rows),
                "skipped_count": len(skipped_rows),
                "first_export_sample_id": args.sample_id_start if exported_rows else None,
                "next_export_sample_id": next_export_id,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
