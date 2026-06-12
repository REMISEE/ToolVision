#!/usr/bin/env python3
"""Prepare or run standalone CoT leak rewrites with Qwen.

Input:
- *.needs_edit.jsonl from annotate_complex_leak_signals.py

Behavior:
- For each edit target, locate the source text:
  - executor_step -> steps/.../executor_cot.md
  - final_answer -> <think>...</think> inside messages.json
- Export a readable work item folder with:
  - metadata.json
  - source_raw.md
  - source_marked.md
  - prompt_user.txt
- Optionally call Qwen to rewrite the text with no task/image context.

Outputs:
- <output-dir>/items/... per-target work items
- <output-dir>/results.jsonl summary of prepared/generated rewrites
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from offline_sft_pipeline.pipelines.api_text_multimodal import (
    DEFAULT_QWEN_MODEL,
    assistant_text_from_chat_response,
    chat_completions_text,
    env_qwen_config,
    is_placeholder_api_key,
)


PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"
DEFAULT_SYSTEM_PROMPT_FILE = "cot_leak_rewrite_system_v01.txt"
_THINK_BLOCK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", flags=re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True, help="Input *.needs_edit.jsonl.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write rewrite work items.")
    parser.add_argument(
        "--system-prompt-file",
        default=DEFAULT_SYSTEM_PROMPT_FILE,
        help="Filename under offline_sft_pipeline/prompts/.",
    )
    parser.add_argument(
        "--model",
        default="",
        help=f"Override model name. Default uses env or {DEFAULT_QWEN_MODEL}.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Optional limit on number of edit targets to prepare/run.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only export source/prompt files; do not call the model.",
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def extract_think_text(content: str) -> str:
    text = str(content or "")
    match = _THINK_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def load_messages(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_trajectory(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def mark_sentences(text: str, sentences_to_edit: list[str]) -> str:
    marked = str(text or "")
    for sentence in sorted(sentences_to_edit, key=len, reverse=True):
        if not sentence:
            continue
        escaped = re.escape(sentence)
        replacement = f"\n\n<<<NEEDS_EDIT>>>\n{sentence}\n<<<END_NEEDS_EDIT>>>\n\n"
        marked, _ = re.subn(escaped, replacement, marked, count=1)
    return marked.strip()


def build_user_prompt(*, item: dict[str, Any], original_text: str, marked_text: str) -> str:
    categories = ", ".join(item["categories"]) or "(none)"
    matches = ", ".join(item["matches"]) or "(none)"
    sentences = "\n".join(f"- {sentence}" for sentence in item["sentences_to_edit"]) or "- (none)"
    return (
        "Rewrite this reasoning text to remove leakage.\n\n"
        f"Message kind: {item['message_kind']}\n"
        f"Leak categories: {categories}\n"
        f"Matched strings: {matches}\n"
        "Sentences that need editing:\n"
        f"{sentences}\n\n"
        "Original text with edit markers:\n"
        f"{marked_text}\n\n"
        "Original raw text:\n"
        f"{original_text}\n"
    )


def parse_rewrite_response(text: str) -> tuple[str | None, list[str], str]:
    raw = str(text or "").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return None, [], raw
        payload = json.loads(match.group(0))

    rewrite_text = payload.get("rewrite_text")
    if not isinstance(rewrite_text, str) or not rewrite_text.strip():
        return None, [], raw

    edit_summary_raw = payload.get("edit_summary")
    if isinstance(edit_summary_raw, list):
        edit_summary = [str(item).strip() for item in edit_summary_raw if str(item).strip()]
    else:
        edit_summary = []
    return rewrite_text.strip(), edit_summary, raw


def resolve_source_text(
    *,
    row: dict[str, Any],
    target: dict[str, Any],
) -> tuple[str, Path | None]:
    message_id = str(target["message_id"] or "")
    message_kind = str(target["message_kind"] or "")

    if message_kind == "executor_step":
        trajectory = load_trajectory(trajectory_path_for_row(row))
        for step in trajectory.get("steps") or []:
            if str(step.get("assistant_message_id") or "") != message_id:
                continue
            cot_rel = str(step.get("executor_cot_path") or "").strip()
            if not cot_rel:
                break
            cot_path = trajectory_path_for_row(row).parent / cot_rel
            return cot_path.read_text(encoding="utf-8"), cot_path.resolve()
        raise KeyError(f"Could not resolve executor cot for message_id={message_id!r}.")

    if message_kind == "final_answer":
        messages_path = messages_path_for_row(row)
        for message in load_messages(messages_path):
            if str(message.get("message_id") or "") != message_id:
                continue
            return extract_think_text(str(message.get("content") or "")), None
        raise KeyError(f"Could not resolve final_answer message_id={message_id!r}.")

    raise ValueError(f"Unsupported message_kind: {message_kind!r}")


def item_dir_for(*, output_dir: Path, row: dict[str, Any], target: dict[str, Any]) -> Path:
    return (
        output_dir
        / "items"
        / str(row["dataset"])
        / str(row["sample_id"])
        / str(row["trajectory_id"])
        / str(target["message_id"])
    )


def main() -> None:
    args = parse_args()
    system_prompt_path = PROMPT_ROOT / args.system_prompt_file
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    rows = load_jsonl(args.input_jsonl.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = env_qwen_config()
    model = args.model.strip() or str(env["model"] or DEFAULT_QWEN_MODEL)
    base_url = str(env["base_url"])
    api_key = env["api_key"]
    timeout_s = float(env["timeout_s"])

    prepared_count = 0
    generated_count = 0
    result_rows: list[dict[str, Any]] = []

    for row in rows:
        for target in row.get("edit_targets") or []:
            if args.max_items and prepared_count >= int(args.max_items):
                break

            original_text, source_path = resolve_source_text(row=row, target=target)
            marked_text = mark_sentences(original_text, list(target.get("sentences_to_edit") or []))
            user_prompt = build_user_prompt(item=target, original_text=original_text, marked_text=marked_text)

            work_dir = item_dir_for(output_dir=output_dir, row=row, target=target)
            work_dir.mkdir(parents=True, exist_ok=True)

            metadata = {
                "dataset": row["dataset"],
                "sample_id": row["sample_id"],
                "trajectory_id": row["trajectory_id"],
                "message_id": target["message_id"],
                "message_kind": target["message_kind"],
                "categories": list(target.get("categories") or []),
                "matches": list(target.get("matches") or []),
                "sentences_to_edit": list(target.get("sentences_to_edit") or []),
                "source_path": str(source_path) if source_path is not None else None,
                "messages_path": str(messages_path_for_row(row)),
            }
            (work_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            (work_dir / "source_raw.md").write_text(original_text, encoding="utf-8")
            (work_dir / "source_marked.md").write_text(marked_text, encoding="utf-8")
            (work_dir / "prompt_user.txt").write_text(user_prompt, encoding="utf-8")

            result_row: dict[str, Any] = {
                "dataset": row["dataset"],
                "sample_id": row["sample_id"],
                "trajectory_id": row["trajectory_id"],
                "message_id": target["message_id"],
                "message_kind": target["message_kind"],
                "work_dir": str(work_dir),
                "source_path": str(source_path) if source_path is not None else None,
                "prepared": True,
                "generated": False,
            }

            if not args.prepare_only:
                if is_placeholder_api_key(api_key):
                    raise RuntimeError(
                        "OFFLINE_SFT_QWEN_API_KEY is missing or placeholder. Use --prepare-only or set a real key."
                    )
                response_text, raw_payload = chat_completions_text(
                    base_url=base_url,
                    api_key=api_key or "",
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    timeout_s=timeout_s,
                )
                rewrite_text, edit_summary, raw_text = parse_rewrite_response(response_text)
                (work_dir / "model_response.txt").write_text(raw_text, encoding="utf-8")
                (work_dir / "model_raw_payload.json").write_text(
                    json.dumps(raw_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                if rewrite_text is not None:
                    (work_dir / "rewrite_text.md").write_text(rewrite_text, encoding="utf-8")
                    (work_dir / "rewrite_result.json").write_text(
                        json.dumps(
                            {
                                "rewrite_text": rewrite_text,
                                "edit_summary": edit_summary,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    result_row["generated"] = True
                    result_row["rewrite_path"] = str((work_dir / "rewrite_text.md"))
                    result_row["edit_summary"] = edit_summary
                    generated_count += 1
                else:
                    result_row["generated"] = False
                    result_row["parse_failed"] = True

            result_rows.append(result_row)
            prepared_count += 1

        if args.max_items and prepared_count >= int(args.max_items):
            break

    write_jsonl(output_dir / "results.jsonl", result_rows)
    summary = {
        "input_jsonl": str(args.input_jsonl.resolve()),
        "output_dir": str(output_dir),
        "system_prompt_file": str(system_prompt_path),
        "model": model,
        "prepare_only": bool(args.prepare_only),
        "prepared_count": prepared_count,
        "generated_count": generated_count,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
