#!/usr/bin/env python3
"""Annotate leak signals on complex answered trajectories.

This script is intentionally narrow and prompt-source-based.

Detection policy:
- executor_step:
  - only check leakage of upstream planner / hidden planning context
- final_answer:
  - only check leakage of planner forced-answer handoff / judge / supplied answer

It does NOT modify source trajectory files.
It does NOT delete rows.
It only appends annotation fields to the output jsonl.

Additional convenience outputs:
- <output_prefix>.needs_edit.jsonl: only rows that require edits
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, TypedDict


DETECTOR_VERSION = "prompt_source_v1"


class RuleSpec(TypedDict):
    category: str
    pattern: str
    rule_source: str


class Hit(TypedDict):
    category: str
    match: str
    sentence: str
    snippet: str
    rule_source: str


class EditTarget(TypedDict):
    message_id: str | None
    message_kind: str
    categories: list[str]
    sentences_to_edit: list[str]
    matches: list[str]
    rule_sources: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Input jsonl from collect_correct_answered_trajectories.py",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output prefix; writes .jsonl and .summary.json",
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


_THINK_BLOCK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", flags=re.IGNORECASE | re.DOTALL)


def extract_think_text(content: str) -> str:
    text = str(content or "")
    match = _THINK_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\n+", "\n", str(text or "").strip())
    if not normalized:
        return []
    pieces = re.split(r"(?<=[.!?。！？])\s+|\n+", normalized)
    return [piece.strip() for piece in pieces if piece.strip()]


_EXECUTOR_RULE_SPECS: list[RuleSpec] = [
    {
        "category": "executor_planner_reference",
        "pattern": r"\bplanner\b",
        "rule_source": "executor hidden context contains planner_global_chain_cot; executor output should not mention planner",
    },
    {
        "category": "executor_planning_meta_reference",
        "pattern": r"\bupstream planning\b",
        "rule_source": "executor prompt forbids exposing upstream planning process",
    },
    {
        "category": "executor_planning_meta_reference",
        "pattern": r"\bglobal cot\b",
        "rule_source": "executor hidden context field label: High-level reasoning / Global CoT",
    },
    {
        "category": "executor_planning_meta_reference",
        "pattern": r"\bsuggestion cot\b",
        "rule_source": "executor hidden context field label: Branch rationale / Suggestion CoT",
    },
    {
        "category": "executor_planning_meta_reference",
        "pattern": r"\bstep goal\b",
        "rule_source": "executor hidden context field label: Local objective / Step Goal",
    },
    {
        "category": "executor_planning_meta_reference",
        "pattern": r"\bexecutor instruction\b",
        "rule_source": "executor hidden context field label: executor instruction",
    },
    {
        "category": "executor_planning_meta_reference",
        "pattern": r"\bbranch rationale\b",
        "rule_source": "executor hidden context field label: Branch rationale",
    },
    {
        "category": "executor_planning_meta_reference",
        "pattern": r"\blocal objective\b",
        "rule_source": "executor hidden context field label: Local objective",
    },
    {
        "category": "executor_planning_meta_reference",
        "pattern": r"\bhigh-level reasoning\b",
        "rule_source": "executor hidden context field label: High-level reasoning",
    },
    {
        "category": "executor_planning_meta_reference",
        "pattern": r"\bfollowing (?:the )?planner'?s suggestion\b",
        "rule_source": "executor prompt forbids quoting planner / suggestion context",
    },
    {
        "category": "executor_planning_meta_reference",
        "pattern": r"\baccording to the planner\b",
        "rule_source": "executor prompt forbids quoting planner context",
    },
    {
        "category": "executor_planning_meta_reference",
        "pattern": r"\bplanner suggests\b",
        "rule_source": "executor prompt forbids quoting planner context",
    },
]

_FINAL_ANSWER_RULE_SPECS: list[RuleSpec] = [
    {
        "category": "final_answer_judge_reference",
        "pattern": r"\bjudge reference\b",
        "rule_source": "planner forced-answer handoff exposes judge-derived block; final answer should not mention it",
    },
    {
        "category": "final_answer_judge_reference",
        "pattern": r"\bjudge evaluation\b",
        "rule_source": "planner forced-answer handoff exposes judge-derived block; final answer should not mention it",
    },
    {
        "category": "final_answer_judge_reference",
        "pattern": r"\blatest judge\b",
        "rule_source": "planner control block contains 'Latest judge overall_score'",
    },
    {
        "category": "final_answer_judge_reference",
        "pattern": r"\bjudged candidate answer\b",
        "rule_source": "planner control block contains 'Latest judged candidate answer'",
    },
    {
        "category": "final_answer_judge_reference",
        "pattern": r"\bcandidate answer\b",
        "rule_source": "planner control block contains 'Latest judged candidate answer'",
    },
    {
        "category": "final_answer_judge_reference",
        "pattern": r"\boverall_score\b",
        "rule_source": "planner control block contains 'Latest judge overall_score'",
    },
    {
        "category": "final_answer_judge_reference",
        "pattern": r"\bsuccessful_model_count\b",
        "rule_source": "planner control block contains 'successful_model_count'",
    },
    {
        "category": "final_answer_judge_reference",
        "pattern": r"\bforced final-answer reason\b",
        "rule_source": "planner control block contains 'Forced final-answer reason'",
    },
    {
        "category": "final_answer_judge_reference",
        "pattern": r"\bforced final answer\b",
        "rule_source": "planner control block contains forced-final-answer handoff",
    },
    {
        "category": "final_answer_answer_source_reference",
        "pattern": r"\breference answer\b",
        "rule_source": "planner final answer should not cite a supplied answer source",
    },
    {
        "category": "final_answer_answer_source_reference",
        "pattern": r"\btarget answer\b",
        "rule_source": "planner final answer should not cite a supplied answer source",
    },
    {
        "category": "final_answer_answer_source_reference",
        "pattern": r"\bprovided answer\b",
        "rule_source": "planner final answer should not cite a supplied answer source",
    },
    {
        "category": "final_answer_answer_source_reference",
        "pattern": r"\bcorrect answer\b",
        "rule_source": "planner final answer should not cite a supplied answer source",
    },
    {
        "category": "final_answer_answer_source_reference",
        "pattern": r"\bground[\s-]?truth\b",
        "rule_source": "planner final answer should not cite a supplied answer source",
    },
    {
        "category": "final_answer_answer_source_reference",
        "pattern": r"\bsupplied answer\b",
        "rule_source": "planner final answer should not cite a supplied answer source",
    },
    {
        "category": "final_answer_prompt_policy_reference",
        "pattern": r"\bround policy\b",
        "rule_source": "planner final answer should not expose round-policy block",
    },
    {
        "category": "final_answer_prompt_policy_reference",
        "pattern": r"\bprompt instructions\b",
        "rule_source": "planner final answer should not expose prompt instruction block",
    },
    {
        "category": "final_answer_prompt_policy_reference",
        "pattern": r"\bmust exactly equal\b",
        "rule_source": "planner control block contains exact-match answer constraint",
    },
    {
        "category": "final_answer_prompt_policy_reference",
        "pattern": r"\baligned with that exact answer\b",
        "rule_source": "planner control block contains think-alignment constraint",
    },
    {
        "category": "final_answer_prompt_policy_reference",
        "pattern": r"\binput is malformed\b",
        "rule_source": "planner control block contains malformed-input exception",
    },
]

_RULES_BY_MESSAGE_KIND = {
    "executor_step": [
        {
            "category": spec["category"],
            "regex": re.compile(spec["pattern"], flags=re.IGNORECASE),
            "rule_source": spec["rule_source"],
        }
        for spec in _EXECUTOR_RULE_SPECS
    ],
    "final_answer": [
        {
            "category": spec["category"],
            "regex": re.compile(spec["pattern"], flags=re.IGNORECASE),
            "rule_source": spec["rule_source"],
        }
        for spec in _FINAL_ANSWER_RULE_SPECS
    ],
}


def detect_hits(*, message_kind: str, text: str) -> list[Hit]:
    rules = _RULES_BY_MESSAGE_KIND.get(message_kind, [])
    hits: list[Hit] = []
    for sentence in split_sentences(text):
        for rule in rules:
            regex = rule["regex"]
            for match in regex.finditer(sentence):
                hits.append(
                    {
                        "category": str(rule["category"]),
                        "match": match.group(0),
                        "sentence": sentence,
                        "snippet": sentence,
                        "rule_source": str(rule["rule_source"]),
                    }
                )

    unique_hits: list[Hit] = []
    seen: set[tuple[str, str, str]] = set()
    for item in hits:
        key = (item["category"], item["match"].lower(), item["sentence"])
        if key in seen:
            continue
        seen.add(key)
        unique_hits.append(item)
    return unique_hits


def build_edit_target(
    *,
    message_id: str | None,
    message_kind: str,
    hits: list[Hit],
) -> EditTarget:
    categories = sorted({item["category"] for item in hits})
    sentences_to_edit: list[str] = []
    matches: list[str] = []
    rule_sources: list[str] = []

    seen_sentences: set[str] = set()
    seen_matches: set[str] = set()
    seen_sources: set[str] = set()

    for item in hits:
        sentence = item["sentence"]
        if sentence not in seen_sentences:
            seen_sentences.add(sentence)
            sentences_to_edit.append(sentence)

        match = item["match"]
        lowered_match = match.lower()
        if lowered_match not in seen_matches:
            seen_matches.add(lowered_match)
            matches.append(match)

        source = item["rule_source"]
        if source not in seen_sources:
            seen_sources.add(source)
            rule_sources.append(source)

    return {
        "message_id": message_id,
        "message_kind": message_kind,
        "categories": categories,
        "sentences_to_edit": sentences_to_edit,
        "matches": matches,
        "rule_sources": rule_sources,
    }


def main() -> None:
    args = parse_args()
    input_rows = load_jsonl(args.input_jsonl.resolve())
    output_rows: list[dict[str, Any]] = []
    needs_edit_rows: list[dict[str, Any]] = []

    summary = {
        "detector_version": DETECTOR_VERSION,
        "input_jsonl": str(args.input_jsonl.resolve()),
        "total_rows": len(input_rows),
        "rows_with_signal": 0,
        "rows_without_signal": 0,
        "rows_executor_only": 0,
        "rows_final_only": 0,
        "rows_executor_and_final": 0,
        "category_counts": {},
        "message_kind_counts": {},
    }

    for row in input_rows:
        msg_path = messages_path_for_row(row)
        messages = json.loads(msg_path.read_text(encoding="utf-8"))

        assistant_checks: list[dict[str, Any]] = []
        edit_targets: list[EditTarget] = []
        row_categories: set[str] = set()
        row_has_signal = False
        executor_leak_signal = False
        final_answer_leak_signal = False

        for message in messages:
            if str(message.get("role") or "") != "assistant":
                continue
            metadata = message.get("metadata") or {}
            message_kind = str(metadata.get("message_kind") or "").strip()
            if message_kind not in {"executor_step", "final_answer"}:
                continue

            think_text = extract_think_text(str(message.get("content") or ""))
            hits = detect_hits(message_kind=message_kind, text=think_text)
            categories = sorted({item["category"] for item in hits})

            if hits:
                row_has_signal = True
                row_categories.update(categories)
                if message_kind == "executor_step":
                    executor_leak_signal = True
                elif message_kind == "final_answer":
                    final_answer_leak_signal = True
                summary["message_kind_counts"][message_kind] = summary["message_kind_counts"].get(message_kind, 0) + 1
                for category in categories:
                    summary["category_counts"][category] = summary["category_counts"].get(category, 0) + 1
                edit_targets.append(
                    build_edit_target(
                        message_id=message.get("message_id"),
                        message_kind=message_kind,
                        hits=hits,
                    )
                )

            assistant_checks.append(
                {
                    "message_id": message.get("message_id"),
                    "message_kind": message_kind,
                    "hit_count": len(hits),
                    "categories": categories,
                    "hits_preview": hits[:10],
                }
            )

        if executor_leak_signal and final_answer_leak_signal:
            leak_scope = "executor_and_final"
        elif executor_leak_signal:
            leak_scope = "executor_only"
        elif final_answer_leak_signal:
            leak_scope = "final_only"
        else:
            leak_scope = "none"

        output_row = {
            "detector_version": DETECTOR_VERSION,
            "leak_signal": row_has_signal,
            "leak_scope": leak_scope,
            "executor_leak_signal": executor_leak_signal,
            "final_answer_leak_signal": final_answer_leak_signal,
            "leak_categories": sorted(row_categories),
            "edit_targets": edit_targets,
            "assistant_checks": assistant_checks,
            **dict(row),
        }
        output_rows.append(output_row)

        if row_has_signal:
            summary["rows_with_signal"] += 1
            if leak_scope == "executor_only":
                summary["rows_executor_only"] += 1
            elif leak_scope == "final_only":
                summary["rows_final_only"] += 1
            elif leak_scope == "executor_and_final":
                summary["rows_executor_and_final"] += 1
            needs_edit_rows.append(output_row)
        else:
            summary["rows_without_signal"] += 1

    output_prefix = args.output_prefix.resolve()
    write_jsonl(output_prefix.with_suffix(".jsonl"), output_rows)
    write_jsonl(output_prefix.with_suffix(".needs_edit.jsonl"), needs_edit_rows)
    output_prefix.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
