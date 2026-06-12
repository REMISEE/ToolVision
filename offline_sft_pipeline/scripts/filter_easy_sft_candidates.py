#!/usr/bin/env python3
"""Filter easy-pipeline outputs into high-confidence SFT candidates.

Current policy:
1. Keep only rows with exit_code == 0.
2. Keep only rows whose direct_answer exactly matches the teacher answer from
   export_images/output_easy/<dataset>/samples.jsonl.
3. Drop rows whose planner CoT or raw assistant text appears contaminated by
   prompt / teacher-answer leakage.

Outputs:
- <output_prefix>.jsonl: concise one-line audit for every batch row
- <output_prefix>.kept.jsonl: only rows that passed all checks
- <output_prefix>.summary.json: aggregate counts
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_EXPORT_ROOT = _REPO_ROOT / "export_images" / "output_easy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-log",
        type=Path,
        required=True,
        help="Path to easy_pipeline/.../batch_run_summary.jsonl",
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        default=_DEFAULT_EXPORT_ROOT,
        help="Root containing output_easy/<dataset>/samples.jsonl",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output prefix; writes .jsonl / .kept.jsonl / .summary.json",
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


def load_jsonl_row_by_sample_id(path: Path, sample_id: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if not text:
                continue
            row = json.loads(text)
            if str(row.get("sample_id") or "").strip() == sample_id:
                return row
    raise ValueError(f"sample_id not found in {path}: {sample_id!r}")


def reference_answer_from_row(row: dict[str, Any], *, dataset_dir_name: str) -> str:
    meta = row.get("metadata") or {}
    dataset_name = dataset_dir_name.strip().lower()
    if dataset_name == "textvqa":
        value = meta.get("model_filtered_resps")
        if value is None or not str(value).strip():
            raise ValueError("textvqa row missing metadata.model_filtered_resps")
        return str(value).strip()

    answer = row.get("answer")
    if isinstance(answer, list):
        if not answer:
            raise ValueError("answer list is empty")
        return str(answer[0]).strip()
    if answer is None or not str(answer).strip():
        raise ValueError("answer is empty")
    return str(answer).strip()


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


_LEAKAGE_PATTERNS: list[tuple[str, str]] = [
    ("teacher_reference", r"\bteacher answer\b"),
    ("teacher_reference", r"\breference answer\b"),
    ("teacher_reference", r"\btarget answer\b"),
    ("teacher_reference", r"\bground[\s-]?truth\b"),
    ("teacher_reference", r"\bprovided answer\b"),
    ("teacher_reference", r"\bexpected answer\b"),
    ("teacher_reference", r"\bcorrect answer\b"),
    ("teacher_reference", r"\bdesignated correct answer\b"),
    ("prompt_reference", r"\bthe prompt (?:says|states|gives|provides)\b"),
    ("prompt_reference", r"\bthe prompt also\b"),
    ("prompt_reference", r"\bthe prompt includes\b"),
    ("prompt_reference", r"\bthe user prompt\b"),
    ("prompt_reference", r"\buser prompt includes\b"),
    ("prompt_reference", r"\bprompt includes\b"),
    ("prompt_reference", r"\bprompt contains\b"),
    ("prompt_reference", r"\bprompt specifies\b"),
    ("prompt_reference", r"\baccording to the prompt\b"),
    ("prompt_reference", r"\bfrom the prompt\b"),
    ("prompt_reference", r"\bin the prompt\b"),
    ("prompt_reference", r"\bsupplied in the prompt\b"),
    ("prompt_reference", r"\bgiven in the prompt\b"),
    ("prompt_reference", r"\bprovided in the prompt\b"),
    ("instruction_reference", r"\bthe instruction (?:says|states|requires)\b"),
    ("instruction_reference", r"\bconstraint\b"),
    ("instruction_reference", r"\banswer constraint\b"),
    ("instruction_reference", r"\bconstraint specifying\b"),
    ("instruction_reference", r"\bconstraint that the answer\b"),
    ("instruction_reference", r"\bI will follow this constraint\b"),
    ("instruction_reference", r"\bmust be exactly\b"),
    ("instruction_reference", r"\bshould be exactly\b"),
    ("instruction_reference", r"\banswer should be\b"),
    ("instruction_reference", r"\banswer must be\b"),
    ("instruction_reference", r"\bexactly this string\b"),
    ("instruction_reference", r"\breturn mode\b"),
    ("instruction_reference", r"\bjson object\b"),
    ("instruction_reference", r"\bmode=\"answer\"\b"),
    ("leakage_verbatim", r"\bdo not mention that the target answer was supplied\b"),
    ("leakage_verbatim", r"\bthe final json field [`'\"]?answer[`'\"]? must be exactly\b"),
    ("answer_source", r"\bthe answer was supplied\b"),
    ("answer_source", r"\bthe answer is supplied\b"),
    ("answer_source", r"\bthe answer is given\b"),
    ("answer_source", r"\bthe answer was given\b"),
    ("answer_source", r"\bprompt gave the answer\b"),
    ("answer_source", r"\bthe prompt already contains the answer\b"),
    ("answer_source", r"\bthe prompt tells me the answer\b"),
    ("answer_source", r"\bthe target string\b"),
    ("answer_source", r"\bconfirms my visual count\b"),
    ("answer_source", r"\bconfirms this count\b"),
    ("answer_source", r"\baligns with my visual inspection\b"),
    ("cn_prompt_reference", r"提示(?:词|里|中)"),
    ("cn_prompt_reference", r"根据提示"),
    ("cn_prompt_reference", r"用户提示"),
    ("cn_prompt_reference", r"题目已经给出答案"),
    ("cn_prompt_reference", r"提示中给了答案"),
    ("cn_teacher_reference", r"标准答案"),
    ("cn_teacher_reference", r"参考答案"),
    ("cn_teacher_reference", r"目标答案"),
    ("cn_teacher_reference", r"正确答案"),
    ("cn_instruction_reference", r"约束"),
    ("cn_instruction_reference", r"必须严格"),
    ("cn_instruction_reference", r"必须输出"),
    ("cn_instruction_reference", r"必须完全一致"),
    ("cn_instruction_reference", r"必须是这个字符串"),
]

_LEAKAGE_REGEXES = [(label, re.compile(pattern, flags=re.IGNORECASE)) for label, pattern in _LEAKAGE_PATTERNS]


def detect_leakage(*texts: str) -> list[dict[str, str]]:
    joined = "\n".join(text for text in texts if text)
    hits: list[dict[str, str]] = []
    for label, regex in _LEAKAGE_REGEXES:
        for match in regex.finditer(joined):
            snippet = joined[max(0, match.start() - 40) : min(len(joined), match.end() + 80)]
            hits.append({"category": label, "match": match.group(0), "snippet": snippet})
    unique_hits: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in hits:
        key = (item["category"], item["match"], item["snippet"])
        if key in seen:
            continue
        seen.add(key)
        unique_hits.append(item)
    return unique_hits


def main() -> None:
    args = parse_args()
    batch_rows = load_jsonl(args.batch_log.resolve())
    export_root = args.export_root.resolve()
    audit_rows: list[dict[str, Any]] = []
    kept_rows: list[dict[str, Any]] = []

    for row in batch_rows:
        dataset = normalize_text(row.get("dataset"))
        sample_id = normalize_text(row.get("sample_id"))
        line_index = row.get("line_index")
        exit_code = int(row.get("exit_code", -1))

        audit: dict[str, Any] = {
            "dataset": dataset,
            "sample_id": sample_id,
            "line_index": line_index,
            "exit_code": exit_code,
            "status": "drop",
            "drop_reasons": [],
            "teacher_answer": "",
            "pred_answer": "",
            "cot_pollution_hit_count": 0,
            "cot_pollution_categories": [],
        }

        if exit_code != 0:
            audit["drop_reasons"].append("exit_code_nonzero")
            audit_rows.append(audit)
            continue

        src_path = export_root / dataset / "samples.jsonl"
        try:
            src_row = load_jsonl_row_by_sample_id(src_path, sample_id)
            teacher_answer = reference_answer_from_row(src_row, dataset_dir_name=dataset)
        except Exception as exc:
            audit["drop_reasons"].append(f"teacher_lookup_failed:{exc}")
            audit_rows.append(audit)
            continue

        summary = row.get("summary") or {}
        pred_answer = normalize_text(summary.get("direct_answer"))
        audit["teacher_answer"] = teacher_answer
        audit["pred_answer"] = pred_answer
        audit["answer_exact_match"] = pred_answer == teacher_answer
        if pred_answer != teacher_answer:
            audit["drop_reasons"].append("answer_mismatch")

        planner_output_path = normalize_text(summary.get("planner_output_path"))
        planner_output: dict[str, Any] = {}
        if planner_output_path:
            planner_output = json.loads(Path(planner_output_path).read_text(encoding="utf-8"))
        cot_text = normalize_text(planner_output.get("global_chain_cot"))

        artifact_files = summary.get("artifact_files") or {}
        assistant_text = ""
        assistant_path = normalize_text(artifact_files.get("planner_assistant_text"))
        if assistant_path and Path(assistant_path).is_file():
            assistant_text = Path(assistant_path).read_text(encoding="utf-8")

        leakage_hits = detect_leakage(cot_text, assistant_text)
        audit["cot_pollution_hit_count"] = len(leakage_hits)
        audit["cot_pollution_categories"] = sorted({item["category"] for item in leakage_hits})
        if leakage_hits:
            audit["drop_reasons"].append("cot_polluted")
            audit["cot_pollution_hits_preview"] = leakage_hits[:10]

        if not audit["drop_reasons"]:
            audit["status"] = "keep"
            kept_rows.append(audit)

        audit_rows.append(audit)

    output_prefix = args.output_prefix.resolve()
    write_jsonl(output_prefix.with_suffix(".jsonl"), audit_rows)
    write_jsonl(output_prefix.with_suffix(".kept.jsonl"), kept_rows)

    summary = {
        "batch_log": str(args.batch_log.resolve()),
        "export_root": str(export_root),
        "total_rows": len(audit_rows),
        "kept_rows": len(kept_rows),
        "dropped_rows": len(audit_rows) - len(kept_rows),
        "drop_reason_counts": {},
    }
    for row in audit_rows:
        for reason in row.get("drop_reasons") or []:
            summary["drop_reason_counts"][reason] = summary["drop_reason_counts"].get(reason, 0) + 1

    output_prefix.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
