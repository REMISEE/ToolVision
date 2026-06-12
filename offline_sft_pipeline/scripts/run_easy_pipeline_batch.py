#!/usr/bin/env python3
"""Run easy planner for the first N rows of each dataset under export_images/output_easy.

Default datasets: fsc147, cavqa_multichoice, textvqa, gqa_002 (GQA easy export uses the ``gqa_002`` folder only).
TextVQA teacher answer: ``metadata.model_filtered_resps`` (see ``jsonl_samples.reference_answer_from_row``).

Use ``--all`` to process **every** line in each ``samples.jsonl`` (full run; one API call per line).

Writes per-run artifacts under:
``<output-root>/<YYYY-MM-DD_HHMMSS>/<dataset>/line_XX_<sample_id>/`` (local time; one folder per invocation).
Use ``--run-date YYYY-MM-DD`` for a date-only folder (same day reruns overwrite).
Use ``--flat-output`` for legacy ``<output-root>/<dataset>/...``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from offline_sft_pipeline.pipelines.easy_question_pipeline.jsonl_samples import (  # noqa: E402
    load_jsonl_row,
    resolve_row_for_easy_planner,
)
from offline_sft_pipeline.pipelines.easy_question_pipeline.run_job import run_easy_planner_job  # noqa: E402

DEFAULT_EXPORT_ROOT = _REPO_ROOT / "export_images" / "output_easy"
DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "offline_sft_pipeline" / "outputs" / "easy_pipeline"
DEFAULT_DATASETS = ("fsc147", "cavqa_multichoice", "textvqa", "gqa_002")


def _safe_dir_suffix(sample_id: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", sample_id.strip())
    return s[:200] if len(s) > 200 else s


def _nonempty_line_count(jsonl_path: Path) -> int:
    n = 0
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _line_index_width(total_lines: int) -> int:
    if total_lines <= 0:
        return 2
    return max(2, len(str(total_lines - 1)))


def _run_subdir_name(run_date: str | None) -> str:
    """Default: local timestamp ``YYYY-MM-DD_HHMMSS``. If ``run_date`` is set: date-only ``YYYY-MM-DD``."""
    if run_date is not None and str(run_date).strip():
        tag = str(run_date).strip()
        date.fromisoformat(tag)
        return tag
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--export-root",
        type=Path,
        default=DEFAULT_EXPORT_ROOT,
        help="Same as run_easy_planner_sample: root for dataset folders (output_easy).",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent directory; by default a date+time subfolder is added (see --run-date).",
    )
    p.add_argument(
        "--run-date",
        type=str,
        default=None,
        help="If set: folder name YYYY-MM-DD only (no time). If unset: default folder is local "
        "timestamp YYYY-MM-DD_HHMMSS so each run gets its own directory. Ignored with --flat-output.",
    )
    p.add_argument(
        "--flat-output",
        action="store_true",
        help="Write directly under output-root/<dataset>/... without a date subfolder (legacy).",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset directory names under export-root (each must have samples.jsonl).",
    )
    p.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of lines from the top of each samples.jsonl (ignored if --all).",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Run every nonempty line in each dataset's samples.jsonl (full export). Costs one API call per line.",
    )
    p.add_argument("--system-prompt-file", default="planner_system_v05.txt", help="Planner prompt filename.")
    p.add_argument("--prompt-root", type=Path, default=None, help="Override prompts directory.")
    p.add_argument(
        "--fresh-batch-log",
        action="store_true",
        help="Truncate batch_run_summary.jsonl under the run directory before running (default: append).",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    export_root = args.export_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.flat_output:
        run_base = output_root
    else:
        run_base = output_root / _run_subdir_name(args.run_date)
    run_base.mkdir(parents=True, exist_ok=True)
    print(f"[batch] run_base={run_base}", file=sys.stderr)
    names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    if not names:
        print("No datasets in --datasets", file=sys.stderr)
        return 2

    batch_log_path = run_base / "batch_run_summary.jsonl"
    if args.fresh_batch_log and batch_log_path.is_file():
        batch_log_path.unlink()
    failures: list[dict[str, object]] = []

    for ds in names:
        jsonl_path = export_root / ds / "samples.jsonl"
        if not jsonl_path.is_file():
            print(f"[batch] skip (no file): {jsonl_path}", file=sys.stderr)
            failures.append({"dataset": ds, "error": "missing samples.jsonl", "path": str(jsonl_path)})
            continue

        if args.all:
            n_lines = _nonempty_line_count(jsonl_path)
            if n_lines <= 0:
                print(f"[batch] skip (empty jsonl): {jsonl_path}", file=sys.stderr)
                failures.append({"dataset": ds, "error": "empty samples.jsonl", "path": str(jsonl_path)})
                continue
            idx_width = _line_index_width(n_lines)
            line_range = range(n_lines)
            print(f"[batch] dataset={ds} full run: {n_lines} lines", file=sys.stderr)
        else:
            c = int(args.count)
            if c < 1:
                print("--count must be >= 1 unless using --all", file=sys.stderr)
                return 2
            idx_width = _line_index_width(c)
            line_range = range(c)

        for line_index in line_range:
            try:
                row = load_jsonl_row(jsonl_path, sample_id=None, line_index=line_index)
                image_path, question, ref, sample_id = resolve_row_for_easy_planner(
                    export_root,
                    row,
                    dataset_dir_name=ds,
                )
                sub = f"line_{line_index:0{idx_width}d}_{_safe_dir_suffix(sample_id)}"
                out_dir = run_base / ds / sub
                code, summary = run_easy_planner_job(
                    image_path=image_path,
                    question=question,
                    reference_answer=ref,
                    sample_id=sample_id,
                    output_dir=out_dir,
                    answer_instruction=None,
                    system_prompt_file=args.system_prompt_file,
                    prompt_root=args.prompt_root,
                    print_summary_json=False,
                )
                record = {
                    "dataset": ds,
                    "line_index": line_index,
                    "sample_id": sample_id,
                    "exit_code": code,
                    "output_dir": str(out_dir),
                    "summary": summary,
                }
                with batch_log_path.open("a", encoding="utf-8") as bf:
                    bf.write(json.dumps(record, ensure_ascii=False) + "\n")
                if code != 0:
                    failures.append(record)
                    print(f"[batch] exit {code} dataset={ds} line={line_index} -> {out_dir}", file=sys.stderr)
                else:
                    print(f"[batch] ok dataset={ds} line={line_index} -> {out_dir}", file=sys.stderr)
            except Exception as exc:
                err = {"dataset": ds, "line_index": line_index, "error": str(exc)}
                failures.append(err)
                with batch_log_path.open("a", encoding="utf-8") as bf:
                    bf.write(json.dumps(err, ensure_ascii=False) + "\n")
                print(f"[batch] FAIL dataset={ds} line={line_index}: {exc}", file=sys.stderr)

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "run_base": str(run_base),
                "run_subdir": None if args.flat_output else run_base.name,
                "batch_log": str(batch_log_path),
                "failures": failures,
            },
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
