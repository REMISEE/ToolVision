#!/usr/bin/env python3
"""Easy-question planner: single-round MUST_ANSWER + teacher string.

Reads a row from ``export_images/output_easy`` (or ``output``) ``/<dataset>/samples.jsonl``,
or accepts explicit ``--image`` / ``--question`` / ``--reference-answer`` for ad-hoc tests.
TextVQA rows use ``metadata.model_filtered_resps`` as the teacher answer when loading by dataset name.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root: offline_sft_pipeline/scripts/ -> parents[2] = ToolVision
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from offline_sft_pipeline.pipelines.easy_question_pipeline import DEFAULT_OUTPUT_EASY_HINT  # noqa: E402
from offline_sft_pipeline.pipelines.easy_question_pipeline.jsonl_samples import (  # noqa: E402
    load_jsonl_row,
    resolve_row_for_easy_planner,
)
from offline_sft_pipeline.pipelines.easy_question_pipeline.run_job import run_easy_planner_job  # noqa: E402

DEFAULT_EXPORT_ROOT = _REPO_ROOT / "export_images" / "output_easy"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "offline_sft_pipeline" / "outputs" / "easy_pipeline"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Single-sample easy planner (reference answer in prompt). "
            "Load from export_images/output_easy/.../samples.jsonl by default, or pass explicit paths. "
            + DEFAULT_OUTPUT_EASY_HINT
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_argument_group("data source (prefer dataset/jsonl)")
    src.add_argument(
        "--export-root",
        type=Path,
        default=DEFAULT_EXPORT_ROOT,
        help="Root that contains dataset folders (e.g. export_images/output_easy); image paths in jsonl are relative to this.",
    )
    src.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset folder name under export-root (e.g. fsc147). Loads export-root/<dataset>/samples.jsonl.",
    )
    src.add_argument(
        "--samples-jsonl",
        type=Path,
        default=None,
        help="Full path to samples.jsonl (overrides --dataset).",
    )
    src.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Select row by sample_id (from jsonl).",
    )
    src.add_argument(
        "--line-index",
        type=int,
        default=None,
        help="Select row by 0-based line index in jsonl (ignored if --sample-id is set). "
        "If neither sample-id nor line-index is given, uses the first line.",
    )

    manual = p.add_argument_group("manual (skip jsonl)")
    manual.add_argument("--image", type=Path, default=None, help="Input image file.")
    manual.add_argument("--question", type=str, default=None, help="Question text.")
    manual.add_argument("--reference-answer", type=str, default=None, help="Teacher / GT answer string.")

    p.add_argument(
        "--answer-instruction",
        default=None,
        help="Optional extra format hint; often already embedded in question from jsonl.",
    )
    p.add_argument(
        "--system-prompt-file",
        default="planner_system_v05.txt",
        help="Filename under offline_sft_pipeline/prompts/.",
    )
    p.add_argument(
        "--sample-id-override",
        type=str,
        default=None,
        help="Override sample_id in PlannerClientRequest when using jsonl (default: from row).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for planner_request_messages.json, raw response, assistant text, planner_output.json.",
    )
    p.add_argument(
        "--prompt-root",
        default=None,
        type=Path,
        help="Override prompts directory (default: offline_sft_pipeline/prompts).",
    )
    return p


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, str, str, str]:
    """Returns image_path, question, reference_answer, sample_id."""
    use_jsonl = args.samples_jsonl is not None or args.dataset is not None
    use_manual = args.image is not None or args.question is not None or args.reference_answer is not None

    if use_jsonl and use_manual:
        raise SystemExit("Use either (--dataset/--samples-jsonl) or (--image/--question/--reference-answer), not both.")

    if use_jsonl:
        if args.samples_jsonl is not None:
            jsonl_path = args.samples_jsonl.resolve()
        elif args.dataset is not None:
            jsonl_path = (args.export_root / args.dataset / "samples.jsonl").resolve()
        else:
            raise SystemExit("Internal error: jsonl mode without path")
        if not jsonl_path.is_file():
            raise SystemExit(f"samples.jsonl not found: {jsonl_path}")

        row = load_jsonl_row(jsonl_path, sample_id=args.sample_id, line_index=args.line_index)
        dataset_hint: str | None = args.dataset
        if dataset_hint is None and args.samples_jsonl is not None:
            dataset_hint = jsonl_path.parent.name
        image_path, question, ref, sid = resolve_row_for_easy_planner(
            args.export_root.resolve(),
            row,
            dataset_dir_name=dataset_hint,
        )
        if args.sample_id_override:
            sid = args.sample_id_override.strip()
        if args.sample_id is None and args.line_index is None:
            print(f"[easy_planner] Using first line of {jsonl_path}", file=sys.stderr)
        return image_path, question, ref, sid

    if use_manual:
        if not (args.image and args.question and args.reference_answer):
            raise SystemExit(
                "With manual mode, provide all of: --image, --question, --reference-answer "
                "(or use --dataset / --samples-jsonl to read from export data)."
            )
        sid = args.sample_id_override or "easy__debug__0"
        return args.image.resolve(), args.question.strip(), args.reference_answer.strip(), sid

    raise SystemExit(
        "Provide data from export: --dataset fsc147 [--sample-id ID | --line-index N], "
        "or --samples-jsonl PATH, "
        "or manual: --image ... --question ... --reference-answer ..."
    )


def main() -> int:
    args = build_parser().parse_args()
    image_path, question, reference_answer, sample_id = _resolve_inputs(args)
    code, _ = run_easy_planner_job(
        image_path=image_path,
        question=question,
        reference_answer=reference_answer,
        sample_id=sample_id,
        output_dir=args.output_dir.resolve(),
        answer_instruction=args.answer_instruction,
        system_prompt_file=args.system_prompt_file,
        prompt_root=args.prompt_root,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
