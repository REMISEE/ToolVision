"""Easy-question pipeline: single-round planner with reference answer (teacher) and optional I/O dumps."""

from __future__ import annotations

from offline_sft_pipeline.pipelines.easy_question_pipeline.backend import (
    EASY_SAVE_FULL_BASE64_ENV,
    EasyApiTextPlannerBackend,
    env_easy_save_full_base64,
)
from offline_sft_pipeline.pipelines.easy_question_pipeline.client import EasyPlannerClient, EasyPlannerRunArtifacts
from offline_sft_pipeline.pipelines.easy_question_pipeline.jsonl_samples import (
    load_jsonl_row,
    reference_answer_from_row,
    resolve_row_for_easy_planner,
)
from offline_sft_pipeline.pipelines.easy_question_pipeline.messages import (
    build_easy_reference_answer_block,
    planner_to_openai_messages_easy,
)
from offline_sft_pipeline.pipelines.easy_question_pipeline.run_job import run_easy_planner_job

DEFAULT_OUTPUT_EASY_HINT = (
    "Data root: export_images/output_easy/<dataset>/samples.jsonl. "
    "Batch: scripts/run_easy_pipeline_batch.py (default 10 lines × each dataset). "
    "Single: scripts/run_easy_planner_sample.py --dataset <name>. "
    "Outputs default to offline_sft_pipeline/outputs/easy_pipeline/."
)

__all__ = [
    "DEFAULT_OUTPUT_EASY_HINT",
    "EASY_SAVE_FULL_BASE64_ENV",
    "EasyApiTextPlannerBackend",
    "EasyPlannerClient",
    "EasyPlannerRunArtifacts",
    "build_easy_reference_answer_block",
    "env_easy_save_full_base64",
    "load_jsonl_row",
    "planner_to_openai_messages_easy",
    "reference_answer_from_row",
    "resolve_row_for_easy_planner",
    "run_easy_planner_job",
]
