#!/usr/bin/env python3
"""Run **OrchestratorV01** end-to-end with a **real HTTP planner** (``ApiTextBackend`` + Qwen).

This is the normal “call the API model” path: ``orchestrator.run(sample)`` → ``PlannerClient.run``
→ ``ApiTextBackend.generate`` → ``chat/completions``. Executor/judge/runtime stay fake/scripted
so the run focuses on **real planner I/O**.

Reads ``offline_sft_pipeline/example/question.json`` (question + image path), materializes the
sample in the store, then runs branching orchestration.

**Environment**

- ``OFFLINE_SFT_QWEN_API_KEY`` — required (unless ``OFFLINE_SFT_API_DRY_RUN=1``, which skips HTTP).
- ``OFFLINE_SFT_QWEN_BASE_URL``, ``OFFLINE_SFT_QWEN_MODEL``, … optional.

**After the run, where to look**

- Parsed planner output (structured JSON): ``store/samples/<sample_id>/trajectories/<traj_id>/planner/round_*.json``
- Conversation state: ``.../trajectories/<traj_id>/messages.json``
- Run pointer: ``<run_root>/run_summary.json``

**See the exact HTTP payload + raw model text** (stderr): pass ``--planner-debug`` or set
``OFFLINE_SFT_PLANNER_DEBUG=1`` before running. That prints OpenAI-style ``messages`` (images
shortened) and the **full assistant string** returned by the API.

Usage::

    cd ToolVision
    export OFFLINE_SFT_QWEN_API_KEY=...
    unset OFFLINE_SFT_API_DRY_RUN
    python offline_sft_pipeline/scripts/run_example_real_planner.py \\
        --output-dir offline_sft_pipeline/outputs/example_real_planner \\
        --planner-debug

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offline_sft_pipeline.core.models import Budget, RootImage, RootSample
from offline_sft_pipeline.core.store import OfflineTrajectoryStore
from offline_sft_pipeline.pipelines.backends import ApiTextBackend, FakeJudgeBackend, FakeTextBackend
from offline_sft_pipeline.pipelines.executor_client import ExecutorClient
from offline_sft_pipeline.pipelines.judge_client import JudgeClient
from offline_sft_pipeline.pipelines.orchestrator_v01 import OrchestratorConfig, OrchestratorV01
from offline_sft_pipeline.pipelines.planner_client import PlannerClient
from offline_sft_pipeline.pipelines.scripted_components import RuntimeSpec, ScriptedRuntime
from offline_sft_pipeline.pipelines.tool_capabilities_io import load_tool_capabilities_from_file

EXAMPLE_DIR = PIPELINE_ROOT / "example"
QUESTION_PATH = EXAMPLE_DIR / "question.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Orchestrator with real planner + example/question.json + example image.")
    p.add_argument(
        "--planner-debug",
        action="store_true",
        help="Set OFFLINE_SFT_PLANNER_DEBUG=1 for this process (stderr: HTTP messages + full model text).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(PIPELINE_ROOT / "outputs" / "example_real_planner_runs"),
        help="Base directory; one run folder is created inside.",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run folder name. Default: utc timestamp prefix example_real__YYYYMMDDTHHMMSSZ",
    )
    p.add_argument(
        "--sample-id",
        type=str,
        default="example__weight_plates",
        help="sample_id stored under store/samples/<sample_id>/",
    )
    return p.parse_args()


def build_run_id(raw: str) -> str:
    t = str(raw or "").strip()
    if t:
        return t
    return "example_real__" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_example_question_and_image() -> tuple[str, Path]:
    if not QUESTION_PATH.is_file():
        raise FileNotFoundError(f"missing {QUESTION_PATH}")
    payload = json.loads(QUESTION_PATH.read_text(encoding="utf-8"))
    q = str(payload.get("question") or "").strip()
    rel = payload.get("image")
    if not q or not rel:
        raise ValueError(f"{QUESTION_PATH} must contain non-empty 'question' and 'image'")
    img = (EXAMPLE_DIR / str(rel)).resolve()
    if not img.is_file():
        raise FileNotFoundError(f"example image not found: {img}")
    return q, img


def main() -> None:
    args = parse_args()
    if args.planner_debug:
        os.environ["OFFLINE_SFT_PLANNER_DEBUG"] = "1"
    run_id = build_run_id(args.run_id)
    output_base = Path(args.output_dir).expanduser().resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    run_root = output_base / run_id
    if run_root.exists():
        raise FileExistsError(f"run directory already exists: {run_root}")
    run_root.mkdir(parents=True)

    store_root = run_root / "store"
    question, image_path = load_example_question_and_image()

    sample = RootSample(
        sample_id=args.sample_id,
        question=question,
        images=[RootImage(image_id="root_0", path=str(image_path))],
        metadata={"source": "offline_sft_pipeline/example/question.json"},
    )

    caps = load_tool_capabilities_from_file()
    config = OrchestratorConfig(
        tool_capabilities=caps,
        planner_suggestion_count=3,
        default_budget=Budget(remaining_rounds=3),
    )

    # Real HTTP planner; fake executor text; fake judge; scripted runtime with fallback for any child id.
    orchestrator = OrchestratorV01(
        store=OfflineTrajectoryStore(store_root),
        planner_client=PlannerClient(backend=ApiTextBackend()),
        executor_client=ExecutorClient(backend=FakeTextBackend()),
        judge_client=JudgeClient(backend=FakeJudgeBackend(overall_score=0.75)),
        runtime=ScriptedRuntime(
            {},
            default_spec=RuntimeSpec(
                text="Synthetic runtime output for example_real_planner (scripted fallback).",
                helper_names=["ground_box"],
                image_label="example_step",
            ),
        ),
        config=config,
    )

    result = orchestrator.run(sample)

    sample_dir = store_root / "samples" / sample.sample_id
    summary = {
        "run_id": run_id,
        "run_root": str(run_root),
        "store_root": str(store_root),
        "example_question_path": str(QUESTION_PATH),
        "example_image": str(image_path),
        "sample_id": result.sample_id,
        "root_trajectory_id": result.root_trajectory_id,
        "all_trajectory_ids": result.all_trajectory_ids,
        "terminal_trajectory_ids": result.terminal_trajectory_ids,
        "inspect": {
            "samples": str(sample_dir),
            "hint_planner_round_json": "Under each trajectories/<traj_id>/planner/round_*.json (parsed PlannerOutput).",
            "hint_messages_json": "Under each trajectories/<traj_id>/messages.json",
            "hint_stderr_debug": "Use --planner-debug or OFFLINE_SFT_PLANNER_DEBUG=1 to log HTTP messages + raw assistant text.",
        },
    }
    out_json = run_root / "run_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
