#!/usr/bin/env python3
"""Probe context-mode step judges on synthetic and saved-rollout cases."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from recipe.codevision.rewards.step_answerability import (
    StepAnswerabilityJudgeClient,
    coerce_json_list,
    compute_step_answerability_delta,
)


DEFAULT_API_MODELS = "qwen3.6-plus,qwen3.5-397b-a17b,qwen3-vl-plus,qwen3.7-plus"


@dataclass(slots=True)
class EndpointSpec:
    name: str
    base_url: str
    model: str
    api_key_env: str
    request_body: dict[str, Any]


@dataclass(slots=True)
class ProbeState:
    label: str
    messages: list[dict[str, Any]]
    images: list[Any]


@dataclass(slots=True)
class ProbeCase:
    name: str
    question: str
    ground_truth: str
    answer_instruction: str
    states: list[ProbeState]


def load_log_env(path: str) -> dict[str, str]:
    env: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return env
    text = p.read_text(errors="ignore")
    pattern = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:(['\"])(.*?)\2|([^\s#]+))", re.M)
    for match in pattern.finditer(text):
        env[match.group(1)] = match.group(3) if match.group(3) is not None else match.group(4)
    return env


def safe_env_name(name: str) -> str:
    return "STEP_JUDGE_PROBE_" + re.sub(r"[^A-Za-z0-9_]", "_", name).upper()


def chat_root(base_url: str) -> str:
    return base_url.rstrip("/")


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def text_image(
    text: str,
    *,
    size: tuple[int, int] = (720, 420),
    font_size: int = 60,
    xy: tuple[int, int] | None = None,
    bg: str = "white",
    fg: str = "black",
) -> Image.Image:
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    fnt = font(font_size)
    if xy is None:
        bbox = draw.textbbox((0, 0), text, font=fnt)
        xy = ((size[0] - (bbox[2] - bbox[0])) // 2, (size[1] - (bbox[3] - bbox[1])) // 2)
    draw.text(xy, text, fill=fg, font=fnt)
    return img


def base_messages(question: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a visual question answering assistant. You may think step by step and call "
                "the code_image_tool when visual inspection is needed."
            ),
        },
        {"role": "user", "content": f"<image>{question}"},
    ]


def with_tool_step(
    messages: list[dict[str, Any]],
    *,
    thought: str,
    code: str,
    tool_text: str,
    tool_image: bool,
) -> list[dict[str, Any]]:
    out = list(messages)
    out.append(
        {
            "role": "assistant",
            "content": (
                f"<think>{thought}</think>\n"
                "<tool_call>\n"
                + json.dumps(
                    {
                        "name": "code_image_tool",
                        "arguments": {"code": code, "description": thought, "image_index": 0},
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n</tool_call>"
            ),
        }
    )
    content: list[dict[str, Any]] = []
    if tool_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": tool_text})
    out.append({"role": "tool", "content": content})
    return out


def build_cases(long_repeats: int) -> list[ProbeCase]:
    tiny_scene = text_image("invoice area", size=(960, 640), font_size=32)
    draw = ImageDraw.Draw(tiny_scene)
    draw.text((795, 545), "AX17", fill="black", font=font(16))
    crop_ax17 = text_image("AX17", size=(420, 180), font_size=86)

    q1 = "What is the code printed in the bottom-right label?"
    m0 = base_messages(q1)
    m1 = with_tool_step(
        m0,
        thought="The label is tiny, so I will crop the bottom-right area.",
        code="crop = _call_manual_crop(760, 520, 940, 620)",
        tool_text="Manual crop returned the bottom-right label region.",
        tool_image=True,
    )

    q2 = "What is the final access code?"
    blank = text_image("No visible code", size=(720, 420), font_size=42)
    long_messages = base_messages(q2)
    for idx in range(long_repeats):
        long_messages = with_tool_step(
            long_messages,
            thought=f"Attempt {idx}: inspect another irrelevant region.",
            code=f"crop = _call_manual_crop({idx}, {idx}, {idx + 20}, {idx + 20})",
            tool_text=("This crop is irrelevant. " * 20).strip(),
            tool_image=False,
        )
    long_messages = with_tool_step(
        long_messages,
        thought="The OCR result from the final crop contains the answer.",
        code="ocr = _call_ocr()",
        tool_text="OCR result: the final access code is 73.",
        tool_image=False,
    )

    q3 = "What color is the status word?"
    red_img = text_image("STATUS: RED", size=(720, 420), font_size=72, fg="red")
    bad_tool = with_tool_step(
        base_messages(q3),
        thought="I will OCR the status region.",
        code="ocr = _call_ocr()",
        tool_text="OCR result: BLUE.",
        tool_image=False,
    )

    q4 = "What animal is shown?"
    cat_img = text_image("CAT", size=(720, 420), font_size=92)
    step1 = with_tool_step(
        base_messages(q4),
        thought="The object is unclear; crop the empty top-left area first.",
        code="crop = _call_manual_crop(0, 0, 100, 100)",
        tool_text="The crop is blank and does not reveal the animal.",
        tool_image=False,
    )
    step2 = with_tool_step(
        step1,
        thought="The previous crop failed; crop the central label.",
        code="crop = _call_manual_crop(200, 100, 520, 340)",
        tool_text="The crop text clearly reads CAT.",
        tool_image=True,
    )
    cat_crop = text_image("CAT", size=(360, 160), font_size=86)

    return [
        ProbeCase(
            name="visual_crop_tiny_text",
            question=q1,
            ground_truth="ax17",
            answer_instruction="Answer with the code only.",
            states=[
                ProbeState("baseline_before_tools", m0, [tiny_scene]),
                ProbeState("after_tool_step_1", m1, [tiny_scene, crop_ax17]),
            ],
        ),
        ProbeCase(
            name="long_context_late_text_reveal",
            question=q2,
            ground_truth="73",
            answer_instruction="Answer with the access code only.",
            states=[
                ProbeState("baseline_before_tools", base_messages(q2), [blank]),
                ProbeState("after_tool_step_final", long_messages, [blank]),
            ],
        ),
        ProbeCase(
            name="bad_tool_should_not_improve",
            question=q3,
            ground_truth="red",
            answer_instruction="Answer with one color word.",
            states=[
                ProbeState("baseline_before_tools", base_messages(q3), [red_img]),
                ProbeState("after_bad_tool_step", bad_tool, [red_img]),
            ],
        ),
        ProbeCase(
            name="multistep_late_visual_reveal",
            question=q4,
            ground_truth="cat",
            answer_instruction="Answer with one word.",
            states=[
                ProbeState("baseline_before_tools", base_messages(q4), [cat_img]),
                ProbeState("after_tool_step_1", step1, [cat_img]),
                ProbeState("after_tool_step_2", step2, [cat_img, cat_crop]),
            ],
        ),
    ]


def endpoint_specs(args: argparse.Namespace) -> list[EndpointSpec]:
    specs: list[EndpointSpec] = []
    log_env = load_log_env(args.env_file)

    if args.include_dashscope:
        base_url = args.dashscope_base_url or log_env.get(
            "OFFLINE_SFT_QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        api_key = args.dashscope_api_key or log_env.get("OFFLINE_SFT_QWEN_API_KEY", "")
        if api_key:
            for model in [item.strip() for item in args.api_models.split(",") if item.strip()]:
                env_name = safe_env_name(model)
                os.environ[env_name] = api_key
                specs.append(
                    EndpointSpec(
                        name=model.replace(".", "_").replace("-", "_"),
                        base_url=base_url,
                        model=model,
                        api_key_env=env_name,
                        request_body={"enable_thinking": False, "temperature": args.temperature},
                    )
                )

    if args.judge_host:
        os.environ.setdefault("JUDGE_LOCAL_API_KEY", args.local_api_key)
        host = args.judge_host
        local_members = [
            ("qwen3_vl_2b", 19080, "qwen3-vl-2b-step-judge"),
            ("qwen3_vl_4b", 19090, "qwen3-vl-4b-step-judge"),
            ("qwen3_vl_8b", 19100, "qwen3-vl-8b-step-judge"),
            ("qwen3_vl_32b", 19110, "qwen3-vl-32b-step-judge"),
        ]
        if args.include_8b_test:
            local_members.append(("qwen3_vl_8b_test", 19120, "qwen3-vl-8b-step-judge-test"))
        for name, port, model in local_members:
            specs.append(
                EndpointSpec(
                    name=name,
                    base_url=f"http://{host}:{port}/v1",
                    model=model,
                    api_key_env="JUDGE_LOCAL_API_KEY",
                    request_body={"temperature": args.temperature},
                )
            )

    if args.gateway_base_url:
        os.environ.setdefault("STEP_JUDGE_API_KEY", args.gateway_api_key)
        specs.append(
            EndpointSpec(
                name="committee_gateway",
                base_url=args.gateway_base_url,
                model=args.gateway_model,
                api_key_env="STEP_JUDGE_API_KEY",
                request_body={"temperature": args.temperature},
            )
        )
    return specs


def extra_info(case: ProbeCase) -> dict[str, Any]:
    return {
        "source_dataset": "gqa",
        "reward_family": "exact",
        "answer_type": "short_text",
        "question": case.question,
    }


def run_probe(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases = build_cases(args.long_repeats)
    specs = endpoint_specs(args)
    if not specs:
        print("No endpoints configured. Use --judge-host, --gateway-base-url, or DashScope env in --env-file.")
        return []

    rows: list[dict[str, Any]] = []
    tau_values = [float(x) for x in args.tau_values.split(",") if x.strip()]
    for spec in specs:
        client = StepAnswerabilityJudgeClient.from_mapping(
            {
                "enable": True,
                "base_url": spec.base_url,
                "model": spec.model,
                "api_key_env": spec.api_key_env,
                "timeout_s": args.timeout_s,
                "max_retries": args.max_retries,
                "num_judgments": args.num_judgments,
                "aggregation": "mean",
                "prompt_mode": "context",
                "max_images": args.max_images,
                "max_context_chars": args.max_context_chars,
                "max_observation_chars": args.max_observation_chars,
                "request_body": spec.request_body,
            }
        )
        for case in cases:
            state_records = []
            scores = []
            started = time.perf_counter()
            for state in case.states:
                record = client.score_state(
                    data_source="gqa",
                    ground_truth=case.ground_truth,
                    extra_info=extra_info(case),
                    question=case.question,
                    answer_instruction=case.answer_instruction,
                    state_label=state.label,
                    observation_text="",
                    images=state.images,
                    context_messages=state.messages,
                    tools=[{"type": "function", "function": {"name": "code_image_tool"}}],
                )
                state_records.append(record)
                scores.append(record.get("score"))
            deltas = {
                str(tau): compute_step_answerability_delta(scores, [True] * (len(scores) - 1), tau=tau, cap=1.0)
                for tau in tau_values
            }
            row = {
                "endpoint": spec.name,
                "model": spec.model,
                "case": case.name,
                "scores": scores,
                "final_answers": [r.get("final_answer") for r in state_records],
                "errors": [r.get("error") for r in state_records],
                "latency_s": round(time.perf_counter() - started, 3),
                "delta_by_tau": {
                    tau: {
                        "raw_delta": deltas[tau]["raw_delta"],
                        "capped_delta": deltas[tau]["capped_delta"],
                        "step_gains": deltas[tau]["step_gains"],
                    }
                    for tau in deltas
                },
            }
            rows.append(row)
            print(
                f"{spec.name:24s} {case.name:30s} scores={scores} "
                f"answers={row['final_answers']} tau0.1={row['delta_by_tau'].get('0.1', {})} "
                f"errors={sum(1 for e in row['errors'] if e)} latency={row['latency_s']}"
            )
    summarize_probe(rows)
    return rows


def summarize_probe(rows: list[dict[str, Any]]) -> None:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(row["case"], []).append(row)
    print("\nGradient summary:")
    for case_name, case_rows in by_case.items():
        final_scores = [row["scores"][-1] for row in case_rows if row["scores"] and row["scores"][-1] is not None]
        endpoints = [row["endpoint"] for row in case_rows]
        uniq = sorted({float(x) for x in final_scores})
        print(f"  {case_name}: endpoints={len(endpoints)} final_score_levels={uniq}")


def rollout_files(path: str) -> list[str]:
    p = Path(path)
    if p.is_dir():
        return sorted(glob.glob(str(p / "*.jsonl")))
    return [path]


def member_group(name: str) -> str:
    return re.sub(r"_[ab]$", "", name)


def analyze_rollout_records(args: argparse.Namespace) -> None:
    files: list[str] = []
    for path in args.rollout_records:
        files.extend(rollout_files(path))
    rows = []
    for fp in files:
        with open(fp, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    print(f"Loaded rollout rows={len(rows)} files={len(files)}")

    member_scores: dict[str, list[float]] = {}
    member_errors: dict[str, int] = {}
    member_seen: dict[str, int] = {}
    group_positive: dict[str, int] = {}
    group_sequences: dict[str, int] = {}
    drop_patterns = [p for p in args.drop_member_regex.split(",") if p]

    for row in rows:
        records = coerce_json_list(row.get("step_answerability_records"))
        per_member: dict[str, list[float | None]] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            for judgment in record.get("judgments") or []:
                if not isinstance(judgment, dict):
                    continue
                for item in judgment.get("committee_judgments") or []:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or item.get("member") or item.get("model") or "?")
                    if any(re.search(pattern, name) for pattern in drop_patterns):
                        continue
                    group = member_group(name)
                    member_seen[name] = member_seen.get(name, 0) + 1
                    if item.get("error"):
                        member_errors[name] = member_errors.get(name, 0) + 1
                        score = None
                    else:
                        score = item.get("score")
                        if score is not None:
                            member_scores.setdefault(name, []).append(float(score))
                    per_member.setdefault(group, []).append(None if score is None else float(score))
        for group, seq in per_member.items():
            if len(seq) < 2:
                continue
            group_sequences[group] = group_sequences.get(group, 0) + 1
            delta = compute_step_answerability_delta(seq, [True] * (len(seq) - 1), tau=args.records_tau, cap=1.0)
            if float(delta["raw_delta"]) > 0:
                group_positive[group] = group_positive.get(group, 0) + 1

    print("\nMember score distribution:")
    for name in sorted(member_seen):
        scores = member_scores.get(name, [])
        err = member_errors.get(name, 0)
        mean = statistics.mean(scores) if scores else None
        levels = sorted({round(float(x), 4) for x in scores})
        level_preview = levels[:10] + (["..."] if len(levels) > 10 else [])
        print(
            f"  {name:24s} seen={member_seen[name]:4d} errors={err:4d} "
            f"valid={len(scores):4d} mean={None if mean is None else round(mean, 4)} levels={level_preview}"
        )

    print(f"\nPositive per-member-group answerability delta, tau={args.records_tau}:")
    for group in sorted(group_sequences):
        seq_n = group_sequences[group]
        pos = group_positive.get(group, 0)
        print(f"  {group:20s} positive={pos:4d}/{seq_n:4d} pct={100.0 * pos / max(1, seq_n):5.2f}%")


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default="/mnt/cpfs/delinmao/log1")
    parser.add_argument("--include-dashscope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dashscope-base-url", default="")
    parser.add_argument("--dashscope-api-key", default="")
    parser.add_argument("--api-models", default=DEFAULT_API_MODELS)
    parser.add_argument("--judge-host", default="", help="DLC judge pod IP for local 2B/4B/8B/32B endpoints.")
    parser.add_argument("--local-api-key", default="local-step-judge-key")
    parser.add_argument("--include-8b-test", action="store_true")
    parser.add_argument("--gateway-base-url", default="")
    parser.add_argument("--gateway-model", default="step-judge-committee")
    parser.add_argument("--gateway-api-key", default="committee-step-judge-key")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--num-judgments", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--max-observation-chars", type=int, default=12000)
    parser.add_argument("--max-context-chars", type=int, default=60000)
    parser.add_argument("--long-repeats", type=int, default=24)
    parser.add_argument("--tau-values", default="0,0.05,0.1")
    parser.add_argument("--jsonl-out", default="")
    parser.add_argument("--rollout-records", nargs="*", default=[])
    parser.add_argument("--records-only", action="store_true")
    parser.add_argument("--records-tau", type=float, default=0.1)
    parser.add_argument("--drop-member-regex", default="122b")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rollout_records:
        analyze_rollout_records(args)
        if args.records_only:
            return 0
    rows = run_probe(args)
    write_jsonl(args.jsonl_out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
