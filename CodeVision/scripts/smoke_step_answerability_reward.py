#!/usr/bin/env python3
"""Smoke tests for step answerability judge and reward shaping.

This script starts local OpenAI-compatible mock services. It does not require a
real model or vLLM, but exercises the same HTTP, answer extraction, rule scoring,
and delta aggregation paths used by RL rollout.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from PIL import Image

from recipe.codevision.rewards.step_answerability import (
    StepAnswerabilityJudgeClient,
    compute_step_answerability_delta,
)


COMMON_EXTRA_INFO = {
    "source_dataset": "gqa",
    "reward_family": "exact",
    "answer_type": "short_text",
    "question": "What animal is shown?",
}

COMMON_SCORE_KWARGS = {
    "data_source": "gqa",
    "ground_truth": "cat",
    "extra_info": COMMON_EXTRA_INFO,
    "question": "What animal is shown?",
    "answer_instruction": "Answer with one word.",
    "images": [],
}


class MockJudgeHandler(BaseHTTPRequestHandler):
    answers_by_state: dict[str, list[str]] = {}
    status_code: int = 200
    state_counts: dict[str, int] = {}

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        payload = json.loads(body)
        prompt_text = self._prompt_text(payload)

        if self.status_code != 200:
            raw = json.dumps({"error": "mock failure"}).encode("utf-8")
            self.send_response(self.status_code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        state_label = self._state_label(prompt_text)
        answers = self.answers_by_state.get(state_label) or self.answers_by_state.get("*") or ["cat"]
        answer_idx = self.state_counts.get(state_label, 0)
        self.state_counts[state_label] = answer_idx + 1
        answer = answers[min(answer_idx, len(answers) - 1)]

        response = {
            "id": "mock-step-judge",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"<answer>{answer}</answer>"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }
        raw = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _prompt_text(self, payload: dict[str, Any]) -> str:
        text_parts = []
        for msg in payload.get("messages", []):
            content = msg.get("content")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
        return "\n".join(text_parts)

    def _state_label(self, prompt_text: str) -> str:
        for state_label in self.answers_by_state:
            if state_label != "*" and state_label in prompt_text:
                return state_label
        return "*"


class MockServer:
    def __init__(self, answers_by_state: dict[str, list[str]], status_code: int = 200):
        self.handler_cls = type(
            "ScopedMockJudgeHandler",
            (MockJudgeHandler,),
            {"answers_by_state": answers_by_state, "status_code": status_code, "state_counts": {}},
        )
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self.handler_cls)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "MockServer":
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.server.shutdown()
        self.server.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"


def assert_close(name: str, actual: float, expected: float, eps: float = 1e-8) -> None:
    if abs(float(actual) - float(expected)) > eps:
        raise AssertionError(f"{name}: actual={actual}, expected={expected}")


def make_client(base_url: str, **overrides: Any) -> StepAnswerabilityJudgeClient:
    cfg = {
        "enable": True,
        "base_url": base_url,
        "model": "mock-judge",
        "timeout_s": 5,
        "max_retries": 0,
    }
    cfg.update(overrides)
    return StepAnswerabilityJudgeClient.from_mapping(cfg)


def test_endpoint_forms() -> None:
    with MockServer({"baseline_before_tools": ["dog"], "after_tool_step_1": ["cat"]}) as server:
        for suffix in ("", "/v1", "/v1/chat/completions"):
            client = make_client(server.url + suffix)
            baseline = client.score_state(
                **COMMON_SCORE_KWARGS,
                state_label="baseline_before_tools",
                observation_text="No tool has been used yet.",
            )
            after = client.score_state(
                **COMMON_SCORE_KWARGS,
                state_label="after_tool_step_1",
                observation_text="Tool observation: crop shows a cat.",
            )
            delta = compute_step_answerability_delta([baseline["score"], after["score"]], [True], tau=0.1, cap=0.5)
            assert_close(f"baseline score {suffix}", baseline["score"], 0.0)
            assert_close(f"after score {suffix}", after["score"], 1.0)
            assert_close(f"capped delta {suffix}", delta["capped_delta"], 0.5)


def test_no_improvement_gets_zero() -> None:
    delta = compute_step_answerability_delta([1.0, 1.0, 1.0], [True, True], tau=0.1, cap=0.5)
    assert_close("no improvement raw", delta["raw_delta"], 0.0)
    assert_close("no improvement capped", delta["capped_delta"], 0.0)


def test_invalid_step_gets_no_positive_gain() -> None:
    delta = compute_step_answerability_delta([0.0, 1.0], [False], tau=0.1, cap=0.5)
    assert_close("invalid step raw", delta["raw_delta"], 0.0)
    assert_close("invalid step capped", delta["capped_delta"], 0.0)
    if delta["valid_count"] != 0:
        raise AssertionError(f"invalid step valid_count={delta['valid_count']}, expected 0")


def test_missing_baseline_gets_no_positive_gain() -> None:
    delta = compute_step_answerability_delta([None, 1.0, 1.0], [True, True], tau=0.1, cap=0.5)
    assert_close("missing baseline raw", delta["raw_delta"], 0.0)
    assert_close("missing baseline capped", delta["capped_delta"], 0.0)
    if not delta.get("missing_baseline"):
        raise AssertionError("missing baseline should be explicit")


def test_multistep_repeats_do_not_farm_reward() -> None:
    delta = compute_step_answerability_delta([0.0, 0.7, 0.75, 0.75, 1.0], [True, True, True, True], tau=0.1, cap=1.0)
    assert_close("multistep gain 0", delta["step_gains"][0], 0.6)
    assert_close("multistep repeat gain 1", delta["step_gains"][1], 0.0)
    assert_close("multistep repeat gain 2", delta["step_gains"][2], 0.0)
    assert_close("multistep final gain", delta["step_gains"][3], 0.15)
    assert_close("multistep raw", delta["raw_delta"], 0.75)


def test_judge_failure_is_recorded_not_raised() -> None:
    with MockServer({"*": ["cat"]}, status_code=500) as server:
        client = make_client(server.url)
        record = client.score_state(
            **COMMON_SCORE_KWARGS,
            state_label="after_tool_step_1",
            observation_text="Tool observation: crop shows a cat.",
        )
        if record["score"] is not None:
            raise AssertionError(f"failure score={record['score']}, expected None")
        if not record["error"]:
            raise AssertionError("failure should record error")


def test_repeated_judgments_mean_and_reward_scale() -> None:
    with MockServer({"baseline_before_tools": ["dog", "dog"], "after_tool_step_1": ["cat", "dog"]}) as server:
        client = make_client(server.url, num_judgments=2, aggregation="mean")
        baseline = client.score_state(
            **COMMON_SCORE_KWARGS,
            state_label="baseline_before_tools",
            observation_text="No tool has been used yet.",
        )
        after = client.score_state(
            **COMMON_SCORE_KWARGS,
            state_label="after_tool_step_1",
            observation_text="Tool observation: crop shows a cat.",
        )
        delta = compute_step_answerability_delta([baseline["score"], after["score"]], [True], tau=0.1, cap=0.5)
        r_step = 0.2 * float(delta["capped_delta"])

        assert_close("repeated baseline", baseline["score"], 0.0)
        assert_close("repeated after mean", after["score"], 0.5)
        assert_close("repeated after std", after["score_std"], 0.5)
        if after["judgment_count"] != 2:
            raise AssertionError(f"judgment_count={after['judgment_count']}, expected 2")
        assert_close("repeated raw delta", delta["raw_delta"], 0.4)
        assert_close("repeated R_step", r_step, 0.08)


def test_committee_payload_averages_successes_only() -> None:
    client = make_client("http://unused")
    committee = client._score_committee_payload(
        raw_answer="",
        raw_payload={
            "committee_judgments": [
                {"name": "judge_ok", "raw_answer": "<answer>cat</answer>"},
                {"name": "judge_wrong", "raw_answer": "<answer>dog</answer>"},
                {"name": "judge_failed", "error": "timeout", "raw_answer": ""},
            ]
        },
        **{
            "data_source": COMMON_SCORE_KWARGS["data_source"],
            "ground_truth": COMMON_SCORE_KWARGS["ground_truth"],
            "extra_info": COMMON_SCORE_KWARGS["extra_info"],
        },
    )
    scores = [float(item["score"]) for item in committee if item.get("score") is not None]
    assert_close("committee mean", client._aggregate_scores(scores), 0.5)
    if len(scores) != 2:
        raise AssertionError(f"committee valid scores={len(scores)}, expected 2")
    failed = [item for item in committee if item.get("name") == "judge_failed"]
    if not failed or failed[0].get("score") is not None:
        raise AssertionError("failed committee member should not receive a score")


def test_committee_group_mean_weights_model_groups_once() -> None:
    client = make_client("http://unused", aggregation="group_mean")
    committee = client._score_committee_payload(
        raw_answer="",
        raw_payload={
            "committee_judgments": [
                {"name": "qwen3_vl_2b_a", "score_group": "qwen3_vl_2b", "raw_answer": "<answer>dog</answer>"},
                {"name": "qwen3_vl_2b_b", "score_group": "qwen3_vl_2b", "raw_answer": "<answer>dog</answer>"},
                {"name": "qwen3_vl_32b_a", "score_group": "qwen3_vl_32b", "raw_answer": "<answer>cat</answer>"},
                {"name": "api_failed", "score_group": "qwen36plus_api", "error": "timeout", "raw_answer": ""},
            ]
        },
        **{
            "data_source": COMMON_SCORE_KWARGS["data_source"],
            "ground_truth": COMMON_SCORE_KWARGS["ground_truth"],
            "extra_info": COMMON_SCORE_KWARGS["extra_info"],
        },
    )
    score, meta = client._aggregate_committee_scores(committee)
    assert_close("committee group mean", score, 0.5)
    if meta.get("success_group_count") != 2:
        raise AssertionError(f"group count={meta.get('success_group_count')}, expected 2")
    if meta.get("group_scores", {}).get("qwen3_vl_2b") != 0.0:
        raise AssertionError(f"2b group score={meta.get('group_scores')}")
    if meta.get("group_scores", {}).get("qwen3_vl_32b") != 1.0:
        raise AssertionError(f"32b group score={meta.get('group_scores')}")


def test_context_prompt_contains_rollout_history_without_gt() -> None:
    client = make_client("http://unused", prompt_mode="context", max_context_chars=20000)
    messages = client._build_messages(
        question="What code is visible?",
        answer_instruction="Answer with the code only.",
        state_label="after_tool_step_1",
        observation_text="legacy snapshot observation",
        images=[Image.new("RGB", (32, 32), "white")],
        tools=[{"type": "function", "function": {"name": "code_image_tool"}}],
        context_messages=[
            {"role": "system", "content": "System prompt with tool-use policy."},
            {"role": "user", "content": "<image>What code is visible?"},
            {
                "role": "assistant",
                "content": "<think>I should crop the label.</think><tool_call>{\"name\":\"code_image_tool\"}</tool_call>",
            },
            {"role": "tool", "content": "OCR crop result says AX-17."},
        ],
    )
    payload_text = json.dumps(messages, ensure_ascii=False)
    required = ["System prompt with tool-use policy", "code_image_tool", "<tool_call>", "OCR crop result says AX-17"]
    for item in required:
        if item not in payload_text:
            raise AssertionError(f"context prompt missing {item!r}: {payload_text[:500]}")
    if "image_url" not in payload_text:
        raise AssertionError("context prompt did not materialize string <image> placeholder")
    if "cat" in payload_text:
        raise AssertionError("ground truth leaked into context prompt")
    if any(message.get("role") == "tool" for message in messages):
        raise AssertionError("context prompt should normalize tool role for OpenAI-compatible APIs")


def main() -> int:
    tests = [
        test_endpoint_forms,
        test_no_improvement_gets_zero,
        test_invalid_step_gets_no_positive_gain,
        test_missing_baseline_gets_no_positive_gain,
        test_multistep_repeats_do_not_farm_reward,
        test_judge_failure_is_recorded_not_raised,
        test_repeated_judgments_mean_and_reward_scale,
        test_committee_payload_averages_successes_only,
        test_committee_group_mean_weights_model_groups_once,
        test_context_prompt_contains_rollout_history_without_gt,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
