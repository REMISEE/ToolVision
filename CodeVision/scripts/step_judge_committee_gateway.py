#!/usr/bin/env python3
"""OpenAI-compatible gateway for step-answerability judge committees.

The gateway forwards one chat/completions request to multiple judge backends and
returns all successful/failed member outputs in ``committee_judgments``. It does
not score answers itself; the RL reward process owns task-specific scoring.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import requests


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def as_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def as_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def chat_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    if root.endswith("/v1"):
        return f"{root}/chat/completions"
    return f"{root}/v1/chat/completions"


def extract_content(payload: dict[str, Any]) -> str:
    try:
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") if isinstance(choice, dict) else {}
        return str(message.get("content", "") if isinstance(message, dict) else "")
    except Exception:
        return ""


def load_json_env(name: str, default: Any) -> Any:
    text = os.getenv(name, "").strip()
    if not text:
        return default
    return json.loads(text)


@dataclass(slots=True)
class JudgeMember:
    name: str
    base_url: str
    model: str
    api_key: str = ""
    api_key_env: str = ""
    temperature: float | None = None
    timeout_s: float = 90.0
    max_retries: int = 0
    enabled: bool = True
    request_body: dict[str, Any] | None = None
    concurrency_group: str = ""
    score_group: str = ""
    max_concurrency: int = 0

    @classmethod
    def from_mapping(cls, item: dict[str, Any]) -> "JudgeMember":
        api_key_env = str(item.get("api_key_env") or "").strip()
        api_key = str(item.get("api_key") or "").strip()
        if api_key_env and not api_key:
            api_key = os.getenv(api_key_env, "")
        request_body = item.get("request_body") if isinstance(item.get("request_body"), dict) else {}
        return cls(
            name=str(item.get("name") or item.get("model") or "judge"),
            base_url=str(item.get("base_url") or "").strip(),
            model=str(item.get("model") or "").strip(),
            api_key=api_key,
            api_key_env=api_key_env,
            temperature=None if item.get("temperature") is None else as_float(item.get("temperature"), 0.25),
            timeout_s=as_float(item.get("timeout_s"), as_float(os.getenv("COMMITTEE_TIMEOUT_S"), 90.0)),
            max_retries=as_int(item.get("max_retries"), as_int(os.getenv("COMMITTEE_MAX_RETRIES"), 0)),
            enabled=as_bool(item.get("enabled", True), True),
            request_body=dict(request_body or {}),
            concurrency_group=str(item.get("concurrency_group") or "").strip(),
            score_group=str(item.get("score_group") or item.get("concurrency_group") or "").strip(),
            max_concurrency=max(0, as_int(item.get("max_concurrency"), 0)),
        )

    @property
    def usable(self) -> bool:
        return bool(self.enabled and self.base_url and self.model)


class CommitteeGateway:
    def __init__(self) -> None:
        judges_raw = load_json_env("COMMITTEE_JUDGES_JSON", None)
        judges_file = os.getenv("COMMITTEE_JUDGES_FILE", "").strip()
        if judges_raw is None and judges_file:
            judges_raw = json.loads(Path(judges_file).read_text())
        if not isinstance(judges_raw, list):
            judges_raw = []
        self.members = [JudgeMember.from_mapping(item) for item in judges_raw if isinstance(item, dict)]
        self.members = [member for member in self.members if member.usable]
        self.model_name = os.getenv("COMMITTEE_MODEL_NAME", "step-judge-committee")
        self.api_key = os.getenv("COMMITTEE_API_KEY", "").strip()
        self.max_workers = max(1, as_int(os.getenv("COMMITTEE_MAX_WORKERS"), max(1, len(self.members))))
        self.min_successes = max(1, as_int(os.getenv("COMMITTEE_MIN_SUCCESSES"), 1))
        self.require_all = as_bool(os.getenv("COMMITTEE_REQUIRE_ALL"), False)
        self.min_success_groups = max(0, as_int(os.getenv("COMMITTEE_MIN_SUCCESS_GROUPS"), 0))
        self.required_groups = self._parse_csv_env("COMMITTEE_REQUIRED_GROUPS")
        self.required_any_groups = self._parse_csv_env("COMMITTEE_REQUIRED_ANY_GROUPS")
        self.default_temperature = as_float(os.getenv("COMMITTEE_DEFAULT_TEMPERATURE"), 0.25)
        self.log_path = os.getenv("COMMITTEE_LOG_JSONL", "").strip()
        self.max_inflight_requests = max(0, as_int(os.getenv("COMMITTEE_MAX_INFLIGHT_REQUESTS"), 0))
        self.request_semaphore = self._build_request_semaphore()
        self.group_semaphores = self._build_group_semaphores()

    def _build_request_semaphore(self) -> threading.BoundedSemaphore | None:
        if self.max_inflight_requests <= 0:
            return None
        return threading.BoundedSemaphore(self.max_inflight_requests)

    def _build_group_semaphores(self) -> dict[str, threading.BoundedSemaphore]:
        limits: dict[str, int] = {}
        for member in self.members:
            group = self._member_group(member)
            if member.max_concurrency <= 0:
                continue
            if group in limits:
                limits[group] = min(limits[group], member.max_concurrency)
            else:
                limits[group] = member.max_concurrency
        return {group: threading.BoundedSemaphore(limit) for group, limit in limits.items() if limit > 0}

    def _member_group(self, member: JudgeMember) -> str:
        return member.concurrency_group or member.base_url.rstrip("/")

    def _member_score_group(self, member: JudgeMember) -> str:
        return member.score_group or self._member_group(member)

    def _parse_csv_env(self, name: str) -> set[str]:
        raw = os.getenv(name, "")
        return {item.strip() for item in raw.split(",") if item.strip()}

    def check_auth(self, headers: Any) -> bool:
        if not self.api_key:
            return True
        auth = str(headers.get("Authorization", ""))
        return auth == f"Bearer {self.api_key}"

    def call_member_bounded(self, member: JudgeMember, request_body: dict[str, Any]) -> dict[str, Any]:
        semaphore = self.group_semaphores.get(self._member_group(member))
        if semaphore is None:
            return self.call_member(member, request_body)
        semaphore.acquire()
        try:
            return self.call_member(member, request_body)
        finally:
            semaphore.release()

    def call_member(self, member: JudgeMember, request_body: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        body = dict(request_body)
        body["model"] = member.model
        body["stream"] = False
        body["temperature"] = member.temperature if member.temperature is not None else self.default_temperature
        body.update(member.request_body or {})
        url = chat_url(member.base_url)
        headers = {"Content-Type": "application/json"}
        if member.api_key:
            headers["Authorization"] = f"Bearer {member.api_key}"

        last_error: Exception | None = None
        for attempt in range(max(0, member.max_retries) + 1):
            try:
                response = requests.post(url, headers=headers, json=body, timeout=member.timeout_s)
                response.raise_for_status()
                payload = response.json()
                content = extract_content(payload)
                return {
                    "name": member.name,
                    "model": member.model,
                    "score_group": self._member_score_group(member),
                    "concurrency_group": self._member_group(member),
                    "temperature": body.get("temperature"),
                    "success": True,
                    "raw_answer": content,
                    "content": content,
                    "usage": payload.get("usage", {}) if isinstance(payload, dict) else {},
                    "latency_s": round(time.perf_counter() - started, 3),
                    "attempt": attempt,
                    "error": None,
                }
            except Exception as exc:
                last_error = exc
        return {
            "name": member.name,
            "model": member.model,
            "score_group": self._member_score_group(member),
            "concurrency_group": self._member_group(member),
            "temperature": body.get("temperature"),
            "success": False,
            "raw_answer": "",
            "content": "",
            "usage": {},
            "latency_s": round(time.perf_counter() - started, 3),
            "attempt": max(0, member.max_retries),
            "error": str(last_error),
        }

    def chat_completions(self, request_body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if self.request_semaphore is not None:
            self.request_semaphore.acquire()
        try:
            return self._chat_completions_inner(request_body)
        finally:
            if self.request_semaphore is not None:
                self.request_semaphore.release()

    def _chat_completions_inner(self, request_body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        started = time.perf_counter()
        if not self.members:
            return 503, {"error": {"message": "committee has no enabled judge members"}}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.call_member_bounded, member, request_body) for member in self.members]
            judgments = [future.result() for future in concurrent.futures.as_completed(futures)]
        judgments.sort(key=lambda item: item.get("name", ""))
        successes = [item for item in judgments if item.get("success") and str(item.get("content", "")).strip()]
        success_groups = {
            str(item.get("score_group") or item.get("concurrency_group") or item.get("name") or "")
            for item in successes
        }
        missing_required_groups = sorted(group for group in self.required_groups if group not in success_groups)
        required_any_satisfied = not self.required_any_groups or bool(success_groups & self.required_any_groups)
        required_successes = len(self.members) if self.require_all else self.min_successes
        enough_successes = len(successes) >= required_successes
        enough_groups = not self.min_success_groups or len(success_groups) >= self.min_success_groups
        status = 200 if (
            enough_successes
            and enough_groups
            and not missing_required_groups
            and required_any_satisfied
        ) else 502
        content = str(successes[0].get("content", "")) if successes else ""
        usage = self._merge_usage([item.get("usage", {}) for item in successes])

        response = {
            "id": f"chatcmpl-committee-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop" if successes else "error",
                }
            ],
            "usage": usage,
            "committee_judgments": judgments,
            "committee_success_count": len(successes),
            "committee_total_count": len(judgments),
            "committee_required_success_count": required_successes,
            "committee_success_groups": sorted(success_groups),
            "committee_success_group_count": len(success_groups),
            "committee_min_success_group_count": self.min_success_groups,
            "committee_required_groups": sorted(self.required_groups),
            "committee_required_any_groups": sorted(self.required_any_groups),
            "committee_latency_s": round(time.perf_counter() - started, 3),
        }
        if status != 200:
            reasons = []
            if not enough_successes:
                reasons.append("not enough successful committee judge responses")
            if not enough_groups:
                reasons.append("not enough successful committee judge groups")
            if missing_required_groups:
                reasons.append(f"missing required groups: {','.join(missing_required_groups)}")
            if not required_any_satisfied:
                reasons.append("missing any required strong group")
            response["error"] = {"message": "; ".join(reasons) or "committee quorum failed"}
        self._log_response(response)
        return status, response

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.model_name,
            "require_all": self.require_all,
            "min_successes": self.min_successes,
            "min_success_groups": self.min_success_groups,
            "required_groups": sorted(self.required_groups),
            "required_any_groups": sorted(self.required_any_groups),
            "max_workers": self.max_workers,
            "max_inflight_requests": self.max_inflight_requests,
            "bounded_groups": sorted(self.group_semaphores),
            "members": [
                {
                    "name": member.name,
                    "model": member.model,
                    "base_url": member.base_url,
                    "concurrency_group": self._member_group(member),
                    "score_group": self._member_score_group(member),
                    "max_concurrency": member.max_concurrency,
                }
                for member in self.members
            ],
        }

    def models(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model_name,
                    "object": "model",
                    "owned_by": "toolvision",
                }
            ],
        }

    def _merge_usage(self, usages: list[dict[str, Any]]) -> dict[str, int]:
        merged: dict[str, int] = {}
        for usage in usages:
            if not isinstance(usage, dict):
                continue
            for key, value in usage.items():
                if isinstance(value, int):
                    merged[key] = merged.get(key, 0) + value
        return merged

    def _log_response(self, response: dict[str, Any]) -> None:
        if not self.log_path:
            return
        try:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            summary = {
                "created": response.get("created"),
                "success_count": response.get("committee_success_count"),
                "total_count": response.get("committee_total_count"),
                "required_success_count": response.get("committee_required_success_count"),
                "success_group_count": response.get("committee_success_group_count"),
                "success_groups": response.get("committee_success_groups"),
                "latency_s": response.get("committee_latency_s"),
                "members": [
                    {
                        "name": item.get("name"),
                        "score_group": item.get("score_group"),
                        "success": item.get("success"),
                        "latency_s": item.get("latency_s"),
                        "error": item.get("error"),
                    }
                    for item in response.get("committee_judgments", [])
                ],
            }
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        except Exception:
            return


GATEWAY = CommitteeGateway()


class QueuedThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = max(128, as_int(os.getenv("COMMITTEE_REQUEST_QUEUE_SIZE"), 256))


class Handler(BaseHTTPRequestHandler):
    server_version = "StepJudgeCommitteeGateway/1.0"

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/health", "/healthy"}:
            self.write_json(200, GATEWAY.health())
            return
        if self.path in {"/v1/models", "/models"}:
            self.write_json(200, GATEWAY.models())
            return
        self.write_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/v1/chat/completions", "/chat/completions"}:
            self.write_json(404, {"error": {"message": "not found"}})
            return
        if not GATEWAY.check_auth(self.headers):
            self.write_json(401, {"error": {"message": "unauthorized"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            status, payload = GATEWAY.chat_completions(body)
            self.write_json(status, payload)
        except Exception as exc:
            self.write_json(500, {"error": {"message": str(exc)}})

    def log_message(self, fmt: str, *args: Any) -> None:
        if as_bool(os.getenv("COMMITTEE_ACCESS_LOG"), False):
            super().log_message(fmt, *args)

    def write_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    host = os.getenv("COMMITTEE_HOST", "0.0.0.0")
    port = as_int(os.getenv("COMMITTEE_PORT"), 19200)
    print(f"Starting step judge committee gateway on {host}:{port}", flush=True)
    print(f"Committee members: {[member.name for member in GATEWAY.members]}", flush=True)
    server = QueuedThreadingHTTPServer((host, port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
