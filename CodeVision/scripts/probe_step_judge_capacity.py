#!/usr/bin/env python3
"""Wait for a DLC step-judge service and probe safe judge concurrency."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


DIRECT_ENDPOINTS = [
    ("qwen3_vl_2b", 19080, "qwen3-vl-2b-step-judge"),
    ("qwen3_vl_4b", 19090, "qwen3-vl-4b-step-judge"),
    ("qwen3_vl_8b", 19100, "qwen3-vl-8b-step-judge"),
    ("qwen3_vl_32b", 19110, "qwen3-vl-32b-step-judge"),
    ("qwen3_vl_8b_test", 19120, "qwen3-vl-8b-step-judge-test"),
]


@dataclass(frozen=True)
class ProbeResult:
    target: str
    level: int
    requests: int
    success: int
    failures: int
    wall_s: float
    mean_s: float | None
    p50_s: float | None
    p95_s: float | None
    errors: dict[str, int]

    @property
    def ok_rate(self) -> float:
        return self.success / max(1, self.requests)


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} {message}", flush=True)


def load_json_from_dlc(output: str) -> dict[str, Any]:
    start = output.find("{")
    if start < 0:
        raise ValueError(f"dlc output has no JSON object: {output[:200]!r}")
    return json.loads(output[start:])


def get_job(job_id: str) -> dict[str, Any]:
    output = subprocess.check_output(
        [
            "dlc",
            "-r",
            "cn-wulanchabu",
            "-e",
            "pai-dlc.cn-wulanchabu.aliyuncs.com",
            "get",
            "job",
            job_id,
        ],
        text=True,
        timeout=30,
    )
    return load_json_from_dlc(output)


def pod_ip(job: dict[str, Any]) -> str:
    for pod in job.get("Pods") or []:
        ip = pod.get("PodIp") or pod.get("PodIP")
        if ip:
            return str(ip)
    return ""


def wait_for_running(job_id: str, *, poll_s: float, max_wait_s: float) -> str:
    started = time.time()
    last_status = ""
    while True:
        job = get_job(job_id)
        status = str(job.get("Status") or "")
        reason = str(job.get("ReasonCode") or "")
        ip = pod_ip(job)
        if status != last_status or ip:
            log(f"job={job_id} status={status} reason={reason} ip={ip or '<none>'}")
            last_status = status
        if status == "Running" and ip:
            return ip
        if status in {"Failed", "Stopped", "Succeeded"}:
            raise RuntimeError(f"job ended before probe: status={status} reason={reason}")
        if time.time() - started > max_wait_s:
            raise TimeoutError(f"waited {max_wait_s:.0f}s for {job_id}, last status={status} reason={reason}")
        time.sleep(poll_s)


def wait_for_http(url: str, *, timeout_s: float, poll_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if 200 <= response.status_code < 500:
                return
        except Exception:
            pass
        time.sleep(poll_s)
    raise TimeoutError(f"service did not answer in time: {url}")


def image_data_url() -> str:
    try:
        from PIL import Image

        img = Image.new("RGB", (96, 96), (220, 32, 32))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()
    except Exception:
        raw = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
        )
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def judge_like_messages() -> list[dict[str, Any]]:
    context = "Prior rollout context: " + ("The model inspected the image and tool output. " * 180)
    return [
        {
            "role": "system",
            "content": (
                "You are a strict visual question answering judge. Continue from the current "
                "state and output only one final answer inside <answer>...</answer>."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": context + "\nWhat is the dominant color of the square?"},
                {"type": "image_url", "image_url": {"url": image_data_url()}},
            ],
        },
    ]


def post_chat(
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout_s: float,
    request_idx: int,
) -> tuple[bool, float, str]:
    del request_idx
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        chat_url = url
    elif url.endswith("/v1"):
        chat_url = f"{url}/chat/completions"
    else:
        chat_url = f"{url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": judge_like_messages(),
        "temperature": 0.2,
        "max_tokens": 48,
    }
    started = time.perf_counter()
    try:
        response = requests.post(
            chat_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout_s,
        )
        elapsed = time.perf_counter() - started
        if not response.ok:
            return False, elapsed, f"http_{response.status_code}"
        content = str((response.json().get("choices") or [{}])[0].get("message", {}).get("content", ""))
        if not content.strip():
            return False, elapsed, "empty_content"
        return True, elapsed, "ok"
    except Exception as exc:
        return False, time.perf_counter() - started, type(exc).__name__


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
    return ordered[idx]


def run_ramp(
    *,
    target: str,
    base_url: str,
    model: str,
    api_key: str,
    levels: list[int],
    per_level_multiplier: int,
    timeout_s: float,
    output_jsonl: Path,
    stop_on_bad: bool,
) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    for level in levels:
        total = max(level, level * per_level_multiplier)
        log(f"probe target={target} concurrency={level} requests={total}")
        started = time.perf_counter()
        latencies: list[float] = []
        errors: dict[str, int] = {}
        success = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=level) as executor:
            futures = [
                executor.submit(
                    post_chat,
                    base_url=base_url,
                    model=model,
                    api_key=api_key,
                    timeout_s=timeout_s,
                    request_idx=i,
                )
                for i in range(total)
            ]
            for future in concurrent.futures.as_completed(futures):
                ok, elapsed, label = future.result()
                if ok:
                    success += 1
                    latencies.append(elapsed)
                else:
                    errors[label] = errors.get(label, 0) + 1
        wall = time.perf_counter() - started
        result = ProbeResult(
            target=target,
            level=level,
            requests=total,
            success=success,
            failures=total - success,
            wall_s=round(wall, 3),
            mean_s=round(statistics.mean(latencies), 3) if latencies else None,
            p50_s=None if not latencies else round(percentile(latencies, 0.50) or 0.0, 3),
            p95_s=None if not latencies else round(percentile(latencies, 0.95) or 0.0, 3),
            errors=errors,
        )
        results.append(result)
        with output_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")
        log(
            "result "
            f"target={target} concurrency={level} ok={success}/{total} "
            f"wall={result.wall_s}s p95={result.p95_s} errors={errors}"
        )
        if stop_on_bad and (result.ok_rate < 0.95 or (result.p95_s is not None and result.p95_s > timeout_s * 0.8)):
            log(f"stop ramp for {target}: ok_rate={result.ok_rate:.3f} p95={result.p95_s}")
            break
    return results


def summarize(results: list[ProbeResult], output_path: Path) -> None:
    stable: dict[str, int] = {}
    for result in results:
        if result.ok_rate >= 0.95 and (result.p95_s is None or result.p95_s < 120):
            stable[result.target] = max(stable.get(result.target, 0), result.level)
    output_path.write_text(json.dumps({"stable_concurrency": stable}, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"summary={output_path} stable={stable}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--local-api-key", default="local-step-judge-key")
    parser.add_argument("--committee-api-key", default="committee-step-judge-key")
    parser.add_argument("--gateway-model", default="step-judge-committee")
    parser.add_argument("--poll-s", type=float, default=30)
    parser.add_argument("--max-wait-s", type=float, default=48 * 3600)
    parser.add_argument("--direct-levels", default="1,2,4,8,16,32")
    parser.add_argument("--gateway-levels", default="1,2,4,8,16,32")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "capacity_results.jsonl"
    summary_path = output_dir / "capacity_summary.json"

    ip = wait_for_running(args.job_id, poll_s=args.poll_s, max_wait_s=args.max_wait_s)
    (output_dir / "judge_host.txt").write_text(ip + "\n", encoding="utf-8")

    log(f"waiting for gateway health ip={ip}")
    wait_for_http(f"http://{ip}:19200/health", timeout_s=3600)
    try:
        health = requests.get(f"http://{ip}:19200/health", timeout=10).json()
        (output_dir / "gateway_health.json").write_text(json.dumps(health, indent=2), encoding="utf-8")
        log(
            f"gateway require_all={health.get('require_all')} "
            f"max_inflight={health.get('max_inflight_requests')} groups={health.get('bounded_groups')}"
        )
    except Exception as exc:
        log(f"gateway health json failed: {exc}")

    all_results: list[ProbeResult] = []
    direct_levels = [int(x) for x in args.direct_levels.split(",") if x.strip()]
    for name, port, model in DIRECT_ENDPOINTS:
        try:
            wait_for_http(f"http://{ip}:{port}/health", timeout_s=1800)
            levels = direct_levels
            if name == "qwen3_vl_32b":
                levels = [level for level in direct_levels if level <= 8]
            all_results.extend(
                run_ramp(
                    target=name,
                    base_url=f"http://{ip}:{port}/v1",
                    model=model,
                    api_key=args.local_api_key,
                    levels=levels,
                    per_level_multiplier=2,
                    timeout_s=240,
                    output_jsonl=output_jsonl,
                    stop_on_bad=True,
                )
            )
        except Exception as exc:
            log(f"direct probe failed target={name}: {type(exc).__name__}: {exc}")

    gateway_levels = [int(x) for x in args.gateway_levels.split(",") if x.strip()]
    try:
        all_results.extend(
            run_ramp(
                target="committee_gateway",
                base_url=f"http://{ip}:19200/v1",
                model=args.gateway_model,
                api_key=args.committee_api_key,
                levels=gateway_levels,
                per_level_multiplier=1,
                timeout_s=420,
                output_jsonl=output_jsonl,
                stop_on_bad=True,
            )
        )
    except Exception as exc:
        log(f"gateway probe failed: {type(exc).__name__}: {exc}")

    summarize(all_results, summary_path)
    log("probe complete")


if __name__ == "__main__":
    main()
