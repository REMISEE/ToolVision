#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import struct
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path


def build_png_base64(width: int = 32, height: int = 32, rgb: tuple[int, int, int] = (255, 255, 255)) -> str:
    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + chunk_type
            + data
            + struct.pack("!I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    row = b"\x00" + bytes(rgb) * width
    raw = row * height
    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return base64.b64encode(png).decode("ascii")


def image_file_base64(path: str) -> str | None:
    image_path = Path(path)
    if not image_path.is_file():
        return None
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def post_json(url: str, payload: dict[str, object], timeout_s: float) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {raw[:1000]}") from exc
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected response from {url}: {raw[:500]}")
    return data


def wait_for_http(url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=min(3.0, timeout_s)):
                return
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for service health {url}: {last_error}")


def warmup_ocr(*, host: str, port: int, timeout_s: float, image_b64: str) -> None:
    payload = {
        "file": image_b64,
        "fileType": 1,
        "visualize": False,
    }
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            data = post_json(f"http://{host}:{port}/ocr", payload, min(30.0, timeout_s))
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            time.sleep(1.0)
    else:
        raise RuntimeError(f"Timed out waiting for OCR service: {last_error}") from last_error

    if int(data.get("errorCode", -1)) != 0:
        raise RuntimeError(f"OCR warmup failed: {data}")


def warmup_groundedsam2(*, host: str, port: int, timeout_s: float, image_b64: str) -> None:
    wait_for_http(f"http://{host}:{port}/healthy", timeout_s)
    payload = {
        "file": image_b64,
        "file_type": 1,
        "operation": "box",
        "text_prompt": "object.",
        "box_threshold": 0.35,
        "text_threshold": 0.25,
    }
    data = post_json(f"http://{host}:{port}/infer", payload, timeout_s)
    if int(data.get("errorCode", -1)) != 0:
        raise RuntimeError(f"GroundedSAM2 warmup failed: {data}")


def warmup_depth(
    *,
    host: str,
    port: int,
    timeout_s: float,
    image_b64: str,
    groundedsam2_host: str | None = None,
    groundedsam2_port: int | None = None,
    groundedsam2_image_b64: str | None = None,
) -> None:
    wait_for_http(f"http://{host}:{port}/healthy", timeout_s)
    payload = {
        "file": image_b64,
        "file_type": 1,
        "operation": "estimate",
        "vis_mode": "overlay",
        "stat": "median",
        "padding": 0,
    }
    data = post_json(f"http://{host}:{port}/infer", payload, timeout_s)
    if int(data.get("errorCode", -1)) != 0:
        raise RuntimeError(f"Depth warmup failed: {data}")
    if groundedsam2_host is None or groundedsam2_port is None:
        return
    ground_payload = {
        "file": groundedsam2_image_b64 or image_b64,
        "file_type": 1,
        "operation": "ground_depth",
        "text_prompt": "object.",
        "vis_mode": "overlay",
        "stat": "median",
        "padding": 0,
    }
    ground_data = post_json(f"http://{host}:{port}/infer", ground_payload, timeout_s)
    if int(ground_data.get("errorCode", -1)) != 0:
        raise RuntimeError(f"Depth ground_depth warmup failed: {ground_data}")


def warmup_countgd(*, host: str, port: int, timeout_s: float, image_b64: str) -> None:
    wait_for_http(f"http://{host}:{port}/healthy", timeout_s)
    payload = {
        "file": image_b64,
        "file_type": 1,
        "text_prompt": "object",
        "confidence_thresh": 0.23,
        "visualize": "heatmap",
    }
    data = post_json(f"http://{host}:{port}/infer", payload, timeout_s)
    if int(data.get("errorCode", -1)) != 0:
        raise RuntimeError(f"CountGD warmup failed: {data}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm up CodeVision external services by sending minimal inference requests.")
    parser.add_argument("target", choices=["ocr", "groundedsam2", "depth", "countgd", "all"])
    parser.add_argument("--ocr-host", default="127.0.0.1")
    parser.add_argument("--ocr-port", type=int, default=8080)
    parser.add_argument("--groundedsam2-host", default="127.0.0.1")
    parser.add_argument("--groundedsam2-port", type=int, default=8081)
    parser.add_argument("--depth-host", default="127.0.0.1")
    parser.add_argument("--depth-port", type=int, default=8082)
    parser.add_argument("--countgd-host", default="127.0.0.1")
    parser.add_argument("--countgd-port", type=int, default=8083)
    parser.add_argument(
        "--groundedsam2-image-path",
        default="/mnt/cpfs/delinmao/data/raw/ref_l4/images/objects365_v2_01058344.jpg",
        help="Optional real image for GroundedSAM2 warmup; falls back to a generated PNG if missing.",
    )
    parser.add_argument("--timeout-s", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_b64 = build_png_base64()
    groundedsam2_image_b64 = image_file_base64(args.groundedsam2_image_path) or image_b64

    tasks: list[tuple[str, callable]] = []
    if args.target in {"ocr", "all"}:
        tasks.append(
            (
                "ocr",
                lambda: warmup_ocr(
                    host=args.ocr_host,
                    port=args.ocr_port,
                    timeout_s=args.timeout_s,
                    image_b64=image_b64,
                ),
            )
        )
    if args.target in {"groundedsam2", "all"}:
        tasks.append(
            (
                "groundedsam2",
                lambda: warmup_groundedsam2(
                    host=args.groundedsam2_host,
                    port=args.groundedsam2_port,
                    timeout_s=args.timeout_s,
                    image_b64=groundedsam2_image_b64,
                ),
            )
        )
    if args.target in {"depth", "all"}:
        tasks.append(
            (
                "depth",
                lambda: warmup_depth(
                    host=args.depth_host,
                    port=args.depth_port,
                    timeout_s=args.timeout_s,
                    image_b64=image_b64,
                    groundedsam2_host=args.groundedsam2_host if args.target == "all" else None,
                    groundedsam2_port=args.groundedsam2_port if args.target == "all" else None,
                    groundedsam2_image_b64=groundedsam2_image_b64,
                ),
            )
        )
    if args.target in {"countgd", "all"}:
        tasks.append(
            (
                "countgd",
                lambda: warmup_countgd(
                    host=args.countgd_host,
                    port=args.countgd_port,
                    timeout_s=args.timeout_s,
                    image_b64=image_b64,
                ),
            )
        )

    for name, fn in tasks:
        started = time.time()
        fn()
        elapsed = time.time() - started
        print(f"[warmup] {name} ok in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
