import argparse
import threading
from typing import Any

from flask import Flask, jsonify, request

from .codec import base64_to_image, images_to_base64
from .runner import DepthProRunner


def build_runner_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "depth_pro_root": args.depth_pro_root,
        "checkpoint_path": args.checkpoint_path,
        "device": args.device,
        "cache_size": args.cache_size,
        "request_timeout": args.request_timeout,
        "default_text_prompt": args.default_text_prompt,
        "groundedsam2_base_url": args.groundedsam2_base_url,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
    }


def create_app(config: dict[str, Any]) -> Flask:
    app = Flask(__name__)
    runner = DepthProRunner(config)
    runner_lock = threading.Lock()

    @app.get("/healthy")
    def healthy():
        return jsonify({"status": "ok"})

    @app.post("/infer")
    def infer():
        payload = request.get_json(force=True, silent=False) or {}
        image_b64 = payload.get("file")
        if not isinstance(image_b64, str) or not image_b64:
            return jsonify({"errorCode": 400, "errorMsg": "file is required", "result": None}), 400

        try:
            image = base64_to_image(image_b64)
            kwargs = {k: v for k, v in payload.items() if k not in {"file", "file_type"}}
            with runner_lock:
                result = runner.infer(image, kwargs)
        except Exception as exc:
            return jsonify({"errorCode": 500, "errorMsg": str(exc), "result": None}), 500

        return jsonify(
            {
                "errorCode": 0,
                "errorMsg": "Success",
                "result": {
                    "images": images_to_base64(result.get("images", [])),
                    "text": result.get("text", ""),
                    "meta": result.get("meta", {}),
                },
            }
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth Pro HTTP service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--depth-pro-root", default="../ml-depth-pro-main")
    parser.add_argument("--checkpoint-path", default="checkpoints/depth_pro.pt")
    parser.add_argument("--device", default="")
    parser.add_argument("--cache-size", type=int, default=8)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--default-text-prompt", default="object.")
    parser.add_argument("--groundedsam2-base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(build_runner_config(args))
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
