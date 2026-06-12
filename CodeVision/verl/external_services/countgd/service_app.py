import argparse
import threading
from typing import Any

from flask import Flask, jsonify, request

from .codec import base64_to_image, images_to_base64
from .runner import CountGDRunner


def build_runner_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "countgd_root": args.countgd_root,
        "device": args.device,
        "config_path": args.config_path,
        "pretrain_model_path": args.pretrain_model_path,
        "text_encoder_type": args.text_encoder_type,
        "default_confidence_thresh": args.default_confidence_thresh,
        "default_visualize": args.default_visualize,
        "heatmap_sigma": args.heatmap_sigma,
    }


def create_app(config: dict[str, Any]) -> Flask:
    app = Flask(__name__)
    runner = CountGDRunner(config)
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
    parser = argparse.ArgumentParser(description="CountGD HTTP service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--countgd-root", default="CountGD")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config-path", default="config/cfg_fsc147_vit_b.py")
    parser.add_argument("--pretrain-model-path", default="checkpoints/checkpoint_fsc147_best.pth")
    parser.add_argument("--text-encoder-type", default="checkpoints/bert-base-uncased")
    parser.add_argument("--default-confidence-thresh", type=float, default=0.23)
    parser.add_argument("--default-visualize", default="heatmap")
    parser.add_argument("--heatmap-sigma", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(build_runner_config(args))
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
