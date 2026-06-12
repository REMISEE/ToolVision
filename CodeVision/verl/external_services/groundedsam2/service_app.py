import argparse
import threading
from typing import Any

from flask import Flask, jsonify, request

from .codec import base64_to_image, images_to_base64
from .runner import GroundedSAM2Runner


def build_runner_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "device": args.device,
        "default_text_prompt": args.default_text_prompt,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "sam2_checkpoint": args.sam2_checkpoint,
        "sam2_model_config": args.sam2_model_config,
        "grounding_dino_config": args.grounding_dino_config,
        "grounding_dino_checkpoint": args.grounding_dino_checkpoint,
    }


def create_app(config: dict[str, Any]) -> Flask:
    app = Flask(__name__)
    runner = GroundedSAM2Runner(config)
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
            app.logger.exception("GroundedSAM2 inference failed")
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
    parser = argparse.ArgumentParser(description="GroundedSAM2 HTTP service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--default-text-prompt", default="object.")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--sam2-checkpoint", default="../Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--sam2-model-config", default="../Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument(
        "--grounding-dino-config",
        default="../Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument(
        "--grounding-dino-checkpoint",
        default="../Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(build_runner_config(args))
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
