"""
CodeImageTool OCR-only 演示脚本

用途：
1) 验证基础 PIL 路径（basic_pil）
2) 验证 PaddleOCR helper（ocr_assist）
3) 导出 OCR 结构化结果（ocr_probe.json）便于人工核查
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import ray
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.tools.code_image_tool import CodeImageTool
from verl.tools.schemas import OpenAIFunctionToolSchema


GENERAL_OCR_MODEL_NAMES = {"paddleocr", "paddleocr_v5", "ocr"}


def build_tool_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "code_image_tool",
                "description": "OCR-only demo schema.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "description": {"type": "string"},
                        "image_index": {"type": "integer"},
                    },
                    "required": ["code", "description", "image_index"],
                },
            },
        }
    )


def build_tool_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "type": "native",
        "num_workers": 2,
        "rate_limit": 6,
        "timeout": 180,
        "max_code_length": 4096,
        "enable_global_rate_limit": True,
        "enable_external_model_functions": True,
        "external_call_mode": args.external_call_mode,
        "ocr_model_name": args.ocr_model_name,
        "external_services": {
            "paddleocr": {
                "base_url": args.ocr_serving_base_url,
                "request_timeout": 180,
                "default_file_type": 1,
                "visualize": args.save_vis == "all",
                "line_y_threshold": args.line_y_threshold,
            },
            "paddleocr_vl": {
                "serving_base_url": args.ocr_serving_base_url,
                "request_timeout": 180,
                "default_file_type": 1,
                "visualize": args.save_vis == "all",
                "restructure_pages": False,
            },
        },
    }


def build_ocr_request_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.ocr_model_name in GENERAL_OCR_MODEL_NAMES:
        return {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": True,
            "text_det_limit_side_len": args.text_det_limit_side_len,
            "text_det_limit_type": args.text_det_limit_type,
            "text_det_thresh": args.text_det_thresh,
            "text_det_box_thresh": args.text_det_box_thresh,
            "text_rec_score_thresh": args.text_rec_score_thresh,
            "visualize": args.save_vis == "all",
        }
    return {
        "visualize": args.save_vis == "all",
    }


def format_python_kwargs(kwargs: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in kwargs.items())


def demo_cases(args: argparse.Namespace) -> list[dict[str, str]]:
    basic_code = """
# 基础 PIL 路径
result = image.rotate(90, expand=True)
"""

    ocr_kwargs_expr = format_python_kwargs(build_ocr_request_kwargs(args))
    ocr_code = f"""
# OCR helper 调用
ocr = _call_ocr_assist({ocr_kwargs_expr})
print("OCR text preview:", ocr["text"][:400])

# 为了在保存图上看到成功痕迹，把文本摘要叠到图像顶部
from PIL import ImageDraw
result = ocr["image"]
"""

    return [
        {"name": "basic_pil", "description": "基础 PIL 验证", "code": basic_code},
        {"name": "ocr_assist", "description": "PaddleOCR 验证", "code": ocr_code},
    ]


def dump_ocr_probe(
    tool: CodeImageTool,
    args: argparse.Namespace,
    image_path: str,
    out_dir: Path,
):
    image = Image.open(image_path).convert("RGB")
    payload_kwargs = build_ocr_request_kwargs(args)
    payload = tool._call_external_model(
        tool.ocr_model_name,
        image,
        payload_kwargs,
    )
    save_path = out_dir / "ocr_probe.json"
    save_path.write_text(
        json.dumps(
            {
                "text": payload.get("text", ""),
                "meta": payload.get("meta", {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.save_raw_json:
        raw_json_path = out_dir / "ocr_probe_raw.json"
        raw_json = payload.get("meta", {}).get("ocr_result", {})
        raw_json_path.write_text(json.dumps(raw_json, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[info] ocr_probe_raw_saved={raw_json_path}")
    print(f"[info] ocr_probe_saved={save_path}")
    print(f"[info] ocr_probe_text_preview={str(payload.get('text', ''))[:300]}")


async def run(args: argparse.Namespace):
    ray.init(ignore_reinit_error=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tool = CodeImageTool(config=build_tool_config(args), tool_schema=build_tool_schema())
    instance_id, _ = await tool.create(create_kwargs={"image": [args.image]})
    ocr_case_success = False
    try:
        for idx, case in enumerate(demo_cases(args)):
            print(f"\n=== Run Case {idx}: {case['name']} ===")
            response, reward, metrics = await tool.execute(
                instance_id,
                {"code": case["code"], "description": case["description"], "image_index": 0},
            )
            print(f"reward={reward}")
            print(f"metrics={metrics}")
            if response.text:
                print(f"text={response.text[:500]}")
            if response.image:
                out_path = out_dir / f"{idx:02d}_{case['name']}.png"
                response.image[0].save(out_path)
                print(f"saved_image={out_path}")
            if case["name"] == "ocr_assist" and bool(metrics.get("success")):
                ocr_case_success = True

        # 再补一次直接 OCR 调用，导出结构化结果文件，方便核查
        if args.dump_ocr_json and ocr_case_success:
            dump_ocr_probe(tool, args, args.image, out_dir)
        elif args.dump_ocr_json and not ocr_case_success:
            print("[warn] skip ocr_probe dump because ocr_assist case failed.")
    finally:
        await tool.release(instance_id)
        ray.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CodeImageTool OCR-only demo.")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径或 URL。")
    parser.add_argument("--out-dir", type=str, default="outputs/code_image_tool_ocr_only")

    # OCR 服务配置
    parser.add_argument("--ocr-serving-base-url", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--external-call-mode", choices=["service"], default="service")
    parser.add_argument("--ocr-model-name", choices=["paddleocr", "paddleocr_v5", "ocr", "paddleocr_vl"], default="paddleocr")
    parser.add_argument("--text-det-limit-side-len", type=int, default=960, help="text_det_limit_side_len.")
    parser.add_argument("--text-det-limit-type", default="max", choices=["min", "max"], help="text_det_limit_type.")
    parser.add_argument("--text-det-thresh", type=float, default=0.4, help="text_det_thresh.")
    parser.add_argument("--text-det-box-thresh", type=float, default=0.7, help="text_det_box_thresh.")
    parser.add_argument("--text-rec-score-thresh", type=float, default=0.6, help="text_rec_score_thresh.")
    parser.add_argument("--line-y-threshold", type=float, default=0.6, help="Relative threshold for grouping boxes into lines.")
    parser.add_argument("--save-vis", choices=["none", "all"], default="none", help="Whether to request OCR visualization images.")
    parser.add_argument("--save-raw-json", action="store_true", help="Also save one raw OCR JSON payload for probe output.")

    parser.add_argument("--dump-ocr-json", action="store_true", help="导出 ocr_probe.json。")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
