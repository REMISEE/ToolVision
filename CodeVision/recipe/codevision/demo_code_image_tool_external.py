"""
CodeImageTool 外部模型功能演示脚本（含详细注释）

这个脚本用于本地快速验证：
1) 旧能力仍可用（纯 PIL 代码）
2) OCR helper: _call_ocr_assist
3) GroundedSAM2 helper:
   - _call_ground_box
   - _call_sam_mask
   - _call_dino_crop
   - _call_blur_bg
4) 兼容别名 _call_focus（等价于 _call_ground_box）

注意：
- 该脚本是“工具层验证”，不依赖训练流程。
- 如果外部依赖/权重不完整，相关 case 会报错，但基础 case 仍可运行。
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
    """构建最小工具 schema，便于直接实例化 CodeImageTool。"""
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "code_image_tool",
                "description": "Demo schema for local CodeImageTool verification.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to process image."},
                        "description": {"type": "string", "description": "Description."},
                        "image_index": {"type": "integer", "description": "Index of image."},
                    },
                    "required": ["code", "description", "image_index"],
                },
            },
        }
    )


def build_tool_config(args: argparse.Namespace) -> dict[str, Any]:
    """
    构建工具配置。
    这里直接复用你项目里约定的字段名，便于和 YAML 对齐。
    """
    return {
        "type": "native",
        "num_workers": 4,
        "rate_limit": 10,
        "timeout": 180,
        "max_code_length": 4096,
        "enable_global_rate_limit": True,
        "enable_external_model_functions": not args.disable_external,
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
            "grounded_sam2": {
                "base_url": args.grounded_sam2_base_url,
                "request_timeout": 180,
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


def demo_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    """
    定义 demo case。
    每个 case 都是“给 code_image_tool 的一段 code”。
    """
    # Case 0: 纯 PIL 旋转，验证兼容性（不依赖任何外部模型）
    basic_code = """
# 兼容性 smoke test：纯 PIL 图像处理
result = image.rotate(90, expand=True)
"""

    # Case 1: OCR 辅助
    ocr_kwargs_expr = format_python_kwargs(build_ocr_request_kwargs(args))
    ocr_code = f"""
# OCR: 返回字典中包含 image/text/meta
ocr = _call_ocr_assist({ocr_kwargs_expr})
print("OCR text preview:", ocr["text"][:300])
result = ocr["image"]
"""

    # Case 2: 只画检测框（Grounding）
    box_code = f"""
# Grounded box: 在原图上画锚框
box_res = _call_ground_box({args.focus_prompt!r}, box_threshold={args.box_threshold}, text_threshold={args.text_threshold})
print(box_res["text"])
result = box_res["image"]
"""

    # Case 3: SAM2 分割高亮（mask 半透明）
    mask_code = f"""
# SAM2 mask: 高亮前景区域并保留锚框
mask_res = _call_sam_mask(
    {args.focus_prompt!r},
    box_threshold={args.box_threshold},
    text_threshold={args.text_threshold},
    multimask_output={str(args.multimask_output)},
    mask_alpha={args.mask_alpha},
)
print(mask_res["text"])
result = mask_res["image"]
"""

    # Case 4: 基于 box/mask 裁剪
    crop_code = f"""
# DINO crop: 返回裁剪图（默认取 detection_index=0）
crop_res = _call_dino_crop(
    {args.focus_prompt!r},
    based_on={args.crop_based_on!r},
    detection_index={args.crop_detection_index},
    max_crops={args.crop_max_crops},
    padding={args.crop_padding},
    multimask_output={str(args.multimask_output)},
)
print(crop_res["text"])
result = crop_res["image"]
"""

    # Case 5: 背景模糊
    blur_bg_code = f"""
# Blur background: 前景保留，背景高斯模糊
blur_res = _call_blur_bg(
    {args.focus_prompt!r},
    blur_radius={args.blur_radius},
    box_threshold={args.box_threshold},
    text_threshold={args.text_threshold},
    multimask_output={str(args.multimask_output)},
)
print(blur_res["text"])
result = blur_res["image"]
"""

    # Case 6: 兼容旧别名 _call_focus（等价于 _call_ground_box）
    focus_alias_code = f"""
# Backward compatibility: _call_focus 仍可用
focus = _call_focus({args.focus_prompt!r}, box_threshold={args.box_threshold}, text_threshold={args.text_threshold})
print(focus["text"])
result = focus["image"]
"""

    return [
        {"name": "basic_pil", "description": "纯 PIL 兼容测试", "code": basic_code},
        {"name": "ocr_assist", "description": "PaddleOCR 辅助", "code": ocr_code},
        {"name": "ground_box", "description": "Grounding 框可视化", "code": box_code},
        {"name": "sam_mask", "description": "SAM2 掩码高亮", "code": mask_code},
        {"name": "dino_crop", "description": "基于 box/mask 裁剪", "code": crop_code},
        {"name": "blur_bg", "description": "背景模糊", "code": blur_bg_code},
        {"name": "focus_alias", "description": "_call_focus 兼容别名", "code": focus_alias_code},
    ]


def select_cases(cases: list[dict[str, Any]], cases_arg: str) -> list[dict[str, Any]]:
    raw = str(cases_arg or "all").strip().lower()
    if raw in {"all", "*"}:
        return cases
    wanted = [x.strip() for x in raw.split(",") if x.strip()]
    by_name = {c["name"]: c for c in cases}
    unknown = [x for x in wanted if x not in by_name]
    if unknown:
        raise ValueError(f"Unknown case(s): {unknown}. Available: {list(by_name.keys())}")
    return [by_name[x] for x in wanted]


def dump_ocr_probe(tool: CodeImageTool, args: argparse.Namespace, image_path: str, out_dir: Path):
    """Directly call OCR service helper and dump structured payload for verification."""
    image = Image.open(image_path).convert("RGB")
    payload = tool._call_external_model(tool.ocr_model_name, image, build_ocr_request_kwargs(args))
    dump_path = out_dir / "ocr_probe.json"
    dump_path.write_text(
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
    print(f"[info] ocr_probe_saved={dump_path}")
    print(f"[info] ocr_probe_text_preview={str(payload.get('text', ''))[:300]}")


async def run(args: argparse.Namespace):
    """
    核心执行流程：
    1) 初始化 Ray
    2) 创建 CodeImageTool
    3) 对同一张输入图依次执行 demo case
    4) 保存每个 case 返回图
    """
    ray.init(ignore_reinit_error=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tool = CodeImageTool(config=build_tool_config(args), tool_schema=build_tool_schema())
    instance_id, _ = await tool.create(create_kwargs={"image": [args.image]})

    cases_to_run = select_cases(demo_cases(args), args.cases)

    try:
        for idx, case in enumerate(cases_to_run):
            print(f"\n=== Run Case {idx}: {case['name']} ===")
            response, reward, metrics = await tool.execute(
                instance_id,
                {
                    "code": case["code"],
                    "description": case["description"],
                    "image_index": 0,
                },
            )
            print(f"reward={reward}")
            print(f"metrics={metrics}")
            if response.text:
                print(f"text={response.text[:500]}")
            if response.image:
                save_path = out_dir / f"{idx:02d}_{case['name']}.png"
                response.image[0].save(save_path)
                print(f"saved_image={save_path}")
            if case["name"] == "ocr_assist" and bool(metrics.get("success")) and args.dump_ocr_probe:
                dump_ocr_probe(tool, args, args.image, out_dir)
    finally:
        await tool.release(instance_id)
        ray.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CodeImageTool external model helper demo.")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径或 URL。")
    parser.add_argument("--out-dir", type=str, default="outputs/code_image_tool_demo")
    parser.add_argument("--disable-external", action="store_true", help="禁用外部模型 helper，只测基础路径。")
    parser.add_argument(
        "--cases",
        type=str,
        default="all",
        help="选择运行的 case：all 或逗号分隔列表（如 ground_box,sam_mask）。",
    )
    parser.add_argument(
        "--dump-ocr-probe",
        action="store_true",
        help="当 ocr_assist 成功时，额外导出 ocr_probe.json。",
    )
    parser.add_argument("--external-call-mode", type=str, default="service", choices=["service"])

    # OCR server 参数
    parser.add_argument("--ocr-serving-base-url", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--ocr-model-name", choices=["paddleocr", "paddleocr_v5", "ocr", "paddleocr_vl"], default="paddleocr")
    parser.add_argument("--text-det-limit-side-len", type=int, default=960, help="text_det_limit_side_len.")
    parser.add_argument("--text-det-limit-type", default="max", choices=["min", "max"], help="text_det_limit_type.")
    parser.add_argument("--text-det-thresh", type=float, default=0.4, help="text_det_thresh.")
    parser.add_argument("--text-det-box-thresh", type=float, default=0.7, help="text_det_box_thresh.")
    parser.add_argument("--text-rec-score-thresh", type=float, default=0.6, help="text_rec_score_thresh.")
    parser.add_argument("--line-y-threshold", type=float, default=0.6, help="Relative threshold for grouping boxes into lines.")
    parser.add_argument("--save-vis", choices=["none", "all"], default="none", help="Whether to request OCR visualization images.")
    parser.add_argument("--save-raw-json", action="store_true", help="Also save one raw OCR JSON payload for probe output.")
    parser.add_argument("--grounded-sam2-base-url", type=str, default="http://127.0.0.1:8081")

    # GroundedSAM2 参数
    parser.add_argument("--focus-prompt", type=str, default="car. tire.")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--multimask-output", action="store_true", help="开启 SAM2 多 mask 输出并自动取最佳 mask。")
    parser.add_argument("--mask-alpha", type=float, default=0.45)
    parser.add_argument("--blur-radius", type=float, default=8.0)

    # dino_crop 参数
    parser.add_argument("--crop-based-on", type=str, default="box", choices=["box", "mask"])
    parser.add_argument("--crop-detection-index", type=int, default=0)
    parser.add_argument("--crop-max-crops", type=int, default=1)
    parser.add_argument("--crop-padding", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
