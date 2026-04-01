from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offline_sft_pipeline.runtime.code_image_runtime_wrapper import (
    CodeImageRuntimeWrapper,
    build_default_code_image_tool_config,
)
from offline_sft_pipeline.runtime.types import ArtifactRef, RuntimeStepRequest

DEFAULT_OUTPUT_DIR = REPO_ROOT / "offline_sft_pipeline" / "outputs" / "runtime_helper_smoke"
DEFAULT_INPUT_IMAGE = REPO_ROOT / "CodeVision" / "tmp_demo_input.png"
DEFAULT_CASES = ("ocr_only", "sam_mask", "mask_crop_then_ocr")
GENERAL_OCR_MODEL_NAMES = {"paddleocr", "paddleocr_v5", "ocr"}


def build_fallback_demo_image(path: Path) -> None:
    image = Image.new("RGB", (640, 360), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((250, 150, 390, 210), outline="black", width=3)
    draw.text((270, 175), "year's", fill="black")
    image.save(path)


def prepare_input_image(output_path: Path, source_path: Path | None) -> Path:
    if source_path is None:
        source_path = DEFAULT_INPUT_IMAGE if DEFAULT_INPUT_IMAGE.exists() else None

    if source_path is not None:
        source_path = source_path.expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"input image not found: {source_path}")
        with Image.open(source_path) as image:
            image.convert("RGB").save(output_path, format="PNG")
        return output_path

    build_fallback_demo_image(output_path)
    return output_path


def build_ocr_request_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.ocr_model_name in GENERAL_OCR_MODEL_NAMES:
        return {
            "use_doc_orientation_classify": args.use_doc_orientation_classify,
            "use_doc_unwarping": args.use_doc_unwarping,
            "use_textline_orientation": args.use_textline_orientation,
            "text_det_limit_side_len": args.text_det_limit_side_len,
            "text_det_limit_type": args.text_det_limit_type,
            "text_det_thresh": args.text_det_thresh,
            "text_det_box_thresh": args.text_det_box_thresh,
            "text_rec_score_thresh": args.text_rec_score_thresh,
            "return_word_box": args.return_word_box,
            "visualize": args.ocr_visualize,
        }
    return {
        "visualize": args.ocr_visualize,
    }


def format_python_kwargs(kwargs: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in kwargs.items())


def build_case_spec(case_name: str, args: argparse.Namespace) -> dict[str, str]:
    prompt = args.focus_prompt
    crop_padding = int(args.crop_padding)
    crop_based_on = args.crop_based_on
    ocr_kwargs_expr = format_python_kwargs(build_ocr_request_kwargs(args))
    ocr_call = f"_call_ocr_assist({ocr_kwargs_expr})"
    crop_ocr_call = f"_call_ocr_assist(image_obj=crop[\"image\"], {ocr_kwargs_expr})"

    if case_name == "ocr_only":
        return {
            "description": "Call OCR helper on the visible image and return the OCR result image.",
            "cot": "Use OCR helper on the current image and inspect whether wrapper metadata is preserved.\n",
            "code": "\n".join(
                [
                    f"ocr = {ocr_call}",
                    'print("ocr_text:", ocr["text"])',
                    'print("ocr_meta_keys:", list((ocr.get("meta") or {}).keys()))',
                    'result = ocr["image"]',
                ]
            )
            + "\n",
        }

    if case_name == "sam_mask":
        return {
            "description": "Call GroundedSAM2 mask helper on the target sign and return the highlighted mask image.",
            "cot": "Use SAM mask helper on the sign-like target so grounded_sam2 text/meta/annotations propagate through runtime_result.\n",
            "code": "\n".join(
                [
                    f"mask = _call_sam_mask({prompt!r}, multimask_output=False)",
                    'print("mask_text:", mask["text"])',
                    'print("mask_meta:", mask.get("meta"))',
                    'result = mask["image"]',
                ]
            )
            + "\n",
        }

    if case_name == "mask_crop_then_ocr":
        return {
            "description": "Crop the sign region with GroundedSAM2 and then run OCR on the crop.",
            "cot": "First crop the sign target with grounded_sam2, then OCR the crop and inspect helper trace ordering.\n",
            "code": "\n".join(
                [
                    f"crop = _call_dino_crop({prompt!r}, based_on={crop_based_on!r}, max_crops=1, padding={crop_padding})",
                    'print("crop_text:", crop["text"])',
                    f"ocr = {crop_ocr_call}",
                    'print("crop_ocr_text:", ocr["text"])',
                    'result = crop["image"]',
                ]
            )
            + "\n",
        }

    raise ValueError(f"Unknown case: {case_name}")


def parse_cases(raw_cases: str) -> list[str]:
    normalized = str(raw_cases or "all").strip().lower()
    if normalized in {"all", "*"}:
        return list(DEFAULT_CASES)

    cases = [item.strip() for item in normalized.split(",") if item.strip()]
    unknown = [item for item in cases if item not in DEFAULT_CASES]
    if unknown:
        raise ValueError(f"Unknown case(s): {unknown}. Available: {list(DEFAULT_CASES)}")
    return cases


def build_external_services_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "paddleocr": {
            "base_url": args.ocr_base_url,
            "request_timeout": args.service_timeout,
            "default_file_type": 1,
            "visualize": args.ocr_visualize,
            "line_y_threshold": args.line_y_threshold,
        },
        "grounded_sam2": {
            "base_url": args.grounded_sam2_base_url,
            "request_timeout": args.service_timeout,
        },
    }


def build_summary_entry(case_name: str, step_idx: int, runtime_result: dict[str, Any], runtime_result_path: str | None) -> dict[str, Any]:
    images = runtime_result.get("images") or []
    image_paths = [str(item.get("path")) for item in images if isinstance(item, dict) and item.get("path")]
    text = str(runtime_result.get("text") or "")
    error = runtime_result.get("error")
    return {
        "case": case_name,
        "step_idx": step_idx,
        "runtime_result_path": runtime_result_path,
        "success": bool(runtime_result.get("success")),
        "observed_helper_call_count": int(runtime_result.get("observed_helper_call_count") or 0),
        "image_paths": image_paths,
        "text_preview": text[:300],
        "error": error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run runtime-wrapper smoke tests that exercise OCR and GroundedSAM2 helpers."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=str(DEFAULT_INPUT_IMAGE),
        help="Input image path. Defaults to CodeVision/tmp_demo_input.png.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to place smoke outputs.",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="all",
        help="Which cases to run: all or comma-separated list from ocr_only,sam_mask,mask_crop_then_ocr.",
    )
    parser.add_argument("--sample-id", type=str, default="sample_runtime_helper_smoke")
    parser.add_argument("--trajectory-id", type=str, default="traj_runtime_helper_smoke")
    parser.add_argument("--round-idx", type=int, default=0)
    parser.add_argument("--start-step-idx", type=int, default=1)
    parser.add_argument("--ocr-base-url", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--grounded-sam2-base-url", type=str, default="http://127.0.0.1:8081")
    parser.add_argument("--ocr-model-name", type=str, default="paddleocr")
    parser.add_argument("--service-timeout", type=int, default=180)
    parser.add_argument("--line-y-threshold", type=float, default=0.6)
    parser.add_argument(
        "--use-doc-orientation-classify",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass use_doc_orientation_classify to OCR helper.",
    )
    parser.add_argument(
        "--use-doc-unwarping",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass use_doc_unwarping to OCR helper.",
    )
    parser.add_argument(
        "--use-textline-orientation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass use_textline_orientation to OCR helper.",
    )
    parser.add_argument("--text-det-limit-side-len", type=int, default=960)
    parser.add_argument("--text-det-limit-type", type=str, default="max", choices=["min", "max"])
    parser.add_argument("--text-det-thresh", type=float, default=0.4)
    parser.add_argument("--text-det-box-thresh", type=float, default=0.7)
    parser.add_argument("--text-rec-score-thresh", type=float, default=0.6)
    parser.add_argument(
        "--return-word-box",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass return_word_box to OCR helper.",
    )
    parser.add_argument("--focus-prompt", type=str, default="sign.")
    parser.add_argument("--crop-based-on", type=str, default="box", choices=["box", "mask"])
    parser.add_argument("--crop-padding", type=int, default=4)
    parser.add_argument(
        "--ocr-visualize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to request OCR visualization images from the OCR service.",
    )
    parser.add_argument(
        "--print-tool-schema",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the wrapper tool schema before running steps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = parse_cases(args.cases)

    output_dir = Path(args.output_dir).expanduser().resolve()
    artifacts_dir = output_dir / "artifacts"
    steps_dir = output_dir / "steps"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    steps_dir.mkdir(parents=True, exist_ok=True)

    input_image_path = prepare_input_image(
        artifacts_dir / "root_0.png",
        Path(args.image).expanduser() if args.image else None,
    )

    wrapper = CodeImageRuntimeWrapper(
        build_default_code_image_tool_config(
            enable_external_model_functions=True,
            external_services=build_external_services_config(args),
            ocr_model_name=args.ocr_model_name,
        )
    )

    if args.print_tool_schema:
        print(json.dumps(wrapper.tool_schema.model_dump(mode="json"), ensure_ascii=False, indent=2))

    results: list[dict[str, Any]] = []
    try:
        for index, case_name in enumerate(cases, start=0):
            step_idx = args.start_step_idx + index
            step_dir = steps_dir / f"step_{step_idx:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)

            executor_code_path = step_dir / "executor_code.py"
            executor_cot_path = step_dir / "executor_cot.md"
            case_spec = build_case_spec(case_name, args)
            executor_code_path.write_text(case_spec["code"], encoding="utf-8")
            executor_cot_path.write_text(case_spec["cot"], encoding="utf-8")

            request = RuntimeStepRequest(
                sample_id=args.sample_id,
                trajectory_id=args.trajectory_id,
                round_idx=args.round_idx,
                step_idx=step_idx,
                executor_cot_path=str(executor_cot_path),
                executor_code_path=str(executor_code_path),
                visible_images=[ArtifactRef(artifact_id="img_root_0", path=str(input_image_path))],
                image_index=0,
                step_output_dir=str(step_dir),
            )

            output = wrapper.run_step_sync(request)
            results.append(
                build_summary_entry(
                    case_name=case_name,
                    step_idx=step_idx,
                    runtime_result=output.runtime_result,
                    runtime_result_path=output.runtime_result_path,
                )
            )
    finally:
        wrapper.close_sync()

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({"cases": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary_path": str(summary_path),
                "cases": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
