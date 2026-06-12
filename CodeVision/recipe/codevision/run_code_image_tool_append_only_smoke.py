from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import ray
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.tools.code_image_tool import CodeImageTool
from verl.tools.schemas import OpenAIFunctionToolSchema


def build_tool_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "code_image_tool",
                "description": "Smoke-test schema for append-only image timeline validation.",
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


def build_tool_config() -> dict[str, Any]:
    return {
        "num_workers": 1,
        "rate_limit": 4,
        "timeout": 30,
        "max_code_length": 2000,
        "enable_external_model_functions": False,
        "enable_global_rate_limit": False,
        "external_call_mode": "service",
        "external_services": {},
    }


async def run_smoke() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_path = root / "root.png"
        ray_tmp = root / "ray"
        Image.new("RGB", (8, 6), color=(255, 0, 0)).save(image_path)

        ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=1, _temp_dir=str(ray_tmp))
        tool = CodeImageTool(config=build_tool_config(), tool_schema=build_tool_schema())
        instance_id, _ = await tool.create(create_kwargs={"image": [str(image_path)]})

        try:
            first_response, first_reward, first_metrics = await tool.execute(
                instance_id,
                {
                    "code": "result = image.transpose(Image.FLIP_LEFT_RIGHT)",
                    "description": "Flip the root image.",
                    "image_index": 0,
                },
            )
            images_after_first = len(tool._instance_dict[instance_id]["images"])

            second_response, second_reward, second_metrics = await tool.execute(
                instance_id,
                {
                    "code": "result = image.rotate(90, expand=True)",
                    "description": "Rotate the appended image.",
                    "image_index": 1,
                },
            )
            images_after_second = len(tool._instance_dict[instance_id]["images"])

            return {
                "first_success": bool(first_metrics.get("success")),
                "first_reward": float(first_reward),
                "first_input_image_index": first_metrics.get("input_image_index"),
                "first_output_image_index": first_metrics.get("output_image_index"),
                "images_after_first": images_after_first,
                "second_success": bool(second_metrics.get("success")),
                "second_reward": float(second_reward),
                "second_input_image_index": second_metrics.get("input_image_index"),
                "second_output_image_index": second_metrics.get("output_image_index"),
                "images_after_second": images_after_second,
                "second_result_size": list(second_response.image[0].size) if second_response.image else None,
                "first_response_has_image": bool(first_response.image),
                "second_response_has_image": bool(second_response.image),
            }
        finally:
            await tool.release(instance_id)
            ray.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test that CodeImageTool appends each successful output into the per-instance image timeline."
    )
    parser.parse_args()
    payload = asyncio.run(run_smoke())
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
