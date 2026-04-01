from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offline_sft_pipeline.runtime.code_image_runtime_wrapper import (
    CodeImageRuntimeWrapper,
    build_default_code_image_tool_config,
)
from offline_sft_pipeline.runtime.types import ArtifactRef, RuntimeStepRequest


def build_demo_image(path: Path) -> None:
    image = Image.new("RGB", (160, 96), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, 148, 84), outline="black", width=2)
    draw.text((20, 36), "runtime smoke", fill="black")
    image.save(path)


def build_executor_code(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                'print("runtime smoke stdout")',
                "result = image.rotate(90, expand=True)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one single-step runtime wrapper smoke test.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="offline_sft_pipeline/outputs/runtime_smoke",
        help="Directory to place smoke files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    step_dir = output_dir / "steps" / "step_001"
    artifacts_dir = output_dir / "artifacts"
    step_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    input_image_path = artifacts_dir / "root_0.png"
    executor_code_path = step_dir / "executor_code.py"
    executor_cot_path = step_dir / "executor_cot.md"

    build_demo_image(input_image_path)
    build_executor_code(executor_code_path)
    executor_cot_path.write_text("Smoke test thought placeholder.\n", encoding="utf-8")

    request = RuntimeStepRequest(
        sample_id="sample_runtime_smoke",
        trajectory_id="traj_runtime_smoke",
        round_idx=0,
        step_idx=1,
        executor_cot_path=str(executor_cot_path),
        executor_code_path=str(executor_code_path),
        visible_images=[ArtifactRef(artifact_id="img_root_0", path=str(input_image_path))],
        image_index=0,
        step_output_dir=str(step_dir),
    )

    wrapper = CodeImageRuntimeWrapper(
        build_default_code_image_tool_config(enable_external_model_functions=False),
    )
    output = wrapper.run_step_sync(request)
    wrapper.close_sync()

    print(json.dumps(output.runtime_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
