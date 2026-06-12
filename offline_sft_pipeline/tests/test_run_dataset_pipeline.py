from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from offline_sft_pipeline.core.models import Budget, RootImage, RootSample
from offline_sft_pipeline.scripts.run_dataset_pipeline import _resolve_root_sample_image_paths, budget_for_sample


class RunDatasetPipelinePathResolutionTest(unittest.TestCase):
    def test_resolves_dataset_prefixed_relative_path_from_parent_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_dir = root / "export_images" / "output" / "fsc147"
            existing_image = root / "export_images" / "output" / "fsc147" / "images" / "fsc147_7692.jpg"
            existing_image.parent.mkdir(parents=True, exist_ok=True)
            existing_image.write_bytes(b"test")

            payload = {
                "sample_id": "fsc147__train__7692",
                "images": [{"image_id": "img_0", "path": "fsc147/images/fsc147_7692.jpg"}],
            }

            resolved = _resolve_root_sample_image_paths(payload, base_dir=base_dir)

            self.assertEqual(resolved["images"][0]["path"], str(existing_image.resolve()))

    def test_high_conf_exact_match_datasets_use_three_step_budget(self) -> None:
        sample = RootSample(
            sample_id="gqa__train__demo",
            question="What is shown?",
            images=[RootImage(image_id="img0", path="/tmp/fake.png")],
            metadata={"source_dataset": "gqa"},
        )

        budget = budget_for_sample(sample, default_budget=Budget(remaining_exec_steps=6))

        self.assertEqual(budget.remaining_exec_steps, 3)


if __name__ == "__main__":
    unittest.main()
