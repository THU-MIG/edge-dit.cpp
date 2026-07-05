from __future__ import annotations

import os
import unittest
from pathlib import Path

from edge_dit import Engine


@unittest.skipUnless(
    os.environ.get("EDGE_DIT_RUN_INTEGRATION") == "1",
    "set EDGE_DIT_RUN_INTEGRATION=1 to run real native smoke tests",
)
class OptionalRealSmokeTests(unittest.TestCase):
    def test_generate_image_with_real_library_and_model(self) -> None:
        library_path = os.environ.get("EDGE_DIT_LIBRARY")
        model_path = os.environ.get("EDGE_DIT_MODEL_PATH")
        if not library_path:
            self.skipTest("EDGE_DIT_LIBRARY is required for integration smoke tests")
        if not model_path:
            self.skipTest("EDGE_DIT_MODEL_PATH is required for integration smoke tests")

        output_path = Path(
            os.environ.get(
                "EDGE_DIT_INTEGRATION_OUTPUT",
                "/tmp/edge_dit_python_integration_test.png",
            )
        )

        with Engine(
            model_path=model_path,
            backend=os.environ.get("EDGE_DIT_BACKEND", "cuda"),
            offload_params_to_cpu=True,
            keep_text_encoder_on_cpu=True,
            max_vram_gb=float(os.environ.get("EDGE_DIT_MAX_VRAM_GB", "8.0")),
            _library_path=library_path,
        ) as engine:
            images = engine.generate_image(
                prompt=os.environ.get("EDGE_DIT_PROMPT", "python integration smoke teapot"),
                width=int(os.environ.get("EDGE_DIT_WIDTH", "256")),
                height=int(os.environ.get("EDGE_DIT_HEIGHT", "256")),
                steps=int(os.environ.get("EDGE_DIT_STEPS", "1")),
                seed=int(os.environ.get("EDGE_DIT_SEED", "42")),
            )

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].size, (256, 256))
        images[0].save(output_path)
        self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
