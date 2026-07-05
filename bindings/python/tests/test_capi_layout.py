from __future__ import annotations

import unittest

from edge_dit import _capi


class CApiLayoutTests(unittest.TestCase):
    def test_context_params_contains_public_fields(self) -> None:
        field_names = [name for name, _ctype in _capi.EdContextParams._fields_]
        self.assertIn("model_path", field_names)
        self.assertIn("diffusion_model_path", field_names)
        self.assertIn("vae_path", field_names)
        self.assertIn("flash_attention", field_names)
        self.assertIn("cfg_parallel_size", field_names)

    def test_image_generation_params_contains_expected_fields(self) -> None:
        field_names = [name for name, _ctype in _capi.EdImageGenerationParams._fields_]
        self.assertEqual(field_names[0], "prompt")
        self.assertIn("negative_prompt", field_names)
        self.assertIn("sample", field_names)
        self.assertIn("loras", field_names)

    def test_sample_params_contains_cache_fields(self) -> None:
        field_names = [name for name, _ctype in _capi.EdSampleParams._fields_]
        self.assertIn("cache_mode", field_names)
        self.assertIn("cache_reuse_threshold", field_names)
        self.assertIn("cache_scm_mask", field_names)


if __name__ == "__main__":
    unittest.main()

