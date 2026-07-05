from __future__ import annotations

import unittest

from edge_dit import EngineConfig, ImageRequest, InvalidArgumentError


class ConfigValidationTests(unittest.TestCase):
    def test_engine_config_requires_model_or_components(self) -> None:
        with self.assertRaises(InvalidArgumentError):
            EngineConfig()

    def test_engine_config_accepts_component_form(self) -> None:
        config = EngineConfig(
            diffusion_model_path="diffusion.safetensors",
            vae_path="vae.safetensors",
            clip_l_path="clip_l.safetensors",
            t5xxl_path="t5xxl.safetensors",
        )
        self.assertEqual(config.vae_path, "vae.safetensors")

    def test_image_request_requires_prompt(self) -> None:
        with self.assertRaises(InvalidArgumentError):
            ImageRequest(prompt="")

    def test_image_request_accepts_batch_size_alias(self) -> None:
        request = ImageRequest.from_kwargs(prompt="hello", batch_size=2)
        self.assertEqual(request.batch_count, 2)

    def test_image_request_rejects_invalid_cache_window(self) -> None:
        with self.assertRaises(InvalidArgumentError):
            ImageRequest(prompt="hello", cache_start_percent=0.9, cache_end_percent=0.2)

    def test_image_request_rejects_invalid_cache_taylor_order(self) -> None:
        with self.assertRaises(InvalidArgumentError):
            ImageRequest(prompt="hello", cache_taylorseer_n_derivatives=0)

    def test_image_request_rejects_unknown_output_type(self) -> None:
        with self.assertRaises(InvalidArgumentError):
            ImageRequest(prompt="hello", output_type="tensor")


if __name__ == "__main__":
    unittest.main()
