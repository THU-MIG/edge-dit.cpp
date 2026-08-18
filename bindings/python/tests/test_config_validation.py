from __future__ import annotations

import unittest

from PIL import Image

from edge_dit import AudioInput, EngineConfig, ImageRequest, InvalidArgumentError, RefVideoInput, VideoRequest


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

    def test_engine_config_accepts_minimax_component_form(self) -> None:
        config = EngineConfig(
            diffusion_model_path="dit.safetensors",
            vae_path="vae.safetensors",
            audio_vae_path="audio-vae.safetensors",
            llm_path="qwen.gguf",
            minimax_h3_stage_lifecycle=True,
        )
        self.assertEqual(config.llm_path, "qwen.gguf")

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

    def test_image_request_rejects_empty_ref_images(self) -> None:
        with self.assertRaises(InvalidArgumentError):
            ImageRequest(prompt="hello", ref_images=[])

    def test_image_request_rejects_non_image_init_image(self) -> None:
        with self.assertRaises(InvalidArgumentError):
            ImageRequest(prompt="hello", init_image="not-an-image")

    def test_image_request_accepts_input_images(self) -> None:
        request = ImageRequest(
            prompt="hello",
            init_image=Image.new("RGB", (4, 5)),
            ref_images=[Image.new("RGB", (6, 7))],
        )
        self.assertEqual(request.init_image.size, (4, 5))
        self.assertEqual(len(request.ref_images or []), 1)

    def test_video_request_requires_frames_to_be_positive(self) -> None:
        with self.assertRaises(InvalidArgumentError):
            VideoRequest(prompt="hello", frames=0)

    def test_video_request_rejects_unknown_output_type(self) -> None:
        with self.assertRaises(InvalidArgumentError):
            VideoRequest(prompt="hello", output_type="tensor")

    def test_video_request_accepts_minimax_reference_inputs(self) -> None:
        audio = AudioInput(samples=[0.0, 0.1], sample_rate=16000)
        request = VideoRequest(
            prompt="hello",
            init_image=Image.new("RGB", (4, 4)),
            end_image=Image.new("RGB", (4, 4)),
            ref_images=[Image.new("RGB", (4, 4))],
            ref_videos=[RefVideoInput([Image.new("RGB", (4, 4))], audio=audio)],
            ref_audios=[audio],
            ref_image_size="match",
        )
        self.assertEqual(request.ref_image_size, "match")


if __name__ == "__main__":
    unittest.main()
