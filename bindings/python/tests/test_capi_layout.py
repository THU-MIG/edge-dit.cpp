from __future__ import annotations

import unittest

from edge_dit import _capi


class CApiLayoutTests(unittest.TestCase):
    def test_context_params_contains_public_fields(self) -> None:
        field_names = [name for name, _ctype in _capi.EdContextParams._fields_]
        self.assertIn("model_path", field_names)
        self.assertIn("diffusion_model_path", field_names)
        self.assertIn("vae_path", field_names)
        self.assertEqual(field_names[field_names.index("vae_path") + 1], "audio_vae_path")
        self.assertEqual(
            field_names[field_names.index("text_encoder_offload") + 1],
            "minimax_h3_stage_lifecycle",
        )
        self.assertIn("flash_attention", field_names)
        self.assertIn("qwen_image_zero_cond_t", field_names)
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

    def test_video_generation_params_contains_expected_fields(self) -> None:
        field_names = [name for name, _ctype in _capi.EdVideoGenerationParams._fields_]
        self.assertEqual(field_names[0], "prompt")
        self.assertIn("frames", field_names)
        self.assertIn("end_image", field_names)
        self.assertIn("ref_images", field_names)
        self.assertIn("ref_videos", field_names)
        self.assertIn("ref_audios", field_names)
        self.assertIn("sample", field_names)
        self.assertIn("high_noise_sample", field_names)

    def test_video_output_contains_audio_fields(self) -> None:
        field_names = [name for name, _ctype in _capi.EdVideo._fields_]
        self.assertEqual(
            field_names,
            ["frames", "frame_count", "audio", "audio_sample_count", "audio_channels", "audio_sample_rate"],
        )


if __name__ == "__main__":
    unittest.main()
