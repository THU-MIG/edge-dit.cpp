from __future__ import annotations

import ctypes
import unittest
from unittest.mock import patch

from edge_dit import EdgeDitClosedError, Engine, GenerationError, ImageRequest, ModelLoadError
from edge_dit._capi import EdContext, EdImage, EdImageBatch, EdImageGenerationParams


class FakeLib:
    def __init__(
        self,
        *,
        create_ok: bool = True,
        root: bool = True,
        generate_status: int = 0,
        last_error: bytes = b"native error",
    ) -> None:
        self.create_ok = create_ok
        self.root = root
        self.generate_status = generate_status
        self.last_error = last_error
        self.create_calls = 0
        self.free_context_calls = 0
        self.free_batch_calls = 0
        self.generated_prompts: list[str] = []
        self._ctx = ctypes.pointer(EdContext()) if create_ok else None
        self._keepalive: list[object] = []

    def ed_context_params_init(self, params) -> None:
        return None

    def ed_image_generation_params_init(self, params) -> None:
        return None

    def ed_create_context(self, params):
        self.create_calls += 1
        return self._ctx

    def ed_free_context(self, ctx) -> None:
        self.free_context_calls += 1

    def ed_generate_image(self, ctx, params, out) -> int:
        request = ctypes.cast(params, ctypes.POINTER(EdImageGenerationParams)).contents
        self.generated_prompts.append(request.prompt.decode("utf-8"))
        if self.generate_status != 0:
            return self.generate_status

        raw = (ctypes.c_uint8 * 3)(255, 0, 0)
        image = EdImage(
            width=1,
            height=1,
            channels=3,
            data=ctypes.cast(raw, ctypes.POINTER(ctypes.c_uint8)),
        )
        images = (EdImage * 1)(image)
        self._keepalive.extend([raw, images])

        batch = ctypes.cast(out, ctypes.POINTER(EdImageBatch)).contents
        batch.images = ctypes.cast(images, ctypes.POINTER(EdImage))
        batch.count = 1
        return 0

    def ed_free_image_batch(self, batch) -> None:
        self.free_batch_calls += 1
        native_batch = ctypes.cast(batch, ctypes.POINTER(EdImageBatch)).contents
        native_batch.images = ctypes.POINTER(EdImage)()
        native_batch.count = 0

    def ed_get_last_error(self, ctx):
        return self.last_error

    def ed_context_parallel_is_root(self, ctx) -> bool:
        return self.root


class EngineLifecycleTests(unittest.TestCase):
    def test_engine_close_is_idempotent(self) -> None:
        fake = FakeLib()
        with patch("edge_dit.engine.load_capi", return_value=fake):
            engine = Engine(model_path="demo-model")
            engine.close()
            engine.close()
        self.assertEqual(fake.free_context_calls, 1)

    def test_generate_image_returns_pil_images(self) -> None:
        fake = FakeLib()
        with patch("edge_dit.engine.load_capi", return_value=fake):
            with Engine(model_path="demo-model") as engine:
                images = engine.generate_image(ImageRequest(prompt="teapot"))
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].size, (1, 1))
        self.assertEqual(fake.generated_prompts, ["teapot"])
        self.assertEqual(fake.free_batch_calls, 1)

    def test_generate_image_supports_numpy_output(self) -> None:
        fake = FakeLib()
        with patch("edge_dit.engine.load_capi", return_value=fake):
            with Engine(model_path="demo-model") as engine:
                images = engine.generate_image(ImageRequest(prompt="teapot", output_type="numpy"))
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].shape, (1, 1, 3))
        self.assertEqual(images[0].tolist(), [[[255, 0, 0]]])

    def test_cache_fields_are_forwarded_to_native_request(self) -> None:
        fake = FakeLib()
        captured: dict[str, object] = {}

        original_generate = fake.ed_generate_image

        def wrapped_generate(ctx, params, out) -> int:
            request = ctypes.cast(params, ctypes.POINTER(EdImageGenerationParams)).contents
            captured["cache_mode"] = request.sample.cache_mode
            captured["cache_reuse_threshold"] = request.sample.cache_reuse_threshold
            captured["cache_start_percent"] = request.sample.cache_start_percent
            captured["cache_end_percent"] = request.sample.cache_end_percent
            captured["cache_error_decay_rate"] = request.sample.cache_error_decay_rate
            captured["cache_use_relative_threshold"] = request.sample.cache_use_relative_threshold
            captured["cache_reset_error_on_compute"] = request.sample.cache_reset_error_on_compute
            captured["cache_Fn_compute_blocks"] = request.sample.cache_Fn_compute_blocks
            captured["cache_Bn_compute_blocks"] = request.sample.cache_Bn_compute_blocks
            captured["cache_residual_diff_threshold"] = request.sample.cache_residual_diff_threshold
            captured["cache_max_accumulated_residual_diff"] = (
                request.sample.cache_max_accumulated_residual_diff
            )
            captured["cache_max_warmup_steps"] = request.sample.cache_max_warmup_steps
            captured["cache_max_cached_steps"] = request.sample.cache_max_cached_steps
            captured["cache_max_continuous_cached_steps"] = (
                request.sample.cache_max_continuous_cached_steps
            )
            captured["cache_taylorseer_n_derivatives"] = request.sample.cache_taylorseer_n_derivatives
            captured["cache_taylorseer_skip_interval"] = request.sample.cache_taylorseer_skip_interval
            captured["cache_scm_mask"] = (
                request.sample.cache_scm_mask.decode("utf-8")
                if request.sample.cache_scm_mask
                else None
            )
            captured["cache_scm_policy_dynamic"] = request.sample.cache_scm_policy_dynamic
            return original_generate(ctx, params, out)

        fake.ed_generate_image = wrapped_generate

        with patch("edge_dit.engine.load_capi", return_value=fake):
            with Engine(model_path="demo-model") as engine:
                engine.generate_image(
                    ImageRequest(
                        prompt="teapot",
                        cache_mode="cache-dit",
                        cache_reuse_threshold=0.8,
                        cache_start_percent=0.2,
                        cache_end_percent=0.9,
                        cache_error_decay_rate=0.5,
                        cache_use_relative_threshold=False,
                        cache_reset_error_on_compute=False,
                        cache_Fn_compute_blocks=12,
                        cache_Bn_compute_blocks=3,
                        cache_residual_diff_threshold=0.07,
                        cache_max_accumulated_residual_diff=1.5,
                        cache_max_warmup_steps=6,
                        cache_max_cached_steps=20,
                        cache_max_continuous_cached_steps=4,
                        cache_taylorseer_n_derivatives=2,
                        cache_taylorseer_skip_interval=1,
                        cache_scm_mask="0011",
                        cache_scm_policy_dynamic=False,
                    )
                )

        self.assertEqual(captured["cache_mode"], 5)
        self.assertAlmostEqual(captured["cache_reuse_threshold"], 0.8, places=6)
        self.assertAlmostEqual(captured["cache_start_percent"], 0.2, places=6)
        self.assertAlmostEqual(captured["cache_end_percent"], 0.9, places=6)
        self.assertAlmostEqual(captured["cache_error_decay_rate"], 0.5, places=6)
        self.assertFalse(captured["cache_use_relative_threshold"])
        self.assertFalse(captured["cache_reset_error_on_compute"])
        self.assertEqual(captured["cache_Fn_compute_blocks"], 12)
        self.assertEqual(captured["cache_Bn_compute_blocks"], 3)
        self.assertAlmostEqual(captured["cache_residual_diff_threshold"], 0.07, places=6)
        self.assertAlmostEqual(captured["cache_max_accumulated_residual_diff"], 1.5, places=6)
        self.assertEqual(captured["cache_max_warmup_steps"], 6)
        self.assertEqual(captured["cache_max_cached_steps"], 20)
        self.assertEqual(captured["cache_max_continuous_cached_steps"], 4)
        self.assertEqual(captured["cache_taylorseer_n_derivatives"], 2)
        self.assertEqual(captured["cache_taylorseer_skip_interval"], 1)
        self.assertEqual(captured["cache_scm_mask"], "0011")
        self.assertFalse(captured["cache_scm_policy_dynamic"])

    def test_generate_after_close_raises(self) -> None:
        fake = FakeLib()
        with patch("edge_dit.engine.load_capi", return_value=fake):
            engine = Engine(model_path="demo-model")
            engine.close()
            with self.assertRaises(EdgeDitClosedError):
                engine.generate_image(prompt="teapot")

    def test_create_context_failure_raises_model_load_error(self) -> None:
        fake = FakeLib(create_ok=False)
        with patch("edge_dit.engine.load_capi", return_value=fake):
            with self.assertRaises(ModelLoadError) as ctx:
                Engine(model_path="demo-model")
        self.assertIn("model_path=demo-model (missing)", str(ctx.exception))
        self.assertIn("verify libstdc++ / GLIBCXX compatibility", str(ctx.exception))

    def test_create_context_failure_includes_backend_context(self) -> None:
        fake = FakeLib(create_ok=False)
        with patch("edge_dit.engine.load_capi", return_value=fake):
            with self.assertRaises(ModelLoadError) as ctx:
                Engine(model_path="demo-model", backend="cuda", max_vram_gb=8.0)
        self.assertIn("backend='cuda'", str(ctx.exception))
        self.assertIn("max_vram_gb=8.0", str(ctx.exception))

    def test_non_root_rank_returns_empty_list(self) -> None:
        fake = FakeLib(root=False)
        with patch("edge_dit.engine.load_capi", return_value=fake):
            with Engine(model_path="demo-model") as engine:
                images = engine.generate_image(prompt="teapot")
        self.assertEqual(images, [])

    def test_generation_error_includes_request_context(self) -> None:
        fake = FakeLib(generate_status=4, last_error=b"image width and height must be positive")
        with patch("edge_dit.engine.load_capi", return_value=fake):
            with Engine(model_path="demo-model", backend="cuda") as engine:
                with self.assertRaises(GenerationError) as ctx:
                    engine.generate_image(
                        ImageRequest(
                            prompt="teapot",
                            width=256,
                            height=256,
                            steps=1,
                            output_type="numpy",
                        )
                    )
        message = str(ctx.exception)
        self.assertIn("image width and height must be positive", message)
        self.assertIn("backend='cuda'", message)
        self.assertIn("size=256x256", message)
        self.assertIn("steps=1", message)
        self.assertIn("output_type='numpy'", message)


if __name__ == "__main__":
    unittest.main()
