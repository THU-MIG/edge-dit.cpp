from __future__ import annotations

import importlib.util
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from edge_dit.config import EngineConfig


REPO_ROOT = Path(__file__).resolve().parents[3]
CONSOLE_ROOT = REPO_ROOT / "bindings/python/frontend/server-console"
PROFILES_DIR = CONSOLE_ROOT / "runtime/profiles"

EXPECTED_DEFAULTS = {
    "flux-dev": ("image", 20, None),
    "flux-kontext": ("image", 20, None),
    "flux-schnell": ("image", 4, None),
    "flux2-klein-4b": ("image", 4, None),
    "kontext-lightning": ("image", 8, None),
    "minimax-h3": ("video", 20, 22),
    "qwen-image": ("image", 30, None),
    "qwen-image-edit": ("image", 30, None),
    "qwen-image-lightning": ("image", 4, None),
    "qwen-image-edit-lightning": ("image", 4, None),
    "sd3-medium": ("image", 20, None),
    "sd35-medium-turbo": ("image", 8, None),
    "wan-t2v": ("video", 30, 41),
    "wan2-t2v-14b": ("video", 50, 41),
    "wan21-t2v-1.3b-distill": ("video", 8, 41),
}


def load_managed_server_module():
    module_path = CONSOLE_ROOT / "runtime/managed_server.py"
    spec = importlib.util.spec_from_file_location("edge_dit_managed_server", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ManagedProfileTests(unittest.TestCase):
    def test_public_server_profile_matrix_matches_benchmark_defaults(self) -> None:
        profile_paths = sorted(PROFILES_DIR.glob("*.json"))
        self.assertEqual({path.stem for path in profile_paths}, set(EXPECTED_DEFAULTS))

        for path in profile_paths:
            with self.subTest(profile=path.stem):
                payload = json.loads(path.read_text(encoding="utf-8"))
                expected_kind, expected_steps, expected_frames = EXPECTED_DEFAULTS[path.stem]
                self.assertEqual(payload["slug"], path.stem)
                self.assertEqual(payload["kind"], expected_kind)
                self.assertTrue(payload["engine"]["auto_allocate"])
                self.assertEqual(payload["request_example"]["steps"], expected_steps)
                if expected_frames is not None:
                    self.assertEqual(payload["request_example"]["frames"], expected_frames)

        minimax = json.loads((PROFILES_DIR / "minimax-h3.json").read_text(encoding="utf-8"))
        frames = minimax["request_example"]["frames"]
        self.assertGreaterEqual(frames, 22)
        self.assertEqual((frames - 5) % 17, 0)

    def test_every_profile_resolves_to_a_valid_engine_config(self) -> None:
        managed_server = load_managed_server_module()
        fake_env = {
            name: f"/models/{name.lower().replace('_', '-')}"
            for name in {
                "EDGE_DIT_FLUX_MODEL_PATH",
                "EDGE_DIT_FLUX_SCHNELL_MODEL_PATH",
                "EDGE_DIT_FLUX_KONTEXT_MODEL_PATH",
                "EDGE_DIT_KONTEXT_LIGHTNING_DIT_PATH",
                "EDGE_DIT_FLUX2_KLEIN_4B_MODEL_PATH",
                "EDGE_DIT_QWEN_IMAGE_MODEL_PATH",
                "EDGE_DIT_QWEN_IMAGE_LIGHTNING_DIT_PATH",
                "EDGE_DIT_QWEN_IMAGE_EDIT_MODEL_PATH",
                "EDGE_DIT_QWEN_IMAGE_EDIT_LIGHTNING_DIT_PATH",
                "EDGE_DIT_SD3_MODEL_PATH",
                "EDGE_DIT_SD35_TURBO_MODEL_PATH",
                "EDGE_DIT_WAN_VIDEO_MODEL_PATH",
                "EDGE_DIT_WAN_14B_MODEL_PATH",
                "EDGE_DIT_WAN_DISTILL_DIT_PATH",
                "EDGE_DIT_MINIMAX_DIT_PATH",
                "EDGE_DIT_MINIMAX_LLM_PATH",
                "EDGE_DIT_MINIMAX_VIDEO_VAE_PATH",
                "EDGE_DIT_MINIMAX_AUDIO_VAE_PATH",
            }
        }

        with patch.dict(os.environ, fake_env, clear=False):
            for path in sorted(PROFILES_DIR.glob("*.json")):
                with self.subTest(profile=path.stem):
                    payload = managed_server._load_profile(path)
                    engine_payload = managed_server._resolve_engine_payload(payload)
                    config = EngineConfig(**engine_payload)
                    self.assertTrue(config.auto_allocate)


if __name__ == "__main__":
    unittest.main()
