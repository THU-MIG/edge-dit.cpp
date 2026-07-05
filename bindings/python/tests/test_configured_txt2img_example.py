from __future__ import annotations

import io
import runpy
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "configured_txt2img.py"
SCRIPT_GLOBALS = runpy.run_path(str(SCRIPT_PATH), run_name="__test__")

build_parser = SCRIPT_GLOBALS["build_parser"]
build_engine_config = SCRIPT_GLOBALS["build_engine_config"]
build_image_request = SCRIPT_GLOBALS["build_image_request"]
load_config_file = SCRIPT_GLOBALS["load_config_file"]
main = SCRIPT_GLOBALS["main"]
parse_args = SCRIPT_GLOBALS["parse_args"]
resolve_output_path = SCRIPT_GLOBALS["resolve_output_path"]

EXPECTED_HELP = textwrap.dedent(
    """\
    usage: configured_txt2img.py [-h] --config CONFIG [--output OUTPUT]

    Run edge-dit text-to-image from a JSON config file

    options:
      -h, --help       show this help message and exit
      --config CONFIG  Path to a JSON config file
      --output OUTPUT  Optional output path override
    """
)


class ConfiguredTxt2ImgExampleTests(unittest.TestCase):
    def test_help_output_snapshot(self) -> None:
        parser = build_parser()
        parser.prog = "configured_txt2img.py"
        self.assertEqual(parser.format_help(), EXPECTED_HELP)

    def test_parse_args_supports_output_override(self) -> None:
        args = parse_args(["--config", "demo.json", "--output", "image.png"])
        self.assertEqual(args.config, "demo.json")
        self.assertEqual(args.output, "image.png")

    def test_load_config_file_expands_environment_variables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                (
                    '{'
                    '"engine": {"model_path": "${EDGE_DIT_MODEL_PATH}", "backend": "cuda"},'
                    '"request": {"prompt": "teapot"}'
                    "}"
                ),
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"EDGE_DIT_MODEL_PATH": "/models/flux"}):
                payload = load_config_file(config_path)
        self.assertEqual(payload["engine"]["model_path"], "/models/flux")

    def test_build_helpers_map_json_payload(self) -> None:
        payload = {
            "engine": {
                "model_path": "/models/flux",
                "backend": "cuda",
                "offload_params_to_cpu": True,
                "keep_text_encoder_on_cpu": True,
                "max_vram_gb": 8.0,
            },
            "request": {
                "prompt": "teapot",
                "width": 256,
                "height": 256,
                "steps": 1,
                "seed": 42,
            },
            "output": "/tmp/out.png",
        }
        config = build_engine_config(payload)
        request = build_image_request(payload)
        output = resolve_output_path(payload)

        self.assertEqual(config.model_path, "/models/flux")
        self.assertEqual(config.backend, "cuda")
        self.assertTrue(config.offload_params_to_cpu)
        self.assertEqual(request.prompt, "teapot")
        self.assertEqual(request.width, 256)
        self.assertEqual(request.steps, 1)
        self.assertEqual(output, "/tmp/out.png")

    def test_output_override_takes_precedence(self) -> None:
        payload = {
            "engine": {"model_path": "/models/flux"},
            "request": {"prompt": "teapot"},
            "output": "default.png",
        }
        self.assertEqual(resolve_output_path(payload, "override.png"), "override.png")

    def test_main_runs_minimal_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                (
                    "{"
                    '"engine": {"model_path": "/models/flux"},'
                    '"request": {"prompt": "teapot"},'
                    '"output": "/tmp/from-config.png"'
                    "}"
                ),
                encoding="utf-8",
            )
            args = parse_args(["--config", str(config_path), "--output", "/tmp/override.png"])
            captured: dict[str, object] = {}

            class FakeImage:
                def save(self, path: str) -> None:
                    captured["saved_path"] = path

            class FakeEngine:
                def __init__(self, config) -> None:
                    captured["config"] = config

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb) -> None:
                    return None

                def generate_image(self, request):
                    captured["request"] = request
                    return [FakeImage()]

            stdout = io.StringIO()
            with patch.dict(main.__globals__, {"Engine": FakeEngine, "parse_args": lambda: args}):
                with redirect_stdout(stdout):
                    result = main()

        self.assertEqual(result, 0)
        self.assertEqual(captured["saved_path"], "/tmp/override.png")
        self.assertEqual(getattr(captured["config"], "model_path"), "/models/flux")
        self.assertEqual(getattr(captured["request"], "prompt"), "teapot")
        self.assertIn("saved image to /tmp/override.png", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
