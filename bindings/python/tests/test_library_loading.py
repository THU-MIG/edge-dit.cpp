from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from edge_dit._lib import (
    EDGE_DIT_DEPENDENCY_DIRS_ENV,
    _common_dependency_dirs,
    load_library,
)


class LibraryLoadingTests(unittest.TestCase):
    def test_load_library_preloads_sibling_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            libedgedit = root / "libedgedit.so"
            libggml = root / "libggml.so"
            libcudnn = root / "libcudnn.so.9"
            for path in (libedgedit, libggml, libcudnn):
                path.touch()

            calls: list[str] = []

            def fake_cdll(path: str, mode: int = 0):
                calls.append(Path(path).name)
                return object()

            with patch("edge_dit._lib.ctypes.CDLL", side_effect=fake_cdll):
                load_library(path=libedgedit)

            self.assertEqual(calls[-1], "libedgedit.so")
            self.assertIn("libggml.so", calls)
            self.assertIn("libcudnn.so.9", calls)

    def test_dependency_dirs_env_is_respected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            depdir = Path(tmpdir)
            with patch.dict("os.environ", {EDGE_DIT_DEPENDENCY_DIRS_ENV: str(depdir)}):
                dirs = _common_dependency_dirs()
            self.assertIn(depdir, dirs)


if __name__ == "__main__":
    unittest.main()

