from __future__ import annotations

import unittest

from edge_dit.enums import resolve_cache_mode, resolve_dtype, resolve_sampler, resolve_scheduler
from edge_dit.errors import InvalidArgumentError


class EnumMappingTests(unittest.TestCase):
    def test_dtype_aliases(self) -> None:
        self.assertEqual(resolve_dtype("q4_k"), 12)
        self.assertEqual(resolve_dtype("bf16"), 30)

    def test_sampler_aliases(self) -> None:
        self.assertEqual(resolve_sampler("dpm++-2m"), 5)
        self.assertEqual(resolve_sampler("ddim"), 10)

    def test_scheduler_aliases(self) -> None:
        self.assertEqual(resolve_scheduler("simple"), 6)

    def test_cache_aliases(self) -> None:
        self.assertEqual(resolve_cache_mode("off"), 0)
        self.assertEqual(resolve_cache_mode("cache-dit"), 5)

    def test_invalid_name_raises(self) -> None:
        with self.assertRaises(InvalidArgumentError):
            resolve_sampler("mystery-sampler")


if __name__ == "__main__":
    unittest.main()

