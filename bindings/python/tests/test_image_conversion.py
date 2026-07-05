from __future__ import annotations

import ctypes
import builtins
import unittest
from unittest.mock import patch

from edge_dit._capi import EdImage, EdImageBatch
from edge_dit.errors import EdgeDitError, UnsupportedImageFormatError
from edge_dit.image import batch_to_numpy_images, batch_to_pil_images, image_to_numpy, image_to_pil


class ImageConversionTests(unittest.TestCase):
    def test_rgb_image_converts_to_pil(self) -> None:
        raw = (ctypes.c_uint8 * 3)(255, 0, 0)
        image = EdImage(width=1, height=1, channels=3, data=ctypes.cast(raw, ctypes.POINTER(ctypes.c_uint8)))
        pil_image = image_to_pil(image)
        self.assertEqual(pil_image.mode, "RGB")
        self.assertEqual(pil_image.size, (1, 1))
        self.assertEqual(pil_image.getpixel((0, 0)), (255, 0, 0))

    def test_batch_conversion(self) -> None:
        raw = (ctypes.c_uint8 * 4)(1, 2, 3, 4)
        image = EdImage(width=1, height=1, channels=4, data=ctypes.cast(raw, ctypes.POINTER(ctypes.c_uint8)))
        images = (EdImage * 1)(image)
        batch = EdImageBatch(images=ctypes.cast(images, ctypes.POINTER(EdImage)), count=1)
        pil_images = batch_to_pil_images(batch)
        self.assertEqual(len(pil_images), 1)
        self.assertEqual(pil_images[0].mode, "RGBA")

    def test_rgb_image_converts_to_numpy(self) -> None:
        raw = (ctypes.c_uint8 * 3)(255, 0, 0)
        image = EdImage(width=1, height=1, channels=3, data=ctypes.cast(raw, ctypes.POINTER(ctypes.c_uint8)))
        array = image_to_numpy(image)
        self.assertEqual(array.shape, (1, 1, 3))
        self.assertEqual(array.tolist(), [[[255, 0, 0]]])

    def test_grayscale_image_converts_to_2d_numpy(self) -> None:
        raw = (ctypes.c_uint8 * 2)(7, 9)
        image = EdImage(width=2, height=1, channels=1, data=ctypes.cast(raw, ctypes.POINTER(ctypes.c_uint8)))
        array = image_to_numpy(image)
        self.assertEqual(array.shape, (1, 2))
        self.assertEqual(array.tolist(), [[7, 9]])

    def test_numpy_batch_conversion(self) -> None:
        raw = (ctypes.c_uint8 * 3)(1, 2, 3)
        image = EdImage(width=1, height=1, channels=3, data=ctypes.cast(raw, ctypes.POINTER(ctypes.c_uint8)))
        images = (EdImage * 1)(image)
        batch = EdImageBatch(images=ctypes.cast(images, ctypes.POINTER(EdImage)), count=1)
        arrays = batch_to_numpy_images(batch)
        self.assertEqual(len(arrays), 1)
        self.assertEqual(arrays[0].tolist(), [[[1, 2, 3]]])

    def test_numpy_output_requires_optional_dependency(self) -> None:
        raw = (ctypes.c_uint8 * 3)(255, 0, 0)
        image = EdImage(width=1, height=1, channels=3, data=ctypes.cast(raw, ctypes.POINTER(ctypes.c_uint8)))
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "numpy":
                raise ImportError("missing numpy")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(EdgeDitError):
                image_to_numpy(image)

    def test_unsupported_channel_count_raises(self) -> None:
        raw = (ctypes.c_uint8 * 2)(0, 0)
        image = EdImage(width=1, height=1, channels=2, data=ctypes.cast(raw, ctypes.POINTER(ctypes.c_uint8)))
        with self.assertRaises(UnsupportedImageFormatError):
            image_to_pil(image)


if __name__ == "__main__":
    unittest.main()
