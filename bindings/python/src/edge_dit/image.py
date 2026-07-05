from __future__ import annotations

import ctypes

from PIL import Image

from ._capi import EdImage, EdImageBatch
from .errors import EdgeDitError, GenerationError, UnsupportedImageFormatError

_PIL_MODE_BY_CHANNELS = {
    1: "L",
    3: "RGB",
    4: "RGBA",
}


def image_to_pil(image: EdImage) -> Image.Image:
    if not image.data or image.width == 0 or image.height == 0:
        raise GenerationError("engine returned an empty image")

    mode = _PIL_MODE_BY_CHANNELS.get(int(image.channels))
    if mode is None:
        raise UnsupportedImageFormatError(
            f"unsupported native image channel count: {image.channels}"
        )

    size = int(image.width) * int(image.height) * int(image.channels)
    raw = ctypes.string_at(image.data, size)
    return Image.frombytes(mode, (int(image.width), int(image.height)), raw)


def image_to_numpy(image: EdImage):
    try:
        import numpy as np
    except ImportError as exc:
        raise EdgeDitError(
            "numpy output requires the optional 'numpy' package to be installed"
        ) from exc

    if not image.data or image.width == 0 or image.height == 0:
        raise GenerationError("engine returned an empty image")

    channels = int(image.channels)
    if channels not in _PIL_MODE_BY_CHANNELS:
        raise UnsupportedImageFormatError(
            f"unsupported native image channel count: {image.channels}"
        )

    width = int(image.width)
    height = int(image.height)
    size = width * height * channels
    raw = ctypes.string_at(image.data, size)
    array = np.frombuffer(raw, dtype=np.uint8)
    if channels == 1:
        return array.reshape((height, width))
    return array.reshape((height, width, channels))


def batch_to_pil_images(batch: EdImageBatch) -> list[Image.Image]:
    if not batch.images or batch.count <= 0:
        return []
    return [image_to_pil(batch.images[index]) for index in range(int(batch.count))]


def batch_to_numpy_images(batch: EdImageBatch) -> list[object]:
    if not batch.images or batch.count <= 0:
        return []
    return [image_to_numpy(batch.images[index]) for index in range(int(batch.count))]
