"""Helpers for working with UI image assets."""

from __future__ import annotations

from functools import lru_cache

from PIL import Image


@lru_cache(maxsize=256)
def get_image_size(image_path: str) -> tuple[int, int]:
    """Return image width and height in pixels."""
    try:
        with Image.open(image_path) as image:
            return image.size
    except OSError:
        return 0, 0
