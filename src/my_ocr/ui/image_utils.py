"""Helpers for working with UI image assets."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image


@lru_cache(maxsize=256)
def get_image_size(image_path: str) -> tuple[int, int]:
    """Return image width and height in pixels."""
    try:
        with Image.open(image_path) as image:
            return image.size
    except OSError:
        return 0, 0


def get_image_source(image_path: str) -> str | bytes:
    """Return a Flet-compatible image source for local desktop and web clients."""
    try:
        return Path(image_path).read_bytes()
    except OSError:
        return image_path
