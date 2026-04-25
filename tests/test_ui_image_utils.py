from __future__ import annotations

from my_ocr.ui.image_utils import get_image_source


def test_get_image_source_reads_updated_bytes_from_same_path(tmp_path) -> None:
    image_path = tmp_path / "page-0001.png"
    image_path.write_bytes(b"first image bytes")

    assert get_image_source(str(image_path)) == b"first image bytes"

    image_path.write_bytes(b"updated image bytes")

    assert get_image_source(str(image_path)) == b"updated image bytes"
