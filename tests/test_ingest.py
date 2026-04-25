from __future__ import annotations

from pathlib import Path

import pytest

from my_ocr.ingest.normalize import normalize_document


def test_normalize_document_copies_directory_images_in_natural_order(tmp_path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    for name in ["page-10.png", "page-2.png", "notes.txt", "page-1.jpg"]:
        path = source / name
        if path.suffix.lower() in {".jpg", ".png"}:
            _image(path)
        else:
            path.write_text("not an image", encoding="utf-8")

    pages_dir = tmp_path / "run" / "pages"
    pages = normalize_document(source, pages_dir)

    assert [Path(page.image_path).name for page in pages] == [
        "page-0001.jpg",
        "page-0002.png",
        "page-0003.png",
    ]
    assert [page.width for page in pages] == [10, 10, 10]
    assert [page.height for page in pages] == [10, 10, 10]
    assert (pages_dir / "page-0001.jpg").exists()
    assert (pages_dir / "page-0002.png").exists()
    assert (pages_dir / "page-0003.png").exists()


def test_normalize_document_rejects_empty_directory(tmp_path) -> None:
    source = tmp_path / "empty"
    source.mkdir()

    with pytest.raises(ValueError, match="No supported images"):
        normalize_document(source, tmp_path / "run" / "pages")


def _image(path: Path) -> None:
    from PIL import Image

    Image.new("RGB", (10, 10), "white").save(path)
