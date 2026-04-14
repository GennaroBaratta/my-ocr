from __future__ import annotations

from pathlib import Path

import pytest

from free_doc_extract.ingest import normalize_document


def test_normalize_document_copies_directory_images_in_natural_order(tmp_path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    for name in ["page-10.png", "page-2.png", "notes.txt", "page-1.jpg"]:
        path = source / name
        path.write_bytes(b"fake")

    run_dir = tmp_path / "run"
    pages = normalize_document(source, run_dir)

    assert [Path(path).name for path in pages] == [
        "page-0001.jpg",
        "page-0002.png",
        "page-0003.png",
    ]
    assert (run_dir / "pages" / "page-0001.jpg").exists()
    assert (run_dir / "pages" / "page-0002.png").exists()
    assert (run_dir / "pages" / "page-0003.png").exists()


def test_normalize_document_rejects_empty_directory(tmp_path) -> None:
    source = tmp_path / "empty"
    source.mkdir()

    with pytest.raises(ValueError, match="No supported images"):
        normalize_document(source, tmp_path / "run")
