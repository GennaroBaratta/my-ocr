from __future__ import annotations

import re
import shutil
from pathlib import Path

from .json_store import ensure_dir

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def normalize_document(input_path: str | Path, run_dir: str | Path) -> list[str]:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Input not found: {source}")

    pages_dir = _reset_pages_dir(Path(run_dir) / "pages")

    if source.is_dir():
        page_paths = sorted(
            (path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
            key=_natural_sort_key,
        )
        if not page_paths:
            raise ValueError(f"No supported images found in directory: {source}")
        return [
            _copy_page(path, pages_dir, index) for index, path in enumerate(page_paths, start=1)
        ]

    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return render_pdf_to_images(source, pages_dir)
    if suffix in IMAGE_SUFFIXES:
        return [_copy_page(source, pages_dir, 1)]

    raise ValueError(f"Unsupported input type: {source.suffix}")


def render_pdf_to_images(pdf_path: str | Path, output_dir: str | Path) -> list[str]:
    pdf_path = Path(pdf_path)
    output_dir = ensure_dir(output_dir)

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PDF rendering requires PyMuPDF. Install with `pip install -e .[pdf]`."
        ) from exc

    rendered: list[str] = []
    with fitz.open(pdf_path) as document:
        for index in range(document.page_count):
            page = document.load_page(index)
            pixmap = page.get_pixmap(dpi=200)
            destination = output_dir / f"page-{index + 1:04d}.png"
            pixmap.save(destination)
            rendered.append(str(destination))
    return rendered


def _copy_page(source: Path, pages_dir: Path, index: int) -> str:
    destination = pages_dir / f"page-{index:04d}{source.suffix.lower()}"
    shutil.copy2(source, destination)
    return str(destination)


def _reset_pages_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    return ensure_dir(path)


def _natural_sort_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]
