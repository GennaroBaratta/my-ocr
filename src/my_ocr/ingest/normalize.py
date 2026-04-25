from __future__ import annotations

from dataclasses import dataclass
import re
import shutil
from pathlib import Path

from my_ocr.support.filesystem import ensure_dir as _ensure_dir
from my_ocr.domain import MissingInputDocument, PageRef

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_RENDER_DPI = 200


@dataclass(frozen=True, slots=True)
class _NormalizedPage:
    page_number: int
    path: Path
    width: int
    height: int


def normalize_document(
    input_path: str | Path, pages_dir: str | Path
) -> list[PageRef]:
    pages = _normalize_pages(input_path, pages_dir)
    return [_page_ref(page) for page in pages]


def _normalize_pages(input_path: str | Path, pages_dir: str | Path) -> list[_NormalizedPage]:
    source = Path(input_path)
    if not source.exists():
        raise MissingInputDocument(f"Input not found: {source}")

    output_dir = _reset_pages_dir(Path(pages_dir))
    if source.is_dir():
        sources = sorted(
            (path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
            key=_natural_sort_key,
        )
        if not sources:
            raise ValueError(f"No supported images found in directory: {source}")
        return [
            _copy_page(path, output_dir, index)
            for index, path in enumerate(sources, 1)
        ]

    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return _render_pdf_pages(source, output_dir)
    if suffix in IMAGE_SUFFIXES:
        return [_copy_page(source, output_dir, 1)]

    raise ValueError(f"Unsupported input type: {source.suffix}")


def _render_pdf_pages(pdf_path: str | Path, output_dir: str | Path) -> list[_NormalizedPage]:
    pdf_path = Path(pdf_path)
    output_dir = _ensure_dir(output_dir)

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PDF rendering requires PyMuPDF. Install with `pip install -e .[pdf]`."
        ) from exc

    rendered: list[_NormalizedPage] = []
    with fitz.open(pdf_path) as document:
        for index in range(document.page_count):
            page = document.load_page(index)
            pixmap = page.get_pixmap(dpi=PDF_RENDER_DPI)
            destination = output_dir / f"page-{index + 1:04d}.png"
            pixmap.save(destination)
            rendered.append(_normalized_page(destination, index + 1))
    return rendered


def _copy_page(source: Path, output_dir: Path, page_number: int) -> _NormalizedPage:
    destination = output_dir / f"page-{page_number:04d}{source.suffix.lower()}"
    shutil.copy2(source, destination)
    return _normalized_page(destination, page_number)


def _page_ref(page: _NormalizedPage) -> PageRef:
    return PageRef(
        page_number=page.page_number,
        image_path=f"pages/{page.path.name}",
        width=page.width,
        height=page.height,
        resolved_path=page.path,
    )


def _normalized_page(path: Path, page_number: int) -> _NormalizedPage:
    width, height = _image_size(path)
    return _NormalizedPage(page_number=page_number, path=path, width=width, height=height)


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Image normalization requires Pillow.") from exc

    with Image.open(path) as image:
        return image.size


def _reset_pages_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    return _ensure_dir(path)


def _natural_sort_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]
