from __future__ import annotations

import re
import shutil
from pathlib import Path

from my_ocr.application.models import PageRef
from my_ocr.application.errors import MissingInputDocument

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class FilesystemDocumentNormalizer:
    def normalize(self, input_path: str | Path, pages_dir: str | Path) -> list[PageRef]:
        source = Path(input_path)
        if not source.exists():
            raise MissingInputDocument(f"Input not found: {source}")

        output_dir = _reset_dir(Path(pages_dir))
        if source.is_dir():
            sources = sorted(
                (path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
                key=_natural_sort_key,
            )
            if not sources:
                raise ValueError(f"No supported images found in directory: {source}")
            return [_copy_page(path, output_dir, index) for index, path in enumerate(sources, 1)]

        suffix = source.suffix.lower()
        if suffix == ".pdf":
            return _render_pdf_to_images(source, output_dir)
        if suffix in IMAGE_SUFFIXES:
            return [_copy_page(source, output_dir, 1)]
        raise ValueError(f"Unsupported input type: {source.suffix}")


def _render_pdf_to_images(pdf_path: Path, output_dir: Path) -> list[PageRef]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PDF rendering requires PyMuPDF. Install with `pip install -e .[pdf]`."
        ) from exc

    rendered: list[PageRef] = []
    with fitz.open(pdf_path) as document:
        for index in range(document.page_count):
            page = document.load_page(index)
            pixmap = page.get_pixmap(dpi=200)
            destination = output_dir / f"page-{index + 1:04d}.png"
            pixmap.save(destination)
            rendered.append(_page_ref(destination, index + 1))
    return rendered


def _copy_page(source: Path, output_dir: Path, page_number: int) -> PageRef:
    destination = output_dir / f"page-{page_number:04d}{source.suffix.lower()}"
    shutil.copy2(source, destination)
    return _page_ref(destination, page_number)


def _page_ref(path: Path, page_number: int) -> PageRef:
    width, height = _image_size(path)
    return PageRef(
        page_number=page_number,
        image_path=f"pages/{path.name}",
        width=width,
        height=height,
        resolved_path=path,
    )


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Image normalization requires Pillow.") from exc

    with Image.open(path) as image:
        return image.size


def _reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _natural_sort_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]
