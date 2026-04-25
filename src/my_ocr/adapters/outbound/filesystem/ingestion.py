from __future__ import annotations

from pathlib import Path

from my_ocr.pipeline.normalize import (
    IMAGE_SUFFIXES as IMAGE_SUFFIXES,
    normalize_document as normalize_document_pages,
    render_pdf_to_images as render_pdf_to_image_pages,
)


def normalize_document(input_path: str | Path, run_dir: str | Path) -> list[str]:
    pages = normalize_document_pages(input_path, Path(run_dir) / "pages", measure_images=False)
    return [str(page.path) for page in pages]


def render_pdf_to_images(pdf_path: str | Path, output_dir: str | Path) -> list[str]:
    pages = render_pdf_to_image_pages(pdf_path, output_dir, measure_images=False)
    return [str(page.path) for page in pages]
