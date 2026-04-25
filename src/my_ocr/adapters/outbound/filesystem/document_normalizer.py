from __future__ import annotations

from pathlib import Path

from my_ocr.pipeline.errors import MissingInputDocument
from my_ocr.models import PageRef
from my_ocr.pipeline.normalize import NormalizedPage, normalize_document


class FilesystemDocumentNormalizer:
    def normalize(self, input_path: str | Path, pages_dir: str | Path) -> list[PageRef]:
        try:
            pages = normalize_document(input_path, pages_dir)
        except FileNotFoundError as exc:
            raise MissingInputDocument(str(exc)) from exc
        return [_page_ref(page) for page in pages]


def _page_ref(page: NormalizedPage) -> PageRef:
    return PageRef(
        page_number=page.page_number,
        image_path=f"pages/{page.path.name}",
        width=page.width,
        height=page.height,
        resolved_path=page.path,
    )
