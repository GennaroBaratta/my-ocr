from __future__ import annotations

from my_ocr.runs.store import RecentRunRecord
from my_ocr.domain import LayoutBlock, ReviewLayout, ReviewPage, RunSnapshot

from .session import BoundingBox, PageData, RecentRunSummary


def recent_run_summary(record: RecentRunRecord) -> RecentRunSummary:
    return RecentRunSummary(
        run_id=record.run_id,
        input_path=record.input_path,
        mtime=record.mtime,
        status=record.status,
    )


def pages_from_snapshot(snapshot: RunSnapshot) -> list[PageData]:
    boxes_by_page = _boxes_from_review(snapshot.review_layout)
    pages: list[PageData] = []
    for index, page in enumerate(snapshot.pages):
        pages.append(
            PageData(
                index=index,
                page_number=page.page_number,
                image_path=str(page.path_for_io),
                relative_image_path=page.image_path,
                boxes=boxes_by_page.get(page.page_number, []),
            )
        )
    return pages


def page_data_to_review_layout(pages: list[PageData]) -> ReviewLayout:
    from my_ocr.ui.image_utils import get_image_size

    review_pages: list[ReviewPage] = []
    for page in pages:
        image_width, image_height = get_image_size(page.image_path)
        review_pages.append(
            ReviewPage(
                page_number=page.page_number,
                image_path=page.relative_image_path or _relative_image_path(
                    page.image_path, page.page_number
                ),
                image_width=image_width,
                image_height=image_height,
                coord_space="pixel",
                blocks=[
                    LayoutBlock(
                        id=box.id,
                        index=index,
                        label=box.label,
                        content=box.content,
                        confidence=box.confidence,
                        bbox=[box.x, box.y, box.x + box.width, box.y + box.height],
                    )
                    for index, box in enumerate(page.boxes)
                ],
            )
        )
    return ReviewLayout(pages=review_pages, status="reviewed")


def _boxes_from_review(review: ReviewLayout | None) -> dict[int, list[BoundingBox]]:
    if review is None:
        return {}
    result: dict[int, list[BoundingBox]] = {}
    for page_index, review_page in enumerate(review.pages):
        result[review_page.page_number] = [
            _box_from_block(block, page_index) for block in review_page.blocks
        ]
    return result


def _box_from_block(block: LayoutBlock, page_index: int) -> BoundingBox:
    x1, y1, x2, y2 = block.bbox
    return BoundingBox(
        id=block.id,
        page_index=page_index,
        x=x1,
        y=y1,
        width=max(0, x2 - x1),
        height=max(0, y2 - y1),
        label=block.label,
        confidence=block.confidence,
        content=block.content,
    )


def _relative_image_path(path: str, page_number: int) -> str:
    marker = "pages/"
    normalized = path.replace("\\", "/")
    if marker in normalized:
        return normalized[normalized.index(marker) :]
    return f"pages/page-{page_number:04d}"
