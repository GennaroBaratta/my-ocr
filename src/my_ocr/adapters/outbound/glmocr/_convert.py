from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from my_ocr.application.dto import (
    ArtifactCopy,
    LayoutBlock,
    OcrPageResult,
    OcrRunResult,
    PageRef,
    ProviderArtifacts,
    ReviewLayout,
    ReviewPage,
)


def review_layout_from_legacy(
    payload: dict[str, Any], pages: list[PageRef], *, status: str = "prepared"
) -> ReviewLayout:
    pages_by_number = {page.page_number: page for page in pages}
    review_pages: list[ReviewPage] = []
    for raw_page in payload.get("pages", []):
        if not isinstance(raw_page, dict):
            continue
        page_number = int(raw_page.get("page_number", len(review_pages) + 1))
        page_ref = pages_by_number.get(page_number)
        size = raw_page.get("image_size", {})
        image_width = int(size.get("width", page_ref.width if page_ref else 0))
        image_height = int(size.get("height", page_ref.height if page_ref else 0))
        review_pages.append(
            ReviewPage(
                page_number=page_number,
                image_path=page_ref.image_path if page_ref else str(raw_page.get("page_path", "")),
                image_width=image_width,
                image_height=image_height,
                coord_space=str(raw_page.get("coord_space", "pixel")),
                provider_path=f"layout/provider/page-{page_number:04d}",
                blocks=[
                    LayoutBlock.from_dict(block)
                    for block in raw_page.get("blocks", [])
                    if isinstance(block, dict)
                ],
            )
        )
    return ReviewLayout(pages=review_pages, status=status)


def legacy_review_payload(review: ReviewLayout, pages: list[PageRef]) -> dict[str, Any]:
    pages_by_number = {page.page_number: page for page in pages}
    payload_pages: list[dict[str, Any]] = []
    for review_page in review.pages:
        page_ref = pages_by_number.get(review_page.page_number)
        page_payload = review_page.to_dict()
        page_payload["page_path"] = str(page_ref.path_for_io if page_ref else review_page.image_path)
        page_payload["source_sdk_json_path"] = review_page.provider_path or ""
        payload_pages.append(page_payload)
    return {
        "version": 2,
        "status": review.status,
        "pages": payload_pages,
        "summary": {"page_count": len(payload_pages)},
    }


def ocr_result_from_legacy(payload: dict[str, Any], markdown: str, pages: list[PageRef]) -> OcrRunResult:
    pages_by_number = {page.page_number: page for page in pages}
    ocr_pages: list[OcrPageResult] = []
    for raw_page in payload.get("pages", []):
        if not isinstance(raw_page, dict):
            continue
        page_number = int(raw_page.get("page_number", len(ocr_pages) + 1))
        page_ref = pages_by_number.get(page_number)
        fallback_path = None
        if raw_page.get("markdown_source") in {"crop_fallback", "full_page_fallback"}:
            fallback_path = f"ocr/fallback/page-{page_number:04d}"
        ocr_pages.append(
            OcrPageResult(
                page_number=page_number,
                image_path=page_ref.image_path if page_ref else str(raw_page.get("page_path", "")),
                markdown=str(raw_page.get("markdown", "")),
                markdown_source=str(raw_page.get("markdown_source", "unknown")),
                provider_path=f"ocr/provider/page-{page_number:04d}",
                fallback_path=fallback_path,
                raw_payload=_relative_raw_payload(raw_page, page_number, fallback_path),
            )
        )
    return OcrRunResult(pages=ocr_pages, markdown=markdown)


def provider_artifacts_from_pages(
    source_root: Path,
    target_root: str,
    pages: list[PageRef],
) -> ProviderArtifacts:
    copies: list[ArtifactCopy] = []
    for page in pages:
        source = source_root / f"page-{page.page_number:04d}"
        if source.exists():
            copies.append(
                ArtifactCopy(source=source, relative_target=f"{target_root}/page-{page.page_number:04d}")
            )
    return ProviderArtifacts(tuple(copies))


def combined_artifacts(*bundles: ProviderArtifacts) -> ProviderArtifacts:
    copies: list[ArtifactCopy] = []
    cleanup_paths: list[Path] = []
    for bundle in bundles:
        copies.extend(bundle.copies)
        cleanup_paths.extend(bundle.cleanup_paths)
    return ProviderArtifacts(tuple(copies), tuple(cleanup_paths))


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _relative_raw_payload(
    raw_page: dict[str, Any], page_number: int, fallback_path: str | None
) -> dict[str, Any]:
    payload = {
        "assessment": raw_page.get("assessment", {}),
        "layout_source": raw_page.get("layout_source"),
    }
    if fallback_path:
        payload["fallback_path"] = fallback_path
    payload["provider_path"] = f"ocr/provider/page-{page_number:04d}"
    return payload
