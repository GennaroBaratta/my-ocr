from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
import sys
from typing import Any

from my_ocr.domain import ArtifactCopy, OcrPageResult, PageRef, ProviderArtifacts
from my_ocr.domain import OcrRuntimeOptions
from my_ocr.ingest.normalize import IMAGE_SUFFIXES
from my_ocr.ocr.ocr_policy import detect_bbox_coord_space
from my_ocr.ocr.scratch_paths import ProviderScratchPaths
from my_ocr.settings import resolve_ocr_api_client as _resolve_configured_ocr_api_client


def emit_layout_profile_warning(diagnostics: dict[str, Any]) -> None:
    warning = diagnostics.get("layout_profile_warning")
    if not isinstance(warning, str) or not warning.strip():
        return
    print(f"Warning: {warning}", file=sys.stderr)


def resolve_ocr_api_client(options: OcrRuntimeOptions) -> tuple[str, str, int]:
    config_model, config_endpoint, config_num_ctx = _resolve_configured_ocr_api_client(
        options.config_path
    )
    return (
        options.model or config_model,
        options.endpoint or config_endpoint,
        options.num_ctx if options.num_ctx is not None else config_num_ctx,
    )


def normalize_page_refs(pages: Sequence[PageRef]) -> tuple[PageRef, ...]:
    candidates = tuple(pages)
    if not candidates:
        raise ValueError("At least one normalized page image is required.")

    for page in candidates:
        page_path = page.path_for_io
        if page_path.is_dir():
            raise ValueError("run_ocr expects page image files, not directories.")
        if not page_path.exists():
            raise FileNotFoundError(f"Page image not found: {page_path}")
        if page_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(
                f"run_ocr expects normalized page images. Unsupported page input: {page_path.name}"
            )
        if page.page_number <= 0:
            raise ValueError(f"Invalid page number {page.page_number} for {page_path}")
    return candidates


def detect_coord_space(blocks: list[dict[str, Any]], page_path: str) -> str:
    if not blocks:
        return "unknown"

    try:
        image_module = import_module("PIL.Image")
    except ImportError:
        return detect_bbox_coord_space(blocks)

    with image_module.open(page_path) as image:
        width, height = image.size
    return detect_bbox_coord_space(blocks, width=width, height=height)


def ocr_page_from_provider_payload(
    raw_page: dict[str, Any], pages_by_number: dict[int, PageRef]
) -> OcrPageResult:
    page_number = int(raw_page.get("page_number", 0))
    page_ref = pages_by_number[page_number]
    fallback_path = None
    if raw_page.get("markdown_source") in {"crop_fallback", "full_page_fallback"}:
        fallback_path = f"ocr/fallback/page-{page_number:04d}"
    return OcrPageResult(
        page_number=page_number,
        image_path=page_ref.image_path,
        markdown=str(raw_page.get("markdown", "")),
        markdown_source=str(raw_page.get("markdown_source", "unknown")),
        provider_path=f"ocr/provider/page-{page_number:04d}",
        fallback_path=fallback_path,
        raw_payload=relative_raw_payload(raw_page, page_number, fallback_path),
    )


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
                ArtifactCopy(
                    source=source,
                    relative_target=f"{target_root}/page-{page.page_number:04d}",
                )
            )
    return ProviderArtifacts(tuple(copies))


def combined_artifacts(*bundles: ProviderArtifacts) -> ProviderArtifacts:
    copies: list[ArtifactCopy] = []
    cleanup_paths: list[Path] = []
    for bundle in bundles:
        copies.extend(bundle.copies)
        cleanup_paths.extend(bundle.cleanup_paths)
    return ProviderArtifacts(tuple(copies), tuple(cleanup_paths))


def with_cleanup(bundle: ProviderArtifacts, cleanup_paths: tuple[Path, ...]) -> ProviderArtifacts:
    return ProviderArtifacts(bundle.copies, bundle.cleanup_paths + cleanup_paths)


def ocr_cleanup_paths(paths: ProviderScratchPaths) -> tuple[Path, ...]:
    return (paths.raw_dir, paths.fallback_dir)


def relative_raw_payload(
    raw_page: dict[str, Any], page_number: int, fallback_path: str | None
) -> dict[str, Any]:
    payload = {
        "assessment": raw_page.get("fallback_assessment", raw_page.get("assessment", {})),
        "layout_source": raw_page.get("layout_source"),
        "sdk_markdown": raw_page.get("sdk_markdown"),
    }
    if fallback_path:
        payload["fallback_path"] = fallback_path
    payload["provider_path"] = f"ocr/provider/page-{page_number:04d}"
    return payload
