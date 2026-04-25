from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from my_ocr.domain.layout import detect_bbox_coord_space, extract_layout_blocks
from my_ocr.domain.page_identity import (
    infer_page_number,
    page_numbers_by_index,
    resolve_page_path_for_number as resolve_page_path_for_number_from_payloads,
)
from my_ocr.domain.review_layout import (
    REVIEW_LAYOUT_VERSION,
    REVIEWED_LAYOUT_APPLY_MODE,
    build_review_layout_payload,
    build_review_page_from_layout,
    load_review_layout_payload,
    review_layout_pages_by_number,
    save_review_layout_payload,
)

from .artifact_payload_paths import (
    count_page_sources,
    load_json,
    ordered_page_records,
    pages_by_number,
    rewrite_path_keys,
    write_json,
    write_merged_markdown,
)
from .run_artifact_publisher import copy_path, remove_paths


def resolve_page_path_for_number(paths: Any, page_number: int, *payloads: Any) -> str:
    payloads_to_check = [*payloads, *load_page_identity_payloads(paths)]
    try:
        return resolve_page_path_for_number_from_payloads(
            paths.run_dir,
            paths.list_page_paths(),
            page_number,
            *payloads_to_check,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Page {page_number} not found in {paths.pages_dir}") from exc


def load_page_identity_payloads(paths: Any) -> list[Any]:
    payloads: list[Any] = []
    if paths.reviewed_layout_path.exists():
        payloads.append(load_review_layout_payload(paths.reviewed_layout_path))
    if paths.ocr_json_path.exists():
        payloads.append(load_json(paths.ocr_json_path))
    return payloads


def copy_review_artifact_snapshot(source_paths: Any, staged_paths: Any) -> None:
    staged_paths.ensure_run_dir()
    for source, target in (
        (source_paths.pages_dir, staged_paths.pages_dir),
        (source_paths.raw_dir, staged_paths.raw_dir),
        (source_paths.reviewed_layout_path, staged_paths.reviewed_layout_path),
    ):
        if source.exists():
            copy_path(source, target)


def copy_reviewed_ocr_artifact_snapshot(source_paths: Any, staged_paths: Any) -> None:
    staged_paths.ensure_run_dir()
    for source, target in (
        (source_paths.raw_dir, staged_paths.raw_dir),
        (source_paths.fallback_dir, staged_paths.fallback_dir),
        (source_paths.ocr_markdown_path, staged_paths.ocr_markdown_path),
        (source_paths.ocr_json_path, staged_paths.ocr_json_path),
        (source_paths.ocr_fallback_path, staged_paths.ocr_fallback_path),
    ):
        if source.exists():
            copy_path(source, target)


def merge_review_page_artifacts(
    source_paths: Any,
    staged_paths: Any,
    partial_paths: Any,
    page_number: int,
) -> None:
    partial_payload = load_review_layout_payload(partial_paths.reviewed_layout_path)
    if partial_payload is None:
        raise FileNotFoundError(f"Missing partial reviewed layout: {partial_paths.reviewed_layout_path}")

    partial_page = review_layout_pages_by_number(partial_payload).get(page_number)
    if partial_page is None:
        raise KeyError(f"Partial reviewed layout is missing page {page_number}")

    rewrite_path_keys(
        partial_page,
        ("page_path", "source_sdk_json_path"),
        str(partial_paths.run_dir),
        str(staged_paths.run_dir),
    )
    copy_page_dir(partial_paths.raw_page_dir(page_number), staged_paths.raw_page_dir(page_number))

    existing_payload = load_review_layout_payload(staged_paths.reviewed_layout_path) or {
        "version": REVIEW_LAYOUT_VERSION,
        "status": "prepared",
        "pages": [],
        "summary": {"page_count": 0},
    }
    existing_pages = existing_payload.get("pages")
    if not isinstance(existing_pages, list):
        existing_pages = []
    existing_by_number = seed_review_pages_for_merge(
        source_paths,
        staged_paths,
        review_layout_pages_by_number(existing_payload),
    )
    existing_by_number[page_number] = partial_page
    ordered_pages = ordered_page_records(existing_pages, existing_by_number, staged_paths.pages_dir)
    status = partial_payload.get("status")
    payload = build_review_layout_payload(
        ordered_pages,
        status=status if isinstance(status, str) and status.strip() else "prepared",
    )
    save_review_layout_payload(staged_paths.reviewed_layout_path, payload)


def seed_review_pages_for_merge(
    source_paths: Any,
    staged_paths: Any,
    review_pages_by_number: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    seeded_pages = dict(review_pages_by_number)
    staged_page_paths = staged_paths.list_page_paths()
    review_payload = {"pages": list(review_pages_by_number.values())}
    ocr_payload = load_json(source_paths.ocr_json_path) if source_paths.ocr_json_path.exists() else None
    page_numbers = page_numbers_by_index(
        staged_paths.run_dir,
        staged_page_paths,
        review_payload,
        ocr_payload,
    )
    expected_page_numbers = [
        page_numbers.get(page_idx, infer_page_number(page_path, page_idx + 1))
        for page_idx, page_path in enumerate(staged_page_paths)
    ]
    missing_page_numbers = [
        page_number for page_number in expected_page_numbers if page_number not in seeded_pages
    ]
    if not missing_page_numbers:
        return seeded_pages

    if not isinstance(ocr_payload, dict):
        raise FileNotFoundError(
            "Missing OCR payload required to preserve untouched review pages: "
            f"{source_paths.ocr_json_path}"
        )

    ocr_pages_by_number = pages_by_number(ocr_payload)
    for missing_page_number in missing_page_numbers:
        ocr_page = ocr_pages_by_number.get(missing_page_number)
        if ocr_page is None:
            raise KeyError(
                "OCR payload is missing untouched page "
                f"{missing_page_number} required for review merge"
            )
        seeded_pages[missing_page_number] = build_review_page_from_ocr_page(
            source_paths,
            staged_paths,
            missing_page_number,
            ocr_page,
            ocr_payload,
        )
    return seeded_pages


def build_review_page_from_ocr_page(
    source_paths: Any,
    staged_paths: Any,
    page_number: int,
    ocr_page: dict[str, Any],
    ocr_payload: dict[str, Any],
) -> dict[str, Any]:
    sdk_json_path = resolve_sdk_json_path_for_page(source_paths, page_number, ocr_page)
    if not sdk_json_path.exists():
        raise FileNotFoundError(
            f"Missing SDK JSON for untouched page {page_number}: {sdk_json_path}"
        )

    page_path = resolve_page_path_for_number(staged_paths, page_number, ocr_payload)
    copy_path(sdk_json_path.parent, staged_paths.raw_page_dir(page_number))
    staged_sdk_json_path = staged_paths.raw_page_dir(page_number) / sdk_json_path.name
    layout = load_json(sdk_json_path)
    blocks = extract_layout_blocks(layout)
    coord_space = detect_coord_space(blocks, page_path)
    image_width, image_height = get_image_size(page_path)
    return build_review_page_from_layout(
        page_number=page_number,
        page_path=page_path,
        source_sdk_json_path=str(staged_sdk_json_path),
        layout=layout,
        coord_space=coord_space,
        image_width=image_width,
        image_height=image_height,
    )


def resolve_sdk_json_path_for_page(
    source_paths: Any,
    page_number: int,
    ocr_page: dict[str, Any],
) -> Path:
    sdk_json_path = ocr_page.get("sdk_json_path")
    if not isinstance(sdk_json_path, str) or not sdk_json_path.strip():
        raise KeyError(f"OCR payload is missing sdk_json_path for untouched page {page_number}")

    candidate = Path(sdk_json_path)
    if candidate.exists():
        return candidate
    if candidate.is_absolute():
        raise FileNotFoundError(candidate)

    resolved = source_paths.run_dir / candidate
    if resolved.exists():
        return resolved
    raise FileNotFoundError(resolved)


def get_image_size(page_path: str | Path) -> tuple[int, int]:
    try:
        image_module = import_module("PIL.Image")
    except ImportError as exc:
        raise RuntimeError("Review page merge requires Pillow.") from exc

    with image_module.open(page_path) as image:
        return image.size


def detect_coord_space(blocks: list[dict[str, Any]], page_path: str | Path) -> str:
    if not blocks:
        return "unknown"

    try:
        image_width, image_height = get_image_size(page_path)
    except OSError:
        return detect_bbox_coord_space(blocks)

    return detect_bbox_coord_space(blocks, width=image_width, height=image_height)


def merge_ocr_page_artifacts(
    staged_paths: Any,
    partial_paths: Any,
    target_paths: Any,
    page_number: int,
) -> None:
    partial_payload = load_json(partial_paths.ocr_json_path)
    if not isinstance(partial_payload, dict):
        raise FileNotFoundError(f"Missing partial OCR payload: {partial_paths.ocr_json_path}")

    partial_pages = pages_by_number(partial_payload)
    partial_page = partial_pages.get(page_number)
    if partial_page is None:
        raise KeyError(f"Partial OCR payload is missing page {page_number}")

    rewrite_path_keys(
        partial_page,
        ("page_path", "sdk_json_path"),
        str(partial_paths.run_dir),
        str(staged_paths.run_dir),
    )
    copy_page_dir(partial_paths.raw_page_dir(page_number), staged_paths.raw_page_dir(page_number))

    existing_payload = load_json(staged_paths.ocr_json_path)
    if not isinstance(existing_payload, dict):
        raise FileNotFoundError(f"Missing existing OCR payload: {staged_paths.ocr_json_path}")
    existing_pages = existing_payload.get("pages")
    if not isinstance(existing_pages, list):
        raise ValueError(f"Invalid OCR pages payload: {staged_paths.ocr_json_path}")

    existing_by_number = pages_by_number(existing_payload)
    existing_by_number[page_number] = partial_page
    merged_pages = ordered_page_records(existing_pages, existing_by_number, target_paths.pages_dir)

    merged_payload = dict(existing_payload)
    merged_payload["pages"] = merged_pages
    merged_summary = {
        "page_count": len(merged_pages),
        "sources": count_page_sources(merged_pages),
    }
    if target_paths.reviewed_layout_path.exists():
        merged_summary["reviewed_layout"] = {
            "path": str(target_paths.reviewed_layout_path),
            "page_count": len(merged_pages),
            "apply_mode": REVIEWED_LAYOUT_APPLY_MODE,
        }
    merged_payload["summary"] = merged_summary
    write_json(staged_paths.ocr_json_path, merged_payload)
    write_merged_markdown(staged_paths.ocr_markdown_path, merged_pages)
    merge_fallback_page_artifacts(staged_paths, partial_paths, existing_pages, page_number)


def merge_fallback_page_artifacts(
    staged_paths: Any,
    partial_paths: Any,
    existing_pages: list[Any],
    page_number: int,
) -> None:
    partial_payload = (
        load_json(partial_paths.ocr_fallback_path)
        if partial_paths.ocr_fallback_path.exists()
        else {}
    )
    partial_pages = pages_by_number(partial_payload if isinstance(partial_payload, dict) else {})
    partial_page = partial_pages.get(page_number)
    partial_fallback_dir = partial_paths.fallback_dir / f"page-{page_number:04d}"
    staged_fallback_dir = staged_paths.fallback_dir / f"page-{page_number:04d}"
    remove_paths([staged_fallback_dir])
    if partial_fallback_dir.exists():
        copy_page_dir(partial_fallback_dir, staged_fallback_dir)

    existing_payload = (
        load_json(staged_paths.ocr_fallback_path)
        if staged_paths.ocr_fallback_path.exists()
        else {}
    )
    existing_by_number = pages_by_number(existing_payload if isinstance(existing_payload, dict) else {})
    if partial_page is None:
        existing_by_number.pop(page_number, None)
    else:
        rewrite_path_keys(
            partial_page,
            ("page_path",),
            str(partial_paths.run_dir),
            str(staged_paths.run_dir),
        )
        chunks = partial_page.get("chunks")
        if isinstance(chunks, list):
            for chunk in chunks:
                if isinstance(chunk, dict):
                    rewrite_path_keys(
                        chunk,
                        ("crop_path", "text_path"),
                        str(partial_paths.run_dir),
                        str(staged_paths.run_dir),
                    )
        existing_by_number[page_number] = partial_page

    merged_pages = ordered_page_records(existing_pages, existing_by_number, staged_paths.pages_dir)
    if not merged_pages:
        remove_paths([staged_paths.ocr_fallback_path, staged_paths.fallback_dir])
        return
    write_json(
        staged_paths.ocr_fallback_path,
        {"pages": merged_pages, "summary": {"page_count": len(merged_pages)}},
    )


def copy_page_dir(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing staged page directory: {source}")
    copy_path(source, target)
