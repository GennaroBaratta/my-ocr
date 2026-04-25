from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from my_ocr.domain.page_identity import infer_page_number


def load_json(path: str | Path) -> Any:
    adapter = import_module("my_ocr.adapters.outbound.filesystem.json_store")
    return adapter.load_json(path)


def write_json(path: str | Path, payload: Any) -> None:
    adapter = import_module("my_ocr.adapters.outbound.filesystem.json_store")
    adapter.write_json(path, payload)


def normalize_published_ocr_artifacts(
    paths: Any, *, source_run_dir: str | Path, target_run_dir: str | Path
) -> None:
    normalize_published_ocr_json(
        paths.ocr_json_path,
        source_run_dir=source_run_dir,
        target_run_dir=target_run_dir,
    )
    normalize_published_fallback_json(
        paths.ocr_fallback_path,
        source_run_dir=source_run_dir,
        target_run_dir=target_run_dir,
    )


def normalize_published_ocr_json(
    payload_path: Path, *, source_run_dir: str | Path, target_run_dir: str | Path
) -> None:
    normalize_published_json_paths(
        payload_path,
        source_run_dir=source_run_dir,
        target_run_dir=target_run_dir,
        page_keys=("page_path", "sdk_json_path"),
    )


def normalize_published_review_json(
    payload_path: Path, *, source_run_dir: str | Path, target_run_dir: str | Path
) -> None:
    if not payload_path.exists():
        return

    payload = load_json(payload_path)
    if not isinstance(payload, dict):
        return

    source_prefix = str(source_run_dir)
    target_prefix = str(target_run_dir)
    pages = payload.get("pages")
    if isinstance(pages, list):
        for page in pages:
            if isinstance(page, dict):
                rewrite_path_keys(
                    page,
                    ("page_path", "source_sdk_json_path"),
                    source_prefix,
                    target_prefix,
                )
    write_json(payload_path, payload)


def normalize_published_fallback_json(
    payload_path: Path, *, source_run_dir: str | Path, target_run_dir: str | Path
) -> None:
    normalize_published_json_paths(
        payload_path,
        source_run_dir=source_run_dir,
        target_run_dir=target_run_dir,
        page_keys=("page_path",),
        chunk_keys=("crop_path", "text_path"),
    )


def normalize_published_json_paths(
    payload_path: Path,
    *,
    source_run_dir: str | Path,
    target_run_dir: str | Path,
    page_keys: tuple[str, ...],
    chunk_keys: tuple[str, ...] = (),
) -> None:
    if not payload_path.exists():
        return

    payload = load_json(payload_path)
    if not isinstance(payload, dict):
        return

    source_prefix = str(source_run_dir)
    target_prefix = str(target_run_dir)
    pages = payload.get("pages")
    if isinstance(pages, list):
        for page in pages:
            if not isinstance(page, dict):
                continue
            rewrite_path_keys(page, page_keys, source_prefix, target_prefix)
            chunks = page.get("chunks")
            if not chunk_keys or not isinstance(chunks, list):
                continue
            for chunk in chunks:
                if isinstance(chunk, dict):
                    rewrite_path_keys(chunk, chunk_keys, source_prefix, target_prefix)
    write_json(payload_path, payload)


def rewrite_path_keys(
    payload: dict[str, Any], keys: tuple[str, ...], source_prefix: str, target_prefix: str
) -> None:
    for key in keys:
        if key in payload:
            payload[key] = rewrite_path_value(payload.get(key), source_prefix, target_prefix)


def rewrite_path_value(value: Any, source_prefix: str, target_prefix: str) -> Any:
    if isinstance(value, str) and source_prefix in value:
        return value.replace(source_prefix, target_prefix)
    return value


def ordered_page_records(
    existing_pages: list[Any],
    pages_by_number: dict[int, dict[str, Any]],
    pages_dir: Path,
) -> list[dict[str, Any]]:
    ordered_numbers: list[int] = []
    for page in existing_pages:
        if not isinstance(page, dict):
            continue
        page_number = page.get("page_number")
        if isinstance(page_number, int) and page_number > 0 and page_number not in ordered_numbers:
            ordered_numbers.append(page_number)

    page_paths = (
        sorted(str(path) for path in pages_dir.iterdir() if path.is_file())
        if pages_dir.exists()
        else []
    )
    for page_idx, page_path in enumerate(page_paths):
        page_number = infer_page_number(page_path, page_idx + 1)
        if page_number not in ordered_numbers:
            ordered_numbers.append(page_number)
    for page_number in sorted(pages_by_number):
        if page_number not in ordered_numbers:
            ordered_numbers.append(page_number)
    return [
        pages_by_number[page_number]
        for page_number in ordered_numbers
        if page_number in pages_by_number
    ]


def pages_by_number(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return {}
    indexed: dict[int, dict[str, Any]] = {}
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_number = page.get("page_number")
        if isinstance(page_number, int) and page_number > 0:
            indexed[page_number] = page
    return indexed


def count_page_sources(pages: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for page in pages:
        source = page.get("markdown_source")
        if isinstance(source, str) and source:
            counts[source] = counts.get(source, 0) + 1
    return counts


def write_merged_markdown(path: Path, pages: list[dict[str, Any]]) -> None:
    markdown_parts = []
    for page in pages:
        markdown = page.get("markdown")
        if isinstance(markdown, str) and markdown.strip():
            markdown_parts.append(markdown.strip())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(markdown_parts), encoding="utf-8")
