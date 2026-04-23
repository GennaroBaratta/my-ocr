from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast


def infer_page_number(page_path: str | Path, fallback_number: int) -> int:
    match = re.fullmatch(r"page-(\d+)", Path(page_path).stem)
    if match is None:
        return fallback_number
    return int(match.group(1))


def iter_payload_pages(payload: Any) -> list[tuple[int, dict[str, Any]]]:
    if not isinstance(payload, dict):
        return []
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return []
    result: list[tuple[int, dict[str, Any]]] = []
    for fallback_idx, page in enumerate(pages):
        if isinstance(page, dict):
            result.append((fallback_idx, cast(dict[str, Any], page)))
    return result


def page_numbers_by_index(
    run_dir: str | Path,
    page_paths: list[str],
    *payloads: Any,
) -> dict[int, int]:
    page_numbers = {
        page_idx: infer_page_number(page_path, page_idx + 1)
        for page_idx, page_path in enumerate(page_paths)
    }
    for payload in payloads:
        for fallback_idx, page_data in iter_payload_pages(payload):
            page_number = page_data.get("page_number")
            if not isinstance(page_number, int) or page_number <= 0:
                continue
            page_idx = payload_page_index(run_dir, page_data, fallback_idx, page_paths)
            page_numbers[page_idx] = page_number
    return page_numbers


def resolve_page_path_for_number(
    run_dir: str | Path,
    page_paths: list[str],
    page_number: int,
    *payloads: Any,
) -> str:
    for payload in payloads:
        for fallback_idx, page_data in iter_payload_pages(payload):
            payload_page_number = page_data.get("page_number")
            if payload_page_number != page_number:
                continue
            payload_page_path = page_data.get("page_path")
            page_idx = payload_page_index(run_dir, page_data, fallback_idx, page_paths)
            resolved = resolve_page_path(
                run_dir,
                page_idx,
                payload_page_path if isinstance(payload_page_path, str) else None,
                page_paths,
            )
            if resolved is not None:
                return resolved

    for page_idx, page_path in enumerate(page_paths):
        if infer_page_number(page_path, page_idx + 1) == page_number:
            return page_path

    raise FileNotFoundError(f"Page {page_number} not found")


def payload_page_index(
    run_dir: str | Path,
    page_data: dict[str, Any],
    fallback_idx: int,
    page_paths: list[str],
) -> int:
    payload_page_path = page_data.get("page_path")
    if isinstance(payload_page_path, str):
        page_idx = find_page_index_by_path(run_dir, payload_page_path, page_paths)
        if page_idx is not None:
            return page_idx

    page_number = page_data.get("page_number")
    if isinstance(page_number, int) and page_number > 0:
        page_idx = find_page_index_by_number(page_number, page_paths)
        if page_idx is not None:
            return page_idx

    return fallback_idx


def find_page_index_by_path(
    run_dir: str | Path,
    page_path: str,
    page_paths: list[str],
) -> int | None:
    lookup_strings = path_candidates(run_dir, page_path)
    lookup_names = {Path(page_path).name}

    for page_idx, candidate_path in enumerate(page_paths):
        candidate_name = Path(candidate_path).name
        if candidate_path in lookup_strings or candidate_name in lookup_names:
            return page_idx

    return None


def find_page_index_by_number(page_number: int, page_paths: list[str]) -> int | None:
    for page_idx, page_path in enumerate(page_paths):
        if infer_page_number(page_path, page_idx + 1) == page_number:
            return page_idx
    return None


def resolve_page_path(
    run_dir: str | Path,
    page_idx: int,
    payload_page_path: str | None,
    page_paths: list[str],
) -> str | None:
    if 0 <= page_idx < len(page_paths):
        return page_paths[page_idx]
    if not payload_page_path:
        return None

    path = Path(payload_page_path)
    if path.exists():
        return str(path)

    candidate = Path(run_dir) / path
    if candidate.exists():
        return str(candidate)

    return None


def paths_match(run_dir: str | Path, left: str, right: str) -> bool:
    return bool(path_candidates(run_dir, left).intersection(path_candidates(run_dir, right)))


def path_candidates(run_dir: str | Path, raw_path: str) -> set[str]:
    candidate = Path(raw_path)
    candidates = {raw_path, str(candidate), candidate.name}
    if not candidate.is_absolute():
        candidates.add(str(Path(run_dir) / candidate))
    return candidates
