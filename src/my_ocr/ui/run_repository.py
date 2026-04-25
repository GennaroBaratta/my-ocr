from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from my_ocr.application.artifacts.run_paths import RunPaths
from my_ocr.domain.layout import normalize_bbox
from my_ocr.domain.page_identity import (
    infer_page_number,
    iter_payload_pages,
    page_numbers_by_index,
    paths_match,
    payload_page_index,
    resolve_page_path,
)
from my_ocr.domain.review_layout import (
    build_review_layout_payload,
    load_review_layout_payload,
    save_review_layout_payload,
)

from .image_utils import get_image_size
from .session import BoundingBox, PageData, RecentRunSummary


@dataclass(frozen=True)
class LoadedRunData:
    run_id: str
    run_paths: RunPaths
    pages: list[PageData]
    ocr_markdown: str
    extraction_json: dict[str, Any]


class RunRepository:
    def __init__(self, run_root: str) -> None:
        self.run_root = run_root

    def list_recent_runs(self) -> list[RecentRunSummary]:
        runs_dir = Path(self.run_root)
        recent_runs: list[RecentRunSummary] = []
        if not runs_dir.exists() or not runs_dir.is_dir():
            return recent_runs
        try:
            run_dirs = sorted(runs_dir.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True)
        except OSError:
            return recent_runs

        for run_dir in run_dirs:
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue

            meta_path = run_dir / "meta.json"
            input_path = ""
            if meta_path.exists():
                try:
                    payload = json.loads(meta_path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        raw_input_path = payload.get("input_path", "")
                        input_path = raw_input_path if isinstance(raw_input_path, str) else ""
                except (json.JSONDecodeError, OSError):
                    pass

            preds_dir = run_dir / "predictions"
            try:
                has_preds = preds_dir.exists() and any(preds_dir.iterdir())
            except OSError:
                has_preds = False

            recent_runs.append(
                RecentRunSummary(
                    run_id=run_dir.name,
                    input_path=input_path,
                    mtime=run_dir.stat().st_mtime,
                    status="extracted" if has_preds else "pending",
                )
            )

        return recent_runs

    def load_run(self, run_id: str) -> LoadedRunData:
        run_paths = RunPaths.from_named_run(run_id, run_root=self.run_root)
        page_paths = run_paths.list_page_paths()
        review_payload = self._load_review_payload(run_paths)
        ocr_payload = self._load_ocr_payload(run_paths)
        page_numbers = page_numbers_by_index(
            run_paths.run_dir,
            page_paths,
            ocr_payload,
            review_payload,
        )
        boxes_by_page = self._load_boxes(run_paths, page_paths, review_payload, ocr_payload)
        pages = [
            PageData(
                index=page_index,
                page_number=page_numbers.get(
                    page_index,
                    infer_page_number(page_path, page_index + 1),
                ),
                image_path=page_path,
                boxes=boxes_by_page.get(page_index, []),
            )
            for page_index, page_path in enumerate(page_paths)
        ]
        ocr_markdown = self._load_markdown(run_paths, ocr_payload)
        extraction_json = self._load_extraction_json(run_paths)
        return LoadedRunData(
            run_id=run_id,
            run_paths=run_paths,
            pages=pages,
            ocr_markdown=ocr_markdown,
            extraction_json=extraction_json,
        )

    def save_reviewed_layout(self, run_paths: RunPaths | None, pages: list[PageData]) -> None:
        if not run_paths:
            return

        page_paths = [page.image_path for page in pages]
        review_payload = self._load_review_payload(run_paths)
        ocr_payload = self._load_ocr_payload(run_paths)
        review_pages: list[dict[str, Any]] = []
        for page in pages:
            image_width, image_height = get_image_size(page.image_path)
            review_pages.append(
                {
                    "page_number": page.page_number,
                    "page_path": page.image_path,
                    "image_size": {"width": image_width, "height": image_height},
                    "coord_space": "pixel",
                    "source_sdk_json_path": self._review_source_sdk_json_path(
                        run_paths,
                        page,
                        page_paths,
                        review_payload,
                        ocr_payload,
                    ),
                    "blocks": [
                        {
                            "id": box.id,
                            "index": block_index,
                            "label": box.label,
                            "content": box.content,
                            "confidence": box.confidence,
                            "bbox": [
                                round(box.x),
                                round(box.y),
                                round(box.x + box.width),
                                round(box.y + box.height),
                            ],
                        }
                        for block_index, box in enumerate(page.boxes)
                    ],
                }
            )

        payload = build_review_layout_payload(review_pages, status="reviewed")
        save_review_layout_payload(run_paths.reviewed_layout_path, payload)

    def layout_profile_warning(self, run_paths: RunPaths | None) -> str | None:
        if not run_paths or not run_paths.meta_path.exists():
            return None
        try:
            payload = json.loads(run_paths.meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(payload, dict):
            return None
        diagnostics = payload.get("layout_diagnostics")
        if not isinstance(diagnostics, dict):
            return None
        warning = cast(dict[str, Any], diagnostics).get("layout_profile_warning")
        return warning if isinstance(warning, str) and warning.strip() else None

    def _load_boxes(
        self,
        run_paths: RunPaths,
        page_paths: list[str],
        review_payload: dict[str, Any] | None,
        ocr_payload: Any,
    ) -> dict[int, list[BoundingBox]]:
        if review_payload is not None:
            review_boxes = self._load_review_boxes(run_paths, review_payload, page_paths)
            if review_boxes:
                return review_boxes

        if ocr_payload is None:
            return {}

        result: dict[int, list[BoundingBox]] = {}
        for page_idx, page_layout, coord_space, page_path in self._iter_page_layouts(
            run_paths, ocr_payload, page_paths
        ):
            resolved_page_path = resolve_page_path(
                run_paths.run_dir,
                page_idx,
                page_path,
                page_paths,
            )
            image_width, image_height = (
                get_image_size(resolved_page_path) if resolved_page_path else (0, 0)
            )
            boxes: list[BoundingBox] = []
            for box_idx, block in enumerate(self._iter_blocks(page_layout)):
                bbox = self._normalize_bbox(
                    block.get("bbox_2d"),
                    image_width=image_width,
                    image_height=image_height,
                    coord_space=coord_space,
                )
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                block_index = block.get("index", box_idx)
                boxes.append(
                    BoundingBox(
                        id=f"p{page_idx}-b{block_index}",
                        page_index=page_idx,
                        x=x1,
                        y=y1,
                        width=x2 - x1,
                        height=y2 - y1,
                        label=block.get("label", "unknown"),
                        content=block.get("content", ""),
                    )
                )
            result[page_idx] = boxes
        return result

    def _load_review_boxes(
        self,
        run_paths: RunPaths,
        payload: dict[str, Any],
        page_paths: list[str],
    ) -> dict[int, list[BoundingBox]]:
        pages = payload.get("pages")
        if not isinstance(pages, list):
            return {}

        result: dict[int, list[BoundingBox]] = {}
        for fallback_idx, page in enumerate(pages):
            if not isinstance(page, dict):
                continue
            page_data = cast(dict[str, Any], page)
            page_idx = payload_page_index(run_paths.run_dir, page_data, fallback_idx, page_paths)
            page_path = page_data.get("page_path")
            resolved_page_path = resolve_page_path(
                run_paths.run_dir,
                page_idx,
                page_path if isinstance(page_path, str) else None,
                page_paths,
            )
            image_width, image_height = (
                get_image_size(resolved_page_path) if resolved_page_path else (0, 0)
            )
            blocks = page_data.get("blocks")
            if not isinstance(blocks, list):
                continue
            boxes: list[BoundingBox] = []
            for block_idx, block in enumerate(blocks):
                if not isinstance(block, dict):
                    continue
                block_data = cast(dict[str, Any], block)
                bbox = self._normalize_bbox(
                    block_data.get("bbox"),
                    image_width=image_width,
                    image_height=image_height,
                    coord_space=page_data.get("coord_space")
                    if isinstance(page_data.get("coord_space"), str)
                    else None,
                )
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                block_index = block_data.get("index", block_idx)
                boxes.append(
                    BoundingBox(
                        id=str(block_data.get("id", f"p{page_idx}-b{block_index}")),
                        page_index=page_idx,
                        x=x1,
                        y=y1,
                        width=x2 - x1,
                        height=y2 - y1,
                        label=str(block_data.get("label", "unknown")),
                        confidence=float(block_data.get("confidence", 1.0)),
                        content=str(block_data.get("content", "")),
                    )
                )
            result[page_idx] = boxes
        return result

    def _iter_page_layouts(
        self,
        run_paths: RunPaths,
        payload: Any,
        page_paths: list[str],
    ) -> list[tuple[int, Any, str | None, str | None]]:
        if isinstance(payload, dict):
            pages = payload.get("pages")
            if isinstance(pages, list):
                page_layouts: list[tuple[int, Any, str | None, str | None]] = []
                for fallback_idx, page in enumerate(pages):
                    if not isinstance(page, dict):
                        page_layouts.append((fallback_idx, page, None, None))
                        continue
                    page_data = cast(dict[str, Any], page)
                    layout = page_data.get("sdk_json")
                    if layout is None:
                        layout_path = page_data.get("sdk_json_path")
                        if isinstance(layout_path, str):
                            layout = self._load_json_path(run_paths, Path(layout_path))
                    page_path = page_data.get("page_path")
                    fallback_assessment = page_data.get("fallback_assessment")
                    coord_space = None
                    if isinstance(fallback_assessment, dict):
                        coord_value = cast(dict[str, Any], fallback_assessment).get(
                            "bbox_coord_space"
                        )
                        if isinstance(coord_value, str):
                            coord_space = coord_value
                    page_idx = payload_page_index(
                        run_paths.run_dir,
                        page_data,
                        fallback_idx,
                        page_paths,
                    )
                    page_layouts.append(
                        (
                            page_idx,
                            layout,
                            coord_space,
                            page_path if isinstance(page_path, str) else None,
                        )
                    )
                return page_layouts
            return []

        if isinstance(payload, list):
            return [
                (page_idx, page_layout, None, None) for page_idx, page_layout in enumerate(payload)
            ]

        return []

    def _load_markdown(self, run_paths: RunPaths, ocr_payload: Any) -> str:
        markdown = (
            run_paths.ocr_markdown_path.read_text(encoding="utf-8")
            if run_paths.ocr_markdown_path.exists()
            else ""
        )
        if markdown.strip():
            return markdown
        return self._extract_markdown_from_ocr_payload(ocr_payload)

    def _load_extraction_json(self, run_paths: RunPaths) -> dict[str, Any]:
        if not run_paths.canonical_prediction_path.exists():
            return {}
        try:
            return cast(
                dict[str, Any],
                json.loads(run_paths.canonical_prediction_path.read_text(encoding="utf-8")),
            )
        except (json.JSONDecodeError, OSError):
            return {}

    def _load_ocr_payload(self, run_paths: RunPaths) -> Any:
        if not run_paths.ocr_json_path.exists():
            return None
        try:
            return json.loads(run_paths.ocr_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _load_review_payload(self, run_paths: RunPaths) -> dict[str, Any] | None:
        return load_review_layout_payload(run_paths.reviewed_layout_path)

    def _load_json_path(self, run_paths: RunPaths, path: Path) -> Any:
        candidate_paths = [path]
        if not path.is_absolute():
            candidate_paths.append(run_paths.run_dir / path)
        for candidate in candidate_paths:
            try:
                if candidate.exists():
                    return json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
        return None

    def _iter_blocks(self, layout: Any) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                blocks.append(node)
                return
            if isinstance(node, list):
                for item in node:
                    walk(item)

        walk(layout)
        return blocks

    def _normalize_bbox(
        self,
        bbox: Any,
        *,
        image_width: int,
        image_height: int,
        coord_space: str | None,
    ) -> tuple[float, float, float, float] | None:
        if image_width > 0 and image_height > 0:
            normalized = normalize_bbox(
                bbox,
                image_width,
                image_height,
                coord_space=coord_space,
            )
            if normalized is not None:
                bbox = normalized
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = (float(value) for value in bbox)
        except (TypeError, ValueError):
            return None
        return x1, y1, x2, y2

    def _extract_markdown_from_ocr_payload(self, payload: Any) -> str:
        if not isinstance(payload, dict):
            return ""
        pages = payload.get("pages")
        if not isinstance(pages, list):
            return ""
        markdown_parts: list[str] = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            markdown = cast(dict[str, Any], page).get("markdown")
            if isinstance(markdown, str) and markdown.strip():
                markdown_parts.append(markdown.strip())
        return "\n\n".join(markdown_parts)

    def _review_source_sdk_json_path(
        self,
        run_paths: RunPaths,
        page: PageData,
        page_paths: list[str],
        review_payload: dict[str, Any] | None,
        ocr_payload: Any,
    ) -> str | None:
        reviewed_page = self._matching_payload_page(run_paths, page, page_paths, review_payload)
        if reviewed_page is not None:
            source_sdk_json_path = reviewed_page.get("source_sdk_json_path")
            if isinstance(source_sdk_json_path, str):
                return source_sdk_json_path

        ocr_page = self._matching_payload_page(run_paths, page, page_paths, ocr_payload)
        if ocr_page is not None:
            source_sdk_json_path = ocr_page.get("sdk_json_path")
            if isinstance(source_sdk_json_path, str):
                return source_sdk_json_path

        return None

    def _matching_payload_page(
        self,
        run_paths: RunPaths,
        target_page: PageData,
        page_paths: list[str],
        payload: Any,
    ) -> dict[str, Any] | None:
        for fallback_idx, page_data in iter_payload_pages(payload):
            payload_page_path = page_data.get("page_path")
            if isinstance(payload_page_path, str) and paths_match(
                run_paths.run_dir, target_page.image_path, payload_page_path
            ):
                return page_data

            payload_page_number = page_data.get("page_number")
            if (
                isinstance(payload_page_number, int)
                and payload_page_number > 0
                and payload_page_number == target_page.page_number
            ):
                return page_data

            page_idx = payload_page_index(
                run_paths.run_dir,
                page_data,
                fallback_idx,
                page_paths,
            )
            if page_idx == target_page.index:
                return page_data

        return None
