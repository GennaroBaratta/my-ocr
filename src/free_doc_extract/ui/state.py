"""Central application state shared across all screens."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from free_doc_extract.ocr_fallback import normalize_bbox
from free_doc_extract.paths import RunPaths
from free_doc_extract.review_artifacts import (
    build_review_layout_payload,
    load_review_layout_payload,
    save_review_layout_payload,
)
from free_doc_extract.settings import (
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_RUN_ROOT,
)

from .image_utils import get_image_size


@dataclass
class BoundingBox:
    id: str
    page_index: int
    x: float
    y: float
    width: float
    height: float
    label: str
    confidence: float = 1.0
    content: str = ""
    selected: bool = False


@dataclass
class PageData:
    index: int
    page_number: int
    image_path: str
    boxes: list[BoundingBox] = field(default_factory=list)


class AppState:
    def __init__(self) -> None:
        # Upload
        self.recent_runs: list[dict] = []

        # Current run
        self.run_id: str | None = None
        self.run_paths: RunPaths | None = None
        self.pages: list[PageData] = []
        self.current_page_index: int = 0
        self.selected_box_id: str | None = None
        self.zoom_level: float = 1.0
        self.is_adding_box: bool = False

        # Processing
        self.processing: bool = False
        self.progress_message: str = ""
        self.error_message: str | None = None

        # Results
        self.ocr_markdown: str = ""
        self.extraction_json: dict = {}
        self.active_result_tab: int = 0

        # Settings
        self.ollama_endpoint: str = DEFAULT_OLLAMA_ENDPOINT
        self.ollama_model: str = DEFAULT_OLLAMA_MODEL
        self.run_root: str = DEFAULT_RUN_ROOT
        self.layout_profile: str = "auto"

    # ── Recent runs ─────────────────────────────────────────────────

    def load_recent_runs(self) -> None:
        runs_dir = Path(self.run_root)
        self.recent_runs.clear()
        if not runs_dir.exists() or not runs_dir.is_dir():
            return
        try:
            run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            return

        for run_dir in run_dirs:
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue
            meta_path = run_dir / "meta.json"
            input_path = ""
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    input_path = meta.get("input_path", "")
                except (json.JSONDecodeError, OSError):
                    pass
            preds_dir = run_dir / "predictions"
            has_preds = preds_dir.exists() and any(preds_dir.iterdir())
            status = "extracted" if has_preds else "pending"
            self.recent_runs.append(
                {
                    "run_id": run_dir.name,
                    "input_path": input_path,
                    "mtime": run_dir.stat().st_mtime,
                    "status": status,
                }
            )

    # ── Load run data ───────────────────────────────────────────────

    def load_run(self, run_id: str) -> None:
        self.run_id = run_id
        self.run_paths = RunPaths.from_named_run(run_id, run_root=self.run_root)
        self._load_pages()
        self._load_results()
        self.current_page_index = 0
        self.selected_box_id = None
        self.is_adding_box = False

    def _load_pages(self) -> None:
        self.pages.clear()
        if not self.run_paths:
            return
        page_paths = self.run_paths.list_page_paths()
        boxes_by_page = self._load_boxes(page_paths)
        for i, pp in enumerate(page_paths):
            self.pages.append(
                PageData(
                    index=i,
                    page_number=_infer_page_number(pp, i + 1),
                    image_path=pp,
                    boxes=boxes_by_page.get(i, []),
                )
            )

    def _load_boxes(self, page_paths: list[str]) -> dict[int, list[BoundingBox]]:
        review_payload = self._load_review_payload()
        if review_payload is not None:
            review_boxes = self._load_review_boxes(review_payload, page_paths)
            if review_boxes:
                return review_boxes

        data = self._load_ocr_payload()
        if data is None:
            return {}

        result: dict[int, list[BoundingBox]] = {}
        for page_idx, page_layout, coord_space, page_path in self._iter_page_layouts(
            data, page_paths
        ):
            resolved_page_path = self._resolve_page_path(page_idx, page_path, page_paths)
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
        self, payload: dict[str, Any], page_paths: list[str]
    ) -> dict[int, list[BoundingBox]]:
        pages = payload.get("pages")
        if not isinstance(pages, list):
            return {}

        result: dict[int, list[BoundingBox]] = {}
        for fallback_idx, page in enumerate(pages):
            if not isinstance(page, dict):
                continue
            page_data = cast(dict[str, Any], page)
            page_idx = self._payload_page_index(page_data, fallback_idx, page_paths)
            page_path = page_data.get("page_path")
            resolved_page_path = self._resolve_page_path(
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
        self, payload: Any, page_paths: list[str]
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
                            layout = self._load_json_path(Path(layout_path))
                    page_path = page_data.get("page_path")
                    fallback_assessment = page_data.get("fallback_assessment")
                    coord_space = None
                    if isinstance(fallback_assessment, dict):
                        coord_value = cast(dict[str, Any], fallback_assessment).get(
                            "bbox_coord_space"
                        )
                        if isinstance(coord_value, str):
                            coord_space = coord_value
                    page_idx = self._payload_page_index(page_data, fallback_idx, page_paths)
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

    def _load_ocr_payload(self) -> Any:
        if not self.run_paths or not self.run_paths.ocr_json_path.exists():
            return None
        try:
            return json.loads(self.run_paths.ocr_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _load_review_payload(self) -> dict[str, Any] | None:
        if not self.run_paths:
            return None
        return load_review_layout_payload(self.run_paths.reviewed_layout_path)

    def _load_json_path(self, path: Path) -> Any:
        candidate_paths = [path]
        if self.run_paths and not path.is_absolute():
            candidate_paths.append(self.run_paths.run_dir / path)
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

    def _load_results(self) -> None:
        if not self.run_paths:
            return
        md_path = self.run_paths.ocr_markdown_path
        self.ocr_markdown = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
        if not self.ocr_markdown.strip():
            self.ocr_markdown = self._extract_markdown_from_ocr_payload(self._load_ocr_payload())
        canonical = self.run_paths.canonical_prediction_path
        if canonical.exists():
            try:
                self.extraction_json = json.loads(canonical.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.extraction_json = {}
        else:
            self.extraction_json = {}

    def layout_profile_warning(self) -> str | None:
        if not self.run_paths or not self.run_paths.meta_path.exists():
            return None
        try:
            payload = json.loads(self.run_paths.meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(payload, dict):
            return None
        diagnostics = payload.get("layout_diagnostics")
        if not isinstance(diagnostics, dict):
            return None
        warning = cast(dict[str, Any], diagnostics).get("layout_profile_warning")
        return warning if isinstance(warning, str) and warning.strip() else None

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

    def _payload_page_index(
        self,
        page_data: dict[str, Any],
        fallback_idx: int,
        page_paths: list[str],
    ) -> int:
        payload_page_path = page_data.get("page_path")
        if isinstance(payload_page_path, str):
            page_idx = self._find_page_index_by_path(payload_page_path, page_paths)
            if page_idx is not None:
                return page_idx

        page_number = page_data.get("page_number")
        if isinstance(page_number, int) and page_number > 0:
            page_idx = self._find_page_index_by_number(page_number, page_paths)
            if page_idx is not None:
                return page_idx

        return fallback_idx

    def _find_page_index_by_path(self, page_path: str, page_paths: list[str]) -> int | None:
        lookup_strings: set[str] = {page_path}
        lookup_names: set[str] = {Path(page_path).name}

        candidate = Path(page_path)
        lookup_strings.add(str(candidate))
        if self.run_paths and not candidate.is_absolute():
            lookup_strings.add(str(self.run_paths.run_dir / candidate))

        for page_idx, candidate_path in enumerate(page_paths):
            candidate_name = Path(candidate_path).name
            if candidate_path in lookup_strings or candidate_name in lookup_names:
                return page_idx

        return None

    def _find_page_index_by_number(self, page_number: int, page_paths: list[str]) -> int | None:
        for page_idx, page_path in enumerate(page_paths):
            if _infer_page_number(page_path, page_idx + 1) == page_number:
                return page_idx
        return None

    def _resolve_page_path(
        self,
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
        if self.run_paths:
            candidate = self.run_paths.run_dir / path
            if candidate.exists():
                return str(candidate)
        return None

    # ── Box operations ──────────────────────────────────────────────

    def select_box(self, box_id: str | None) -> None:
        self.selected_box_id = box_id
        for page in self.pages:
            for box in page.boxes:
                box.selected = box.id == box_id

    def get_selected_box(self) -> BoundingBox | None:
        if not self.selected_box_id:
            return None
        for page in self.pages:
            for box in page.boxes:
                if box.id == self.selected_box_id:
                    return box
        return None

    def update_box(self, box_id: str, **kwargs: object) -> None:
        for page in self.pages:
            for box in page.boxes:
                if box.id == box_id:
                    for k, v in kwargs.items():
                        setattr(box, k, v)
                    self.save_reviewed_layout()
                    return

    def add_box_to_current_page(
        self,
        label: str = "text",
        x: float | None = None,
        y: float | None = None,
        width: float | None = None,
        height: float | None = None,
    ) -> str | None:
        page = self.current_page
        if not page:
            return None
        image_width, image_height = get_image_size(page.image_path)
        default_w, default_h = 240, 80
        if image_width:
            default_w = min(default_w, image_width)
        if image_height:
            default_h = min(default_h, image_height)
        
        final_x = x if x is not None else (max(0, (image_width - default_w) // 2) if image_width else 0)
        final_y = y if y is not None else (max(0, (image_height - default_h) // 2) if image_height else 0)
        final_w = width if width is not None else default_w
        final_h = height if height is not None else default_h

        box_id = self._next_box_id(page.index)
        page.boxes.append(
            BoundingBox(
                id=box_id,
                page_index=page.index,
                x=final_x,
                y=final_y,
                width=final_w,
                height=final_h,
                label=label,
                confidence=1.0,
            )
        )
        self.save_reviewed_layout()
        return box_id

    def _next_box_id(self, page_index: int) -> str:
        existing = {b.id for p in self.pages for b in p.boxes}
        i = len(self.pages[page_index].boxes)
        while True:
            candidate = f"p{page_index}-u{i}"
            if candidate not in existing:
                return candidate
            i += 1

    def remove_box(self, box_id: str) -> None:
        for page in self.pages:
            page.boxes = [b for b in page.boxes if b.id != box_id]
        if self.selected_box_id == box_id:
            self.selected_box_id = None
        self.save_reviewed_layout()

    def save_reviewed_layout(self) -> None:
        if not self.run_paths:
            return

        review_pages: list[dict[str, Any]] = []
        for page in self.pages:
            image_width, image_height = get_image_size(page.image_path)
            review_pages.append(
                {
                    "page_number": page.page_number,
                    "page_path": page.image_path,
                    "image_size": {"width": image_width, "height": image_height},
                    "coord_space": "pixel",
                    "source_sdk_json_path": self._review_source_sdk_json_path(page.index),
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
        save_review_layout_payload(self.run_paths.reviewed_layout_path, payload)

    @property
    def current_page(self) -> PageData | None:
        if 0 <= self.current_page_index < len(self.pages):
            return self.pages[self.current_page_index]
        return None

    def page_number_for_index(self, page_index: int) -> int:
        if 0 <= page_index < len(self.pages):
            return self.pages[page_index].page_number
        return page_index + 1

    @property
    def current_page_number(self) -> int:
        return self.page_number_for_index(self.current_page_index)

    def _review_source_sdk_json_path(self, page_index: int) -> str | None:
        payload = self._load_review_payload()
        if payload is not None:
            pages = payload.get("pages")
            if isinstance(pages, list) and 0 <= page_index < len(pages):
                page = pages[page_index]
                if isinstance(page, dict):
                    source_sdk_json_path = page.get("source_sdk_json_path")
                    if isinstance(source_sdk_json_path, str):
                        return source_sdk_json_path

        ocr_payload = self._load_ocr_payload()
        if not isinstance(ocr_payload, dict):
            return None
        pages = ocr_payload.get("pages")
        if not isinstance(pages, list) or not (0 <= page_index < len(pages)):
            return None
        page = pages[page_index]
        if not isinstance(page, dict):
            return None
        source_sdk_json_path = page.get("sdk_json_path")
        return source_sdk_json_path if isinstance(source_sdk_json_path, str) else None


def _infer_page_number(page_path: str | Path, fallback_number: int) -> int:
    match = re.fullmatch(r"page-(\d+)", Path(page_path).stem)
    if match is None:
        return fallback_number
    return int(match.group(1))
