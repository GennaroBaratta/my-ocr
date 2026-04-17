"""Central application state shared across all screens."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from free_doc_extract.ocr_fallback import normalize_bbox
from free_doc_extract.paths import RunPaths
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

    def _load_pages(self) -> None:
        self.pages.clear()
        if not self.run_paths:
            return
        page_paths = self.run_paths.list_page_paths()
        boxes_by_page = self._load_boxes(page_paths)
        for i, pp in enumerate(page_paths):
            self.pages.append(PageData(index=i, image_path=pp, boxes=boxes_by_page.get(i, [])))

    def _load_boxes(self, page_paths: list[str]) -> dict[int, list[BoundingBox]]:
        data = self._load_ocr_payload()
        if data is None:
            return {}

        result: dict[int, list[BoundingBox]] = {}
        for page_idx, page_layout, coord_space, page_path in self._iter_page_layouts(data):
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

    def _iter_page_layouts(self, payload: Any) -> list[tuple[int, Any, str | None, str | None]]:
        if isinstance(payload, dict):
            pages = payload.get("pages")
            if isinstance(pages, list):
                page_layouts: list[tuple[int, Any, str | None, str | None]] = []
                for fallback_idx, page in enumerate(pages):
                    if not isinstance(page, dict):
                        page_layouts.append((fallback_idx, page, None, None))
                        continue
                    page_data = cast(dict[str, Any], page)
                    page_number = page_data.get("page_number")
                    page_idx = (
                        page_number - 1
                        if isinstance(page_number, int) and page_number > 0
                        else fallback_idx
                    )
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
                    return

    def remove_box(self, box_id: str) -> None:
        for page in self.pages:
            page.boxes = [b for b in page.boxes if b.id != box_id]
        if self.selected_box_id == box_id:
            self.selected_box_id = None

    @property
    def current_page(self) -> PageData | None:
        if 0 <= self.current_page_index < len(self.pages):
            return self.pages[self.current_page_index]
        return None
