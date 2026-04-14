"""Central application state shared across all screens."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from free_doc_extract.paths import RunPaths
from free_doc_extract.settings import (
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_RUN_ROOT,
)


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
        if not runs_dir.exists():
            return
        for run_dir in sorted(
            runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
        ):
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
        boxes_by_page = self._load_boxes()
        for i, pp in enumerate(page_paths):
            self.pages.append(
                PageData(index=i, image_path=pp, boxes=boxes_by_page.get(i, []))
            )

    def _load_boxes(self) -> dict[int, list[BoundingBox]]:
        if not self.run_paths or not self.run_paths.ocr_json_path.exists():
            return {}
        try:
            data = json.loads(
                self.run_paths.ocr_json_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            return {}
        result: dict[int, list[BoundingBox]] = {}
        for page_idx, page_block_groups in enumerate(data):
            boxes: list[BoundingBox] = []
            for block_group in page_block_groups:
                for block in block_group:
                    bbox = block.get("bbox_2d", [0, 0, 0, 0])
                    x1, y1, x2, y2 = bbox
                    boxes.append(
                        BoundingBox(
                            id=f"p{page_idx}-b{block.get('index', 0)}",
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

    def _load_results(self) -> None:
        if not self.run_paths:
            return
        md_path = self.run_paths.ocr_markdown_path
        self.ocr_markdown = (
            md_path.read_text(encoding="utf-8") if md_path.exists() else ""
        )
        canonical = self.run_paths.canonical_prediction_path
        if canonical.exists():
            try:
                self.extraction_json = json.loads(
                    canonical.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                self.extraction_json = {}
        else:
            self.extraction_json = {}

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
