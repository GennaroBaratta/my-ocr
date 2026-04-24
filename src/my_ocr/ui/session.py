from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from my_ocr.paths import RunPaths


@dataclass(frozen=True, slots=True)
class RecentRunSummary:
    run_id: str
    input_path: str
    mtime: float
    status: str


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


@dataclass
class UiSessionState:
    recent_runs: list[RecentRunSummary] = field(default_factory=list)
    run_id: str | None = None
    run_paths: RunPaths | None = None
    pages: list[PageData] = field(default_factory=list)
    current_page_index: int = 0
    selected_box_id: str | None = None
    zoom_level: float = 1.0
    is_adding_box: bool = False
    processing: bool = False
    progress_message: str = ""
    error_message: str | None = None
    ocr_markdown: str = ""
    extraction_json: dict[str, Any] = field(default_factory=dict)
    active_result_tab: int = 0
