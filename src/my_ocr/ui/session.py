from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RecentRunSummary:
    run_id: str
    input_path: str
    mtime: float
    status: str


@dataclass
class BoundingBox:
    """Mutable UI view model for an editable layout box."""

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
    """Mutable UI view model for page review state, not a persistence model."""

    index: int
    page_number: int
    image_path: str
    relative_image_path: str | None = None
    boxes: list[BoundingBox] = field(default_factory=list)


@dataclass
class UiSessionState:
    recent_runs: list[RecentRunSummary] = field(default_factory=list)
    run_id: str | None = None
    run_status: str = "pending"
    current_input_path: str = ""
    pages: list[PageData] = field(default_factory=list)
    current_page_index: int = 0
    selected_box_id: str | None = None
    zoom_level: float = 1.0
    zoom_mode: str = "fit_width"
    zoom_fit_width: float | None = None
    is_adding_box: bool = False
    processing: bool = False
    progress_message: str = ""
    error_message: str | None = None
    ocr_markdown: str = ""
    ocr_json: dict[str, Any] = field(default_factory=dict)
    extraction_json: dict[str, Any] = field(default_factory=dict)
    active_result_tab: int = 0
    layout_warning: str | None = None
