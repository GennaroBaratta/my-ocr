"""Central application state shared across all screens."""

from __future__ import annotations

from my_ocr.bootstrap import (
    BackendServices,
    DEFAULT_RUN_ROOT,
    build_backend_services,
)

from .mappers import pages_from_snapshot, recent_run_summary, status_for_snapshot
from .session import BoundingBox, PageData, UiSessionState
from .zoom import ZOOM_MODE_FIT_WIDTH, ZOOM_MODE_MANUAL, clamp_zoom


class AppState:
    def __init__(self, services: BackendServices | None = None) -> None:
        self.session = UiSessionState()
        self.ollama_endpoint: str | None = None
        self.ollama_model: str | None = None
        self._run_root: str = DEFAULT_RUN_ROOT
        self.services = services or build_backend_services(self._run_root)
        self.layout_profile: str = "auto"
        from .controller import WorkflowController
        from .review_controller import ReviewController

        self.controller = WorkflowController(self)
        self.review_controller = ReviewController(self)

    @property
    def run_root(self) -> str:
        return self._run_root

    @run_root.setter
    def run_root(self, value: str) -> None:
        self._run_root = value
        self.services = build_backend_services(value)
        self.controller = type(self.controller)(self)
        self.review_controller = type(self.review_controller)(self)

    def load_recent_runs(self) -> None:
        self.session.recent_runs = [
            recent_run_summary(record) for record in self.services.read_model.list_recent_runs()
        ]

    def load_run(self, run_id: str) -> None:
        snapshot = self.services.read_model.load_run(run_id)
        self.session.run_id = str(snapshot.run_id)
        self.session.run_status = status_for_snapshot(snapshot)
        self.session.current_input_path = snapshot.manifest.input.path
        self.session.pages = pages_from_snapshot(snapshot)
        self.session.ocr_markdown = snapshot.ocr_result.markdown if snapshot.ocr_result else ""
        self.session.ocr_json = (
            snapshot.ocr_result.model_dump(mode="json") if snapshot.ocr_result else {}
        )
        canonical = snapshot.extraction.get("canonical")
        self.session.extraction_json = canonical if isinstance(canonical, dict) else {}
        self.session.layout_warning = snapshot.manifest.diagnostics.layout.warning
        self.set_current_page_index(0)
        self.select_box(None)
        self.set_add_box_mode(False)

    def load_run_preserving_page_index(self, run_id: str, page_index: int) -> None:
        self.load_run(run_id)
        self.set_current_page_index(page_index)

    def set_current_page_index(self, index: int) -> bool:
        previous = self.session.current_page_index
        if not self.session.pages:
            self.session.current_page_index = 0
        else:
            self.session.current_page_index = min(max(index, 0), len(self.session.pages) - 1)
        return self.session.current_page_index != previous

    def select_next_page(self) -> bool:
        return self.set_current_page_index(self.session.current_page_index + 1)

    def select_previous_page(self) -> bool:
        return self.set_current_page_index(self.session.current_page_index - 1)

    def set_processing(self, active: bool, message: str = "") -> None:
        self.session.processing = active
        self.session.progress_message = message if active else ""

    def set_error(self, message: str | None) -> None:
        self.session.error_message = message

    def set_add_box_mode(self, active: bool) -> None:
        self.session.is_adding_box = active
        if active:
            self.select_box(None)

    def set_zoom_level(self, level: float, mode: str | None = None) -> float:
        self.session.zoom_mode = mode if mode is not None else ZOOM_MODE_MANUAL
        self.session.zoom_level = clamp_zoom(level)
        return self.session.zoom_level

    def set_manual_zoom(self, level: float) -> float:
        return self.set_zoom_level(level, mode=ZOOM_MODE_MANUAL)

    def set_fit_width_zoom(self) -> None:
        self.session.zoom_mode = ZOOM_MODE_FIT_WIDTH

    def set_zoom_available_width(self, width: float | None) -> None:
        self.session.zoom_fit_width = max(0.0, float(width or 0))

    def set_active_result_tab(self, index: int) -> None:
        self.session.active_result_tab = max(0, index)

    def needs_review_before_results(self) -> bool:
        return self.session.run_status == "review_ready"

    def layout_profile_warning(self) -> str | None:
        return self.session.layout_warning

    def select_box(self, box_id: str | None) -> None:
        self.session.selected_box_id = box_id
        for page in self.session.pages:
            for box in page.boxes:
                box.selected = box.id == box_id

    def get_selected_box(self) -> BoundingBox | None:
        if not self.session.selected_box_id:
            return None
        for page in self.session.pages:
            for box in page.boxes:
                if box.id == self.session.selected_box_id:
                    return box
        return None

    @property
    def current_page(self) -> PageData | None:
        if 0 <= self.session.current_page_index < len(self.session.pages):
            return self.session.pages[self.session.current_page_index]
        return None

    def page_number_for_index(self, page_index: int) -> int:
        if 0 <= page_index < len(self.session.pages):
            return self.session.pages[page_index].page_number
        return page_index + 1

    @property
    def current_page_number(self) -> int:
        return self.page_number_for_index(self.session.current_page_index)
