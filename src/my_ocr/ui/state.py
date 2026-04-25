"""Central application state shared across all screens."""

from __future__ import annotations

from my_ocr.pipeline.errors import UnsupportedRunSchema
from my_ocr.models import RunId
from my_ocr.bootstrap import (
    BackendServices,
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_RUN_ROOT,
    build_backend_services,
)

from .image_utils import get_image_size
from .mappers import page_data_to_review_layout, pages_from_snapshot, recent_run_summary
from .session import BoundingBox, PageData, UiSessionState


class AppState:
    def __init__(self, services: BackendServices | None = None) -> None:
        self.session = UiSessionState()
        self.ollama_endpoint: str = DEFAULT_OLLAMA_ENDPOINT
        self.ollama_model: str = DEFAULT_OLLAMA_MODEL
        self._run_root: str = DEFAULT_RUN_ROOT
        self.services = services or build_backend_services(self._run_root)
        self.layout_profile: str = "auto"
        from .controller import WorkflowController

        self.controller = WorkflowController(self)

    @property
    def run_root(self) -> str:
        return self._run_root

    @run_root.setter
    def run_root(self, value: str) -> None:
        self._run_root = value
        self.services = build_backend_services(value)
        self.controller = type(self.controller)(self)

    def load_recent_runs(self) -> None:
        self.session.recent_runs = [
            recent_run_summary(record) for record in self.services.read_model.list_recent_runs()
        ]

    def load_run(self, run_id: str) -> None:
        try:
            snapshot = self.services.read_model.load_run(run_id)
        except UnsupportedRunSchema as exc:
            self.session.run_id = run_id
            self.session.unsupported_run_message = str(exc)
            self.session.pages = []
            self.session.current_input_path = ""
            self.session.ocr_markdown = ""
            self.session.ocr_json = {}
            self.session.extraction_json = {}
            return
        self.session.unsupported_run_message = None
        self.session.run_id = str(snapshot.run_id)
        self.session.current_input_path = snapshot.manifest.input.path
        self.session.pages = pages_from_snapshot(snapshot)
        self.session.ocr_markdown = snapshot.ocr_result.markdown if snapshot.ocr_result else ""
        self.session.ocr_json = (
            snapshot.ocr_result.model_dump(mode="json") if snapshot.ocr_result else {}
        )
        canonical = snapshot.extraction.get("canonical")
        self.session.extraction_json = canonical if isinstance(canonical, dict) else {}
        self.session.layout_warning = snapshot.manifest.diagnostics.layout.warning
        self.session.current_page_index = 0
        self.session.selected_box_id = None
        self.session.is_adding_box = False

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
        existing = {box.id for page in self.session.pages for box in page.boxes}
        index = len(self.session.pages[page_index].boxes)
        while True:
            candidate = f"p{page_index}-u{index}"
            if candidate not in existing:
                return candidate
            index += 1

    def remove_box(self, box_id: str) -> None:
        for page in self.session.pages:
            page.boxes = [box for box in page.boxes if box.id != box_id]
        if self.session.selected_box_id == box_id:
            self.session.selected_box_id = None
        self.save_reviewed_layout()

    def save_reviewed_layout(self) -> None:
        if not self.session.run_id:
            return
        self.services.workflow.save_review_layout(
            RunId(self.session.run_id),
            page_data_to_review_layout(self.session.pages),
        )

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

