from __future__ import annotations

from my_ocr.domain import RunId

from .image_utils import get_image_size
from .mappers import page_data_to_review_layout
from .session import BoundingBox


class ReviewController:
    def __init__(self, state: object) -> None:
        self._state = state

    def add_box_to_current_page(
        self,
        label: str = "text",
        x: float | None = None,
        y: float | None = None,
        width: float | None = None,
        height: float | None = None,
    ) -> str | None:
        page = self._state.current_page
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
        self.save_review_layout()
        return box_id

    def remove_box(self, box_id: str) -> None:
        for page in self._state.session.pages:
            page.boxes = [box for box in page.boxes if box.id != box_id]
        if self._state.session.selected_box_id == box_id:
            self._state.session.selected_box_id = None
        self.save_review_layout()

    def save_review_layout(self) -> None:
        run_id = self._state.session.run_id
        if not run_id:
            return
        self._state.services.workflow.save_review_layout(
            RunId(run_id),
            page_data_to_review_layout(self._state.session.pages),
        )

    def _next_box_id(self, page_index: int) -> str:
        existing = {box.id for page in self._state.session.pages for box in page.boxes}
        index = len(self._state.session.pages[page_index].boxes)
        while True:
            candidate = f"p{page_index}-u{index}"
            if candidate not in existing:
                return candidate
            index += 1
