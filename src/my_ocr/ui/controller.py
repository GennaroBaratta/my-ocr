from __future__ import annotations

import asyncio
from dataclasses import dataclass
import functools

from my_ocr.application.dto import LayoutOptions, OcrOptions, RunId


@dataclass(frozen=True, slots=True)
class UiActionResult:
    run_id: str
    route: str | None = None
    warning: str | None = None


class WorkflowController:
    def __init__(self, state: object) -> None:
        self._state = state

    async def prepare_review(self, input_path: str) -> UiActionResult:
        state = self._state
        result = await asyncio.to_thread(
            functools.partial(
                state.services.workflow.prepare_review,
                input_path=input_path,
                options=_layout_options(state),
            )
        )
        run_id = str(result.snapshot.run_id)
        state.load_run(run_id)
        return UiActionResult(run_id=run_id, route=f"/review/{run_id}", warning=result.warning)

    async def run_reviewed_ocr(self, run_id: str) -> UiActionResult:
        state = self._state
        result = await asyncio.to_thread(
            functools.partial(
                state.services.workflow.run_reviewed_ocr,
                RunId(run_id),
                options=_ocr_options(state),
            )
        )
        state.load_run(str(result.snapshot.run_id))
        return UiActionResult(
            run_id=str(result.snapshot.run_id),
            route=f"/results/{result.snapshot.run_id}",
            warning=result.warning,
        )

    async def redetect_review(self, input_path: str, run_id: str) -> UiActionResult:
        state = self._state
        result = await asyncio.to_thread(
            functools.partial(
                state.services.workflow.prepare_review,
                input_path=input_path,
                run_id=RunId(run_id),
                options=_layout_options(state),
            )
        )
        state.load_run(str(result.snapshot.run_id))
        return UiActionResult(
            run_id=str(result.snapshot.run_id),
            route=f"/review/{result.snapshot.run_id}",
            warning=result.warning,
        )

    async def rerun_page_layout(self, run_id: str, page_number: int) -> UiActionResult:
        state = self._state
        result = await asyncio.to_thread(
            functools.partial(
                state.services.workflow.rerun_page_layout,
                RunId(run_id),
                page_number=page_number,
                options=_layout_options(state),
            )
        )
        state.load_run(str(result.snapshot.run_id))
        return UiActionResult(
            run_id=str(result.snapshot.run_id),
            route=f"/review/{result.snapshot.run_id}",
            warning=result.warning,
        )

    async def rerun_page_ocr(self, run_id: str, page_number: int) -> UiActionResult:
        state = self._state
        result = await asyncio.to_thread(
            functools.partial(
                state.services.workflow.rerun_page_ocr,
                RunId(run_id),
                page_number=page_number,
                options=_ocr_options(state),
            )
        )
        state.load_run(str(result.snapshot.run_id))
        return UiActionResult(run_id=str(result.snapshot.run_id), route=f"/results/{run_id}")


def _layout_options(state: object) -> LayoutOptions:
    return LayoutOptions(layout_profile=state.layout_profile)


def _ocr_options(state: object) -> OcrOptions:
    return OcrOptions(layout_profile=state.layout_profile)
