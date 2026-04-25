from __future__ import annotations

from types import SimpleNamespace

from my_ocr.ui.zoom import ZOOM_MODE_FIT_WIDTH, ZOOM_MODE_MANUAL, toggle_fit_width_zoom


def test_toggle_fit_width_zoom_enables_fit_width_from_manual() -> None:
    state = SimpleNamespace(
        session=SimpleNamespace(
            zoom_mode=ZOOM_MODE_MANUAL,
            zoom_level=1.25,
            zoom_fit_width=132,
        ),
    )

    scale = toggle_fit_width_zoom(state, 100)

    assert state.session.zoom_mode == ZOOM_MODE_FIT_WIDTH
    assert state.session.zoom_level == 1.25
    assert scale == 1.0


def test_toggle_fit_width_zoom_keeps_current_scale_when_disabling_fit_width() -> None:
    state = SimpleNamespace(
        session=SimpleNamespace(
            zoom_mode=ZOOM_MODE_FIT_WIDTH,
            zoom_level=0.5,
            zoom_fit_width=132,
        ),
    )

    scale = toggle_fit_width_zoom(state, 100)

    assert state.session.zoom_mode == ZOOM_MODE_MANUAL
    assert state.session.zoom_level == 1.0
    assert scale == 1.0

