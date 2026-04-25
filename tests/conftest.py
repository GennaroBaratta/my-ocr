"""Shared pytest configuration for the test suite.

Provides an autouse fixture that stubs out the preflight layout-profile
resolver so that tests do not need a real Hugging Face connection or a
downloaded model checkpoint.

Tests in ``test_layout_profile.py`` bypass this by patching
``my_ocr.ocr.layout_profile._load_model_labels`` directly inside each
test case.
"""

from __future__ import annotations

from typing import Any

import pytest


_STUB_LAYOUT_DOTTED: dict[str, Any] = {
    "pipeline.layout.label_task_mapping": {
        "text": ["abstract", "content", "text"],
        "formula": ["formula"],
        "table": ["table"],
        "skip": ["chart", "image"],
        "abandon": ["header", "footer"],
    },
    "pipeline.result_formatter.label_visualization_mapping": {
        "formula": ["formula"],
        "table": ["table"],
        "image": ["chart", "image"],
        "text": ["abstract", "content", "text"],
    },
}

_STUB_LAYOUT_DIAGNOSTICS: dict[str, Any] = {
    "layout_profile_requested": "auto",
    "layout_profile_selected": "pp_doclayout_formula",
    "layout_profile_source": "auto",
    "layout_profile_status": "applied",
    "layout_model_labels": ["formula", "text", "table"],
}


def _stub_resolve(config_path: Any, layout_profile: Any = "auto") -> tuple[dict, dict]:  # noqa: ANN001
    return _STUB_LAYOUT_DOTTED, _STUB_LAYOUT_DIAGNOSTICS


@pytest.fixture(autouse=True)
def _stub_resolve_layout_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace resolve_layout_profile with a no-op stub for the whole suite.

    ``ocr.py`` calls the function via ``_layout_profile_mod.resolve_layout_profile``
    (module attribute), so patching the module attribute is sufficient.
    Direct tests of layout_profile logic patch ``_load_model_labels`` inside
    the layout_profile module, which this fixture does not interfere with.
    """
    import my_ocr.ocr.layout_profile as _lp_mod

    monkeypatch.setattr(_lp_mod, "resolve_layout_profile", _stub_resolve)
