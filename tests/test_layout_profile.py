from __future__ import annotations

from pathlib import Path

import pytest

from my_ocr.cli import build_parser
from my_ocr.ocr.layout_profile import resolve_layout_profile
from my_ocr.ui.state import AppState


def test_missing_config_passthroughs_for_auto() -> None:
    dotted, diagnostics = resolve_layout_profile("missing.yaml", "auto")

    assert dotted == {}
    assert diagnostics["layout_profile_status"] == "passthrough_missing_config"


def test_missing_config_raises_for_override() -> None:
    with pytest.raises(FileNotFoundError):
        resolve_layout_profile("missing.yaml", "pp_doclayout_formula")


def test_explicit_layout_profile_is_forwarded_by_cli_parser() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "ocr",
            "input.pdf",
            "--layout-profile",
            "pp_doclayout_formula",
        ]
    )

    assert args.layout_profile == "pp_doclayout_formula"


def test_cli_layout_profile_default_is_auto() -> None:
    parser = build_parser()

    args = parser.parse_args(["run", "input.pdf"])

    assert args.layout_profile == "auto"


def test_cli_rejects_unknown_layout_profile() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["ocr", "input.pdf", "--layout-profile", "unknown"])


def test_app_state_default_layout_profile_is_auto() -> None:
    assert AppState().layout_profile == "auto"


def test_app_state_layout_profile_can_be_overridden(tmp_path: Path) -> None:
    state = AppState()
    state.run_root = str(tmp_path)

    state.layout_profile = "pp_doclayout_formula"

    assert state.layout_profile == "pp_doclayout_formula"
