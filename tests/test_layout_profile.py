"""Tests for the preflight layout-profile resolver.

 Covers:
- auto detection for pp_doclayout_formula (labels include "formula")
- auto detection for pp_doclayout_split_formula (labels include display/inline_formula)
- unknown label set → passthrough diagnostics without overlay
- explicit override mismatch → RuntimeError
- explicit override that matches auto-detected profile → accepted
- diagnostics and overlays are returned correctly
- Workflow entrypoints forward layout_profile to run_ocr / prepare_review_artifacts
- CLI parser exposes --layout-profile with correct choices
- AppState carries layout_profile field (default "auto")
- write_run_metadata persists layout_diagnostics when present
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import free_doc_extract.layout_profile as lp_mod
from free_doc_extract.layout_profile import resolve_layout_profile


# ── Helpers ──────────────────────────────────────────────────────────────────


def _fake_auto_config(labels: list[str]) -> MagicMock:
    """Return a mock that imitates transformers.AutoConfig with the given id2label."""
    cfg = MagicMock()
    cfg.id2label = {i: lbl for i, lbl in enumerate(labels)}
    return cfg


def _make_config_yaml(tmp_path: Path, *, model_dir: str = "some/model") -> Path:
    config = tmp_path / "local.yaml"
    config.write_text(
        f"""\
pipeline:
  maas:
    enabled: false
  layout:
    model_dir: {model_dir}
    label_task_mapping:
      text: [text]
      table: [table]
      formula: [formula, display_formula, inline_formula]
  result_formatter:
    label_visualization_mapping:
      text: [text]
      table: [table]
      formula: [formula, display_formula, inline_formula]
""",
        encoding="utf-8",
    )
    return config


# ── Unit tests: profile detection from label sets ─────────────────────────────


def test_auto_detects_pp_doclayout_formula(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _make_config_yaml(tmp_path)
    labels = ["formula", "text", "table", "abstract"]

    with patch.object(lp_mod, "_load_model_labels", return_value=labels):
        dotted, diagnostics = resolve_layout_profile(config, "auto")

    assert diagnostics["layout_profile_selected"] == "pp_doclayout_formula"
    assert diagnostics["layout_profile_source"] == "auto"
    assert diagnostics["layout_profile_status"] == "applied"
    assert diagnostics["layout_profile_requested"] == "auto"
    assert diagnostics["layout_model_labels"] == labels
    assert dotted["pipeline.layout.label_task_mapping"]["formula"] == ["formula"]
    assert dotted["pipeline.result_formatter.label_visualization_mapping"]["formula"] == ["formula"]


def test_auto_detects_pp_doclayout_split_formula(tmp_path: Path) -> None:
    config = _make_config_yaml(tmp_path)
    labels = ["display_formula", "inline_formula", "text", "table"]

    with patch.object(lp_mod, "_load_model_labels", return_value=labels):
        dotted, diagnostics = resolve_layout_profile(config, "auto")

    assert diagnostics["layout_profile_selected"] == "pp_doclayout_split_formula"
    assert diagnostics["layout_profile_source"] == "auto"
    assert diagnostics["layout_profile_status"] == "applied"
    assert dotted["pipeline.layout.label_task_mapping"]["formula"] == [
        "display_formula",
        "inline_formula",
    ]
    assert dotted["pipeline.result_formatter.label_visualization_mapping"]["formula"] == [
        "display_formula",
        "inline_formula",
    ]


def test_auto_passthrough_on_unknown_label_set(tmp_path: Path) -> None:
    config = _make_config_yaml(tmp_path)
    labels = ["text", "table", "chart", "image"]

    with patch.object(lp_mod, "_load_model_labels", return_value=labels):
        dotted, diagnostics = resolve_layout_profile(config, "auto")

    assert dotted == {}
    assert diagnostics["layout_profile_source"] == "auto"
    assert diagnostics["layout_profile_status"] == "passthrough_existing_config"
    assert diagnostics["layout_profile_requested"] == "auto"
    assert diagnostics["layout_model_labels"] == labels
    assert "Proceeding with existing config/default mappings." in diagnostics[
        "layout_profile_warning"
    ]


def test_explicit_override_matches_auto_detected(tmp_path: Path) -> None:
    config = _make_config_yaml(tmp_path)
    labels = ["formula", "text", "table"]

    with patch.object(lp_mod, "_load_model_labels", return_value=labels):
        dotted, diagnostics = resolve_layout_profile(config, "pp_doclayout_formula")

    assert diagnostics["layout_profile_selected"] == "pp_doclayout_formula"
    assert diagnostics["layout_profile_source"] == "override"
    assert diagnostics["layout_profile_status"] == "applied"
    assert diagnostics["layout_profile_requested"] == "pp_doclayout_formula"


def test_explicit_override_mismatch_fails(tmp_path: Path) -> None:
    """Requesting split_formula for a model with plain 'formula' label must raise."""
    config = _make_config_yaml(tmp_path)
    labels = ["formula", "text", "table"]

    with patch.object(lp_mod, "_load_model_labels", return_value=labels):
        with pytest.raises(RuntimeError, match="not compatible"):
            resolve_layout_profile(config, "pp_doclayout_split_formula")


def test_explicit_override_mismatch_fails_other_direction(tmp_path: Path) -> None:
    """Requesting formula for a model with display_formula / inline_formula must raise."""
    config = _make_config_yaml(tmp_path)
    labels = ["display_formula", "inline_formula", "text", "table"]

    with patch.object(lp_mod, "_load_model_labels", return_value=labels):
        with pytest.raises(RuntimeError, match="not compatible"):
            resolve_layout_profile(config, "pp_doclayout_formula")


def test_none_layout_profile_behaves_as_auto(tmp_path: Path) -> None:
    config = _make_config_yaml(tmp_path)
    labels = ["formula", "text"]

    with patch.object(lp_mod, "_load_model_labels", return_value=labels):
        dotted, diagnostics = resolve_layout_profile(config, None)

    assert diagnostics["layout_profile_source"] == "auto"
    assert diagnostics["layout_profile_selected"] == "pp_doclayout_formula"
    assert diagnostics["layout_profile_requested"] == "auto"


def test_missing_model_dir_raises(tmp_path: Path) -> None:
    config = tmp_path / "local.yaml"
    config.write_text("pipeline:\n  layout:\n    model_dir:\n", encoding="utf-8")

    dotted, diagnostics = resolve_layout_profile(config, "auto")

    assert dotted == {}
    assert diagnostics["layout_profile_status"] == "passthrough_existing_config"
    assert "model_dir is missing or empty" in diagnostics["layout_profile_warning"]


def test_missing_config_passthroughs_for_auto() -> None:
    dotted, diagnostics = resolve_layout_profile("/nonexistent/path/config.yaml", "auto")

    assert dotted == {}
    assert diagnostics["layout_profile_status"] == "passthrough_missing_config"
    assert diagnostics["layout_profile_requested"] == "auto"
    assert "does not exist" in diagnostics["layout_profile_warning"]


def test_missing_config_raises_for_override() -> None:
    with pytest.raises(FileNotFoundError, match="Config not found"):
        resolve_layout_profile("/nonexistent/path/config.yaml", "pp_doclayout_formula")


def test_unreadable_model_labels_raises(tmp_path: Path) -> None:
    config = _make_config_yaml(tmp_path)

    with patch.object(lp_mod, "_load_model_labels", side_effect=RuntimeError("HF unavailable")):
        dotted, diagnostics = resolve_layout_profile(config, "auto")

    assert dotted == {}
    assert diagnostics["layout_profile_status"] == "passthrough_unreadable_model_metadata"
    assert "HF unavailable" in diagnostics["layout_profile_warning"]


def test_unreadable_model_labels_raise_for_override(tmp_path: Path) -> None:
    config = _make_config_yaml(tmp_path)

    with patch.object(lp_mod, "_load_model_labels", side_effect=RuntimeError("HF unavailable")):
        with pytest.raises(RuntimeError, match="HF unavailable"):
            resolve_layout_profile(config, "pp_doclayout_formula")


def test_empty_id2label_raises(tmp_path: Path) -> None:
    config = _make_config_yaml(tmp_path)

    with patch.object(lp_mod, "_load_model_labels", return_value=[]):
        dotted, diagnostics = resolve_layout_profile(config, "auto")

    assert dotted == {}
    assert diagnostics["layout_profile_status"] == "passthrough_existing_config"
    assert diagnostics["layout_model_labels"] == []
    assert "did not find a built-in specialization" in diagnostics["layout_profile_warning"]


# ── Diagnostics are written into run metadata ─────────────────────────────────


def test_write_run_metadata_includes_layout_diagnostics(tmp_path: Path) -> None:
    from free_doc_extract.workflows import write_run_metadata

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "pages").mkdir()
    (run_dir / "pages" / "page-0001.png").write_bytes(b"img")

    written: dict[str, Any] = {}

    def capture_write(path: Path | str, payload: Any) -> None:
        written[str(path)] = payload

    diagnostics = {
        "layout_profile_requested": "auto",
        "layout_profile_selected": "pp_doclayout_formula",
        "layout_profile_source": "auto",
        "layout_profile_status": "applied",
        "layout_model_labels": ["formula", "text", "table"],
    }
    write_run_metadata(
        run_dir,
        "input.pdf",
        [str(run_dir / "pages" / "page-0001.png")],
        {
            "raw_dir": str(run_dir / "ocr_raw"),
            "config_path": "config/local.yaml",
            "layout_device": "cpu",
            "layout_diagnostics": diagnostics,
        },
        write_json_fn=capture_write,
    )

    meta_key = str(run_dir / "meta.json")
    assert meta_key in written
    meta = written[meta_key]
    assert meta["layout_diagnostics"] == diagnostics


def test_write_run_metadata_omits_layout_diagnostics_when_absent(tmp_path: Path) -> None:
    from free_doc_extract.workflows import write_run_metadata

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "pages").mkdir()
    (run_dir / "pages" / "page-0001.png").write_bytes(b"img")

    written: dict[str, Any] = {}

    def capture_write(path: Path | str, payload: Any) -> None:
        written[str(path)] = payload

    write_run_metadata(
        run_dir,
        "input.pdf",
        [str(run_dir / "pages" / "page-0001.png")],
        {
            "raw_dir": str(run_dir / "ocr_raw"),
            "config_path": "config/local.yaml",
            "layout_device": "cpu",
        },
        write_json_fn=capture_write,
    )

    meta = written[str(run_dir / "meta.json")]
    assert "layout_diagnostics" not in meta


# ── Workflow tests: layout_profile is forwarded ───────────────────────────────


def test_run_ocr_workflow_forwards_layout_profile(tmp_path: Path) -> None:
    from free_doc_extract.workflows import run_ocr_workflow
    from tests.support import normalize_to_single_page, write_basic_ocr_outputs

    run_root = tmp_path / "runs"
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF-1.4")
    captured: dict[str, Any] = {}

    def fake_run_ocr(_pages, run_dir_arg, *, config_path, layout_device, layout_profile, **_kw):
        captured["layout_profile"] = layout_profile
        return {
            **write_basic_ocr_outputs(Path(run_dir_arg)),
            "config_path": config_path,
            "layout_device": layout_device,
        }

    run_ocr_workflow(
        str(source),
        run="lp-fwd",
        run_root=str(run_root),
        normalize_document_fn=normalize_to_single_page,
        run_ocr_fn=fake_run_ocr,
        layout_profile="pp_doclayout_formula",
    )

    assert captured["layout_profile"] == "pp_doclayout_formula"


def test_prepare_review_workflow_forwards_layout_profile(tmp_path: Path) -> None:
    from free_doc_extract.workflows import prepare_review_workflow
    from tests.support import normalize_to_single_page

    run_root = tmp_path / "runs"
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF-1.4")
    captured: dict[str, Any] = {}

    def fake_prepare_review(_pages, run_dir_arg, *, config_path, layout_device, layout_profile, **_kw):
        captured["layout_profile"] = layout_profile
        run_dir_path = Path(run_dir_arg)
        run_dir_path.mkdir(parents=True, exist_ok=True)
        layout_path = run_dir_path / "reviewed_layout.json"
        layout_path.write_text(
            json.dumps({"version": 1, "status": "prepared", "pages": [], "summary": {"page_count": 0}}),
            encoding="utf-8",
        )
        return {
            "reviewed_layout": {},
            "raw_dir": str(run_dir_path / "ocr_raw"),
            "config_path": config_path,
            "layout_device": layout_device,
            "reviewed_layout_path": str(layout_path),
        }

    prepare_review_workflow(
        str(source),
        run="lp-fwd-review",
        run_root=str(run_root),
        normalize_document_fn=normalize_to_single_page,
        prepare_review_artifacts_fn=fake_prepare_review,
        layout_profile="pp_doclayout_split_formula",
    )

    assert captured["layout_profile"] == "pp_doclayout_split_formula"


def test_run_reviewed_ocr_workflow_forwards_layout_profile(tmp_path: Path) -> None:
    from free_doc_extract.workflows import run_reviewed_ocr_workflow
    from tests.support import (
        normalize_to_single_page,
        write_basic_ocr_outputs,
        build_ocr_page,
        write_reviewed_layout,
    )

    run_root = tmp_path / "runs"
    run_dir = run_root / "lp-rev"
    page_path = Path(normalize_to_single_page("ignored", run_dir)[0])
    raw_dir = run_dir / "ocr_raw" / "page-0001"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "page-0001_model.json").write_text("{}", encoding="utf-8")
    write_reviewed_layout(run_dir, page_path=str(page_path))
    (run_dir / "meta.json").write_text(
        json.dumps({"input_path": "doc.pdf"}), encoding="utf-8"
    )
    captured: dict[str, Any] = {}

    def fake_run_ocr(
        _pages, run_dir_arg, *, config_path, layout_device, layout_profile, reviewed_layout_path, **_kw
    ):
        captured["layout_profile"] = layout_profile
        result = write_basic_ocr_outputs(
            Path(run_dir_arg),
            json_payload={
                "pages": [
                    build_ocr_page(
                        page_path=str(page_path),
                        sdk_json_path=str(raw_dir / "page-0001_model.json"),
                    )
                ],
                "summary": {"page_count": 1, "sources": {"sdk_markdown": 1}},
            },
        )
        return {
            **result,
            "config_path": config_path,
            "layout_device": layout_device,
        }

    run_reviewed_ocr_workflow(
        "lp-rev",
        run_root=str(run_root),
        run_ocr_fn=fake_run_ocr,
        layout_profile="pp_doclayout_formula",
    )

    assert captured["layout_profile"] == "pp_doclayout_formula"


# ── CLI parser tests ──────────────────────────────────────────────────────────


def test_cli_ocr_parser_accepts_layout_profile() -> None:
    from free_doc_extract.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["ocr", "input.pdf", "--layout-profile", "pp_doclayout_formula"]
    )
    assert args.layout_profile == "pp_doclayout_formula"


def test_cli_run_parser_accepts_layout_profile() -> None:
    from free_doc_extract.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["run", "input.pdf", "--layout-profile", "pp_doclayout_split_formula"]
    )
    assert args.layout_profile == "pp_doclayout_split_formula"


def test_cli_layout_profile_default_is_auto() -> None:
    from free_doc_extract.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["ocr", "input.pdf"])
    assert args.layout_profile == "auto"


def test_cli_rejects_unknown_layout_profile() -> None:
    from free_doc_extract.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["ocr", "input.pdf", "--layout-profile", "bad_profile"])


# ── AppState / settings tests ─────────────────────────────────────────────────


def test_app_state_default_layout_profile_is_auto() -> None:
    from free_doc_extract.ui.state import AppState

    state = AppState()
    assert state.layout_profile == "auto"


def test_app_state_layout_profile_can_be_overridden() -> None:
    from free_doc_extract.ui.state import AppState

    state = AppState()
    state.layout_profile = "pp_doclayout_formula"
    assert state.layout_profile == "pp_doclayout_formula"
