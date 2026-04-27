from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from my_ocr.settings import InferenceProviderConfig
from my_ocr.runs.store import FilesystemRunReadModel
from my_ocr.runs.store import FilesystemRunStore
from my_ocr.use_cases.invalidation import ocr_result_updated_policy
from my_ocr.domain import ArtifactCopy, LayoutBlock, PageRef, ReviewLayout, ReviewPage, RunId
from my_ocr.domain import ProviderArtifacts
from my_ocr.domain import OcrPageResult, OcrRunResult
from my_ocr.ui.components.ollama_status import OllamaStatus
from my_ocr.ui.controller import _ocr_options
from my_ocr.ui.components.settings_dialog import _normalize_ocr_override
from my_ocr.ui.features.upload.screen import _build_inference_status_badge
from my_ocr.ui.session import BoundingBox, PageData
from my_ocr.ui.state import AppState


def test_app_state_loads_v3_run_and_saves_review_layout(tmp_path: Path) -> None:
    _seed_run(tmp_path / "runs", "demo")
    state = AppState()
    state.run_root = str(tmp_path / "runs")

    state.load_run("demo")
    box_id = state.review_controller.add_box_to_current_page(
        label="table", x=1, y=2, width=3, height=4
    )

    persisted = json.loads(
        (tmp_path / "runs" / "demo" / "layout" / "review.json").read_text(encoding="utf-8")
    )
    assert box_id
    assert state.session.run_id == "demo"
    assert state.session.pages[0].relative_image_path == "pages/page-0001.png"
    assert persisted["status"] == "reviewed"
    assert persisted["pages"][0]["image_path"] == "pages/page-0001.png"
    assert persisted["pages"][0]["blocks"][0]["label"] == "table"


def test_load_run_resets_selection_and_add_box_mode(tmp_path: Path) -> None:
    _seed_run(tmp_path / "runs", "demo")
    state = AppState()
    state.run_root = str(tmp_path / "runs")
    state.set_add_box_mode(True)
    state.set_current_page_index(4)
    state.select_box("missing")

    state.load_run("demo")

    assert state.session.is_adding_box is False
    assert state.session.current_page_index == 0
    assert state.session.selected_box_id is None


def test_page_selection_commands_clamp_to_available_pages() -> None:
    state = AppState()
    state.session.pages = [
        PageData(index=0, page_number=1, image_path="page-1.png"),
        PageData(index=1, page_number=2, image_path="page-2.png"),
        PageData(index=2, page_number=3, image_path="page-3.png"),
    ]

    assert state.set_current_page_index(10) is True
    assert state.session.current_page_index == 2
    assert state.select_next_page() is False
    assert state.session.current_page_index == 2
    assert state.select_previous_page() is True
    assert state.session.current_page_index == 1
    assert state.set_current_page_index(-4) is True
    assert state.session.current_page_index == 0
    assert state.select_previous_page() is False


def test_load_run_preserving_page_index_restores_valid_page(tmp_path: Path) -> None:
    _seed_run_with_pages(tmp_path / "runs", "demo", page_count=3)
    state = AppState()
    state.run_root = str(tmp_path / "runs")
    state.set_add_box_mode(True)
    state.select_box("missing")

    state.load_run_preserving_page_index("demo", 2)

    assert state.session.current_page_index == 2
    assert state.session.is_adding_box is False
    assert state.session.selected_box_id is None

    state.load_run_preserving_page_index("demo", 99)

    assert state.session.current_page_index == 2


def test_processing_error_zoom_and_add_box_commands() -> None:
    state = AppState()
    state.session.pages = [
        PageData(
            index=0,
            page_number=1,
            image_path="page-1.png",
            boxes=[
                BoundingBox(
                    id="box-1",
                    page_index=0,
                    x=0,
                    y=0,
                    width=1,
                    height=1,
                    label="text",
                )
            ],
        )
    ]
    state.select_box("box-1")

    state.set_processing(True, "Working")
    state.set_error("Failed")
    zoom_level = state.set_zoom_level(20.0)
    state.set_add_box_mode(True)

    assert state.session.processing is True
    assert state.session.progress_message == "Working"
    assert state.session.error_message == "Failed"
    assert zoom_level == 3.0
    assert state.session.zoom_level == 3.0
    assert state.session.zoom_mode == "manual"
    assert state.session.is_adding_box is True
    assert state.session.selected_box_id is None
    assert not state.session.pages[0].boxes[0].selected

    state.set_processing(False)
    state.set_error(None)

    assert state.session.processing is False
    assert state.session.progress_message == ""
    assert state.session.error_message is None


def test_set_zoom_level_accepts_explicit_mode() -> None:
    state = AppState()

    zoom_level = state.set_zoom_level(0.1, mode="fit_width")

    assert zoom_level == 0.25
    assert state.session.zoom_level == 0.25
    assert state.session.zoom_mode == "fit_width"


def test_read_model_loads_seeded_v3_run_without_migrating_relative_paths(
    tmp_path: Path,
) -> None:
    _seed_reviewed_v3_run(tmp_path / "runs", "demo")
    run_dir = tmp_path / "runs" / "demo"
    manifest_before = (run_dir / "run.json").read_text(encoding="utf-8")
    review_before = (run_dir / "layout" / "review.json").read_text(encoding="utf-8")

    snapshot = FilesystemRunReadModel(tmp_path / "runs").load_run("demo")

    assert (run_dir / "run.json").read_text(encoding="utf-8") == manifest_before
    assert (run_dir / "layout" / "review.json").read_text(encoding="utf-8") == review_before
    assert snapshot.pages[0].image_path == "pages/page-0001.png"
    assert snapshot.pages[0].resolved_path == run_dir / "pages" / "page-0001.png"
    assert snapshot.review_layout is not None
    assert snapshot.review_layout.pages[0].image_path == "pages/page-0001.png"
    assert snapshot.review_layout.pages[0].provider_path == "layout/provider/page-0001"


def test_app_state_loads_ocr_complete_run_without_extraction_as_ocr_first(
    tmp_path: Path,
) -> None:
    _seed_ocr_complete_run(tmp_path / "runs", "demo")
    state = AppState()
    state.run_root = str(tmp_path / "runs")

    state.load_run("demo")

    assert not (tmp_path / "runs" / "demo" / "extraction").exists()
    assert state.session.ocr_markdown == "# OCR"
    assert state.session.ocr_json["pages"][0]["markdown"] == "# OCR"
    assert state.session.extraction_json == {}


def test_app_state_marks_review_ready_runs_as_needing_review_before_results(
    tmp_path: Path,
) -> None:
    _seed_run(tmp_path / "runs", "demo")
    state = AppState()
    state.run_root = str(tmp_path / "runs")

    state.load_run("demo")

    assert state.session.run_status == "review_ready"
    assert state.needs_review_before_results() is True


def test_app_state_marks_ocr_complete_runs_as_results_ready(tmp_path: Path) -> None:
    _seed_ocr_complete_run(tmp_path / "runs", "demo")
    state = AppState()
    state.run_root = str(tmp_path / "runs")

    state.load_run("demo")

    assert state.session.run_status == "ocr_complete"
    assert state.needs_review_before_results() is False


def test_run_root_setter_rebuilds_services_for_recent_runs(tmp_path: Path) -> None:
    _seed_run(tmp_path / "runs-a", "a")
    _seed_run(tmp_path / "runs-b", "b")
    state = AppState()

    state.run_root = str(tmp_path / "runs-a")
    state.load_recent_runs()
    assert [run.run_id for run in state.session.recent_runs] == ["a"]

    state.run_root = str(tmp_path / "runs-b")
    state.load_recent_runs()
    assert [run.run_id for run in state.session.recent_runs] == ["b"]


def test_ocr_options_include_ui_ollama_settings() -> None:
    state = AppState()
    state.layout_profile = "pp_doclayout_formula"
    state.ollama_model = "manual-model"
    state.ollama_endpoint = "http://manual.example/api/generate"

    options = _ocr_options(state)

    assert options.layout_profile == "pp_doclayout_formula"
    assert options.model == "manual-model"
    assert options.endpoint == "http://manual.example/api/generate"


def test_app_state_uses_nullable_ollama_override_fields() -> None:
    state = AppState()

    assert state.ollama_model is None
    assert state.ollama_endpoint is None


def test_fresh_ocr_options_use_default_layout_profile_and_no_overrides() -> None:
    state = AppState()

    options = _ocr_options(state)

    assert options.layout_profile == "auto"
    assert options.model is None
    assert options.endpoint is None


def test_ocr_options_preserve_endpoint_only_explicit_override() -> None:
    state = AppState()
    state.ollama_endpoint = "http://manual.example/api/generate"

    options = _ocr_options(state)

    assert options.layout_profile == "auto"
    assert options.model is None
    assert options.endpoint == "http://manual.example/api/generate"


def test_ocr_options_preserve_nullable_explicit_overrides() -> None:
    state = AppState()
    state.layout_profile = "pp_doclayout_formula"
    state.ollama_model = "manual-model"

    options = _ocr_options(state)

    assert options.layout_profile == "pp_doclayout_formula"
    assert options.model == "manual-model"
    assert options.endpoint is None


def test_normalize_ocr_override_treats_blank_input_as_no_override() -> None:
    assert _normalize_ocr_override(None) is None
    assert _normalize_ocr_override("") is None
    assert _normalize_ocr_override("   ") is None
    assert _normalize_ocr_override("  http://manual.example/api/generate  ") == (
        "http://manual.example/api/generate"
    )


def test_upload_status_badge_uses_openai_compatible_config_without_ollama_fallback() -> None:
    state = cast(
        AppState,
        SimpleNamespace(
            services=SimpleNamespace(
                inference_config=_inference_config(
                    provider="openai_compatible",
                    base_url="http://localhost:8000/v1",
                    endpoint="http://localhost:8000/v1/chat/completions",
                )
            ),
            ollama_endpoint=None,
        ),
    )

    badge = _build_inference_status_badge(state)

    assert isinstance(badge, OllamaStatus)
    assert badge._endpoint == "http://localhost:8000/v1/chat/completions"
    assert badge._probe_endpoint == "http://localhost:8000/v1"
    assert badge._label.value == "OpenAI-compatible: Checking…"
    assert state.ollama_endpoint is None


def test_upload_status_badge_keeps_explicit_endpoint_override_nullable_for_ocr() -> None:
    state = cast(
        AppState,
        SimpleNamespace(
            services=SimpleNamespace(
                inference_config=_inference_config(
                    provider="ollama",
                    base_url="http://localhost:11434",
                    endpoint="http://localhost:11434/api/generate",
                )
            ),
            ollama_endpoint="http://manual.example/api/generate",
        ),
    )

    badge = _build_inference_status_badge(state)

    assert badge._endpoint == "http://manual.example/api/generate"
    assert badge._probe_endpoint == "http://manual.example"
    assert badge._label.value == "Ollama: Checking…"
    assert state.ollama_endpoint == "http://manual.example/api/generate"


def test_inference_status_badge_uses_provider_accurate_ready_and_offline_labels() -> None:
    badge = OllamaStatus(
        provider="openai_compatible",
        endpoint="http://localhost:8000/v1/chat/completions",
    )

    badge._apply_status(True)
    assert badge._label.value == "OpenAI-compatible: Ready"

    badge._apply_status(False)
    assert badge._label.value == "OpenAI-compatible: Offline"


def _seed_run(run_root: Path, run_id: str) -> None:
    store = FilesystemRunStore(run_root)
    workspace = store.start_run("input.pdf", RunId(run_id))
    page_path = workspace.work_dir / "pages" / "page-0001.png"
    _write_png(page_path)
    store.publish_prepared_run(
        workspace,
        [
            PageRef(
                page_number=1,
                image_path="pages/page-0001.png",
                width=10,
                height=10,
                resolved_path=page_path,
            )
        ],
        ReviewLayout(pages=[], status="prepared"),
        ProviderArtifacts.empty(),
    )


def _inference_config(*, provider: str, base_url: str, endpoint: str) -> InferenceProviderConfig:
    return InferenceProviderConfig(
        provider=provider,
        model="test-model",
        base_url=base_url,
        endpoint=endpoint,
        timeout_seconds=3,
    )


def _seed_run_with_pages(run_root: Path, run_id: str, *, page_count: int) -> None:
    store = FilesystemRunStore(run_root)
    workspace = store.start_run("input.pdf", RunId(run_id))
    pages: list[PageRef] = []
    for page_number in range(1, page_count + 1):
        page_path = workspace.work_dir / "pages" / f"page-{page_number:04d}.png"
        _write_png(page_path)
        pages.append(
            PageRef(
                page_number=page_number,
                image_path=f"pages/page-{page_number:04d}.png",
                width=10,
                height=10,
                resolved_path=page_path,
            )
        )
    store.publish_prepared_run(
        workspace,
        pages,
        ReviewLayout(pages=[], status="prepared"),
        ProviderArtifacts.empty(),
    )


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PNG_BYTES)


_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
    b"\x00\x0cIDATx\x9cc``\x00\x00\x00\x04\x00\x01\xf6\x178U"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _seed_reviewed_v3_run(run_root: Path, run_id: str) -> None:
    store = FilesystemRunStore(run_root)
    workspace = store.start_run("input.pdf", RunId(run_id))
    page_path = workspace.work_dir / "pages" / "page-0001.png"
    _write_png(page_path)
    provider_dir = workspace.work_dir / "layout-provider" / "page-0001"
    provider_dir.mkdir(parents=True, exist_ok=True)
    (provider_dir / "layout.json").write_text("{}", encoding="utf-8")
    store.publish_prepared_run(
        workspace,
        [
            PageRef(
                page_number=1,
                image_path="pages/page-0001.png",
                width=10,
                height=10,
                resolved_path=page_path,
            )
        ],
        ReviewLayout(
            pages=[
                ReviewPage(
                    page_number=1,
                    image_path="pages/page-0001.png",
                    image_width=10,
                    image_height=10,
                    provider_path="layout/provider/page-0001",
                    blocks=[LayoutBlock(id="p1-b0", index=0, label="text", bbox=[1, 2, 8, 9])],
                )
            ],
            status="reviewed",
        ),
        ProviderArtifacts((ArtifactCopy(provider_dir, "layout/provider/page-0001"),)),
    )


def _seed_ocr_complete_run(run_root: Path, run_id: str) -> None:
    _seed_run(run_root, run_id)
    store = FilesystemRunStore(run_root)
    ocr_result = OcrRunResult(
        pages=[
            OcrPageResult(
                page_number=1,
                image_path="pages/page-0001.png",
                markdown="# OCR",
                markdown_source="sdk_markdown",
                provider_path="ocr/provider/page-0001",
            )
        ],
        markdown="# OCR",
    )
    manifest = store.open_run(RunId(run_id)).manifest
    store.write_ocr_result(RunId(run_id), ocr_result, ProviderArtifacts.empty())
    store.apply_invalidation_plan(
        RunId(run_id),
        ocr_result_updated_policy(manifest, diagnostics=ocr_result.diagnostics),
    )
