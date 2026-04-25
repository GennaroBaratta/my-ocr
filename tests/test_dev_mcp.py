# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import cast

import anyio
import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from starlette.testclient import TestClient

from tools.dev_mcp.app import create_app
from tools.dev_mcp.config import DevMcpConfig
import tools.dev_mcp.services as services_module
from tools.dev_mcp.services import (
    FeedbackBundleStore,
    OCRWorkflowRunner,
    RunStateReader,
    UIProcessManager,
)


def test_health_endpoint_and_mcp_route_exist(tmp_path) -> None:
    repo_root = build_repo_fixture(tmp_path)
    app = create_app(repo_root)

    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        health_response = client.get("/healthz")
        assert health_response.status_code == 200
        assert health_response.json()["mcp_path"] == "/mcp"

        mcp_response = client.options("/mcp/")
        assert mcp_response.status_code in {200, 204, 400, 405}


def test_start_stop_ui_and_status(tmp_path, monkeypatch) -> None:
    repo_root = build_repo_fixture(tmp_path)
    config = DevMcpConfig.from_repo_root(repo_root)
    config = DevMcpConfig(
        repo_root=config.repo_root,
        host=config.host,
        port=config.port,
        ui_command_override=(
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
        ),
    )
    monkeypatch.setattr(services_module, "_http_ready", lambda _url, timeout_seconds=2.0: True)
    manager = UIProcessManager(config)

    started = manager.start_ui()
    assert started["ok"] is True
    assert started["running"] is True
    assert started["pid"] is not None

    status = manager.ui_status()
    assert status["running"] is True
    assert status["command"][0] == sys.executable
    assert status["ui_url"] == config.ui_base_url

    stopped = manager.stop_ui()
    assert stopped["ok"] is True
    assert stopped["running"] is False


def test_read_run_state_collects_expected_artifacts(tmp_path) -> None:
    repo_root = build_repo_fixture(tmp_path)
    run_dir = seed_run(repo_root / "data" / "runs" / "demo-ui")
    reader = RunStateReader(DevMcpConfig.from_repo_root(repo_root))

    result = reader.read_run_state("demo-ui")

    assert result["ok"] is True
    assert result["page_count"] == 1
    assert result["meta"]["input_path"] == "data/raw/demo.pdf"
    assert result["reviewed_layout"]["status"] == "reviewed"
    assert result["ocr"]["page_count"] == 1
    assert result["predictions"][0]["name"] == "rules.json"
    assert result["predictions"][0]["top_level_keys"] == ["document_type"]
    assert (
        result["artifact_paths"]["ocr_json"]["repo_relative_path"]
        == "data/runs/demo-ui/ocr/pages.json"
    )
    assert run_dir.exists()


def test_run_ocr_returns_run_state_for_repo_local_pdf(tmp_path, monkeypatch) -> None:
    repo_root = build_repo_fixture(tmp_path)
    seed_raw_pdf(repo_root / "data" / "raw" / "demo.pdf")
    run_dir = seed_run(repo_root / "data" / "runs" / "demo-run")
    captured: dict[str, object] = {}

    def fake_prepare_review(*, input_path, run_id):
        captured["input_path"] = input_path
        captured["run"] = run_id
        return SimpleNamespace(snapshot=SimpleNamespace(run_id=run_id, run_dir=run_dir))

    def fake_run_reviewed_ocr(run_id):
        captured["ocr_run"] = run_id
        return SimpleNamespace(snapshot=SimpleNamespace(run_id=run_id, run_dir=run_dir))

    monkeypatch.setattr(
        services_module,
        "build_backend_services",
        lambda run_root: SimpleNamespace(
            workflow=SimpleNamespace(
                prepare_review=fake_prepare_review,
                run_reviewed_ocr=fake_run_reviewed_ocr,
            ),
        ),
    )
    runner = OCRWorkflowRunner(DevMcpConfig.from_repo_root(repo_root))

    result = runner.run_ocr(input_path="data/raw/demo.pdf", run_id="demo-run")

    assert result["ok"] is True
    assert result["run_id"] == "demo-run"
    assert result["input_path"] == "data/raw/demo.pdf"
    assert result["run_state"]["run_id"] == "demo-run"
    assert captured == {
        "input_path": str((repo_root / "data" / "raw" / "demo.pdf").resolve()),
        "run": services_module.RunId("demo-run"),
        "ocr_run": services_module.RunId("demo-run"),
    }


def test_run_ocr_rejects_non_pdf_or_outside_data_raw(tmp_path) -> None:
    repo_root = build_repo_fixture(tmp_path)
    seed_raw_pdf(repo_root / "data" / "raw" / "demo.pdf")
    text_path = repo_root / "data" / "raw" / "notes.txt"
    text_path.write_text("hello", encoding="utf-8")
    outside_pdf = repo_root / "docs" / "outside.pdf"
    outside_pdf.parent.mkdir(parents=True, exist_ok=True)
    outside_pdf.write_bytes(b"%PDF-1.4 outside")

    runner = OCRWorkflowRunner(DevMcpConfig.from_repo_root(repo_root))

    with pytest.raises(services_module.RepoPathError, match="OCR input must be a PDF"):
        runner.run_ocr(input_path="data/raw/notes.txt")

    with pytest.raises(services_module.RepoPathError, match="outside repository root"):
        runner.run_ocr(input_path="../outside.pdf")

    with pytest.raises(
        services_module.RepoPathError, match="OCR input must be a PDF under data/raw"
    ):
        runner.run_ocr(input_path=str(outside_pdf))


def test_run_ocr_rejects_overlapping_requests(tmp_path) -> None:
    repo_root = build_repo_fixture(tmp_path)
    seed_raw_pdf(repo_root / "data" / "raw" / "demo.pdf")
    runner = OCRWorkflowRunner(DevMcpConfig.from_repo_root(repo_root))
    assert runner._run_lock.acquire(blocking=False) is True
    try:
        with pytest.raises(RuntimeError, match="OCR is already running in the dev MCP"):
            runner.run_ocr(input_path="data/raw/demo.pdf", run_id="demo-run")
    finally:
        runner._run_lock.release()


def test_run_ocr_rejects_invalid_new_run_id(tmp_path) -> None:
    repo_root = build_repo_fixture(tmp_path)
    seed_raw_pdf(repo_root / "data" / "raw" / "demo.pdf")
    runner = OCRWorkflowRunner(DevMcpConfig.from_repo_root(repo_root))

    with pytest.raises(services_module.RepoPathError, match="Invalid run_id"):
        runner.run_ocr(input_path="data/raw/demo.pdf", run_id="nested/demo")


def test_save_feedback_bundle_copies_screenshot_and_lists_bundles(tmp_path) -> None:
    repo_root = build_repo_fixture(tmp_path)
    seed_run(repo_root / "data" / "runs" / "demo-ui")
    screenshot = tmp_path / "capture.png"
    screenshot.write_bytes(b"png-bytes")

    store = FeedbackBundleStore(DevMcpConfig.from_repo_root(repo_root))

    async def save_bundle() -> dict[str, object]:
        return await store.save_feedback_bundle(
            run_id="demo-ui",
            summary="Toolbar spacing still feels cramped.",
            issues=[
                {"severity": "medium", "location": "results.toolbar", "note": "Buttons wrap early."}
            ],
            notes="Captured after reviewing the demo run.",
            screenshot_paths=[str(screenshot)],
            related_paths=["data/runs/demo-ui/layout/review.json"],
        )

    saved = cast(dict[str, object], anyio.run(save_bundle))

    assert saved["ok"] is True
    manifest_path = Path(cast(str, saved["manifest_path"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "demo-ui"
    assert manifest["screenshots"][0]["original_path"] == str(screenshot.resolve())
    assert Path(repo_root / manifest["screenshots"][0]["copied_path"]).exists()

    listed = store.list_feedback_bundles(run_id="demo-ui")
    assert listed["ok"] is True
    assert listed["bundles"][0]["run_id"] == "demo-ui"
    assert listed["bundles"][0]["screenshot_count"] == 1


def test_save_feedback_bundle_can_capture_screenshot_autonomously(tmp_path, monkeypatch) -> None:
    repo_root = build_repo_fixture(tmp_path)
    seed_run(repo_root / "data" / "runs" / "demo-ui")
    config = DevMcpConfig.from_repo_root(repo_root)

    class FakePage:
        async def goto(self, url: str, wait_until: str, timeout: int) -> None:
            self.url = url
            self.wait_until = wait_until
            self.timeout = timeout

        async def wait_for_selector(self, selector: str, timeout: int, state: str) -> None:
            self.selector = selector
            self.selector_timeout = timeout
            self.selector_state = state

        async def wait_for_timeout(self, timeout: int) -> None:
            self.settle_timeout = timeout

        async def screenshot(self, *, path: str, full_page: bool) -> None:
            Path(path).write_bytes(b"captured-png")
            self.screenshot_path = path
            self.full_page = full_page

    class FakeBrowser:
        def __init__(self) -> None:
            self.page = FakePage()

        async def new_page(self, viewport: dict[str, int]) -> FakePage:
            self.viewport = viewport
            return self.page

        async def close(self) -> None:
            return None

    class FakePlaywrightContext:
        def __init__(self) -> None:
            self.browser = FakeBrowser()

            async def launch(headless: bool = True) -> FakeBrowser:
                _ = headless
                return self.browser

            self.chromium = SimpleNamespace(launch=launch)

        async def __aenter__(self) -> "FakePlaywrightContext":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    fake_playwright_context = FakePlaywrightContext()
    fake_sync_api = SimpleNamespace(
        Error=RuntimeError,
        async_playwright=lambda: fake_playwright_context,
    )

    monkeypatch.setattr(services_module.importlib, "import_module", lambda name: fake_sync_api)

    class FakeUIManager:
        def __init__(self, ui_url: str) -> None:
            self.ui_url = ui_url

        def ui_status(self) -> dict[str, object]:
            return {"running": True, "ui_url": self.ui_url}

        def start_ui(self) -> dict[str, object]:
            raise AssertionError("start_ui should not be called when UI is already running")

    store = FeedbackBundleStore(
        config,
        ui_manager=cast(UIProcessManager, FakeUIManager(config.ui_base_url)),
    )

    async def save_bundle() -> dict[str, object]:
        return await store.save_feedback_bundle(
            run_id="demo-ui",
            summary="Capture the results screen",
            issues=[{"severity": "medium", "location": "results", "note": "Needs review"}],
            capture_route="/results/demo-ui",
        )

    saved = cast(dict[str, object], anyio.run(save_bundle))

    manifest = json.loads(Path(cast(str, saved["manifest_path"])).read_text(encoding="utf-8"))
    screenshot = manifest["screenshots"][0]
    assert screenshot["kind"] == "captured"
    assert screenshot["route"] == "/results/demo-ui"
    assert screenshot["ui_url"] == config.ui_base_url
    assert fake_playwright_context.browser.page.url == f"{config.ui_base_url}/results/demo-ui"
    assert Path(repo_root / screenshot["copied_path"]).read_bytes() == b"captured-png"


def test_save_feedback_bundle_can_autostart_ui_before_capture(tmp_path, monkeypatch) -> None:
    repo_root = build_repo_fixture(tmp_path)
    seed_run(repo_root / "data" / "runs" / "demo-ui")
    config = DevMcpConfig.from_repo_root(repo_root)

    class FakePage:
        async def goto(self, url: str, wait_until: str, timeout: int) -> None:
            self.url = url

        async def wait_for_selector(self, selector: str, timeout: int, state: str) -> None:
            self.selector = selector
            self.state = state

        async def wait_for_timeout(self, timeout: int) -> None:
            self.timeout = timeout

        async def screenshot(self, *, path: str, full_page: bool) -> None:
            Path(path).write_bytes(b"captured-after-start")

    class FakeBrowser:
        def __init__(self) -> None:
            self.page = FakePage()

        async def new_page(self, viewport: dict[str, int]) -> FakePage:
            self.viewport = viewport
            return self.page

        async def close(self) -> None:
            return None

    class FakePlaywrightContext:
        def __init__(self) -> None:
            self.browser = FakeBrowser()

            async def launch(headless: bool = True) -> FakeBrowser:
                _ = headless
                return self.browser

            self.chromium = SimpleNamespace(launch=launch)

        async def __aenter__(self) -> "FakePlaywrightContext":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(
        services_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            Error=RuntimeError, async_playwright=lambda: FakePlaywrightContext()
        ),
    )

    class FakeUIManager:
        def __init__(self, ui_url: str) -> None:
            self.ui_url = ui_url
            self.started = False

        def ui_status(self) -> dict[str, object]:
            return {"running": self.started, "ui_url": self.ui_url}

        def start_ui(self) -> dict[str, object]:
            self.started = True
            return {"running": True, "ui_url": self.ui_url, "summary": "started"}

    fake_ui_manager = FakeUIManager(config.ui_base_url)
    store = FeedbackBundleStore(config, ui_manager=cast(UIProcessManager, fake_ui_manager))

    async def save_bundle() -> dict[str, object]:
        return await store.save_feedback_bundle(
            run_id="demo-ui",
            summary="Autostart capture",
            issues=[{"severity": "low", "location": "results", "note": "autostarted"}],
            capture_route="/results/demo-ui",
        )

    saved = cast(dict[str, object], anyio.run(save_bundle))
    manifest = json.loads(Path(cast(str, saved["manifest_path"])).read_text(encoding="utf-8"))
    assert fake_ui_manager.started is True
    assert manifest["screenshots"][0]["kind"] == "captured"


def test_save_feedback_bundle_raises_clean_error_when_ui_cannot_start(tmp_path) -> None:
    repo_root = build_repo_fixture(tmp_path)
    seed_run(repo_root / "data" / "runs" / "demo-ui")
    config = DevMcpConfig.from_repo_root(repo_root)

    class FakeUIManager:
        def __init__(self, ui_url: str) -> None:
            self.ui_url = ui_url

        def ui_status(self) -> dict[str, object]:
            return {"running": False, "ui_url": self.ui_url}

        def start_ui(self) -> dict[str, object]:
            return {"running": False, "ui_url": self.ui_url, "summary": "port unavailable"}

    store = FeedbackBundleStore(
        config,
        ui_manager=cast(UIProcessManager, FakeUIManager(config.ui_base_url)),
    )

    async def save_bundle() -> None:
        await store.save_feedback_bundle(
            run_id="demo-ui",
            summary="Should fail cleanly",
            issues=[{"severity": "low", "location": "results", "note": "failure path"}],
            capture_route="/results/demo-ui",
        )

    try:
        anyio.run(save_bundle)
    except RuntimeError as exc:
        assert str(exc) == "Could not start UI for screenshot capture: port unavailable"
    else:
        raise AssertionError("Expected a RuntimeError when UI startup fails")


def test_stdio_transport_lists_expected_tools(tmp_path) -> None:
    repo_root = build_repo_fixture(tmp_path)
    seed_run(repo_root / "data" / "runs" / "demo-ui")

    async def exercise_stdio_transport() -> None:
        params = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "tools.dev_mcp",
                "--transport",
                "stdio",
                "--repo-root",
                str(repo_root),
            ],
            cwd=Path(__file__).resolve().parents[1],
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
        tool_names = sorted(tool.name for tool in tools.tools)
        assert tool_names == [
            "health",
            "list_feedback_bundles",
            "project_info",
            "read_run_state",
            "run_ocr",
            "save_feedback_bundle",
            "start_ui",
            "stop_ui",
            "ui_status",
        ]

    anyio.run(exercise_stdio_transport)


def build_repo_fixture(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "runs").mkdir(parents=True)
    (repo_root / "data" / "raw").mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8"
    )
    return repo_root


def seed_raw_pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4 demo")
    return path


def seed_run(run_dir: Path) -> Path:
    pages_dir = run_dir / "pages"
    extraction_dir = run_dir / "extraction"
    pages_dir.mkdir(parents=True, exist_ok=True)
    extraction_dir.mkdir(parents=True, exist_ok=True)

    (pages_dir / "page-0001.png").write_bytes(b"page")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": run_dir.name,
                "input": {"path": "data/raw/demo.pdf", "name": "demo.pdf", "kind": "pdf"},
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "pages": [
                    {
                        "page_number": 1,
                        "image_path": "pages/page-0001.png",
                        "width": 10,
                        "height": 10,
                    }
                ],
                "status": {"layout": "reviewed", "ocr": "complete", "extraction": "rules"},
                "diagnostics": {"layout": {}, "ocr": {}, "extraction": {}},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "layout").mkdir(exist_ok=True)
    (run_dir / "layout" / "review.json").write_text(
        json.dumps(
            {
                "version": 2,
                "status": "reviewed",
                "pages": [{"page_number": 1, "image_path": "pages/page-0001.png", "blocks": []}],
                "summary": {"page_count": 1},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "ocr").mkdir(exist_ok=True)
    (run_dir / "ocr" / "pages.json").write_text(
        json.dumps({"pages": [{"page_number": 1}], "summary": {"page_count": 1}}),
        encoding="utf-8",
    )
    (run_dir / "ocr" / "markdown.md").write_text("# OCR", encoding="utf-8")
    (extraction_dir / "rules.json").write_text(
        json.dumps({"document_type": "report"}),
        encoding="utf-8",
    )
    return run_dir
