# pyright: reportMissingImports=false
from __future__ import annotations

import contextlib
import asyncio
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from .config import DevMcpConfig
from .services import (
    FeedbackBundleStore,
    OCRWorkflowRunner,
    RepoPathError,
    RunStateReader,
    UIProcessManager,
)


def build_mcp(config: DevMcpConfig) -> FastMCP:
    ui_manager = UIProcessManager(config)
    run_state_reader = RunStateReader(config)
    ocr_runner = OCRWorkflowRunner(config, run_state_reader=run_state_reader)
    feedback_store = FeedbackBundleStore(config, ui_manager=ui_manager)

    mcp = FastMCP(
        "my-ocr-dev-mcp",
        json_response=True,
        stateless_http=True,
        streamable_http_path="/",
    )

    @mcp.tool()
    def health() -> dict[str, Any]:
        """Report sidecar and UI process health."""
        status = ui_manager.ui_status()
        return {
            "ok": True,
            "summary": "Dev MCP sidecar is healthy",
            "server": "my-ocr-dev-mcp",
            "host": config.host,
            "port": config.port,
            "ui": status,
            "detected_stack": ["python", "flet", "pytest", "uv"],
            "next_suggested_actions": [
                "Call project_info for repo-local paths and commands.",
                "Call start_ui if the Flet app is not already running.",
            ],
        }

    @mcp.tool()
    def project_info() -> dict[str, Any]:
        """Return repo-rooted project metadata for the local UX loop."""
        config.ensure_runtime_dirs()
        return {
            "ok": True,
            "summary": "Loaded project-local MCP metadata",
            "repo_root": str(config.repo_root),
            "run_root": str(config.run_root),
            "runtime_root": str(config.runtime_root),
            "feedback_root": str(config.feedback_root),
            "ui_url": config.ui_base_url,
            "ui_command": list(config.ui_command),
            "detected_stack": ["python", "flet", "pytest", "uv"],
            "next_suggested_actions": [
                "Use run_ocr with a repo-local PDF under data/raw to generate a new OCR run.",
                "Use start_ui to launch the local UX session.",
                "Use save_feedback_bundle with capture_route to save an autonomous screenshot bundle.",
            ],
        }

    @mcp.tool()
    def start_ui() -> dict[str, Any]:
        """Start the local Flet UI process using the repo's existing command."""
        return ui_manager.start_ui()

    @mcp.tool()
    def stop_ui() -> dict[str, Any]:
        """Stop the tracked local Flet UI process."""
        return ui_manager.stop_ui()

    @mcp.tool()
    def ui_status() -> dict[str, Any]:
        """Report whether the tracked local Flet UI process is running."""
        return ui_manager.ui_status()

    @mcp.tool()
    def read_run_state(run_id: str) -> dict[str, Any]:
        """Inspect the known run artifacts under data/runs/<run_id>."""
        return run_state_reader.read_run_state(run_id)

    @mcp.tool()
    async def run_ocr(input_path: str, run_id: str | None = None) -> dict[str, Any]:
        """Run OCR end to end for a repo-local PDF under data/raw."""
        return await asyncio.to_thread(ocr_runner.run_ocr, input_path=input_path, run_id=run_id)

    @mcp.tool()
    async def save_feedback_bundle(
        run_id: str,
        summary: str,
        issues: list[dict[str, Any]],
        notes: str = "",
        screenshot_paths: list[str] | None = None,
        capture_route: str | None = None,
        wait_for_selector: str | None = None,
        full_page: bool = True,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
        related_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """Save a structured UX feedback bundle rooted in .dev-mcp/feedback."""
        return await feedback_store.save_feedback_bundle(
            run_id=run_id,
            summary=summary,
            issues=issues,
            notes=notes,
            screenshot_paths=screenshot_paths,
            capture_route=capture_route,
            wait_for_selector=wait_for_selector,
            full_page=full_page,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            related_paths=related_paths,
        )

    @mcp.tool()
    def list_feedback_bundles(run_id: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List saved UX feedback bundles, optionally filtered by run_id."""
        return feedback_store.list_feedback_bundles(run_id=run_id, limit=limit)

    return mcp


def create_app(
    repo_root: str | Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    ui_host: str = "127.0.0.1",
    ui_port: int = 8550,
) -> Starlette:
    config = DevMcpConfig.from_repo_root(
        repo_root or Path(__file__).resolve().parents[2],
        host=host,
        port=port,
        ui_host=ui_host,
        ui_port=ui_port,
    )
    mcp = build_mcp(config)

    async def healthz(_request: Request) -> Response:
        return JSONResponse(
            {
                "ok": True,
                "status": "ok",
                "server": "my-ocr-dev-mcp",
                "mcp_path": "/mcp",
                "repo_root": str(config.repo_root),
            }
        )

    async def index(_request: Request) -> Response:
        return JSONResponse(
            {
                "ok": True,
                "summary": "Local dev-only MCP sidecar for the Flet UX feedback loop.",
                "health_path": "/healthz",
                "mcp_path": "/mcp",
            }
        )

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette):
        async with mcp.session_manager.run():
            yield

    return Starlette(
        routes=[
            Route("/", index, methods=["GET"]),
            Route("/healthz", healthz, methods=["GET"]),
            Mount("/mcp", app=mcp.streamable_http_app()),
        ],
        lifespan=lifespan,
        exception_handlers={
            RepoPathError: _handle_repo_path_error,
            FileNotFoundError: _handle_not_found_error,
        },
    )


async def _handle_repo_path_error(_request: Request, exc: Exception) -> Response:
    return JSONResponse({"ok": False, "summary": str(exc)}, status_code=400)


async def _handle_not_found_error(_request: Request, exc: Exception) -> Response:
    return JSONResponse({"ok": False, "summary": str(exc)}, status_code=404)
