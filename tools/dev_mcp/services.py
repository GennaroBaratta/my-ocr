from __future__ import annotations

import json
import importlib
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from typing import Any

from my_ocr.workflows import run_ocr_workflow

from .config import DevMcpConfig


def utc_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def _slug(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("-")
    slug = "".join(allowed).strip("-._")
    return slug or "bundle"


class RepoPathError(ValueError):
    """Raised when a requested path escapes the repository sandbox."""


@dataclass(frozen=True, slots=True)
class RepoPaths:
    config: DevMcpConfig

    def require_run_id(self, run_id: str) -> Path:
        if not run_id or Path(run_id).name != run_id:
            raise RepoPathError(f"Invalid run_id: {run_id!r}")
        run_dir = (self.config.run_root / run_id).resolve()
        self._require_within(self.config.run_root, run_dir)
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir

    def validate_run_id(self, run_id: str) -> str:
        if not run_id or Path(run_id).name != run_id:
            raise RepoPathError(f"Invalid run_id: {run_id!r}")
        run_dir = (self.config.run_root / run_id).resolve()
        self._require_within(self.config.run_root, run_dir)
        return run_id

    def require_repo_path(self, value: str | Path) -> Path:
        raw_path = Path(value)
        candidate = raw_path if raw_path.is_absolute() else self.config.repo_root / raw_path
        resolved = candidate.resolve()
        self._require_within(self.config.repo_root, resolved)
        return resolved

    def rel_to_repo(self, path: str | Path) -> str:
        return Path(path).resolve().relative_to(self.config.repo_root).as_posix()

    @staticmethod
    def _require_within(root: Path, candidate: Path) -> None:
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise RepoPathError(f"Path is outside repository root: {candidate}") from exc


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _try_read_json(path: Path) -> Any | None:
    try:
        return _read_json(path)
    except (json.JSONDecodeError, OSError):
        return None


def _safe_file_info(path: Path, repo_paths: RepoPaths) -> dict[str, Any]:
    return {
        "path": str(path),
        "repo_relative_path": repo_paths.rel_to_repo(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def _tail_text(path: Path, *, max_chars: int = 2000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _http_ready(url: str, *, timeout_seconds: float = 2.0) -> bool:
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            return 200 <= response.status < 300
    except (URLError, TimeoutError, ValueError):
        return False


class UIProcessManager:
    def __init__(self, config: DevMcpConfig) -> None:
        self.config = config
        self.repo_paths = RepoPaths(config)
        self._process: subprocess.Popen[Any] | None = None

    def start_ui(self) -> dict[str, Any]:
        self.config.ensure_runtime_dirs()
        current = self.ui_status()
        if current["running"]:
            current["summary"] = "UI process is already running"
            return current

        stdout_path = self.config.logs_root / "ui.stdout.log"
        stderr_path = self.config.logs_root / "ui.stderr.log"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            with (
                stdout_path.open("a", encoding="utf-8") as stdout_handle,
                stderr_path.open("a", encoding="utf-8") as stderr_handle,
            ):
                popen_kwargs: dict[str, Any] = {
                    "cwd": self.config.repo_root,
                    "stdout": stdout_handle,
                    "stderr": stderr_handle,
                    "env": env,
                }
                if os.name == "nt":
                    popen_kwargs["creationflags"] = (
                        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                        | getattr(subprocess, "DETACHED_PROCESS", 0)
                    )
                else:
                    popen_kwargs["start_new_session"] = True
                process = subprocess.Popen(
                    list(self.config.ui_command),
                    **popen_kwargs,
                )
                self._process = process
        except OSError as exc:
            return {
                "ok": False,
                "summary": "Failed to start UI process",
                "running": False,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "error": str(exc),
                "next_suggested_actions": [
                    "Verify that `uv` is installed and available on PATH.",
                    "Check the configured UI command in project_info before retrying.",
                ],
            }

        time.sleep(0.25)
        exit_code = process.poll()
        if exit_code is not None:
            return {
                "ok": False,
                "summary": "UI process exited immediately",
                "running": False,
                "exit_code": exit_code,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "stderr_excerpt": _tail_text(stderr_path),
                "next_suggested_actions": [
                    "Check the stderr excerpt and verify the project dependencies are installed.",
                    "Run `uv sync --group dev --extra dev-mcp` before retrying.",
                ],
            }

        state = {
            "pid": process.pid,
            "started_at": utc_now(),
            "command": list(self.config.ui_command),
            "cwd": str(self.config.repo_root),
            "ui_url": self.config.ui_base_url,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
        deadline = time.monotonic() + self.config.ui_start_timeout_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return {
                    "ok": False,
                    "summary": "UI process exited before the web UI became reachable",
                    "running": False,
                    "exit_code": process.returncode,
                    "ui_url": self.config.ui_base_url,
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "stderr_excerpt": _tail_text(stderr_path),
                    "next_suggested_actions": [
                        "Check the stderr excerpt and verify the Flet web UI can start locally.",
                        "Verify the configured ui_host/ui_port are available before retrying.",
                    ],
                }
            if _http_ready(self.config.ui_base_url):
                break
            time.sleep(0.25)
        else:
            self.stop_ui(timeout_seconds=1.0)
            return {
                "ok": False,
                "summary": "Timed out waiting for the Flet web UI to become reachable",
                "running": False,
                "ui_url": self.config.ui_base_url,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "stdout_excerpt": _tail_text(stdout_path),
                "stderr_excerpt": _tail_text(stderr_path),
                "next_suggested_actions": [
                    "Check the captured UI logs for startup failures.",
                    "Retry with a different ui_port if another process is using the default.",
                ],
            }
        self.config.ui_state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return self.ui_status(summary="Started local Flet UI process")

    def stop_ui(self, *, timeout_seconds: float = 5.0) -> dict[str, Any]:
        state = self._load_state()
        if state is None:
            return self.ui_status(summary="UI process is already stopped")

        pid = int(state["pid"])
        if not _pid_running(pid):
            self.config.ui_state_path.unlink(missing_ok=True)
            return self.ui_status(summary="UI process was not running")

        try:
            if self._process is not None and self._process.pid == pid:
                self._process.terminate()
                if os.name == "nt":
                    self._process.poll()
                    self._process = None
                    self.config.ui_state_path.unlink(missing_ok=True)
                    return self.ui_status(summary="Stopped local Flet UI process")
                self._process.wait(timeout=timeout_seconds)
                self._process = None
                self.config.ui_state_path.unlink(missing_ok=True)
                return self.ui_status(summary="Stopped local Flet UI process")
            else:
                self._terminate_process(pid)
        except ProcessLookupError:
            self._process = None
            self.config.ui_state_path.unlink(missing_ok=True)
            return self.ui_status(summary="UI process was not running")
        except subprocess.TimeoutExpired:
            if self._process is not None and self._process.pid == pid:
                self._process.kill()
                self._process.wait(timeout=1.0)
                self._process = None
                self.config.ui_state_path.unlink(missing_ok=True)
                return self.ui_status(summary="Killed unresponsive Flet UI process")
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if not _pid_running(pid):
                self._process = None
                self.config.ui_state_path.unlink(missing_ok=True)
                return self.ui_status(summary="Stopped local Flet UI process")
            time.sleep(0.1)

        try:
            if self._process is not None and self._process.pid == pid:
                self._process.kill()
            else:
                self._terminate_process(pid, force=True)
        except ProcessLookupError:
            pass
        self._process = None
        self.config.ui_state_path.unlink(missing_ok=True)
        return self.ui_status(summary="Killed unresponsive Flet UI process")

    @staticmethod
    def _terminate_process(pid: int, *, force: bool = False) -> None:
        if os.name == "nt":
            import ctypes

            process_terminate = 0x0001
            handle = ctypes.windll.kernel32.OpenProcess(process_terminate, False, pid)
            if not handle:
                raise ProcessLookupError(pid)
            try:
                if not ctypes.windll.kernel32.TerminateProcess(handle, 1):
                    raise ProcessLookupError(pid)
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
            return
        if force:
            os.killpg(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
            return
        os.killpg(pid, signal.SIGTERM)

    def ui_status(self, *, summary: str | None = None) -> dict[str, Any]:
        state = self._load_state()
        if state is None:
            stdout_path = self.config.logs_root / "ui.stdout.log"
            stderr_path = self.config.logs_root / "ui.stderr.log"
            return {
                "ok": True,
                "summary": summary or "UI process is not running",
                "running": False,
                "pid": None,
                "ui_url": self.config.ui_base_url,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "stdout_excerpt": _tail_text(stdout_path),
                "stderr_excerpt": _tail_text(stderr_path),
                "next_suggested_actions": [
                    "Call start_ui to launch the Flet UX locally.",
                    "Use read_run_state before capturing a feedback bundle.",
                ],
            }

        pid = int(state["pid"])
        stdout_path = Path(state["stdout_path"])
        stderr_path = Path(state["stderr_path"])
        running = _pid_running(pid)
        if not running:
            self.config.ui_state_path.unlink(missing_ok=True)

        return {
            "ok": True,
            "summary": summary
            or ("UI process is running" if running else "UI process is not running"),
            "running": running,
            "pid": pid if running else None,
            "started_at": state.get("started_at"),
            "ui_url": state.get("ui_url", self.config.ui_base_url),
            "command": state.get("command", []),
            "cwd": state.get("cwd"),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "stdout_excerpt": _tail_text(stdout_path),
            "stderr_excerpt": _tail_text(stderr_path),
            "next_suggested_actions": [
                "Use save_feedback_bundle with capture_route to generate screenshots automatically.",
                "Use stop_ui when the review session is complete.",
            ],
        }

    def _load_state(self) -> dict[str, Any] | None:
        path = self.config.ui_state_path
        if not path.exists():
            return None
        try:
            loaded = _read_json(path)
        except (json.JSONDecodeError, OSError):
            return None
        return loaded if isinstance(loaded, dict) else None


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunStateReader:
    def __init__(self, config: DevMcpConfig) -> None:
        self.config = config
        self.repo_paths = RepoPaths(config)

    def read_run_state(self, run_id: str) -> dict[str, Any]:
        run_dir = self.repo_paths.require_run_id(run_id)
        pages_dir = run_dir / "pages"
        predictions_dir = run_dir / "predictions"

        artifact_paths = {
            "run_dir": _safe_file_info(run_dir, self.repo_paths),
            "meta_json": _safe_file_info(run_dir / "meta.json", self.repo_paths),
            "reviewed_layout_json": _safe_file_info(
                run_dir / "reviewed_layout.json", self.repo_paths
            ),
            "ocr_json": _safe_file_info(run_dir / "ocr.json", self.repo_paths),
            "ocr_markdown": _safe_file_info(run_dir / "ocr.md", self.repo_paths),
            "predictions_dir": _safe_file_info(predictions_dir, self.repo_paths),
        }

        result: dict[str, Any] = {
            "ok": True,
            "summary": f"Collected run state for {run_id}",
            "run_id": run_id,
            "run_dir": str(run_dir),
            "artifact_paths": artifact_paths,
            "page_paths": [],
            "page_count": 0,
            "meta": None,
            "reviewed_layout": None,
            "ocr": None,
            "predictions": [],
            "next_suggested_actions": [
                "Inspect reviewed_layout and predictions before capturing UX feedback.",
                "Attach any external screenshot paths with save_feedback_bundle.",
            ],
        }

        if pages_dir.exists():
            page_paths = [
                self.repo_paths.rel_to_repo(path)
                for path in sorted(path for path in pages_dir.iterdir() if path.is_file())
            ]
            result["page_paths"] = page_paths
            result["page_count"] = len(page_paths)

        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            meta = _try_read_json(meta_path)
            if isinstance(meta, dict):
                result["meta"] = meta

        reviewed_layout_path = run_dir / "reviewed_layout.json"
        if reviewed_layout_path.exists():
            reviewed_layout = _try_read_json(reviewed_layout_path)
            if isinstance(reviewed_layout, dict):
                pages = reviewed_layout.get("pages")
                result["reviewed_layout"] = {
                    "status": reviewed_layout.get("status"),
                    "page_count": len(pages) if isinstance(pages, list) else 0,
                    "summary": reviewed_layout.get("summary"),
                    "path": self.repo_paths.rel_to_repo(reviewed_layout_path),
                }

        ocr_path = run_dir / "ocr.json"
        if ocr_path.exists():
            ocr = _try_read_json(ocr_path)
            if isinstance(ocr, dict):
                pages = ocr.get("pages")
                result["ocr"] = {
                    "page_count": len(pages) if isinstance(pages, list) else 0,
                    "summary": ocr.get("summary"),
                    "path": self.repo_paths.rel_to_repo(ocr_path),
                }

        if predictions_dir.exists():
            predictions = []
            for path in sorted(predictions_dir.iterdir()):
                if not path.is_file():
                    continue
                prediction: dict[str, Any] = {
                    "name": path.name,
                    "path": self.repo_paths.rel_to_repo(path),
                    "size_bytes": path.stat().st_size,
                }
                if path.suffix == ".json":
                    payload = _try_read_json(path)
                    if isinstance(payload, dict):
                        prediction["top_level_keys"] = sorted(payload.keys())
                    elif payload is None:
                        prediction["warning"] = "invalid_json"
                predictions.append(prediction)
            result["predictions"] = predictions

        return result


class OCRWorkflowRunner:
    def __init__(
        self, config: DevMcpConfig, *, run_state_reader: RunStateReader | None = None
    ) -> None:
        self.config = config
        self.repo_paths = RepoPaths(config)
        self.run_state_reader = run_state_reader or RunStateReader(config)
        self._run_lock = threading.Lock()

    def run_ocr(self, *, input_path: str, run_id: str | None = None) -> dict[str, Any]:
        resolved_input_path = self.repo_paths.require_repo_path(input_path)
        raw_root = (self.config.repo_root / "data" / "raw").resolve()
        try:
            resolved_input_path.relative_to(raw_root)
        except ValueError as exc:
            raise RepoPathError(
                f"OCR input must be a PDF under data/raw: {resolved_input_path}"
            ) from exc
        if resolved_input_path.suffix.lower() != ".pdf":
            raise RepoPathError(f"OCR input must be a PDF under data/raw: {resolved_input_path}")
        if not resolved_input_path.exists() or not resolved_input_path.is_file():
            raise FileNotFoundError(f"OCR input file not found: {resolved_input_path}")

        normalized_run_id = self.repo_paths.validate_run_id(run_id) if run_id is not None else None
        repo_relative_input = self.repo_paths.rel_to_repo(resolved_input_path)
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError(
                "OCR is already running in the dev MCP. Wait for it to finish before starting another run."
            )
        try:
            run_dir = run_ocr_workflow(
                str(resolved_input_path),
                run=normalized_run_id,
                run_root=str(self.config.run_root),
                recorded_input_path=repo_relative_input,
            )
        finally:
            self._run_lock.release()
        run_state = self.run_state_reader.read_run_state(run_dir.name)
        return {
            "ok": True,
            "summary": f"Ran OCR for {repo_relative_input}",
            "run_id": run_dir.name,
            "input_path": repo_relative_input,
            "run_dir": str(run_dir),
            "run_state": run_state,
            "next_suggested_actions": [
                "Use read_run_state with this run_id to inspect OCR artifacts again.",
                "Use start_ui and save_feedback_bundle(capture_route=...) to review the result visually.",
            ],
        }


class FeedbackBundleStore:
    def __init__(self, config: DevMcpConfig, *, ui_manager: UIProcessManager | None = None) -> None:
        self.config = config
        self.repo_paths = RepoPaths(config)
        self.ui_manager = ui_manager or UIProcessManager(config)

    async def save_feedback_bundle(
        self,
        *,
        run_id: str,
        summary: str,
        issues: list[dict[str, Any]],
        notes: str | None = None,
        screenshot_paths: list[str] | None = None,
        capture_route: str | None = None,
        wait_for_selector: str | None = None,
        full_page: bool = True,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
        related_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        run_dir = self.repo_paths.require_run_id(run_id)
        self.config.ensure_runtime_dirs()

        bundle_id = f"{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}--{_slug(run_id)}"
        bundle_dir = self.config.feedback_root / bundle_id
        attachments_dir = bundle_dir / "attachments"
        attachments_dir.mkdir(parents=True, exist_ok=True)

        copied_screenshots: list[dict[str, Any]] = []
        if capture_route:
            copied_screenshots.append(
                await self._capture_route_screenshot(
                    bundle_dir=bundle_dir,
                    route=capture_route,
                    wait_for_selector=wait_for_selector,
                    full_page=full_page,
                    viewport_width=viewport_width,
                    viewport_height=viewport_height,
                )
            )
        for index, raw_path in enumerate(screenshot_paths or [], start=1):
            source = Path(raw_path).expanduser().resolve()
            if not source.exists() or not source.is_file():
                raise FileNotFoundError(f"Screenshot path not found: {source}")
            target_name = f"screenshot-external-{index:02d}{source.suffix.lower() or '.bin'}"
            target = attachments_dir / target_name
            shutil.copy2(source, target)
            copied_screenshots.append(
                {
                    "kind": "external",
                    "original_path": str(source),
                    "copied_path": self.repo_paths.rel_to_repo(target),
                    "size_bytes": target.stat().st_size,
                }
            )

        validated_related_paths: list[dict[str, Any]] = []
        for raw_path in related_paths or []:
            resolved = self.repo_paths.require_repo_path(raw_path)
            validated_related_paths.append(
                {
                    "path": str(resolved),
                    "repo_relative_path": self.repo_paths.rel_to_repo(resolved),
                    "exists": resolved.exists(),
                }
            )

        manifest = {
            "bundle_id": bundle_id,
            "created_at": utc_now(),
            "run_id": run_id,
            "run_dir": self.repo_paths.rel_to_repo(run_dir),
            "summary": summary,
            "notes": notes or "",
            "issues": issues,
            "screenshots": copied_screenshots,
            "capture_route": capture_route,
            "related_paths": validated_related_paths,
        }
        manifest_path = bundle_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {
            "ok": True,
            "summary": f"Saved feedback bundle {bundle_id}",
            "bundle_id": bundle_id,
            "manifest_path": str(manifest_path),
            "artifact_paths": [
                str(manifest_path),
                *(item["copied_path"] for item in copied_screenshots),
            ],
            "next_suggested_actions": [
                "Use list_feedback_bundles to review saved UX feedback.",
                "Capture another bundle after each UX review pass to keep screenshot evidence current.",
            ],
        }

    async def _capture_route_screenshot(
        self,
        *,
        bundle_dir: Path,
        route: str,
        wait_for_selector: str | None,
        full_page: bool,
        viewport_width: int | None,
        viewport_height: int | None,
    ) -> dict[str, Any]:
        ui_status = self.ui_manager.ui_status()
        if not ui_status.get("running"):
            started = self.ui_manager.start_ui()
            if not started.get("running"):
                raise RuntimeError(
                    f"Could not start UI for screenshot capture: {started.get('summary')}"
                )
            ui_status = started

        ui_url = str(ui_status.get("ui_url") or self.config.ui_base_url)
        target_route = route if route.startswith("/") else f"/{route}"
        target_url = f"{ui_url.rstrip('/')}{target_route}"
        attachments_dir = bundle_dir / "attachments"
        attachments_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = attachments_dir / "screenshot-01.png"

        try:
            playwright_async_api = importlib.import_module("playwright.async_api")
            playwright_error = getattr(playwright_async_api, "Error")
            async_playwright = getattr(playwright_async_api, "async_playwright")
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is not installed. Run `uv sync --group dev --extra dev-mcp` and `uv run playwright install chromium`."
            ) from exc

        try:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(headless=True)
                page = await browser.new_page(
                    viewport={
                        "width": viewport_width or self.config.default_viewport_width,
                        "height": viewport_height or self.config.default_viewport_height,
                    }
                )
                await page.goto(
                    target_url,
                    wait_until="domcontentloaded",
                    timeout=int(self.config.screenshot_timeout_seconds * 1000),
                )
                await page.wait_for_selector(
                    wait_for_selector or "flt-glass-pane",
                    state="attached",
                    timeout=int(self.config.screenshot_timeout_seconds * 1000),
                )
                await page.wait_for_timeout(int(self.config.screenshot_settle_delay_seconds * 1000))
                await page.screenshot(path=str(screenshot_path), full_page=full_page)
                await browser.close()
        except playwright_error as exc:
            raise RuntimeError(
                f"Could not capture screenshot for route {target_route}: {exc}"
            ) from exc

        return {
            "kind": "captured",
            "route": target_route,
            "ui_url": ui_url,
            "copied_path": self.repo_paths.rel_to_repo(screenshot_path),
            "size_bytes": screenshot_path.stat().st_size,
        }

    def list_feedback_bundles(
        self, *, run_id: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        self.config.ensure_runtime_dirs()
        manifests = sorted(self.config.feedback_root.glob("*/manifest.json"), reverse=True)
        bundles = []
        for manifest_path in manifests:
            payload = _try_read_json(manifest_path)
            if not isinstance(payload, dict):
                continue
            if run_id and payload.get("run_id") != run_id:
                continue
            bundles.append(
                {
                    "bundle_id": payload.get("bundle_id"),
                    "run_id": payload.get("run_id"),
                    "created_at": payload.get("created_at"),
                    "summary": payload.get("summary"),
                    "manifest_path": self.repo_paths.rel_to_repo(manifest_path),
                    "screenshot_count": len(payload.get("screenshots", [])),
                }
            )
            if len(bundles) >= max(limit, 1):
                break

        return {
            "ok": True,
            "summary": f"Found {len(bundles)} feedback bundle(s)",
            "bundles": bundles,
            "next_suggested_actions": [
                "Read a manifest.json file directly for the full saved payload.",
                "Capture another bundle after each UX review pass to keep feedback structured.",
            ],
        }
