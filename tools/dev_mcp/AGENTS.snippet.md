Prefer the project-local dev MCP sidecar for UX review work when `http://127.0.0.1:8765/healthz` is healthy.

- Use `project_info` first to confirm repo-local paths.
- Use `run_ocr` for repo-local PDFs under `data/raw/` when you need a fresh OCR run.
- Use `read_run_state` after `run_ocr` to inspect the produced artifacts and run metadata.
- Use `start_ui` and `ui_status` to manage the Flet review session.
- Use `read_run_state` before `save_feedback_bundle` so UX notes reference concrete run artifacts.
- Prefer `save_feedback_bundle(capture_route=...)` for autonomous screenshot capture.
- Use external `screenshot_paths` only when attaching extra evidence beyond the MCP-captured screenshot.
