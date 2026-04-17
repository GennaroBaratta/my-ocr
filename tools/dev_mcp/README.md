# Dev MCP sidecar

This repo includes a dev-only, localhost-only MCP sidecar for the Flet UX feedback loop.

## What it exposes

- `GET /healthz`
- streamable HTTP MCP at `POST /mcp`
- tools: `health`, `project_info`, `run_ocr`, `start_ui`, `stop_ui`, `ui_status`, `read_run_state`, `save_feedback_bundle`, `list_feedback_bundles`

The sidecar is intentionally narrow:

- it is rooted in this repository
- it does not expose arbitrary shell execution
- it stores its own runtime artifacts under `.dev-mcp/`
- it reuses the existing UI command in web mode so the agent can capture screenshots autonomously

## Install

```bash
uv sync --group dev --extra dev-mcp
uv run playwright install chromium
```

## Start the sidecar over HTTP

```bash
uv run python -m tools.dev_mcp --host 127.0.0.1 --port 8765
```

Health check:

```bash
curl http://127.0.0.1:8765/healthz
```

## Run it as a stdio MCP server for OpenCode

For an OpenCode `type: "local"` MCP entry, the process must speak MCP over stdio:

```bash
uv run python -m tools.dev_mcp --transport stdio
```

Keep HTTP mode for manual health checks and local debugging. Use stdio mode for OpenCode's local MCP transport.

## Run OCR end to end through MCP

Use `run_ocr` for repo-local PDFs under `data/raw/`.

- `input_path` must point to a PDF inside `data/raw`
- `run_id` is optional; omit it to let the workflow generate a fresh run name
- the tool reuses the existing OCR workflow and writes under `data/runs/<run_id>`
- the result can be inspected immediately through `read_run_state`

Example flow:

1. Call `project_info`.
2. Call `run_ocr(input_path="data/raw/PublicWaterMassMailing.pdf")`.
3. Call `read_run_state(run_id=...)` to inspect the generated artifacts.
4. Optionally call `start_ui` and `save_feedback_bundle(capture_route="/results/<run_id>")` to review the output visually.

## Save UX feedback safely

Use `save_feedback_bundle` after reviewing a run.

- `run_id` points at `data/runs/<run_id>`
- `issues` is a structured list you control
- `capture_route` lets the MCP open the Flet web UI and capture a screenshot by itself
- `screenshot_paths` still accepts externally captured screenshots when you want to attach extra evidence
- feedback bundles are saved under `.dev-mcp/feedback/<bundle_id>/manifest.json`

Example autonomous capture flow:

1. Call `start_ui`.
2. Call `save_feedback_bundle` with `capture_route="/results/<run_id>"` or `capture_route="/review/<run_id>"`.
3. Inspect the saved bundle under `.dev-mcp/feedback/<bundle_id>/`.

## Runtime files

The sidecar writes only dev artifacts under `.dev-mcp/`:

- `ui-process.json`
- `logs/ui.stdout.log`
- `logs/ui.stderr.log`
- `feedback/<bundle_id>/...`

## Stop or disable it

- stop the sidecar with `Ctrl+C`
- stop the tracked Flet UI with the `stop_ui` tool
- disable Codex integration by removing the repo-local config snippet you copied from `tools/dev_mcp/codex.mcp.example.json`

## Test it

```bash
uv run pytest tests/test_dev_mcp.py
```
