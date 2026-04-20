# AGENT

## Workflow Rules

Understand → build the best path (delegated based on Agent rules, split and parallelized as much as possible) → execute → verify.

If delegating, launch the specialist in the same turn you mention it.

## Important Project Findings

### OCR pipeline

- Do **not** treat SDK `markdown_result` as the primary source of truth.
- Do **not** rely on SDK `result.json_result` for layout routing.
- Always save SDK artifacts and load layout from saved `ocr_raw/page-XXXX/page-XXXX_model.json`.
- Published `ocr.json` must keep only project-owned fields and reference raw vendor payloads via `sdk_json_path`. Do **not** embed full SDK JSON in `ocr.json`.
- Save SDK artifacts to `ocr_raw/` root; the SDK creates its own `page-XXXX/` subdirectory. Passing a page dir directly causes double nesting.
- `bbox_2d` from GLM-OCR layout is typically in normalized `0..1000` coordinates, not pixels. Scale before cropping.
- If a layout block has no `index`, use block order as the fallback source index.
- The OCR flow is page-wise and should stay page-wise: normalized page images in → per-page OCR/fallback → aggregated `ocr.md` / `ocr.json`.

### OCR fallback behavior

- Current robust order is:
  1. SDK markdown
  2. markdown reconstructed from layout JSON
  3. crop fallback from layout blocks
  4. full-page fallback
- Keep OCR source selection centralized in `plan_page_ocr(...)`; do **not** re-decide the fallback route in multiple modules.
- Use task-specific prompts by label:
  - text-like → `Text Recognition:`
  - table → `Table Recognition:`
  - formula → `Formula Recognition:`
- Tables can be detected correctly in layout even when public SDK outputs look empty.
- Shared HTML/table normalization lives in `text_normalization.py`. Reuse it from both OCR cleanup and structured-extraction input cleanup instead of reimplementing table parsing.
- Normalize recognized HTML table output into plain/Markdown-ish rows before writing chunk `.txt`, page markdown, `ocr.json`, and `ocr.md`.

### Published run artifacts

- OCR runs are staged into a temp dir first, then published.
- Publish/replace only OCR-owned artifacts, not predictions.
- Publish must be transactional: back up current OCR artifacts, move staged artifacts into place, normalize published JSON paths, and restore the backup if publish fails.
- Treat `meta.json` as part of the same publish transaction. A metadata-write failure must roll back published OCR artifacts and restore the previous metadata state.
- Keep the backup alive until all post-publish steps succeed, including metadata writing. Do **not** delete the backup if restore fails.
- Restore from backup non-destructively. If rollback fails mid-restore, the preserved backup must still contain the full pre-publish state.
- For brand-new runs, failures before or during staging/publish must not leave behind an empty run directory that looks valid to downstream commands.
- After publish, rewrite only schema-owned path fields:
  - `ocr.json`: `page_path`, `sdk_json_path`
  - `ocr_fallback.json`: `page_path`, `crop_path`, `text_path`

### Structured extraction

- Structured extraction must use only values explicitly present in OCR text/images; no guessing/inference.
- Clean OCR text for extraction input only (table HTML flattening, tag stripping, light whitespace cleanup). Do not silently rewrite raw OCR artifacts for non-table cleanup.
- Validate structured outputs aggressively before making them canonical.
- Reject outputs that:
  - echo instructions/schema text (for example `Required: ...`)
  - contain hallucinated authors/institution/date not found in OCR text
- If structured extraction is suspicious or fails, prefer `rules.json` as canonical fallback.

### Config and tests

- OCR config parsing now depends on `PyYAML`; existing YAML config files are the supported source of OCR API overrides.
- If a config file exists but contains invalid YAML, raise `RuntimeError` instead of silently falling back to defaults.
- Prefer test builders/helpers in `tests/support.py` for OCR payloads and seeded runs.
- Prefer contract-level assertions in tests; keep only a few full artifact-shape checks.

### Dev MCP and continuous UX feedback

- Prefer the project-local `my-ocr-dev` MCP for OCR runs, UI inspection, and UX feedback whenever it is available and healthy.
- Treat the MCP as the default path for end-to-end UX iteration. Do not fall back to ad hoc CLI plus manual screenshots unless the MCP is broken.
- The app flow is intentionally staged: upload/prepare review -> review layout boxes -> run OCR -> results.
- Treat that staged flow as a product invariant, not just a UI suggestion.
- Intended user workflow:
  1. user uploads a document
  2. app performs the first pass for layout / review artifact preparation
  3. user reviews and can modify layout boxes
  4. user explicitly decides to start OCR
  5. app publishes OCR outputs and shows results
- Do **not** auto-start OCR immediately after upload or review preparation.
- Do **not** bypass the review screen when a run is only layout-prepared; review-ready runs should reopen at `/review/<run_id>`.
- After reviewed OCR completes, the results experience should be OCR-first: `ocr.md`, `ocr.json`, and per-page OCR payloads are the primary outputs.
- Do **not** present reviewed-OCR results as if structured extraction ran automatically.
- Results actions labeled copy/download JSON should operate on OCR JSON unless a future workflow explicitly adds a separate extraction step.
- The default UX loop for future sessions is:
  1. `project_info`
  2. `run_ocr(input_path="data/raw/<file>.pdf")` when a fresh run is needed
  3. `read_run_state(run_id=...)`
  4. `start_ui`
  5. `save_feedback_bundle(capture_route="/review/<run_id>")` or `save_feedback_bundle(capture_route="/results/<run_id>")`
  6. inspect the saved screenshot + manifest under `.dev-mcp/feedback/<bundle_id>/`
  7. repeat the same capture flow after changes for comparison
  8. `stop_ui` when done
- Use `list_feedback_bundles` before another UX pass to inspect prior feedback artifacts.
- The dev MCP currently supports `run_ocr`, `read_run_state`, `start_ui`, `stop_ui`, `ui_status`, `save_feedback_bundle`, and `list_feedback_bundles`.
- `run_ocr` is intentionally sandboxed to PDFs inside `data/raw/`.
- OCR runs inside the MCP are serialized; do not start overlapping OCR requests.
- Autonomous screenshot capture depends on Playwright + Chromium being installed.
- OpenCode local MCP transport for this project is stdio, not HTTP.
- The UI can be launched in web mode with `python -m free_doc_extract.ui --web --host 127.0.0.1 --port 8550`.

### Current UX findings

- The biggest current UX issues are in the results screen.
- Watch for duplicated page navigation cues.
- Watch for weak hierarchy between source preview, extracted content, and export actions.
- Watch for dense markdown/text presentation.
- Watch for export actions visually dominating review actions.
- The next UX iterations should focus on improving results-screen hierarchy first, then repeat the MCP feedback loop and compare bundles.

### Relevant paths

- `tools/dev_mcp/README.md` - dev MCP usage and workflow
- `tools/dev_mcp/AGENTS.snippet.md` - compact MCP usage guidance
- `src/free_doc_extract/ui/` - Flet screens/components/state
- `data/raw/` - allowed OCR input PDFs
- `data/runs/` - OCR/review artifacts
- `.dev-mcp/feedback/` - saved UX feedback bundles
