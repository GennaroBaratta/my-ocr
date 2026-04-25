# Pipeline Refactor Plan

## Target Pipeline Package

Move the document-processing orchestration into `my_ocr.application.pipeline`.

The package should own the application-level pipeline API currently centered on
`my_ocr.application.workflow.DocumentWorkflow`:

- preparing review layouts
- saving edited review layouts
- running OCR from reviewed layouts
- rerunning layout or OCR for a single page
- extracting rules and structured payloads
- running the non-interactive automatic pipeline

`my_ocr.adapters.inbound.cli`, `my_ocr.ui`, and `tools/dev_mcp` should continue to depend on a
stable workflow service returned by `my_ocr.bootstrap.build_backend_services`, even if the concrete
implementation moves from `workflow.py` into the new package.

## Temporary Compatibility Layers

These files are compatibility layers or legacy bridges and should stay narrow while the pipeline is
being moved:

- `src/my_ocr/application/workflow.py` - current orchestration surface. During the move, keep it as
  an import/API shim if callers still import `DocumentWorkflow`.
- `src/my_ocr/application/use_cases/__init__.py` - explicitly marked as a legacy package retained
  for evaluation helpers during the v2 command migration.
- `src/my_ocr/application/use_cases/evaluation.py` - evaluation helper still imported by the CLI and
  tests from the legacy `use_cases` package.
- `src/my_ocr/adapters/outbound/glmocr/_convert.py` - translates legacy GLM-OCR review/OCR payloads
  into project-owned v2 application models.
- `src/my_ocr/adapters/outbound/glmocr/layout_detector.py` - adapter bridge that runs the legacy
  GLM-OCR artifact flow in a temp folder and returns `LayoutDetectionResult`.
- `src/my_ocr/adapters/outbound/glmocr/ocr_engine.py` - adapter bridge that feeds reviewed layouts
  to the legacy GLM-OCR runner and returns `OcrRecognitionResult`.
- `src/my_ocr/adapters/outbound/filesystem/run_paths.py` - adapter-internal legacy run layout used
  only by the GLM-OCR bridge temp folders.
- `src/my_ocr/adapters/outbound/ocr/glmocr_engine.py` - legacy GLM-OCR runner used behind the newer
  application ports; avoid adding new application-level behavior here.

## Constraints

- No CLI command changes. Keep existing `my-ocr` subcommands, arguments, defaults, and exit behavior
  intact: `prepare-review`, `run-reviewed-ocr`, `ocr`, `extract-rules`, `extract-glmocr`, `eval`,
  and `run`.
- No run-folder shape changes. Preserve the v2 folder contract under `data/runs/<run-id>/`,
  including `run.json`, `pages/`, `layout/review.json`, `layout/provider/`, `ocr/markdown.md`,
  `ocr/pages.json`, `ocr/provider/`, `ocr/fallback/`, and `extraction/`.
- No Flet breakage. Keep `my-ocr-ui`, `AppState`, `WorkflowController`, and screen/component flows
  working against `build_backend_services().workflow`.
- Keep adapter-specific legacy formats contained inside adapter compatibility layers. Application
  and UI code should continue to use project-owned models such as `RunSnapshot`, `ReviewLayout`,
  `OcrRunResult`, and `WorkflowResult`.
- Preserve architecture boundaries: domain does not import application/adapters/UI; application does
  not import adapters/UI; UI does not import removed command layers or filesystem run stores directly.

## Required Verification

Run these commands before considering the refactor complete:

```bash
uv run --group dev-mcp pytest
uv run ruff check src tests
```

Equivalent make targets are acceptable when they stay in sync:

```bash
make test
make lint
```
