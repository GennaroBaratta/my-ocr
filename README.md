# my-ocr

<p align="center">
  <img src="docs/screenshots/Screenshot From 2026-04-20 23-39-07.png" alt="my-ocr OCR results workspace" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%20to%203.13-0f172a?style=flat-square" alt="Python 3.11 to 3.13" />
  <img src="https://img.shields.io/badge/runtime-local%20first-0f172a?style=flat-square" alt="Local first" />
  <img src="https://img.shields.io/badge/OCR-GLM--OCR%20%2B%20pluggable%20inference-0f172a?style=flat-square" alt="GLM-OCR and pluggable inference" />
  <img src="https://img.shields.io/badge/UI-Flet-0f172a?style=flat-square" alt="Flet UI" />
  <img src="https://img.shields.io/badge/tests-pytest-0f172a?style=flat-square" alt="pytest" />
</p>

**A local-first document extraction workbench for PDFs and scanned images.**

Drop in a document, **review and correct the detected layout before OCR runs**, then export markdown, JSON, and structured fields. The default runtime is local Ollama. Through `pipeline.inference`, you can also point my-ocr at an OpenAI-compatible or vLLM server you run yourself. No data leaves the machine unless you configure a remote provider.

Most OCR tools hide layout detection. my-ocr makes it a first-class, editable step because layout errors are often the root cause of bad OCR output.

The screenshots below are real GUI captures from a local run of [`docs/demo/PublicWaterMassMailing.pdf`](docs/demo/PublicWaterMassMailing.pdf), a public sample document checked into the repo for reproducible demos.

## Highlights

- **Review-first workflow.** Edit layout boxes per page before OCR becomes canonical.
- **Local-first stack.** OCR and structured extraction default to Ollama on `localhost`, with OpenAI-compatible/vLLM client support through `pipeline.inference`.
- **Reproducible runs.** Every run writes pages, raw OCR payloads, markdown, JSON, metadata, predictions, and reports to a single folder.
- **Two extraction paths.** Compare a deterministic rules baseline against direct structured generation.
- **Evaluation built in.** Gold labels and markdown reports let you measure changes, not guess.

## Pipeline

```text
  PDF / image
      │
      ▼
  ingest ──► normalize pages ──► detect layout
                                      │
                                      ▼
                         ┌── REVIEW & EDIT LAYOUT ──┐   ◄── the differentiator
                         │  (add / remove / retype) │
                         └────────────┬─────────────┘
                                      ▼
                                   run OCR
                                      │
                      ┌───────────────┴───────────────┐
                      ▼                               ▼
              rules extraction              structured extraction
                      │                               │
                      └───────────────┬───────────────┘
                                      ▼
                         evaluate vs. gold labels
```

## Product Walkthrough

### 1. Upload a document

Focused upload screen with a clear drop zone, lightweight run history, and local inference status.

<p align="center">
  <img src="docs/screenshots/Screenshot From 2026-04-20 23-28-05.png" alt="Upload workspace" width="88%" />
</p>

### 2. Review the layout

Inspect detected regions page by page. Add boxes, remove boxes, and fix the layout *before* OCR becomes canonical.

<p align="center">
  <img src="docs/screenshots/Screenshot From 2026-04-20 23-28-54.png" alt="Document review workspace" width="88%" />
</p>

### 3. Adjust region types

Per-box metadata lets you correct block type and coordinates for images, figures, and text before OCR runs.

<p align="center">
  <img src="docs/screenshots/Screenshot From 2026-04-20 23-29-31.png" alt="Layout review properties panel" width="88%" />
</p>

### 4. Inspect OCR results

Document, markdown, JSON, and raw page payloads sit side by side so debugging extraction quality is fast.

<p align="center">
  <img src="docs/screenshots/Screenshot From 2026-04-20 23-28-32.png" alt="Results workspace" width="88%" />
</p>

### 5. Validate the extracted text

Clean reading layout for comparing OCR markdown against the page preview before exporting.

<p align="center">
  <img src="docs/screenshots/Screenshot From 2026-04-20 23-39-07.png" alt="OCR markdown validation workspace" width="88%" />
</p>

## Quick Start

### Requirements

- Python `3.11`, `3.12`, or `3.13`
- [`uv`](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally for the default setup, or a separately managed OpenAI-compatible/vLLM endpoint configured in `config/local.yaml`
- NVIDIA driver for CUDA acceleration on Windows or Linux

### Install

```bash
uv python install 3.11
uv venv --python 3.11
uv sync --group dev --extra pdf --extra glmocr
```

The project pins PyTorch and TorchVision through uv's `tool.uv.sources` so Windows and Linux installs use the CUDA 13.0 wheels from `https://download.pytorch.org/whl/cu130`. On macOS, uv falls back to the normal PyPI wheels because PyTorch does not publish CUDA builds for macOS.

Verify the installed build:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Pull the OCR model

```bash
ollama pull glm-ocr:latest
ollama create glm-ocr-8k:latest -f Modelfile
ollama serve
```

Default OCR model: `glm-ocr-8k:latest`. Default inference provider: `ollama` at `http://localhost:11434`.

### Configure inference

Runtime inference settings live under `pipeline.inference` in `config/local.yaml`:

```yaml
pipeline:
  inference:
    provider: ollama
    base_url: http://localhost:11434
    model: glm-ocr-8k:latest
    num_ctx: 8192
    max_tokens:
    api_key:
    extra:
      keep_alive: 15m
```

For an OpenAI-compatible/vLLM server, point `base_url` at the `/v1` root. my-ocr maps it to `/v1/chat/completions` internally and does not probe the server during bootstrap. This is client support only: my-ocr does not launch or manage vLLM, so keep that server running outside the app.

```yaml
pipeline:
  inference:
    provider: openai_compatible
    base_url: http://localhost:8000/v1
    model: qwen2-vl
    api_key: local-token-if-needed
    max_tokens: 1024
    extra:
      top_k: 5
```

Config field behavior:

- `provider` must be `ollama` or `openai_compatible`.
- `base_url` is the provider root. Ollama resolves to `/api/generate`; OpenAI-compatible resolves to `/chat/completions` under the `/v1` root.
- `api_key` is optional. It is sent as a Bearer token for OpenAI-compatible providers and is normally left blank for local Ollama.
- `max_tokens` maps to Ollama `num_predict` or OpenAI-compatible `max_tokens`.
- `num_ctx` is an Ollama context option. For OpenAI-compatible servers, put provider-specific context settings in `extra` if that server accepts them.
- `extra` carries provider-specific request options. For OpenAI-compatible providers, extra fields are merged into the top-level raw HTTP `/chat/completions` JSON body after adapter-owned fields are built. Extras cannot override adapter-owned fields such as `model`, `messages`, `temperature`, `max_tokens`, or `response_format`. Deprecated `guided_*` fields are rejected.

Structured extraction requests JSON through the project-owned `StructuredOutputRequest` contract, then validates the result locally before it can become canonical. The app rejects schema echoes, invalid fields, and values missing from OCR text or images. If structured extraction looks suspicious or fails validation, deterministic `rules.json` stays canonical.

The layout detector uses `pipeline.layout.model_dir` in `config/local.yaml`, defaulting to `PaddlePaddle/PP-DocLayoutV3_safetensors`. If your Hugging Face cache is empty, Transformers may fetch that checkpoint on the first run. Later runs reuse the local cache, and the current run keeps the loaded detector in memory across pages. Review-first OCR reads the saved `layout/review.json` directly, so Run OCR does not start layout detection again. To pin storage, set `pipeline.layout.model_dir` to a local checkpoint directory.

### Launch the UI

```bash
# Desktop
uv run my-ocr-ui

# Browser
uv run python -m my_ocr.ui --web --host 127.0.0.1 --port 8550
```

### Run the pipeline (CLI)

A sample document ships at `docs/demo/PublicWaterMassMailing.pdf`, so the commands below work on a fresh clone.

| Command | What it does |
| --- | --- |
| `prepare-review <pdf> --run <id>` | Ingest, normalize pages, and prepare editable layout review |
| `run-reviewed-ocr --run <id>` | Run OCR using the saved reviewed layout |
| `ocr <pdf> --run <id>` | Non-interactive automatic layout detection and OCR |
| `extract-rules --run <id>` | Run the deterministic rules extractor |
| `extract-glmocr --run <id>` | Run structured extraction via GLM-OCR |
| `run <pdf> --run <id>` | Non-interactive automatic layout, OCR, and rules extraction |
| `eval --gold-dir ... --pred-dir ... --output ...` | Score predictions against gold labels |

End-to-end example:

```bash
# Automatic layout + OCR + rules extraction in one shot
uv run my-ocr run docs/demo/PublicWaterMassMailing.pdf --run demo001

# Optional: evaluate predictions against hand-labeled gold data
uv run my-ocr eval \
  --gold-dir data/gold \
  --pred-dir data/runs/demo001/extraction \
  --output data/reports/demo001.md
```

### Try other sample documents

Three additional public-sample PDFs ship in `docs/demo/` for quick experimentation:

- [`stub-cv.pdf`](docs/demo/stub-cv.pdf)
- [`stub-invoice.pdf`](docs/demo/stub-invoice.pdf)
- [`stub-research-note.pdf`](docs/demo/stub-research-note.pdf)

## What Gets Written to Disk

Each run lands in `data/runs/<run-id>/`.

```text
data/runs/demo001/
  run.json
  pages/
    page-0001.png
  layout/
    review.json
    provider/
      page-0001/
  ocr/
    markdown.md
    pages.json
    provider/
      page-0001/
    fallback/
      page-0001/
  extraction/
    rules.json
    structured.json
    structured_meta.json
    structured_raw.json
    canonical.json
```

Key files:

- `run.json`: v3 manifest with input metadata, immutable page identities, status, and diagnostics
- `layout/review.json`: page-by-page layout state used by the review step
- `ocr/markdown.md`: merged markdown output for the run
- `ocr/pages.json`: project-owned OCR payload with page records and relative artifact references
- `layout/provider/` and `ocr/provider/`: saved GLM-OCR provider payloads per page
- `extraction/`: rules, structured, raw structured metadata, and canonical extraction outputs

Committed run payloads use paths relative to the run folder. Recent-run discovery only lists folders with a valid v3 `run.json`.

### Artifact schema and migration stance

No artifact schemas changed during the clean-break `pipeline.inference` refactor. Existing v3 run folders continue to load through the current Pydantic models, and no migration or upgrade step is needed. If a future change updates `run.json`, `layout/review.json`, `ocr/pages.json`, or extraction outputs, that change must ship with migration notes, fixture updates, and schema tests.

## Design Decisions & Trade-offs

Choices that shaped the codebase:

- **Local-first over SaaS OCR.** Keeps documents on the machine and makes runs fully reproducible. Cost: latency on dense pages depends on local hardware.
- **Review-first layout correction.** OCR quality is bottlenecked by layout detection, so the UI exposes layout boxes as an editable stage instead of a hidden one. Cost: adds a human step; worth it for noisy scans.
- **Rules baseline alongside LLM extraction.** Gives a deterministic floor and a regression guard when LLM output drifts. Cost: the rules extractor is narrow by design.
- **Run-folder artifacts over a database.** Every run is a self-contained folder that can be zipped, diffed, or shared. Cost: no built-in query layer, swap in a DB if you need one.
- **Flet for the UI.** Python-only stack, no JS toolchain, fast to iterate for a single-machine workbench. Cost: smaller ecosystem than Electron/web stacks.
- **Intentional scope cuts.** No multi-tenant mode, no hosted deployment, no user management. This is a local workbench, not a product.

## Tech Stack

- **UI:** Flet
- **OCR pipeline:** GLM-OCR
- **Inference serving:** Ollama by default; OpenAI-compatible/vLLM endpoints via `pipeline.inference`
- **PDF / image normalization:** Pillow, PyMuPDF
- **Evaluation and reporting:** project-native Python utilities

## Repository Layout

The code is organized around product boundaries: `workflow.py` is the stable facade,
`use_cases/` owns orchestration behind ports, runtime integrations live in their feature
packages, and `runs/` owns run-folder persistence mechanics.

```text
src/my_ocr/
  cli.py
  bootstrap.py
  settings.py
  workflow.py
  domain/
    __init__.py
    _base.py
    artifacts.py
    document.py
    errors.py
    ocr.py
    options.py
    results.py
    review.py
    run.py
  use_cases/
    __init__.py
    ports.py
    review.py
    ocr.py
    extraction.py
  inference/
    __init__.py
    contracts.py
    ollama.py
    openai_compatible.py
  extraction/
    __init__.py
    canonical.py
    evaluation.py
    parse_json.py
    rules.py
    structured.py
    structured_prompt.py
    validation.py
  ingest/
    __init__.py
    normalize.py
    page_identity.py
  ocr/
    __init__.py
    bbox.py
    fallback.py
    glmocr.py
    glmocr_artifacts.py
    glmocr_retry.py
    glmocr_runtime.py
    glmocr_sdk.py
    labels.py
    layout_blocks.py
    layout_profile.py
    ollama_client.py
    ocr_policy.py
    page_processor.py
    review_mapping.py
    scratch_paths.py
    text_cleanup.py
  runs/
    __init__.py
    artifact_io.py
    artifacts.py
    invalidation.py
    manifest.py
    store.py
  support/
    __init__.py
    filesystem.py
    text.py
  ui/
    app.py
    controller.py
    review_controller.py
    session.py
    state.py
    features/
      upload/
      review/
      results/
    components/

data/
  raw/
  gold/
  runs/
  reports/
```

## Testing

The repo ships with a pytest suite covering CLI workflows, OCR glue code, review artifacts, evaluation, architecture boundaries, and the Flet UI components. Default tests are offline and deterministic; they don't require a live Ollama, OpenAI, or vLLM server.

```bash
make test     # uv run pytest
make lint     # uv run ruff check src tests
```

## Make Targets

```bash
make install                 # uv sync with dev + dev-mcp + pdf + glmocr extras
make test                    # run pytest
make lint                    # run ruff
make report RUN_ID=demo001   # regenerate evaluation report for a run
```

## Known Limits

- PDF rasterization requires the `pdf` extra.
- CUDA acceleration requires a compatible NVIDIA driver.
- OCR and structured extraction default to local Ollama. OpenAI-compatible/vLLM support is a client adapter only; run and manage that server separately.
- The rules extractor is intentionally narrow and should be treated as a baseline.
- Very large or visually dense pages can stress local inference latency.
- No CI/CD yet. Tests are run locally via `make test`.

## Dev Note

A localhost-only MCP sidecar lives at [`tools/dev_mcp/README.md`](tools/dev_mcp/README.md) for UX-review workflows and automated feedback capture.
