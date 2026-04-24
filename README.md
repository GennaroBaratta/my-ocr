# my-ocr

<p align="center">
  <img src="docs/screenshots/Screenshot From 2026-04-20 23-39-07.png" alt="my-ocr OCR results workspace" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%20to%203.13-0f172a?style=flat-square" alt="Python 3.11 to 3.13" />
  <img src="https://img.shields.io/badge/runtime-local%20first-0f172a?style=flat-square" alt="Local first" />
  <img src="https://img.shields.io/badge/OCR-GLM--OCR%20%2B%20Ollama-0f172a?style=flat-square" alt="GLM-OCR and Ollama" />
  <img src="https://img.shields.io/badge/UI-Flet-0f172a?style=flat-square" alt="Flet UI" />
  <img src="https://img.shields.io/badge/tests-pytest-0f172a?style=flat-square" alt="pytest" />
</p>

**A local-first document extraction workbench for PDFs and scanned images.**

Drop a document in, **review and correct the detected layout before OCR runs**, then export markdown, JSON, and structured fields. Everything runs against a local Ollama endpoint — no hosted SaaS, no data leaving the machine.

What makes it different: most OCR tools treat layout detection as a hidden step. Here it is a first-class, editable stage, because layout errors are the single largest cause of bad OCR output downstream.

The screenshots below are real GUI captures from a local run of [`docs/demo/PublicWaterMassMailing.pdf`](docs/demo/PublicWaterMassMailing.pdf), a public sample document checked into the repo for reproducible demos.

## Highlights

- **Review-first workflow.** Edit layout boxes per page before OCR becomes canonical.
- **Local stack.** OCR and structured extraction talk to Ollama on `localhost`.
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

Focused upload screen with a clear drop zone, lightweight run history, and local Ollama status.

<p align="center">
  <img src="docs/screenshots/Screenshot From 2026-04-20 23-28-05.png" alt="Upload workspace" width="88%" />
</p>

### 2. Review the layout

The core idea of the project: inspect detected regions, page by page, add boxes, remove boxes, and fix the layout *before* OCR becomes canonical.

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
- [Ollama](https://ollama.com/) running locally
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

Default OCR model: `glm-ocr-8k:latest`. Default OCR endpoint: `http://localhost:11434/api/generate`.

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
| `ocr <pdf> --run <id>` | Ingest, layout-detect, and OCR a document |
| `extract-rules --run <id>` | Run the deterministic rules extractor |
| `extract-glmocr --run <id>` | Run structured extraction via GLM-OCR |
| `run <pdf> --run <id>` | End-to-end: OCR + rules extraction |
| `eval --gold-dir ... --pred-dir ... --output ...` | Score predictions against gold labels |

End-to-end example:

```bash
# OCR + rules extraction in one shot
uv run my-ocr run docs/demo/PublicWaterMassMailing.pdf --run demo001

# Optional: evaluate predictions against hand-labeled gold data
uv run my-ocr eval \
  --gold-dir data/gold \
  --pred-dir data/runs/demo001/predictions \
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
  pages/
  ocr_raw/
  reviewed_layout.json
  ocr.md
  ocr.json
  ocr_fallback.json
  meta.json
  predictions/
```

Key files:

- `reviewed_layout.json` — page-by-page layout state used by the review step
- `ocr.md` — merged markdown output for the run
- `ocr.json` — project-owned OCR payload with page records and references to raw SDK artifacts
- `ocr_raw/` — saved GLM-OCR model payloads per page
- `predictions/` — rules and structured extraction outputs

## Design Decisions & Trade-offs

Choices that shaped this codebase, and what they cost:

- **Local-first over SaaS OCR.** Keeps documents on the machine and makes runs fully reproducible. Cost: latency on dense pages depends on local hardware.
- **Review-first layout correction.** OCR quality is bottlenecked by layout detection, so the UI exposes layout boxes as an editable stage instead of a hidden one. Cost: adds a human step; worth it for noisy scans.
- **Rules baseline alongside LLM extraction.** Gives a deterministic floor and a regression guard when LLM output drifts. Cost: the rules extractor is narrow by design.
- **Run-folder artifacts over a database.** Every run is a self-contained folder that can be zipped, diffed, or shared. Cost: no built-in query layer — swap in a DB if you need one.
- **Flet for the UI.** Python-only stack, no JS toolchain, fast to iterate for a single-machine workbench. Cost: smaller ecosystem than Electron/web stacks.
- **Intentional scope cuts.** No multi-tenant mode, no hosted deployment, no user management. This is a local workbench, not a product.

## Tech Stack

- **UI:** Flet
- **OCR pipeline:** GLM-OCR
- **Local inference serving:** Ollama
- **PDF / image normalization:** Pillow, PyMuPDF
- **Evaluation and reporting:** project-native Python utilities

## Repository Layout

```text
src/my_ocr/
  domain/
    document.py
    layout.py
    page_identity.py
    review_layout.py
    text.py
  application/
    ports.py
    services/
    use_cases/
  adapters/
    inbound/cli.py
    outbound/
      config/
      filesystem/
      llm/
      ocr/
  ui/

data/
  raw/
  gold/
  runs/
  reports/
```

## Testing

The repo ships with a pytest suite covering CLI workflows, OCR glue code, review artifacts, evaluation, architecture boundaries, and the Flet UI components.

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
- OCR and structured extraction both assume a local Ollama setup is available.
- The rules extractor is intentionally narrow and should be treated as a baseline.
- Very large or visually dense pages can stress local inference latency.
- No CI/CD yet — tests are run locally via `make test`.

## Dev Note

A localhost-only MCP sidecar lives at [`tools/dev_mcp/README.md`](tools/dev_mcp/README.md) for UX-review workflows and automated feedback capture.
