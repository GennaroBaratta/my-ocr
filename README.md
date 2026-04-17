# free-doc-extract

Local OCR and document-field extraction pipeline built around GLM-OCR + Ollama.

## What it does

1. Ingest a PDF or image.
2. Normalize input into ordered page images.
3. Run local OCR through GLM-OCR via Ollama.
4. Save OCR markdown, JSON, and raw SDK artifacts.
5. Extract a small fixed schema with:
   - deterministic rules over OCR text
   - direct structured extraction through Ollama `/api/generate`
6. Evaluate predictions against hand-labeled gold annotations.
7. Save reproducible run outputs and Markdown reports.

## Repository layout

```text
free-doc-extract/
  README.md
  pyproject.toml
  Makefile
  config/
    local.yaml
  data/
    raw/
    gold/
    runs/
    reports/
  src/free_doc_extract/
    cli.py
    ingest.py
    ocr.py
    extract_rules.py
    experimental/
      extract_glmocr.py
    schema.py
    evaluate.py
    utils.py
  tests/
    test_schema.py
    test_extract_rules.py
    test_smoke_pipeline.py
```

## Setup

### 1. Create an environment with `uv`

```bash
uv python install 3.11
uv venv --python 3.11
uv sync --group dev --extra pdf --extra glmocr
```

To use the dev-only MCP sidecar for UX review, install its optional extra too:

```bash
uv sync --group dev --extra pdf --extra glmocr --extra dev-mcp
```

Run commands with `uv run`:

```bash
uv run python -m free_doc_extract.cli --help
uv run pytest
```

### Why this is the default

- `glmocr[selfhosted]` is more likely to work on Python 3.11/3.12 than on 3.14.
- `uv sync` installs from `pyproject.toml` directly.
- The `dev` tools are modeled as a `uv` dependency group, while `pdf` and `glmocr` remain optional extras.

### Alternative: editable install with `uv pip`

```bash
uv venv --python 3.11
uv pip install -e .[dev,pdf,glmocr]
```

Do not mix these styles in one step. In particular, avoid:

```bash
uv run pip install -e .[dev,pdf,glmocr]
```

### 2. Pull and serve the model

```bash
ollama pull glm-ocr-8k
# or update config/local.yaml to point at a different local model tag

ollama serve
```

### 3. Configure GLM-OCR

The default local config is already checked in at `config/local.yaml` and points to Ollama at `localhost:11434` using `/api/generate`.

## Commands

### OCR a document

```bash
uv run python -m free_doc_extract.cli ocr data/raw/sample.pdf --run demo001
```

Artifacts are written to `data/runs/demo001/`:

- `pages/`
- `ocr_raw/`
- `ocr.md`
- `ocr.json`
- `meta.json`

`ocr.json` stores project-owned OCR results per page and references the saved GLM-OCR payload via
`sdk_json_path`. Raw vendor JSON remains under `ocr_raw/` and is not embedded into `ocr.json`.

### Run rule-based extraction over OCR

```bash
uv run python -m free_doc_extract.cli extract-rules --run demo001
```

Writes:

- `data/runs/demo001/predictions/rules.json`

### Run direct structured extraction through Ollama

This path is experimental and lives under `src/free_doc_extract/experimental/`.

```bash
uv run python -m free_doc_extract.cli extract-glmocr --run demo001
```

Writes:

- `data/runs/demo001/predictions/glmocr_structured.json`
- `data/runs/demo001/predictions/glmocr_structured_meta.json`

### Run a small end-to-end pipeline

```bash
uv run python -m free_doc_extract.cli run data/raw/sample.pdf --run demo001
```

This normalizes pages, runs OCR, and writes the rule-based extraction result.

### Evaluate against gold labels

```bash
uv run python -m free_doc_extract.cli eval \
  --gold-dir data/gold \
  --pred-dir data/runs/demo001/predictions \
  --output data/reports/demo001.md
```

## Gold data format

Each gold file should be a JSON document keyed by input stem, for example:

`data/gold/sample.json`

```json
{
  "document_type": "report",
  "title": "Sample Report",
  "authors": ["Ada Lovelace"],
  "institution": "Analytical Engine Institute",
  "date": "2024-01-15",
  "language": "en",
  "summary_line": "A short summary from the document."
}
```

## Make targets

```bash
make install
make test
make lint
make report RUN_ID=demo001
```

## Dev MCP sidecar for UX review

The repo includes a localhost-only MCP sidecar under `tools/dev_mcp/` for iterating on the Flet UX and saving structured feedback bundles, including autonomous screenshot capture through Playwright. See `tools/dev_mcp/README.md` for setup, browser install, start, disable, and test instructions.

## Known limitations

- PDF rasterization requires the `pdf` extra (`PyMuPDF`).
- The experimental direct structured extractor prefers OCR markdown and otherwise falls back to the first page image; very large pages can still stress Ollama.
- Language detection in the rules baseline is heuristic.
- The rule extractor is intentionally simple and should be treated as a baseline, not a production parser.

## Suggested next steps

- Add a hand-labeled dataset under `data/gold/`.
- Record latency and token metadata from Ollama batch runs.
- Add an `error_analysis.md` file once you have model outputs to inspect.
