# Planned Next Tests

## Priority 1

### 1. Ingest: directory page ordering
- Verify `normalize_document()` preserves human page order with filenames like:
  - `page-1.png`
  - `page-2.png`
  - `page-10.png`
- Expected: natural sort, then copied to sequential `page-0001.*`, `page-0002.*`, `page-0003.*`.

### 2. Ingest: single image input
- Verify a single image is copied into `run_dir/pages/`.
- Expected: one normalized page path returned and file exists.

### 3. Ingest: unsupported input type
- Pass an unsupported file extension.
- Expected: raises `ValueError` with a clear message.

### 4. OCR wrapper: empty page list
- Call `run_ocr([])`.
- Expected: raises `ValueError`.

### 5. OCR wrapper: missing GLM-OCR dependency
- Stub import failure for `glmocr`.
- Expected: raises `RuntimeError` with install guidance.

### 6. Structured extraction: response parsing
- Test `_parse_response_json()` with:
  - valid raw JSON
  - fenced ```json blocks
  - extra text around JSON
- Expected: valid object returned in each recoverable case.

## Priority 2

### 7. Structured extraction: bad Ollama response
- Simulate responses with:
  - missing `response`
  - invalid JSON payload
  - non-object JSON
- Expected: clear `RuntimeError` paths.

### 8. Structured extraction: network failure
- Simulate `URLError` / `HTTPError`.
- Expected: user-readable failure messages.

### 9. CLI: `extract-glmocr`
- Stub Ollama call and verify:
  - page images are loaded from `run_dir/pages/`
  - predictions are written to `predictions/glmocr_structured.json`
  - metadata is written to `predictions/glmocr_structured_meta.json`

### 10. CLI: `ocr`
- Stub ingest + OCR and verify `meta.json`, `ocr.md`, and `ocr.json` exist after command execution.

### 11. CLI: missing OCR markdown for rules extraction
- Run `extract-rules` on a run directory without `ocr.md`.
- Expected: `FileNotFoundError`.

## Priority 3

### 12. Evaluation: missing prediction accounting
- Provide gold files where one prediction is absent.
- Expected:
  - document still counted
  - missing prediction count increments
  - missing file name listed in summary

### 13. Evaluation: date normalization
- Compare equivalent date formats such as:
  - `2024-01-15`
  - `15/01/2024`
  - `January 15, 2024`
- Expected: normalized date match passes.

### 14. Rules extractor: sparse OCR text
- Pass short/noisy OCR text.
- Expected: returns schema with empty defaults instead of crashing.

### 15. Rules extractor: document-type heuristics
- Add focused samples for:
  - report
  - article
  - letter
  - form
  - invoice
- Expected: stable type classification for obvious cases.

## Later integration tests

### 16. Real PDF smoke test
- Mark as optional/integration.
- Requires `PyMuPDF` and a tiny sample PDF fixture.
- Expected: PDF renders to page images and pipeline completes.

### 17. Real Ollama smoke test
- Mark as optional/integration.
- Requires local Ollama + `glm-ocr` model.
- Expected: one sample document completes OCR / structured extraction end to end.

### 18. Reproducibility test
- Run the same stubbed pipeline twice.
- Expected: stable file layout and schema-compatible outputs.

## Suggested test files

- `tests/test_ingest.py`
- `tests/test_ocr.py`
- `tests/test_extract_glmocr.py`
- `tests/test_cli.py`
- expand `tests/test_evaluate.py`
