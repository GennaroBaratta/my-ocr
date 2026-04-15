# AGENT

## Workflow Rules

Understand → build the best path (delegated based on Agent rules, split and parallelized as much as possible) → execute → verify.

If delegating, launch the specialist in the same turn you mention it.

## Important Project Findings

### OCR pipeline

- Do **not** treat SDK `markdown_result` as the primary source of truth.
- Do **not** rely on SDK `result.json_result` for layout routing.
- Always save SDK artifacts and load layout from saved `ocr_raw/page-XXXX/page-XXXX_model.json`.
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
- Use task-specific prompts by label:
  - text-like → `Text Recognition:`
  - table → `Table Recognition:`
  - formula → `Formula Recognition:`
- Tables can be detected correctly in layout even when public SDK outputs look empty.
- Normalize recognized HTML table output into plain/Markdown-ish rows before writing chunk `.txt`, page markdown, `ocr.json`, and `ocr.md`.

### Published run artifacts

- OCR runs are staged into a temp dir first, then published.
- After publish, rewrite embedded path fields in `ocr.json` / `ocr_fallback.json` from staging paths to final run paths.
- Limit path rewriting to path-like keys (`*_path`, `*_paths`, `*_dir`) to avoid mutating OCR content.

### Structured extraction

- Structured extraction must use only values explicitly present in OCR text/images; no guessing/inference.
- Clean OCR text for extraction input only (table HTML flattening, tag stripping, light whitespace cleanup). Do not silently rewrite raw OCR artifacts for non-table cleanup.
- Validate structured outputs aggressively before making them canonical.
- Reject outputs that:
  - echo instructions/schema text (for example `Required: ...`)
  - contain hallucinated authors/institution/date not found in OCR text
- If structured extraction is suspicious or fails, prefer `rules.json` as canonical fallback.
