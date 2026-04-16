Deeper maintainability review of HEAD (ae80235): the refactor is headed in a reasonable direction, but it still leaves a few maintainability hazards. The biggest ones are split ownership of OCR recovery decisions, vendor-shaped data leaking into your own artifacts, and tests that are more coupled to internals than behavior.
- medium — src/free_doc_extract/ocr.py:132-205, src/free_doc_extract/workflows.py:270-308, tests/test_ocr.py:95-100,309-313  
  ocr.json now embeds full sdk_json per page even though the same raw payload already lives in ocr_raw/..._model.json. That couples your stable artifacts and tests to GLM-OCR’s schema, inflates payloads, and forces path-rewrite logic to traverse vendor-owned data.  
  Recommendation: keep raw model JSON only in ocr_raw/, and expose a stable reference in ocr.json such as sdk_json_path plus only the derived fields you own.
- medium — src/free_doc_extract/ocr.py:133-179, src/free_doc_extract/ocr_fallback.py:42-85  
  Fallback routing policy is split across modules. assess_crop_fallback() computes use_fallback, but _run_page_ocr() re-decides the path using separate branching. That makes future heuristic changes easy to desynchronize.  
  Recommendation: centralize strategy selection in one helper that returns an explicit outcome like sdk_markdown | layout_json | crop_fallback | full_page_fallback plus assessment metadata.
- medium — src/free_doc_extract/ocr_fallback.py:361-408, src/free_doc_extract/experimental/extract_glmocr.py:134-158  
  Table/HTML cleanup exists twice with different implementations. Those paths normalize the same kind of content, so drift is likely.  
  Recommendation: move table-to-text normalization into one shared helper and reuse it from both places.
- low — src/free_doc_extract/settings.py:30-115  
  resolve_ocr_api_client() now maintains a handwritten YAML fallback parser alongside yaml.safe_load(). That is effectively a second parser with narrower, implicit rules and silent fallback behavior.  
  Recommendation: either require YAML parsing for this feature or move OCR client overrides into a simpler format you fully control.
- low — tests/test_workflows.py:24-44,63-85,164-190,214-276,477-483,395-405,451-465, tests/test_ocr.py:230-250  
  Tests repeat setup/stub code and often assert deeply nested payloads verbatim. That raises churn for harmless refactors.  
  Recommendation: add shared fixtures/builders for fake OCR outputs and prefer contract-level subset assertions, keeping only a few full-payload shape tests.