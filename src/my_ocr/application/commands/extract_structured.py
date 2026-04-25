from __future__ import annotations

from dataclasses import dataclass

from my_ocr.application.dto import RunId, StructuredExtractionOptions, WorkflowResult
from my_ocr.application.errors import StructuredExtractionFailed
from my_ocr.application.ports import RunStore, StructuredExtractor
from my_ocr.application.services.structured_validation import validate_structured_prediction


@dataclass(frozen=True, slots=True)
class ExtractStructuredCommand:
    run_id: RunId
    options: StructuredExtractionOptions = StructuredExtractionOptions()


class ExtractStructured:
    def __init__(self, run_store: RunStore, structured_extractor: StructuredExtractor) -> None:
        self._run_store = run_store
        self._structured_extractor = structured_extractor

    def __call__(self, command: ExtractStructuredCommand) -> WorkflowResult:
        snapshot = self._run_store.open_run(command.run_id)
        markdown = snapshot.ocr_result.markdown if snapshot.ocr_result else None
        try:
            prediction, metadata = self._structured_extractor.extract(
                snapshot.pages,
                markdown_text=markdown,
                options=command.options,
            )
        except Exception as exc:
            raise StructuredExtractionFailed(f"Structured extraction failed: {exc}") from exc

        validation = validate_structured_prediction(prediction, source_text=markdown)
        canonical = prediction
        canonical_source = "structured"
        rules_prediction = snapshot.extraction.get("rules")
        if not validation["ok"] and isinstance(rules_prediction, dict):
            canonical = rules_prediction
            canonical_source = "rules"
        metadata = {**metadata, "canonical_source": canonical_source, "validation": validation}

        tx = self._run_store.begin_update(command.run_id)
        try:
            tx.write_structured_extraction(
                prediction,
                metadata,
                canonical_prediction=canonical,
            )
            return WorkflowResult(snapshot=tx.commit())
        except Exception:
            tx.rollback()
            raise

