from __future__ import annotations

from my_ocr.domain import (
    OcrFailed,
    RunId,
    StructuredExtractionFailed,
    StructuredExtractionOptions,
    WorkflowResult,
)
from my_ocr.extraction.canonical import choose_canonical_prediction
from .ports import RulesExtractor, RunRepository, StructuredExtractor


class ExtractionUseCase:
    def __init__(
        self,
        run_store: RunRepository,
        rules_extractor: RulesExtractor,
        structured_extractor: StructuredExtractor,
    ) -> None:
        self._run_store = run_store
        self._rules_extractor = rules_extractor
        self._structured_extractor = structured_extractor

    def extract_rules(self, run_id: RunId) -> WorkflowResult:
        snapshot = self._run_store.open_run(run_id)
        if snapshot.ocr_result is None or not snapshot.ocr_result.markdown.strip():
            raise OcrFailed("Rules extraction requires OCR markdown.")
        prediction = self._rules_extractor(snapshot.ocr_result.markdown)
        snapshot = self._run_store.write_rules_extraction(run_id, prediction)
        return WorkflowResult(snapshot=snapshot)

    def extract_structured(
        self,
        run_id: RunId,
        options: StructuredExtractionOptions = StructuredExtractionOptions(),
    ) -> WorkflowResult:
        snapshot = self._run_store.open_run(run_id)
        markdown = snapshot.ocr_result.markdown if snapshot.ocr_result else None
        if not markdown or not markdown.strip():
            raise OcrFailed("Structured extraction requires OCR markdown.")

        rules_prediction = snapshot.extraction.get("rules")
        if not isinstance(rules_prediction, dict):
            rules_prediction = self._rules_extractor(markdown)
            snapshot = self._run_store.write_rules_extraction(run_id, rules_prediction)

        try:
            prediction, metadata = self._structured_extractor.extract(
                snapshot.pages,
                markdown_text=markdown,
                options=options,
            )
        except Exception as exc:
            raise StructuredExtractionFailed(f"Structured extraction failed: {exc}") from exc

        canonical, canonical_metadata = choose_canonical_prediction(
            structured_prediction=prediction,
            rules_prediction=rules_prediction,
            source_text=markdown,
        )
        metadata = {**metadata, **canonical_metadata}

        snapshot = self._run_store.write_structured_extraction(
            run_id,
            prediction,
            metadata,
            canonical_prediction=canonical,
        )
        return WorkflowResult(snapshot=snapshot)


__all__ = ["ExtractionUseCase"]
