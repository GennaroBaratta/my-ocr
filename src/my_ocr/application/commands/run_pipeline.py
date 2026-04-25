from __future__ import annotations

from dataclasses import dataclass

from my_ocr.application.dto import LayoutOptions, OcrOptions, RunId, WorkflowResult
from my_ocr.application.ports import DocumentNormalizer, LayoutDetector, OcrEngine, RulesExtractor, RunStore

from .extract_rules import ExtractRules, ExtractRulesCommand
from .prepare_layout_review import PrepareLayoutReview, PrepareLayoutReviewCommand
from .run_reviewed_ocr import RunReviewedOcr, RunReviewedOcrCommand


@dataclass(frozen=True, slots=True)
class RunPipelineCommand:
    input_path: str
    run_id: RunId | None = None
    layout_options: LayoutOptions = LayoutOptions()
    ocr_options: OcrOptions = OcrOptions()


class RunPipeline:
    """Non-interactive fast path: automatic layout review, OCR, then rules extraction."""

    def __init__(
        self,
        run_store: RunStore,
        normalizer: DocumentNormalizer,
        layout_detector: LayoutDetector,
        ocr_engine: OcrEngine,
        rules_extractor: RulesExtractor,
    ) -> None:
        self._prepare = PrepareLayoutReview(run_store, normalizer, layout_detector)
        self._ocr = RunReviewedOcr(run_store, ocr_engine)
        self._rules = ExtractRules(run_store, rules_extractor)

    def __call__(self, command: RunPipelineCommand) -> WorkflowResult:
        prepared = self._prepare(
            PrepareLayoutReviewCommand(
                input_path=command.input_path,
                run_id=command.run_id,
                options=command.layout_options,
            )
        )
        run_id = prepared.snapshot.run_id
        self._ocr(RunReviewedOcrCommand(run_id=run_id, options=command.ocr_options))
        return self._rules(ExtractRulesCommand(run_id=run_id))

