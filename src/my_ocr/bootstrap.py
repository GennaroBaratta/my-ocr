from __future__ import annotations

from dataclasses import dataclass

from my_ocr.adapters.outbound.config.settings import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_LAYOUT_DEVICE,
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_RUN_ROOT,
)
from my_ocr.adapters.outbound.filesystem.document_normalizer import (
    FilesystemDocumentNormalizer,
)
from my_ocr.adapters.outbound.filesystem.run_read_model import FilesystemRunReadModel
from my_ocr.adapters.outbound.filesystem.run_store import FilesystemRunStore
from my_ocr.adapters.outbound.glmocr import GlmOcrEngine, GlmOcrLayoutDetector
from my_ocr.adapters.outbound.llm.structured_extractor import OllamaStructuredExtractor
from my_ocr.application.commands import (
    ExtractRules,
    ExtractStructured,
    PrepareLayoutReview,
    RerunPageLayout,
    RerunPageOcr,
    RunPipeline,
    RunReviewedOcr,
)
from my_ocr.application.services.rules_extractor import extract_from_markdown


class RulesExtractorAdapter:
    def extract(self, markdown: str) -> dict[str, object]:
        return extract_from_markdown(markdown)


@dataclass(frozen=True, slots=True)
class BackendServices:
    run_store: FilesystemRunStore
    read_model: FilesystemRunReadModel
    prepare_layout_review: PrepareLayoutReview
    run_reviewed_ocr: RunReviewedOcr
    rerun_page_layout: RerunPageLayout
    rerun_page_ocr: RerunPageOcr
    extract_rules: ExtractRules
    extract_structured: ExtractStructured
    run_pipeline: RunPipeline


def build_backend_services(run_root: str = DEFAULT_RUN_ROOT) -> BackendServices:
    run_store = FilesystemRunStore(run_root)
    normalizer = FilesystemDocumentNormalizer()
    layout_detector = GlmOcrLayoutDetector()
    ocr_engine = GlmOcrEngine()
    rules_extractor = RulesExtractorAdapter()
    structured_extractor = OllamaStructuredExtractor()
    read_model = FilesystemRunReadModel(run_root)
    return BackendServices(
        run_store=run_store,
        read_model=read_model,
        prepare_layout_review=PrepareLayoutReview(run_store, normalizer, layout_detector),
        run_reviewed_ocr=RunReviewedOcr(run_store, ocr_engine),
        rerun_page_layout=RerunPageLayout(run_store, layout_detector),
        rerun_page_ocr=RerunPageOcr(run_store, ocr_engine),
        extract_rules=ExtractRules(run_store, rules_extractor),
        extract_structured=ExtractStructured(run_store, structured_extractor),
        run_pipeline=RunPipeline(
            run_store,
            normalizer,
            layout_detector,
            ocr_engine,
            rules_extractor,
        ),
    )


__all__ = [
    "BackendServices",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_LAYOUT_DEVICE",
    "DEFAULT_OLLAMA_ENDPOINT",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_RUN_ROOT",
    "build_backend_services",
]

