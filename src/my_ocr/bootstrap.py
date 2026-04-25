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
from my_ocr.adapters.outbound.llm.structured_extractor import OllamaStructuredExtractor
from my_ocr.adapters.outbound.ocr.glmocr_engine import GlmOcrEngine, GlmOcrLayoutDetector
from my_ocr.pipeline.extraction import extract_from_markdown
from my_ocr.pipeline.workflow import DocumentWorkflow


class RulesExtractorAdapter:
    def extract(self, markdown: str) -> dict[str, object]:
        return extract_from_markdown(markdown)


@dataclass(frozen=True, slots=True)
class BackendServices:
    run_store: FilesystemRunStore
    read_model: FilesystemRunReadModel
    workflow: DocumentWorkflow


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
        workflow=DocumentWorkflow(
            run_store,
            normalizer,
            layout_detector,
            ocr_engine,
            rules_extractor,
            structured_extractor,
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
