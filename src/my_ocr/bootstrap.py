from __future__ import annotations

from dataclasses import dataclass

from my_ocr.settings import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_LAYOUT_DEVICE,
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_RUN_ROOT,
)
from my_ocr.normalize import normalize_document
from my_ocr.storage import FilesystemRunReadModel
from my_ocr.storage import FilesystemRunStore
from my_ocr.extraction.structured import OllamaStructuredExtractor
from my_ocr.ocr.glmocr import GlmOcrEngine, GlmOcrLayoutDetector
from my_ocr.extraction.rules import extract_from_markdown
from my_ocr.workflow import DocumentWorkflow


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
            normalize_document,
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
