from __future__ import annotations

from dataclasses import dataclass

from my_ocr.settings import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_LAYOUT_DEVICE,
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_RUN_ROOT,
)
from my_ocr.ingest.normalize import normalize_document
from my_ocr.runs.store import FilesystemRunReadModel
from my_ocr.runs.store import FilesystemRunStore
from my_ocr.extraction.structured import OllamaStructuredExtractor
from my_ocr.ocr.glmocr import GlmOcrEngine, GlmOcrLayoutDetector
from my_ocr.extraction.rules import extract_from_markdown
from my_ocr.workflow import DocumentWorkflow


@dataclass(frozen=True, slots=True)
class BackendServices:
    run_store: FilesystemRunStore
    read_model: FilesystemRunReadModel
    workflow: DocumentWorkflow


def build_backend_services(run_root: str = DEFAULT_RUN_ROOT) -> BackendServices:
    run_store = FilesystemRunStore(run_root)
    layout_detector = GlmOcrLayoutDetector()
    ocr_engine = GlmOcrEngine()
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
            extract_from_markdown,
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
