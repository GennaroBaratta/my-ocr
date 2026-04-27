from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from my_ocr.settings import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_LAYOUT_DEVICE,
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_RUN_ROOT,
    InferenceProviderConfig,
    resolve_inference_provider_config,
)
from my_ocr.inference import InferenceClient, OllamaClient, OpenAICompatibleClient
from my_ocr.ingest.normalize import normalize_document
from my_ocr.runs.store import FilesystemRunReadModel
from my_ocr.runs.store import FilesystemRunStore
from my_ocr.extraction.structured import ProviderNeutralStructuredExtractor
from my_ocr.ocr.glmocr import GlmOcrEngine, GlmOcrLayoutDetector
from my_ocr.extraction.rules import extract_from_markdown
from my_ocr.use_cases.ports import RunRepository
from my_ocr.workflow import DocumentWorkflow


@dataclass(frozen=True, slots=True)
class BackendServices:
    run_store: FilesystemRunStore
    read_model: FilesystemRunReadModel
    workflow: DocumentWorkflow
    inference_config: InferenceProviderConfig
    inference_client: InferenceClient


def build_backend_services(
    run_root: str = DEFAULT_RUN_ROOT,
    *,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> BackendServices:
    inference_config = resolve_inference_provider_config(config_path)
    inference_client = build_inference_client(inference_config)
    run_store = FilesystemRunStore(run_root)
    layout_detector = GlmOcrLayoutDetector()
    ocr_engine = GlmOcrEngine(
        inference_client=inference_client,
        inference_config=inference_config,
    )
    structured_extractor = ProviderNeutralStructuredExtractor(
        inference_client=inference_client,
        inference_config=inference_config,
    )
    read_model = FilesystemRunReadModel(run_root)
    workflow_run_store = cast(RunRepository, run_store)
    return BackendServices(
        run_store=run_store,
        read_model=read_model,
        inference_config=inference_config,
        inference_client=inference_client,
        workflow=DocumentWorkflow(
            workflow_run_store,
            normalize_document,
            layout_detector,
            ocr_engine,
            extract_from_markdown,
            structured_extractor,
        ),
    )


def build_inference_client(config: InferenceProviderConfig) -> InferenceClient:
    if config.provider == "ollama":
        return OllamaClient(
            endpoint=config.endpoint,
            model=config.model,
            timeout=config.timeout_seconds,
            default_max_tokens=config.max_tokens,
            default_num_ctx=config.num_ctx,
            default_extra=config.extra,
        )
    if config.provider == "openai_compatible":
        return OpenAICompatibleClient(
            endpoint=config.endpoint,
            model=config.model,
            timeout=config.timeout_seconds,
            api_key=config.api_key,
            default_max_tokens=config.max_tokens,
            default_extra=config.extra,
        )
    raise RuntimeError(f"Unsupported inference provider: {config.provider}")


__all__ = [
    "BackendServices",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_LAYOUT_DEVICE",
    "DEFAULT_OLLAMA_ENDPOINT",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_RUN_ROOT",
    "build_backend_services",
    "build_inference_client",
]
