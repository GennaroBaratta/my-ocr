"""Domain models and workflow data contracts."""

from __future__ import annotations

from my_ocr.domain._base import SCHEMA_VERSION, StrictModel, utc_now_iso
from my_ocr.domain._base import validate_optional_run_relative_path, validate_run_relative_path
from my_ocr.domain.artifacts import ArtifactCopy, ProviderArtifacts
from my_ocr.domain.document import DocumentFields, FIELD_NAMES, JSON_SCHEMA
from my_ocr.domain.errors import (
    ApplicationError,
    LayoutDetectionFailed,
    MissingInputDocument,
    MissingPage,
    OcrFailed,
    RunCommitFailed,
    RunNotFound,
    StructuredExtractionFailed,
)
from my_ocr.domain.ocr import OcrPageResult, OcrRunResult
from my_ocr.domain.options import OcrRuntimeOptions, StructuredExtractionOptions
from my_ocr.domain.results import LayoutDetectionResult, OcrRecognitionResult, RunSnapshot
from my_ocr.domain.results import WorkflowResult
from my_ocr.domain.review import LayoutBlock, ReviewLayout, ReviewPage
from my_ocr.domain.run import LayoutDiagnostics, PageRef, RunDiagnostics, RunId, RunInput
from my_ocr.domain.run import RunManifest, RunStatus

__all__ = [
    "SCHEMA_VERSION",
    "ApplicationError",
    "ArtifactCopy",
    "DocumentFields",
    "FIELD_NAMES",
    "JSON_SCHEMA",
    "LayoutBlock",
    "LayoutDetectionFailed",
    "LayoutDetectionResult",
    "LayoutDiagnostics",
    "MissingInputDocument",
    "MissingPage",
    "OcrFailed",
    "OcrPageResult",
    "OcrRecognitionResult",
    "OcrRunResult",
    "OcrRuntimeOptions",
    "PageRef",
    "ProviderArtifacts",
    "ReviewLayout",
    "ReviewPage",
    "RunCommitFailed",
    "RunDiagnostics",
    "RunId",
    "RunInput",
    "RunManifest",
    "RunNotFound",
    "RunSnapshot",
    "RunStatus",
    "StrictModel",
    "StructuredExtractionFailed",
    "StructuredExtractionOptions",
    "WorkflowResult",
    "utc_now_iso",
    "validate_optional_run_relative_path",
    "validate_run_relative_path",
]
