from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from my_ocr.adapters.outbound.ocr import glmocr_engine
from my_ocr.application.artifacts import OcrRecognitionResult
from my_ocr.application.models import PageRef, ReviewLayout
from my_ocr.application.options import OcrOptions

from ._convert import (
    combined_artifacts,
    legacy_review_payload,
    ocr_result_from_legacy,
    provider_artifacts_from_pages,
    read_json,
    write_json,
)


class GlmOcrEngine:
    def recognize(
        self,
        pages: list[PageRef],
        review: ReviewLayout | None,
        options: OcrOptions,
    ) -> OcrRecognitionResult:
        temp_dir = Path(tempfile.mkdtemp(prefix="my-ocr-ocr-"))
        review_path: Path | None = None
        try:
            if review is not None:
                review_path = temp_dir / "reviewed_layout.json"
                write_json(review_path, legacy_review_payload(review, pages))
            result = glmocr_engine.run_ocr(
                [str(page.path_for_io) for page in pages],
                temp_dir,
                config_path=options.config_path,
                layout_device=options.layout_device,
                layout_profile=options.layout_profile,
                reviewed_layout_path=review_path,
                page_numbers=[page.page_number for page in pages],
            )
            payload = read_json(temp_dir / "ocr.json")
            markdown = (temp_dir / "ocr.md").read_text(encoding="utf-8")
            ocr_result = ocr_result_from_legacy(payload, markdown, pages)
            ocr_result = type(ocr_result)(
                pages=ocr_result.pages,
                markdown=ocr_result.markdown,
                diagnostics=dict(result.get("layout_diagnostics", {})),
            )
            artifacts = combined_artifacts(
                provider_artifacts_from_pages(temp_dir / "ocr_raw", "ocr/provider", pages),
                provider_artifacts_from_pages(temp_dir / "ocr_fallback", "ocr/fallback", pages),
            )
            artifacts = type(artifacts)(artifacts.copies, (temp_dir,))
            return OcrRecognitionResult(result=ocr_result, artifacts=artifacts)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
