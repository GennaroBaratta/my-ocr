from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from my_ocr.adapters.outbound.ocr import glmocr_engine
from my_ocr.application.artifacts import LayoutDetectionResult
from my_ocr.application.models import LayoutDiagnostics, PageRef
from my_ocr.application.options import LayoutOptions

from ._convert import provider_artifacts_from_pages, read_json, review_layout_from_legacy


class GlmOcrLayoutDetector:
    def detect_layout(
        self, pages: list[PageRef], options: LayoutOptions
    ) -> LayoutDetectionResult:
        temp_dir = Path(tempfile.mkdtemp(prefix="my-ocr-layout-"))
        try:
            result = glmocr_engine.prepare_review_artifacts(
                [str(page.path_for_io) for page in pages],
                temp_dir,
                config_path=options.config_path,
                layout_device=options.layout_device,
                layout_profile=options.layout_profile,
                page_numbers=[page.page_number for page in pages],
            )
            payload = read_json(temp_dir / "reviewed_layout.json")
            layout = review_layout_from_legacy(payload, pages, status="prepared")
            artifacts = provider_artifacts_from_pages(temp_dir / "ocr_raw", "layout/provider", pages)
            artifacts = type(artifacts)(artifacts.copies, (temp_dir,))
            return LayoutDetectionResult(
                layout=layout,
                artifacts=artifacts,
                diagnostics=LayoutDiagnostics.from_dict(result.get("layout_diagnostics")),
            )
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
