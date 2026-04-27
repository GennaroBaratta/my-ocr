from __future__ import annotations

from pathlib import Path
from typing import Any

from my_ocr.domain import ReviewPage
from my_ocr.inference import InferenceClient
from my_ocr.ocr import fallback as _fallback_mod
from my_ocr.ocr import glmocr_artifacts as _artifacts_mod
from my_ocr.ocr import glmocr_retry as _retry_mod
from my_ocr.ocr import glmocr_runtime as _runtime_mod
from my_ocr.ocr import review_mapping as _review_mapping_mod
from my_ocr.ocr.layout_blocks import extract_layout_blocks
from my_ocr.ocr.ocr_policy import plan_page_ocr
from my_ocr.ocr.scratch_paths import ProviderScratchPaths
from my_ocr.ocr.text_cleanup import has_meaningful_text
from my_ocr.support.filesystem import load_json


class GlmOcrPageProcessor:
    def __init__(
        self,
        paths: ProviderScratchPaths,
        *,
        recognizer: InferenceClient,
        model: str | None = None,
    ) -> None:
        self._paths = paths
        self._recognizer = recognizer
        self._model = model

    def process_sdk_page(
        self,
        parser: Any,
        page_path: str,
        page_number: int,
        *,
        parser_cls: type,
        config_path: str,
        layout_device: str,
        layout_dotted: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        outcome = _retry_mod.parse_page_with_cpu_fallback(
            parser,
            page_path,
            parser_cls=parser_cls,
            config_path=config_path,
            layout_device=layout_device,
            layout_dotted=layout_dotted,
        )
        result = outcome.result
        result_dict = result.to_dict() if hasattr(result, "to_dict") else {}
        if isinstance(result_dict, dict) and result_dict.get("error"):
            raise RuntimeError(f"OCR failed for {page_path}: {result_dict['error']}")

        if not hasattr(result, "save"):
            raise RuntimeError("GLM-OCR result.save() is required to load saved *_model.json.")

        sdk_json_path = _artifacts_mod.save_result_to_raw_dir(
            result, self._paths.raw_dir, page_path, page_number
        )

        sdk_markdown = (getattr(result, "markdown_result", "") or "").strip()
        sdk_json = load_json(sdk_json_path)
        blocks = extract_layout_blocks(sdk_json)
        coord_space = _runtime_mod.detect_coord_space(blocks, page_path)
        page_result = self._finalize_page_ocr(
            page_path=page_path,
            page_number=page_number,
            sdk_markdown=sdk_markdown,
            sdk_json_path=sdk_json_path,
            page_layout=sdk_json,
            page_coord_space=coord_space,
            layout_source="sdk_json",
        )
        return page_result, outcome.parser_retired

    def process_review_layout_page(
        self,
        page_path: str,
        page_number: int,
        *,
        review_page: ReviewPage,
    ) -> dict[str, Any]:
        review_layout = _review_mapping_mod.review_layout_payload_from_page(review_page)
        if review_layout is None:
            raise RuntimeError(f"Review layout is missing page {page_number}.")
        page_layout, page_coord_space = review_layout
        sdk_json_path = _artifacts_mod.write_page_layout_to_raw_dir(
            page_layout,
            self._paths.raw_dir,
            page_path,
            page_number,
        )
        return self._finalize_page_ocr(
            page_path=page_path,
            page_number=page_number,
            sdk_markdown="",
            sdk_json_path=sdk_json_path,
            page_layout=page_layout,
            page_coord_space=page_coord_space,
            layout_source="review_layout",
        )

    def _finalize_page_ocr(
        self,
        *,
        page_path: str,
        page_number: int,
        sdk_markdown: str,
        sdk_json_path: Path,
        page_layout: dict[str, Any],
        page_coord_space: str,
        layout_source: str,
    ) -> dict[str, Any]:
        plan = plan_page_ocr(sdk_markdown, page_layout, coord_space=page_coord_space)

        if plan.primary_source == "sdk_markdown":
            markdown_source = plan.primary_source
            final_markdown = sdk_markdown
        elif plan.primary_source == "layout_json":
            markdown_source = plan.primary_source
            final_markdown = plan.layout_markdown
        elif plan.primary_source == "crop_fallback":
            crop_markdown, _ = _fallback_mod.run_crop_fallback_for_page(
                page_path=page_path,
                page_json=page_layout,
                coord_space=page_coord_space,
                page_fallback_dir=self._paths.fallback_page_dir(page_number),
                recognizer=self._recognizer,
                model=self._model,
            )
            crop_markdown = crop_markdown.strip()
            if has_meaningful_text(crop_markdown):
                markdown_source = plan.primary_source
                final_markdown = crop_markdown
            else:
                fallback_source = plan.fallback_source or "full_page_fallback"
                full_page_markdown = self._recognize_full_page_markdown(page_path)
                markdown_source = fallback_source
                final_markdown = full_page_markdown
        else:
            full_page_markdown = self._recognize_full_page_markdown(page_path)
            markdown_source = plan.primary_source
            final_markdown = full_page_markdown

        return {
            "page_number": page_number,
            "page_path": page_path,
            "markdown": final_markdown,
            "markdown_source": markdown_source,
            "sdk_markdown": sdk_markdown,
            "sdk_json_path": str(sdk_json_path),
            "layout_source": layout_source,
            "fallback_assessment": plan.assessment,
        }

    def _recognize_full_page_markdown(self, page_path: str) -> str:
        return _fallback_mod.recognize_full_page(
            page_path,
            recognizer=self._recognizer,
            model=self._model,
        ).strip()
