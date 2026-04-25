from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from my_ocr.ocr import fallback as _fallback_mod
from my_ocr.ocr import glmocr_artifacts as _artifacts_mod
from my_ocr.ocr import glmocr_sdk as _parser_mod
from my_ocr.ocr import glmocr_retry as _retry_mod
from my_ocr.ocr import review_mapping as _review_mod
from my_ocr.ocr import glmocr_runtime as _runtime_mod
from my_ocr.ocr import layout_profile as _layout_profile_mod
from my_ocr.ocr.scratch_paths import ProviderScratchPaths
from my_ocr.domain import (
    LayoutDetectionResult,
    OcrRecognitionResult,
)
from my_ocr.ocr.ocr_policy import (
    extract_layout_blocks,
    has_meaningful_text,
    plan_page_ocr,
)
from my_ocr.support.filesystem import load_json
from my_ocr.domain import (
    LayoutDiagnostics,
    OcrRunResult,
    PageRef,
    ReviewLayout,
    ReviewPage,
)
from my_ocr.domain import OcrRuntimeOptions


class GlmOcrLayoutDetector:
    def detect_layout(
        self, pages: list[PageRef], run_dir: Path, options: OcrRuntimeOptions
    ) -> LayoutDetectionResult:
        return prepare_review_artifacts(
            pages,
            run_dir,
            options=options,
        )


class GlmOcrEngine:
    def recognize(
        self,
        pages: list[PageRef],
        run_dir: Path,
        review: ReviewLayout | None,
        options: OcrRuntimeOptions,
    ) -> OcrRecognitionResult:
        return run_ocr(
            pages,
            run_dir,
            review=review,
            options=options,
        )


def run_ocr(
    pages: Sequence[PageRef],
    run_dir: str | Path,
    *,
    review: ReviewLayout | None = None,
    options: OcrRuntimeOptions = OcrRuntimeOptions(),
) -> OcrRecognitionResult:
    page_inputs = _runtime_mod.normalize_page_refs(pages)

    layout_dotted, layout_diagnostics = _layout_profile_mod.resolve_layout_profile(
        options.config_path, options.layout_profile
    )
    _runtime_mod.emit_layout_profile_warning(layout_diagnostics)

    paths = ProviderScratchPaths.from_run_dir(run_dir)
    paths.ensure_run_dir()
    paths.reset_ocr_artifacts()
    model, endpoint, num_ctx = _runtime_mod.resolve_ocr_api_client(options)
    review_pages = {
        page.page_number: page for page in review.pages
    } if review is not None else {}

    pages: list[dict[str, Any]] = []
    parser_cls: type | None = None
    parser: Any | None = None
    try:
        for page_ref in page_inputs:
            page_number = page_ref.page_number
            page_path = str(page_ref.path_for_io)
            review_page = review_pages.get(page_number)
            if review_page is not None:
                page_result = _run_page_ocr_from_review_layout(
                    page_path,
                    page_number,
                    paths,
                    model=model,
                    endpoint=endpoint,
                    num_ctx=num_ctx,
                    review_page=review_page,
                )
            else:
                if parser_cls is None:
                    parser_cls = _parser_mod.load_glmocr_parser()
                if parser is None:
                    parser = parser_cls(
                        config_path=options.config_path,
                        layout_device=options.layout_device,
                        _dotted=layout_dotted,
                    )
                    parser.__enter__()
                page_result, parser_retired = _run_page_ocr(
                    parser,
                    page_path,
                    page_number,
                    paths,
                    parser_cls=parser_cls,
                    config_path=options.config_path,
                    layout_device=options.layout_device,
                    layout_dotted=layout_dotted,
                    model=model,
                    endpoint=endpoint,
                    num_ctx=num_ctx,
                )
                if parser_retired:
                    parser = None
            pages.append(page_result)
    finally:
        if parser is not None:
            parser.__exit__(None, None, None)

    markdown = "\n\n".join(page["markdown"] for page in pages if page["markdown"].strip())
    page_refs_by_number = {page.page_number: page for page in page_inputs}
    ocr_pages = [
        _runtime_mod.ocr_page_from_provider_payload(raw_page, page_refs_by_number)
        for raw_page in pages
    ]
    result = OcrRunResult(
        pages=ocr_pages,
        markdown=markdown,
        diagnostics=dict(layout_diagnostics),
    )
    page_list = list(page_inputs)
    artifacts = _runtime_mod.combined_artifacts(
        _runtime_mod.provider_artifacts_from_pages(
            paths.raw_dir,
            "ocr/provider",
            page_list,
        ),
        _runtime_mod.provider_artifacts_from_pages(
            paths.fallback_dir,
            "ocr/fallback",
            page_list,
        ),
    )
    artifacts = _runtime_mod.with_cleanup(artifacts, _runtime_mod.ocr_cleanup_paths(paths))
    return OcrRecognitionResult(result=result, artifacts=artifacts)


def prepare_review_artifacts(
    pages: Sequence[PageRef],
    run_dir: str | Path,
    *,
    options: OcrRuntimeOptions = OcrRuntimeOptions(),
) -> LayoutDetectionResult:
    page_inputs = _runtime_mod.normalize_page_refs(pages)

    layout_dotted, layout_diagnostics = _layout_profile_mod.resolve_layout_profile(
        options.config_path, options.layout_profile
    )
    _runtime_mod.emit_layout_profile_warning(layout_diagnostics)

    parser_cls = _parser_mod.load_glmocr_parser()
    paths = ProviderScratchPaths.from_run_dir(run_dir)
    paths.ensure_run_dir()

    review_pages: list[ReviewPage] = []

    parser: Any | None = None
    try:
        for page_ref in page_inputs:
            page_number = page_ref.page_number
            page_path = str(page_ref.path_for_io)
            if parser is None:
                parser = parser_cls(
                    config_path=options.config_path,
                    layout_device=options.layout_device,
                    _dotted=layout_dotted,
                )
                parser.__enter__()
            outcome = _retry_mod.parse_page_with_cpu_fallback(
                parser,
                page_path,
                method_name="parse_layout_only",
                parser_cls=parser_cls,
                config_path=options.config_path,
                layout_device=options.layout_device,
                layout_dotted=layout_dotted,
            )
            if outcome.parser_retired:
                parser = None
            result = outcome.result
            result_dict = result.to_dict() if hasattr(result, "to_dict") else {}
            if isinstance(result_dict, dict) and result_dict.get("error"):
                raise RuntimeError(
                    f"Review preparation failed for {page_path}: {result_dict['error']}"
                )

            if not hasattr(result, "save"):
                raise RuntimeError(
                    "GLM-OCR result.save() is required to load saved *_model.json for review prep."
                )

            sdk_json_path = _artifacts_mod.save_result_to_raw_dir(
                result, paths.raw_dir, page_path, page_number
            )
            sdk_json = load_json(sdk_json_path)
            blocks = extract_layout_blocks(sdk_json)
            coord_space = _runtime_mod.detect_coord_space(blocks, page_path)
            image_width, image_height = _review_mod.get_image_size(page_path)
            review_pages.append(
                _review_mod.review_page_from_provider_layout(
                    page_ref=page_ref,
                    layout=sdk_json,
                    coord_space=coord_space,
                    image_width=image_width,
                    image_height=image_height,
                )
            )
    finally:
        if parser is not None:
            parser.__exit__(None, None, None)

    layout = ReviewLayout(pages=review_pages, status="prepared")
    artifacts = _runtime_mod.with_cleanup(
        _runtime_mod.provider_artifacts_from_pages(
            paths.raw_dir, "layout/provider", list(page_inputs)
        ),
        (paths.raw_dir,),
    )
    return LayoutDetectionResult(
        layout=layout,
        artifacts=artifacts,
        diagnostics=LayoutDiagnostics(dict(layout_diagnostics)),
    )


def _run_page_ocr(
    parser: Any,
    page_path: str,
    page_number: int,
    paths: ProviderScratchPaths,
    *,
    parser_cls: type,
    config_path: str,
    layout_device: str,
    layout_dotted: dict[str, Any],
    model: str,
    endpoint: str,
    num_ctx: int,
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
        result, paths.raw_dir, page_path, page_number
    )

    sdk_markdown = (getattr(result, "markdown_result", "") or "").strip()
    sdk_json = load_json(sdk_json_path)
    blocks = extract_layout_blocks(sdk_json)
    coord_space = _runtime_mod.detect_coord_space(blocks, page_path)
    page_result = _finalize_page_ocr(
        page_path=page_path,
        page_number=page_number,
        paths=paths,
        sdk_markdown=sdk_markdown,
        sdk_json_path=sdk_json_path,
        page_layout=sdk_json,
        page_coord_space=coord_space,
        layout_source="sdk_json",
        model=model,
        endpoint=endpoint,
        num_ctx=num_ctx,
    )
    return page_result, outcome.parser_retired


def _run_page_ocr_from_review_layout(
    page_path: str,
    page_number: int,
    paths: ProviderScratchPaths,
    *,
    model: str,
    endpoint: str,
    num_ctx: int,
    review_page: ReviewPage,
) -> dict[str, Any]:
    review_layout = _review_mod.review_layout_payload_from_page(review_page)
    if review_layout is None:
        raise RuntimeError(f"Review layout is missing page {page_number}.")
    page_layout, page_coord_space = review_layout
    sdk_json_path = _artifacts_mod.write_page_layout_to_raw_dir(
        page_layout,
        paths.raw_dir,
        page_path,
        page_number,
    )
    return _finalize_page_ocr(
        page_path=page_path,
        page_number=page_number,
        paths=paths,
        sdk_markdown="",
        sdk_json_path=sdk_json_path,
        page_layout=page_layout,
        page_coord_space=page_coord_space,
        layout_source="review_layout",
        model=model,
        endpoint=endpoint,
        num_ctx=num_ctx,
    )


def _finalize_page_ocr(
    *,
    page_path: str,
    page_number: int,
    paths: ProviderScratchPaths,
    sdk_markdown: str,
    sdk_json_path: Path,
    page_layout: dict[str, Any],
    page_coord_space: str,
    layout_source: str,
    model: str,
    endpoint: str,
    num_ctx: int,
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
            page_fallback_dir=paths.fallback_page_dir(page_number),
            model=model,
            endpoint=endpoint,
            num_ctx=num_ctx,
        )
        crop_markdown = crop_markdown.strip()
        if has_meaningful_text(crop_markdown):
            markdown_source = plan.primary_source
            final_markdown = crop_markdown
        else:
            fallback_source = plan.fallback_source or "full_page_fallback"
            full_page_markdown = _recognize_full_page_markdown(
                page_path,
                model=model,
                endpoint=endpoint,
                num_ctx=num_ctx,
            )
            markdown_source = fallback_source
            final_markdown = full_page_markdown
    else:
        full_page_markdown = _recognize_full_page_markdown(
            page_path,
            model=model,
            endpoint=endpoint,
            num_ctx=num_ctx,
        )
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

def _recognize_full_page_markdown(
    page_path: str,
    *,
    model: str,
    endpoint: str,
    num_ctx: int,
) -> str:
    return _fallback_mod.recognize_full_page(
        page_path,
        model=model,
        endpoint=endpoint,
        num_ctx=num_ctx,
    ).strip()
