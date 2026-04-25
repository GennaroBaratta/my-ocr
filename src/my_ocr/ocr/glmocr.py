from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
import sys
from typing import Any

from my_ocr.settings import resolve_ocr_api_client
from my_ocr.ingest.normalize import IMAGE_SUFFIXES
from my_ocr.ocr import layout_profile as _layout_profile_mod
from my_ocr.ocr.run_paths import RunPaths
from my_ocr.ocr.glmocr_artifacts import (
    publish_saved_model_json_path as _publish_saved_model_json_path_impl,
    save_result_to_raw_dir as _save_result_to_raw_dir_impl,
    write_page_layout_to_raw_dir as _write_page_layout_to_raw_dir_impl,
)
from my_ocr.ocr.glmocr_parser import (
    build_lazy_glmocr_parser as _build_lazy_glmocr_parser_impl,
    build_raw_json as _build_raw_json_impl,
    load_glmocr_parser as _load_glmocr_parser_impl,
)
from my_ocr.ocr.glmocr_retry import (
    cleanup_after_cuda_oom as _cleanup_after_cuda_oom_impl,
    is_cuda_oom_error as _is_cuda_oom_error_impl,
    parse_page_with_cpu_fallback as _parse_page_with_cpu_fallback_impl,
    should_retry_parse_on_cpu as _should_retry_parse_on_cpu_impl,
)
from my_ocr.ocr.fallback import (
    recognize_full_page,
    run_crop_fallback_for_page,
)
from my_ocr.domain import (
    ArtifactCopy,
    LayoutDetectionResult,
    OcrRecognitionResult,
    ProviderArtifacts,
)
from my_ocr.ocr.planning import (
    detect_bbox_coord_space,
    extract_layout_blocks,
    has_meaningful_text,
    normalize_bbox,
    plan_page_ocr,
)
from my_ocr.support.filesystem import load_json
from my_ocr.domain import (
    LayoutBlock,
    LayoutDiagnostics,
    OcrPageResult,
    OcrRunResult,
    PageRef,
    ReviewLayout,
    ReviewPage,
)
from my_ocr.domain import OcrRuntimeOptions


def _emit_layout_profile_warning(diagnostics: dict[str, Any]) -> None:
    warning = diagnostics.get("layout_profile_warning")
    if not isinstance(warning, str) or not warning.strip():
        return
    print(f"Warning: {warning}", file=sys.stderr)


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
    page_inputs = _normalize_page_refs(pages)

    layout_dotted, layout_diagnostics = _layout_profile_mod.resolve_layout_profile(
        options.config_path, options.layout_profile
    )
    _emit_layout_profile_warning(layout_diagnostics)

    paths = RunPaths.from_run_dir(run_dir)
    paths.ensure_run_dir()
    paths.reset_ocr_artifacts()
    model, endpoint, num_ctx = resolve_ocr_api_client(options.config_path)
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
                page_result, fallback_result = _run_page_ocr_from_review_layout(
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
                    parser_cls = _load_glmocr_parser()
                if parser is None:
                    parser = parser_cls(
                        config_path=options.config_path,
                        layout_device=options.layout_device,
                        _dotted=layout_dotted,
                    )
                    parser.__enter__()
                page_result, fallback_result, parser_retired = _run_page_ocr(
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
                    review_page=None,
                )
                if parser_retired:
                    parser = None
            pages.append(page_result)
            _ = fallback_result
    finally:
        if parser is not None:
            parser.__exit__(None, None, None)

    markdown = "\n\n".join(page["markdown"] for page in pages if page["markdown"].strip())
    page_refs_by_number = {page.page_number: page for page in page_inputs}
    ocr_pages = [
        _ocr_page_from_provider_payload(raw_page, page_refs_by_number)
        for raw_page in pages
    ]
    result = OcrRunResult(
        pages=ocr_pages,
        markdown=markdown,
        diagnostics=dict(layout_diagnostics),
    )
    artifacts = _combined_artifacts(
        _provider_artifacts_from_pages(paths.raw_dir, "ocr/provider", list(page_inputs)),
        _provider_artifacts_from_pages(paths.fallback_dir, "ocr/fallback", list(page_inputs)),
    )
    artifacts = _with_cleanup(artifacts, _ocr_cleanup_paths(paths))
    return OcrRecognitionResult(result=result, artifacts=artifacts)


def prepare_review_artifacts(
    pages: Sequence[PageRef],
    run_dir: str | Path,
    *,
    options: OcrRuntimeOptions = OcrRuntimeOptions(),
) -> LayoutDetectionResult:
    page_inputs = _normalize_page_refs(pages)

    layout_dotted, layout_diagnostics = _layout_profile_mod.resolve_layout_profile(
        options.config_path, options.layout_profile
    )
    _emit_layout_profile_warning(layout_diagnostics)

    parser_cls = _load_glmocr_parser()
    paths = RunPaths.from_run_dir(run_dir)
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
            outcome = _parse_page_with_cpu_fallback(
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

            sdk_json_path = _save_result_to_raw_dir(result, paths.raw_dir, page_path, page_number)
            sdk_json = load_json(sdk_json_path)
            blocks = extract_layout_blocks(sdk_json)
            coord_space = _detect_coord_space(blocks, page_path)
            image_width, image_height = _get_image_size(page_path)
            review_pages.append(
                _review_page_from_provider_layout(
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
    artifacts = _with_cleanup(
        _provider_artifacts_from_pages(paths.raw_dir, "layout/provider", list(page_inputs)),
        (paths.raw_dir,),
    )
    return LayoutDetectionResult(
        layout=layout,
        artifacts=artifacts,
        diagnostics=LayoutDiagnostics(dict(layout_diagnostics)),
    )


def _normalize_page_refs(pages: Sequence[PageRef]) -> tuple[PageRef, ...]:
    candidates = tuple(pages)
    if not candidates:
        raise ValueError("At least one normalized page image is required.")

    for page in candidates:
        page_path = page.path_for_io
        if page_path.is_dir():
            raise ValueError("run_ocr expects page image files, not directories.")
        if not page_path.exists():
            raise FileNotFoundError(f"Page image not found: {page_path}")
        if page_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(
                f"run_ocr expects normalized page images. Unsupported page input: {page_path.name}"
            )
        if page.page_number <= 0:
            raise ValueError(f"Invalid page number {page.page_number} for {page_path}")
    return candidates


def _run_page_ocr(
    parser: Any,
    page_path: str,
    page_number: int,
    paths: RunPaths,
    *,
    parser_cls: type,
    config_path: str,
    layout_device: str,
    layout_dotted: dict[str, Any],
    model: str,
    endpoint: str,
    num_ctx: int,
    review_page: ReviewPage | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    outcome = _parse_page_with_cpu_fallback(
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

    sdk_json_path = _save_result_to_raw_dir(result, paths.raw_dir, page_path, page_number)

    sdk_markdown = (getattr(result, "markdown_result", "") or "").strip()
    sdk_json = load_json(sdk_json_path)
    blocks = extract_layout_blocks(sdk_json)
    coord_space = _detect_coord_space(blocks, page_path)
    review_layout = _resolve_review_layout_page(review_page)
    if review_layout is None:
        page_layout = sdk_json
        page_coord_space = coord_space
        layout_source = "sdk_json"
    else:
        page_layout, page_coord_space = review_layout
        layout_source = "review_layout"

    page_result, fallback_result = _finalize_page_ocr(
        page_path=page_path,
        page_number=page_number,
        paths=paths,
        sdk_markdown=sdk_markdown,
        sdk_json_path=sdk_json_path,
        page_layout=page_layout,
        page_coord_space=page_coord_space,
        layout_source=layout_source,
        model=model,
        endpoint=endpoint,
        num_ctx=num_ctx,
    )
    return page_result, fallback_result, outcome.parser_retired


def _run_page_ocr_from_review_layout(
    page_path: str,
    page_number: int,
    paths: RunPaths,
    *,
    model: str,
    endpoint: str,
    num_ctx: int,
    review_page: ReviewPage,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    review_layout = _resolve_review_layout_page(review_page)
    if review_layout is None:
        raise RuntimeError(f"Review layout is missing page {page_number}.")
    page_layout, page_coord_space = review_layout
    sdk_json_path = _write_page_layout_to_raw_dir(
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
    paths: RunPaths,
    sdk_markdown: str,
    sdk_json_path: Path,
    page_layout: dict[str, Any],
    page_coord_space: str,
    layout_source: str,
    model: str,
    endpoint: str,
    num_ctx: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    plan = plan_page_ocr(sdk_markdown, page_layout, coord_space=page_coord_space)

    fallback_result: dict[str, Any] | None = None

    if plan.primary_source == "sdk_markdown":
        markdown_source = plan.primary_source
        final_markdown = sdk_markdown
    elif plan.primary_source == "layout_json":
        markdown_source = plan.primary_source
        final_markdown = plan.layout_markdown
    elif plan.primary_source == "crop_fallback":
        crop_markdown, recognized_chunks = run_crop_fallback_for_page(
            page_path=page_path,
            page_json=page_layout,
            coord_space=page_coord_space,
            page_fallback_dir=paths.fallback_page_dir(page_number),
            model=model,
            endpoint=endpoint,
            num_ctx=num_ctx,
        )
        crop_markdown = crop_markdown.strip()
        fallback_result = {
            "page_number": page_number,
            "page_path": page_path,
            "assessment": plan.assessment,
            "chunks": recognized_chunks,
        }
        if has_meaningful_text(crop_markdown):
            markdown_source = plan.primary_source
            final_markdown = crop_markdown
            fallback_result["markdown"] = crop_markdown
            fallback_result["markdown_source"] = markdown_source
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
            fallback_result["markdown"] = full_page_markdown
            fallback_result["markdown_source"] = markdown_source
    else:
        full_page_markdown = _recognize_full_page_markdown(
            page_path,
            model=model,
            endpoint=endpoint,
            num_ctx=num_ctx,
        )
        markdown_source = plan.primary_source
        final_markdown = full_page_markdown
        fallback_result = {
            "page_number": page_number,
            "page_path": page_path,
            "assessment": plan.assessment,
            "chunks": [],
            "markdown": full_page_markdown,
            "markdown_source": markdown_source,
        }

    return (
        {
            "page_number": page_number,
            "page_path": page_path,
            "markdown": final_markdown,
            "markdown_source": markdown_source,
            "sdk_markdown": sdk_markdown,
            "sdk_json_path": str(sdk_json_path),
            "layout_source": layout_source,
            "fallback_assessment": plan.assessment,
        },
        fallback_result,
    )


def _load_glmocr_parser() -> type:
    return _load_glmocr_parser_impl()


def _build_lazy_glmocr_parser(
    *,
    load_config: Any,
    page_loader_cls: type,
    layout_detector_cls: type,
    ocr_client_cls: type,
    pipeline_result_cls: type,
    result_formatter_cls: type,
    crop_image_region: Any,
) -> type:
    return _build_lazy_glmocr_parser_impl(
        load_config=load_config,
        page_loader_cls=page_loader_cls,
        layout_detector_cls=layout_detector_cls,
        ocr_client_cls=ocr_client_cls,
        pipeline_result_cls=pipeline_result_cls,
        result_formatter_cls=result_formatter_cls,
        crop_image_region=crop_image_region,
    )


def _parse_page_with_cpu_fallback(
    parser: Any,
    page_path: str,
    *,
    method_name: str = "parse",
    parser_cls: type,
    config_path: str,
    layout_device: str,
    layout_dotted: dict[str, Any],
) -> Any:
    return _parse_page_with_cpu_fallback_impl(
        parser,
        page_path,
        method_name=method_name,
        parser_cls=parser_cls,
        config_path=config_path,
        layout_device=layout_device,
        layout_dotted=layout_dotted,
    )


def _should_retry_parse_on_cpu(exc: Exception, layout_device: str) -> bool:
    return _should_retry_parse_on_cpu_impl(exc, layout_device)


def _is_cuda_oom_error(exc: BaseException) -> bool:
    return _is_cuda_oom_error_impl(exc)


def _cleanup_after_cuda_oom() -> None:
    _cleanup_after_cuda_oom_impl()


def _build_raw_json(grouped_results: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    return _build_raw_json_impl(grouped_results)


def _detect_coord_space(blocks: list[dict[str, Any]], page_path: str) -> str:
    if not blocks:
        return "unknown"

    try:
        image_module = import_module("PIL.Image")
    except ImportError:
        return detect_bbox_coord_space(blocks)

    with image_module.open(page_path) as image:
        width, height = image.size
    return detect_bbox_coord_space(blocks, width=width, height=height)


def _save_result_to_raw_dir(
    result: Any,
    raw_dir: str | Path,
    page_path: str,
    page_number: int,
) -> Path:
    return _save_result_to_raw_dir_impl(result, raw_dir, page_path, page_number)


def _write_page_layout_to_raw_dir(
    page_layout: dict[str, Any],
    raw_dir: str | Path,
    page_path: str,
    page_number: int,
) -> Path:
    return _write_page_layout_to_raw_dir_impl(page_layout, raw_dir, page_path, page_number)


def _publish_saved_model_json_path(
    save_root: str | Path,
    raw_root: str | Path,
    page_path: str,
    page_number: int,
) -> Path:
    return _publish_saved_model_json_path_impl(save_root, raw_root, page_path, page_number)


def _recognize_full_page_markdown(
    page_path: str,
    *,
    model: str,
    endpoint: str,
    num_ctx: int,
) -> str:
    return recognize_full_page(
        page_path,
        model=model,
        endpoint=endpoint,
        num_ctx=num_ctx,
    ).strip()


def _get_image_size(page_path: str | Path) -> tuple[int, int]:
    try:
        image_module = import_module("PIL.Image")
    except ImportError as exc:
        raise RuntimeError("Review preparation requires Pillow.") from exc

    with image_module.open(page_path) as image:
        return image.size


def _resolve_review_layout_page(
    review_page: ReviewPage | None,
) -> tuple[dict[str, Any], str] | None:
    if review_page is None:
        return None
    blocks = [
        {
            "index": block.index,
            "label": block.label,
            "content": block.content,
            "bbox_2d": block.bbox,
        }
        for block in review_page.blocks
    ]
    return {"blocks": blocks}, review_page.coord_space


def _review_page_from_provider_layout(
    *,
    page_ref: PageRef,
    layout: Any,
    coord_space: str,
    image_width: int,
    image_height: int,
) -> ReviewPage:
    return ReviewPage(
        page_number=page_ref.page_number,
        image_path=page_ref.image_path,
        image_width=image_width,
        image_height=image_height,
        coord_space="pixel",
        provider_path=f"layout/provider/page-{page_ref.page_number:04d}",
        blocks=_review_blocks_from_provider_layout(
            page_number=page_ref.page_number,
            layout=layout,
            coord_space=coord_space,
            image_width=image_width,
            image_height=image_height,
        ),
    )


def _review_blocks_from_provider_layout(
    *,
    page_number: int,
    layout: Any,
    coord_space: str,
    image_width: int,
    image_height: int,
) -> list[LayoutBlock]:
    blocks: list[LayoutBlock] = []
    for fallback_index, block in enumerate(extract_layout_blocks(layout)):
        bbox = normalize_bbox(
            block.get("bbox_2d"),
            image_width,
            image_height,
            coord_space=coord_space,
        )
        if bbox is None:
            continue
        block_index = _coerce_block_index(block.get("index"), fallback=fallback_index)
        blocks.append(
            LayoutBlock(
                id=f"p{page_number - 1}-b{block_index}",
                index=block_index,
                label=str(block.get("label", "unknown")),
                content=str(block.get("content", "")),
                confidence=_coerce_confidence(block.get("confidence")),
                bbox=bbox,
            )
        )
    return blocks


def _coerce_block_index(value: Any, *, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_confidence(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def _ocr_page_from_provider_payload(
    raw_page: dict[str, Any], pages_by_number: dict[int, PageRef]
) -> OcrPageResult:
    page_number = int(raw_page.get("page_number", 0))
    page_ref = pages_by_number[page_number]
    fallback_path = None
    if raw_page.get("markdown_source") in {"crop_fallback", "full_page_fallback"}:
        fallback_path = f"ocr/fallback/page-{page_number:04d}"
    return OcrPageResult(
        page_number=page_number,
        image_path=page_ref.image_path,
        markdown=str(raw_page.get("markdown", "")),
        markdown_source=str(raw_page.get("markdown_source", "unknown")),
        provider_path=f"ocr/provider/page-{page_number:04d}",
        fallback_path=fallback_path,
        raw_payload=_relative_raw_payload(raw_page, page_number, fallback_path),
    )


def _provider_artifacts_from_pages(
    source_root: Path,
    target_root: str,
    pages: list[PageRef],
) -> ProviderArtifacts:
    copies: list[ArtifactCopy] = []
    for page in pages:
        source = source_root / f"page-{page.page_number:04d}"
        if source.exists():
            copies.append(
                ArtifactCopy(
                    source=source,
                    relative_target=f"{target_root}/page-{page.page_number:04d}",
                )
            )
    return ProviderArtifacts(tuple(copies))


def _combined_artifacts(*bundles: ProviderArtifacts) -> ProviderArtifacts:
    copies: list[ArtifactCopy] = []
    cleanup_paths: list[Path] = []
    for bundle in bundles:
        copies.extend(bundle.copies)
        cleanup_paths.extend(bundle.cleanup_paths)
    return ProviderArtifacts(tuple(copies), tuple(cleanup_paths))


def _with_cleanup(bundle: ProviderArtifacts, cleanup_paths: tuple[Path, ...]) -> ProviderArtifacts:
    return ProviderArtifacts(bundle.copies, bundle.cleanup_paths + cleanup_paths)


def _ocr_cleanup_paths(paths: RunPaths) -> tuple[Path, ...]:
    return (paths.raw_dir, paths.fallback_dir)


def _relative_raw_payload(
    raw_page: dict[str, Any], page_number: int, fallback_path: str | None
) -> dict[str, Any]:
    payload = {
        "assessment": raw_page.get("fallback_assessment", raw_page.get("assessment", {})),
        "layout_source": raw_page.get("layout_source"),
        "sdk_markdown": raw_page.get("sdk_markdown"),
    }
    if fallback_path:
        payload["fallback_path"] = fallback_path
    payload["provider_path"] = f"ocr/provider/page-{page_number:04d}"
    return payload
