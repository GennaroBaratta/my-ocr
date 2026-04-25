from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
import sys
from typing import Any

from my_ocr.adapters.outbound.config import layout_profile as _layout_profile_mod
from my_ocr.adapters.outbound.config.settings import resolve_ocr_api_client
from my_ocr.adapters.outbound.filesystem.ingestion import IMAGE_SUFFIXES
from my_ocr.adapters.outbound.filesystem.review_layout_store import (
    load_review_layout_payload,
    save_review_layout_payload,
)
from my_ocr.adapters.outbound.filesystem.run_paths import RunPaths
from my_ocr.adapters.outbound.ocr._glmocr_artifacts import (
    publish_saved_model_json_path as _publish_saved_model_json_path_impl,
    save_result_to_raw_dir as _save_result_to_raw_dir_impl,
    write_page_layout_to_raw_dir as _write_page_layout_to_raw_dir_impl,
)
from my_ocr.adapters.outbound.ocr._glmocr_parser import (
    build_lazy_glmocr_parser as _build_lazy_glmocr_parser_impl,
    build_raw_json as _build_raw_json_impl,
    load_glmocr_parser as _load_glmocr_parser_impl,
)
from my_ocr.adapters.outbound.ocr._glmocr_retry import (
    cleanup_after_cuda_oom as _cleanup_after_cuda_oom_impl,
    is_cuda_oom_error as _is_cuda_oom_error_impl,
    parse_page_with_cpu_fallback as _parse_page_with_cpu_fallback_impl,
    should_retry_parse_on_cpu as _should_retry_parse_on_cpu_impl,
)
from my_ocr.adapters.outbound.ocr.fallback_ocr import (
    recognize_full_page,
    run_crop_fallback_for_page,
)
from my_ocr.application.artifacts import (
    ArtifactCopy,
    LayoutDetectionResult,
    OcrRecognitionResult,
    ProviderArtifacts,
)
from my_ocr.domain.layout import (
    detect_bbox_coord_space,
    extract_layout_blocks,
    has_meaningful_text,
    plan_page_ocr,
)
from my_ocr.domain.page_identity import infer_page_number
from my_ocr.domain.review_layout import (
    REVIEWED_LAYOUT_APPLY_MODE,
    build_review_layout_payload,
    build_review_page_from_layout,
    review_layout_pages_by_number,
    review_page_to_layout_payload,
)
from my_ocr.filesystem import load_json, write_json, write_text
from my_ocr.models import (
    LayoutBlock,
    LayoutDiagnostics,
    OcrPageResult,
    OcrRunResult,
    PageRef,
    ReviewLayout,
    ReviewPage,
)
from my_ocr.pipeline.options import LayoutOptions, OcrOptions


def _emit_layout_profile_warning(diagnostics: dict[str, Any]) -> None:
    warning = diagnostics.get("layout_profile_warning")
    if not isinstance(warning, str) or not warning.strip():
        return
    print(f"Warning: {warning}", file=sys.stderr)


class GlmOcrLayoutDetector:
    def detect_layout(
        self, pages: list[PageRef], options: LayoutOptions
    ) -> LayoutDetectionResult:
        run_dir = _run_dir_from_pages(pages)
        result = prepare_review_artifacts(
            [str(page.path_for_io) for page in pages],
            run_dir,
            config_path=options.config_path,
            layout_device=options.layout_device,
            layout_profile=options.layout_profile,
            page_numbers=[page.page_number for page in pages],
        )
        layout = _review_layout_from_legacy(
            _dict_payload(result.get("reviewed_layout")), pages, status="prepared"
        )
        artifacts = _with_cleanup(
            _provider_artifacts_from_pages(run_dir / "ocr_raw", "layout/provider", pages),
            _legacy_layout_cleanup_paths(run_dir),
        )
        return LayoutDetectionResult(
            layout=layout,
            artifacts=artifacts,
            diagnostics=LayoutDiagnostics.from_dict(result.get("layout_diagnostics")),
        )


class GlmOcrEngine:
    def recognize(
        self,
        pages: list[PageRef],
        review: ReviewLayout | None,
        options: OcrOptions,
    ) -> OcrRecognitionResult:
        run_dir = _run_dir_from_pages(pages)
        review_path: Path | None = None
        if review is not None:
            review_path = run_dir / "reviewed_layout.json"
            write_json(review_path, _legacy_review_payload(review, pages))

        result = run_ocr(
            [str(page.path_for_io) for page in pages],
            run_dir,
            config_path=options.config_path,
            layout_device=options.layout_device,
            layout_profile=options.layout_profile,
            reviewed_layout_path=review_path,
            page_numbers=[page.page_number for page in pages],
        )
        ocr_result = _ocr_result_from_legacy(
            _dict_payload(result.get("json")),
            str(result.get("markdown", "")),
            pages,
        )
        ocr_result = OcrRunResult(
            pages=ocr_result.pages,
            markdown=ocr_result.markdown,
            diagnostics=dict(result.get("layout_diagnostics", {})),
        )
        artifacts = _combined_artifacts(
            _provider_artifacts_from_pages(run_dir / "ocr_raw", "ocr/provider", pages),
            _provider_artifacts_from_pages(run_dir / "ocr_fallback", "ocr/fallback", pages),
        )
        artifacts = _with_cleanup(artifacts, _legacy_ocr_cleanup_paths(run_dir))
        return OcrRecognitionResult(result=ocr_result, artifacts=artifacts)


def run_ocr(
    page_paths: Sequence[str | Path],
    run_dir: str | Path,
    *,
    config_path: str = "config/local.yaml",
    layout_device: str = "cuda",
    layout_profile: str | None = "auto",
    reviewed_layout_path: str | Path | None = None,
    page_numbers: Sequence[int] | None = None,
) -> dict[str, Any]:
    page_inputs = _normalize_page_inputs(page_paths, page_numbers=page_numbers)

    layout_dotted, layout_diagnostics = _layout_profile_mod.resolve_layout_profile(
        config_path, layout_profile
    )
    _emit_layout_profile_warning(layout_diagnostics)

    paths = RunPaths.from_run_dir(run_dir)
    paths.ensure_run_dir()
    paths.reset_ocr_artifacts()
    model, endpoint, num_ctx = resolve_ocr_api_client(config_path)
    reviewed_layout_payload = (
        load_review_layout_payload(reviewed_layout_path)
        if reviewed_layout_path is not None
        else None
    )
    reviewed_layout_pages = review_layout_pages_by_number(reviewed_layout_payload)

    pages: list[dict[str, Any]] = []
    fallback_pages: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}

    parser_cls: type | None = None
    parser: Any | None = None
    try:
        for page_number, page_path in page_inputs:
            reviewed_layout_page = reviewed_layout_pages.get(page_number)
            if reviewed_layout_page is not None:
                page_result, fallback_result = _run_page_ocr_from_reviewed_layout(
                    page_path,
                    page_number,
                    paths,
                    model=model,
                    endpoint=endpoint,
                    num_ctx=num_ctx,
                    reviewed_layout_page=reviewed_layout_page,
                    reviewed_layout_path=reviewed_layout_path,
                )
            else:
                if parser_cls is None:
                    parser_cls = _load_glmocr_parser()
                if parser is None:
                    parser = parser_cls(
                        config_path=config_path,
                        layout_device=layout_device,
                        _dotted=layout_dotted,
                    )
                    parser.__enter__()
                page_result, fallback_result, parser_retired = _run_page_ocr(
                    parser,
                    page_path,
                    page_number,
                    paths,
                    parser_cls=parser_cls,
                    config_path=config_path,
                    layout_device=layout_device,
                    layout_dotted=layout_dotted,
                    model=model,
                    endpoint=endpoint,
                    num_ctx=num_ctx,
                    reviewed_layout_page=None,
                    reviewed_layout_path=reviewed_layout_path,
                )
                if parser_retired:
                    parser = None
            pages.append(page_result)
            source = str(page_result["markdown_source"])
            source_counts[source] = source_counts.get(source, 0) + 1
            if fallback_result is not None:
                fallback_pages.append(fallback_result)
    finally:
        if parser is not None:
            parser.__exit__(None, None, None)

    markdown = "\n\n".join(page["markdown"] for page in pages if page["markdown"].strip())
    json_result = {
        "pages": pages,
        "summary": {
            "page_count": len(pages),
            "sources": source_counts,
        },
    }
    if reviewed_layout_payload is not None and reviewed_layout_path is not None:
        json_result["summary"]["reviewed_layout"] = {
            "path": str(reviewed_layout_path),
            "page_count": len(reviewed_layout_pages),
            "apply_mode": REVIEWED_LAYOUT_APPLY_MODE,
        }

    write_text(paths.ocr_markdown_path, markdown)
    write_json(paths.ocr_json_path, json_result)
    if fallback_pages:
        write_json(
            paths.ocr_fallback_path,
            {
                "pages": fallback_pages,
                "summary": {"page_count": len(fallback_pages)},
            },
        )

    return {
        "markdown": markdown,
        "json": json_result,
        "raw_dir": str(paths.raw_dir),
        "config_path": config_path,
        "layout_device": layout_device,
        "layout_diagnostics": layout_diagnostics,
    }


def prepare_review_artifacts(
    page_paths: Sequence[str | Path],
    run_dir: str | Path,
    *,
    config_path: str = "config/local.yaml",
    layout_device: str = "cuda",
    layout_profile: str | None = "auto",
    page_numbers: Sequence[int] | None = None,
) -> dict[str, Any]:
    page_inputs = _normalize_page_inputs(page_paths, page_numbers=page_numbers)

    layout_dotted, layout_diagnostics = _layout_profile_mod.resolve_layout_profile(
        config_path, layout_profile
    )
    _emit_layout_profile_warning(layout_diagnostics)

    parser_cls = _load_glmocr_parser()
    paths = RunPaths.from_run_dir(run_dir)
    paths.ensure_run_dir()

    review_pages: list[dict[str, Any]] = []

    parser: Any | None = None
    try:
        for page_number, page_path in page_inputs:
            if parser is None:
                parser = parser_cls(
                    config_path=config_path,
                    layout_device=layout_device,
                    _dotted=layout_dotted,
                )
                parser.__enter__()
            outcome = _parse_page_with_cpu_fallback(
                parser,
                page_path,
                method_name="parse_layout_only",
                parser_cls=parser_cls,
                config_path=config_path,
                layout_device=layout_device,
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
                build_review_page_from_layout(
                    page_number=page_number,
                    page_path=page_path,
                    source_sdk_json_path=str(sdk_json_path),
                    layout=sdk_json,
                    coord_space=coord_space,
                    image_width=image_width,
                    image_height=image_height,
                )
            )
    finally:
        if parser is not None:
            parser.__exit__(None, None, None)

    reviewed_layout_payload = build_review_layout_payload(review_pages, status="prepared")
    save_review_layout_payload(paths.reviewed_layout_path, reviewed_layout_payload)
    return {
        "reviewed_layout": reviewed_layout_payload,
        "raw_dir": str(paths.raw_dir),
        "config_path": config_path,
        "layout_device": layout_device,
        "reviewed_layout_path": str(paths.reviewed_layout_path),
        "layout_diagnostics": layout_diagnostics,
    }


def _normalize_page_inputs(
    page_paths: Sequence[str | Path],
    *,
    page_numbers: Sequence[int] | None = None,
) -> list[tuple[int, str]]:
    if isinstance(page_paths, (str, Path)):
        candidates = [page_paths]
    else:
        candidates = list(page_paths)

    if not candidates:
        raise ValueError("At least one normalized page image is required.")
    if page_numbers is not None and len(page_numbers) != len(candidates):
        raise ValueError("page_numbers must match the number of page images.")

    normalized: list[tuple[int, str]] = []
    for fallback_number, raw_path in enumerate(candidates, start=1):
        page_path = Path(raw_path)
        if page_path.is_dir():
            raise ValueError("run_ocr expects page image files, not directories.")
        if not page_path.exists():
            raise FileNotFoundError(f"Page image not found: {page_path}")
        if page_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(
                f"run_ocr expects normalized page images. Unsupported page input: {page_path.name}"
            )
        page_number = (
            page_numbers[fallback_number - 1]
            if page_numbers is not None
            else infer_page_number(page_path, fallback_number)
        )
        if page_number <= 0:
            raise ValueError(f"Invalid page number {page_number} for {page_path}")
        normalized.append((page_number, str(page_path)))
    return normalized


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
    reviewed_layout_page: dict[str, Any] | None,
    reviewed_layout_path: str | Path | None,
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
    reviewed_layout = _resolve_reviewed_layout_page(reviewed_layout_page)
    if reviewed_layout is None:
        page_layout = sdk_json
        page_coord_space = coord_space
        layout_source = "sdk_json"
    else:
        page_layout, page_coord_space = reviewed_layout
        layout_source = "reviewed_layout"

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
        reviewed_layout_path=reviewed_layout_path,
        include_reviewed_layout_path=reviewed_layout is not None,
    )
    return page_result, fallback_result, outcome.parser_retired


def _run_page_ocr_from_reviewed_layout(
    page_path: str,
    page_number: int,
    paths: RunPaths,
    *,
    model: str,
    endpoint: str,
    num_ctx: int,
    reviewed_layout_page: dict[str, Any],
    reviewed_layout_path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    reviewed_layout = _resolve_reviewed_layout_page(reviewed_layout_page)
    if reviewed_layout is None:
        raise RuntimeError(f"Reviewed layout is missing page {page_number}.")
    page_layout, page_coord_space = reviewed_layout
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
        layout_source="reviewed_layout",
        model=model,
        endpoint=endpoint,
        num_ctx=num_ctx,
        reviewed_layout_path=reviewed_layout_path,
        include_reviewed_layout_path=reviewed_layout_path is not None,
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
    reviewed_layout_path: str | Path | None,
    include_reviewed_layout_path: bool,
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
            **(
                {"reviewed_layout_path": str(reviewed_layout_path)}
                if include_reviewed_layout_path and reviewed_layout_path is not None
                else {}
            ),
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


def _resolve_reviewed_layout_page(
    reviewed_layout_page: dict[str, Any] | None,
) -> tuple[dict[str, Any], str] | None:
    if reviewed_layout_page is None:
        return None
    return review_page_to_layout_payload(reviewed_layout_page)


def _run_dir_from_pages(pages: list[PageRef]) -> Path:
    if not pages:
        raise ValueError("At least one normalized page image is required.")
    page_path = pages[0].path_for_io
    if page_path.parent.name == "pages":
        return page_path.parent.parent
    return page_path.parent


def _dict_payload(payload: Any) -> dict[str, Any]:
    return payload if isinstance(payload, dict) else {}


def _review_layout_from_legacy(
    payload: dict[str, Any], pages: list[PageRef], *, status: str = "prepared"
) -> ReviewLayout:
    pages_by_number = {page.page_number: page for page in pages}
    review_pages: list[ReviewPage] = []
    for raw_page in payload.get("pages", []):
        if not isinstance(raw_page, dict):
            continue
        page_number = int(raw_page.get("page_number", len(review_pages) + 1))
        page_ref = pages_by_number.get(page_number)
        size = raw_page.get("image_size", {})
        image_width = int(size.get("width", page_ref.width if page_ref else 0))
        image_height = int(size.get("height", page_ref.height if page_ref else 0))
        review_pages.append(
            ReviewPage(
                page_number=page_number,
                image_path=page_ref.image_path if page_ref else str(raw_page.get("page_path", "")),
                image_width=image_width,
                image_height=image_height,
                coord_space=str(raw_page.get("coord_space", "pixel")),
                provider_path=f"layout/provider/page-{page_number:04d}",
                blocks=[
                    LayoutBlock.from_dict(block)
                    for block in raw_page.get("blocks", [])
                    if isinstance(block, dict)
                ],
            )
        )
    return ReviewLayout(pages=review_pages, status=status)


def _legacy_review_payload(review: ReviewLayout, pages: list[PageRef]) -> dict[str, Any]:
    pages_by_number = {page.page_number: page for page in pages}
    payload_pages: list[dict[str, Any]] = []
    for review_page in review.pages:
        page_ref = pages_by_number.get(review_page.page_number)
        page_payload = review_page.to_dict()
        page_payload["page_path"] = str(page_ref.path_for_io if page_ref else review_page.image_path)
        page_payload["source_sdk_json_path"] = review_page.provider_path or ""
        payload_pages.append(page_payload)
    return {
        "version": 2,
        "status": review.status,
        "pages": payload_pages,
        "summary": {"page_count": len(payload_pages)},
    }


def _ocr_result_from_legacy(
    payload: dict[str, Any], markdown: str, pages: list[PageRef]
) -> OcrRunResult:
    pages_by_number = {page.page_number: page for page in pages}
    ocr_pages: list[OcrPageResult] = []
    for raw_page in payload.get("pages", []):
        if not isinstance(raw_page, dict):
            continue
        page_number = int(raw_page.get("page_number", len(ocr_pages) + 1))
        page_ref = pages_by_number.get(page_number)
        fallback_path = None
        if raw_page.get("markdown_source") in {"crop_fallback", "full_page_fallback"}:
            fallback_path = f"ocr/fallback/page-{page_number:04d}"
        ocr_pages.append(
            OcrPageResult(
                page_number=page_number,
                image_path=page_ref.image_path if page_ref else str(raw_page.get("page_path", "")),
                markdown=str(raw_page.get("markdown", "")),
                markdown_source=str(raw_page.get("markdown_source", "unknown")),
                provider_path=f"ocr/provider/page-{page_number:04d}",
                fallback_path=fallback_path,
                raw_payload=_relative_raw_payload(raw_page, page_number, fallback_path),
            )
        )
    return OcrRunResult(pages=ocr_pages, markdown=markdown)


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


def _legacy_layout_cleanup_paths(run_dir: Path) -> tuple[Path, ...]:
    return (run_dir / "ocr_raw", run_dir / "reviewed_layout.json")


def _legacy_ocr_cleanup_paths(run_dir: Path) -> tuple[Path, ...]:
    return (
        run_dir / "ocr_raw",
        run_dir / "ocr_fallback",
        run_dir / "ocr.md",
        run_dir / "ocr.json",
        run_dir / "ocr_fallback.json",
        run_dir / "reviewed_layout.json",
    )


def _relative_raw_payload(
    raw_page: dict[str, Any], page_number: int, fallback_path: str | None
) -> dict[str, Any]:
    payload = {
        "assessment": raw_page.get("assessment", {}),
        "layout_source": raw_page.get("layout_source"),
    }
    if fallback_path:
        payload["fallback_path"] = fallback_path
    payload["provider_path"] = f"ocr/provider/page-{page_number:04d}"
    return payload
