from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
import sys
from typing import Any

from .ingest import IMAGE_SUFFIXES
from .ocr_fallback import (
    detect_bbox_coord_space,
    extract_layout_blocks,
    has_meaningful_text,
    plan_page_ocr,
    recognize_full_page,
    run_crop_fallback_for_page,
)
from .paths import RunPaths
from .review_artifacts import (
    build_review_layout_payload,
    build_review_page_from_layout,
    load_review_layout_payload,
    review_layout_pages_by_number,
    review_page_to_layout_payload,
    save_review_layout_payload,
)
from .settings import resolve_ocr_api_client
from .utils import load_json, write_json, write_text
from . import layout_profile as _layout_profile_mod


def _emit_layout_profile_warning(diagnostics: dict[str, Any]) -> None:
    warning = diagnostics.get("layout_profile_warning")
    if not isinstance(warning, str) or not warning.strip():
        return
    print(f"Warning: {warning}", file=sys.stderr)


def run_ocr(
    page_paths: Sequence[str | Path],
    run_dir: str | Path,
    *,
    config_path: str = "config/local.yaml",
    layout_device: str = "cuda",
    layout_profile: str | None = "auto",
    reviewed_layout_path: str | Path | None = None,
) -> dict[str, Any]:
    normalized_page_paths = _normalize_page_paths(page_paths)

    layout_dotted, layout_diagnostics = _layout_profile_mod.resolve_layout_profile(
        config_path, layout_profile
    )
    _emit_layout_profile_warning(layout_diagnostics)

    parser_cls = _load_glmocr_parser()
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

    with parser_cls(config_path=config_path, layout_device=layout_device, _dotted=layout_dotted) as parser:
        for page_number, page_path in enumerate(normalized_page_paths, start=1):
            page_result, fallback_result = _run_page_ocr(
                parser,
                page_path,
                page_number,
                paths,
                model=model,
                endpoint=endpoint,
                num_ctx=num_ctx,
                reviewed_layout_page=reviewed_layout_pages.get(page_number),
                reviewed_layout_path=reviewed_layout_path,
            )
            pages.append(page_result)
            source = str(page_result["markdown_source"])
            source_counts[source] = source_counts.get(source, 0) + 1
            if fallback_result is not None:
                fallback_pages.append(fallback_result)

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
            "apply_mode": "planning_and_fallback_only",
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
) -> dict[str, Any]:
    normalized_page_paths = _normalize_page_paths(page_paths)

    layout_dotted, layout_diagnostics = _layout_profile_mod.resolve_layout_profile(
        config_path, layout_profile
    )
    _emit_layout_profile_warning(layout_diagnostics)

    parser_cls = _load_glmocr_parser()
    paths = RunPaths.from_run_dir(run_dir)
    paths.ensure_run_dir()

    review_pages: list[dict[str, Any]] = []

    with parser_cls(config_path=config_path, layout_device=layout_device, _dotted=layout_dotted) as parser:
        for page_number, page_path in enumerate(normalized_page_paths, start=1):
            result = parser.parse(page_path)
            result_dict = result.to_dict() if hasattr(result, "to_dict") else {}
            if isinstance(result_dict, dict) and result_dict.get("error"):
                raise RuntimeError(
                    f"Review preparation failed for {page_path}: {result_dict['error']}"
                )

            if not hasattr(result, "save"):
                raise RuntimeError(
                    "GLM-OCR result.save() is required to load saved *_model.json for review prep."
                )

            result.save(output_dir=str(paths.raw_dir))
            sdk_json_path = _resolve_saved_model_json_path(paths.raw_dir, page_path)
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


def _normalize_page_paths(page_paths: Sequence[str | Path]) -> list[str]:
    if isinstance(page_paths, (str, Path)):
        candidates = [page_paths]
    else:
        candidates = list(page_paths)

    if not candidates:
        raise ValueError("At least one normalized page image is required.")

    normalized: list[str] = []
    for raw_path in candidates:
        page_path = Path(raw_path)
        if page_path.is_dir():
            raise ValueError("run_ocr expects page image files, not directories.")
        if not page_path.exists():
            raise FileNotFoundError(f"Page image not found: {page_path}")
        if page_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(
                f"run_ocr expects normalized page images. Unsupported page input: {page_path.name}"
            )
        normalized.append(str(page_path))
    return normalized


def _run_page_ocr(
    parser: Any,
    page_path: str,
    page_number: int,
    paths: RunPaths,
    *,
    model: str,
    endpoint: str,
    num_ctx: int,
    reviewed_layout_page: dict[str, Any] | None,
    reviewed_layout_path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    result = parser.parse(page_path)
    result_dict = result.to_dict() if hasattr(result, "to_dict") else {}
    if isinstance(result_dict, dict) and result_dict.get("error"):
        raise RuntimeError(f"OCR failed for {page_path}: {result_dict['error']}")

    if not hasattr(result, "save"):
        raise RuntimeError("GLM-OCR result.save() is required to load saved *_model.json.")

    result.save(output_dir=str(paths.raw_dir))

    sdk_markdown = (getattr(result, "markdown_result", "") or "").strip()
    sdk_json_path = _resolve_saved_model_json_path(paths.raw_dir, page_path)
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
                if reviewed_layout is not None and reviewed_layout_path is not None
                else {}
            ),
        },
        fallback_result,
    )


def _load_glmocr_parser() -> type:
    try:
        glmocr_module = import_module("glmocr")
    except ImportError as exc:
        raise RuntimeError(
            "GLM-OCR is not installed. Install with `pip install -e .[glmocr]`."
        ) from exc
    return glmocr_module.GlmOcr


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


def _resolve_saved_model_json_path(raw_dir: str | Path, page_path: str) -> Path:
    page_stem = Path(page_path).stem
    model_path = Path(raw_dir) / page_stem / f"{page_stem}_model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing saved GLM-OCR model JSON: {model_path}")
    return model_path


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
