from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
import gc
from importlib import import_module
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from typing import Any

from my_ocr.adapters.outbound.config import layout_profile as _layout_profile_mod
from my_ocr.adapters.outbound.config.settings import resolve_ocr_api_client
from my_ocr.adapters.outbound.filesystem.ingestion import IMAGE_SUFFIXES
from my_ocr.adapters.outbound.filesystem.json_store import load_json, write_json, write_text
from my_ocr.adapters.outbound.filesystem.run_paths import RunPaths
from my_ocr.adapters.outbound.ocr.fallback_ocr import (
    recognize_full_page,
    run_crop_fallback_for_page,
)
from my_ocr.domain.layout import (
    detect_bbox_coord_space,
    extract_layout_blocks,
    has_meaningful_text,
    plan_page_ocr,
)
from my_ocr.domain.page_identity import infer_page_number
from my_ocr.domain.review_layout import (
    build_review_layout_payload,
    build_review_page_from_layout,
    load_review_layout_payload,
    review_layout_pages_by_number,
    review_page_to_layout_payload,
    save_review_layout_payload,
)


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
        for page_number, page_path in page_inputs:
            page_result, fallback_result = _run_page_ocr(
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

    with parser_cls(config_path=config_path, layout_device=layout_device, _dotted=layout_dotted) as parser:
        for page_number, page_path in page_inputs:
            result = _parse_page_with_cpu_fallback(
                parser,
                page_path,
                method_name="parse_layout_only",
                parser_cls=parser_cls,
                config_path=config_path,
                layout_device=layout_device,
                layout_dotted=layout_dotted,
            )
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
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    result = _parse_page_with_cpu_fallback(
        parser,
        page_path,
        parser_cls=parser_cls,
        config_path=config_path,
        layout_device=layout_device,
        layout_dotted=layout_dotted,
    )
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
        config_module = import_module("glmocr.config")
        dataloader_module = import_module("glmocr.dataloader.page_loader")
        layout_module = import_module("glmocr.layout.layout_detector")
        ocr_client_module = import_module("glmocr.ocr_client")
        result_module = import_module("glmocr.parser_result.pipeline_result")
        formatter_module = import_module("glmocr.postprocess.result_formatter")
        image_utils_module = import_module("glmocr.utils.image_utils")
    except ImportError as exc:
        raise RuntimeError(
            "GLM-OCR is not installed. Install with `pip install -e .[glmocr]`."
        ) from exc
    return _build_lazy_glmocr_parser(
        load_config=config_module.load_config,
        page_loader_cls=dataloader_module.PageLoader,
        layout_detector_cls=layout_module.PPDocLayoutDetector,
        ocr_client_cls=ocr_client_module.OCRClient,
        pipeline_result_cls=result_module.PipelineResult,
        result_formatter_cls=formatter_module.ResultFormatter,
        crop_image_region=image_utils_module.crop_image_region,
    )


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
    class LazyGlmOcrParser:
        def __init__(
            self,
            *,
            config_path: str,
            layout_device: str,
            **kwargs: Any,
        ) -> None:
            self._config_path = config_path
            self._config_model = load_config(
                config_path,
                mode="selfhosted",
                layout_device=layout_device,
                **kwargs,
            )
            self._page_loader = page_loader_cls(self._config_model.pipeline.page_loader)
            self._layout_detector = layout_detector_cls(self._config_model.pipeline.layout)
            self._ocr_client: Any | None = None
            self._result_formatter = result_formatter_cls(self._config_model.pipeline.result_formatter)
            self._pipeline_result_cls = pipeline_result_cls
            self._crop_image_region = crop_image_region
            self._started_layout = False
            self._started_ocr = False

        def __enter__(self) -> "LazyGlmOcrParser":
            return self

        def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
            self.close()

        def parse(self, image: str | bytes | Path) -> Any:
            pages, original_images, layout_results, vis_images = self._collect_layout_stage(image)

            grouped_results: list[list[dict[str, Any]]] = []
            cropped_images: dict[tuple[Any, ...], Any] = {}

            for page_idx, (page_image, _unit_idx) in enumerate(pages):
                page_regions = layout_results[page_idx] if page_idx < len(layout_results) else []
                processed_regions: list[dict[str, Any]] = []
                for region in page_regions:
                    processed_region = dict(region)
                    try:
                        polygon = (
                            processed_region.get("polygon")
                            if self._config_model.pipeline.layout.use_polygon
                            else None
                        )
                        cropped = self._crop_image_region(
                            page_image,
                            processed_region["bbox_2d"],
                            polygon,
                        )
                    except Exception:
                        processed_region["content"] = ""
                        processed_regions.append(processed_region)
                        continue

                    if processed_region.get("task_type") == "skip":
                        processed_region["content"] = None
                        bbox = processed_region.get("bbox_2d")
                        if bbox:
                            cropped_images[(page_idx, *bbox)] = cropped
                        processed_regions.append(processed_region)
                        continue

                    self._ensure_ocr_started()
                    request_data = self._page_loader.build_request_from_image(
                        cropped,
                        processed_region.get("task_type", "text"),
                    )
                    ocr_client = self._ocr_client
                    if ocr_client is None:
                        raise RuntimeError("OCR client failed to initialize")
                    response, status_code = ocr_client.process(request_data)
                    if status_code == 200:
                        content = response["choices"][0]["message"]["content"]
                        processed_region["content"] = content.strip() if content else ""
                    else:
                        processed_region["content"] = None
                    processed_regions.append(processed_region)

                grouped_results.append(processed_regions)

            json_result, markdown_result, image_files = self._result_formatter.process(
                grouped_results,
                cropped_images=cropped_images or None,
            )
            raw_json_result = _build_raw_json(grouped_results)
            return self._pipeline_result_cls(
                json_result=json_result,
                markdown_result=markdown_result,
                original_images=original_images,
                image_files=image_files or None,
                raw_json_result=raw_json_result,
                layout_vis_images=vis_images or None,
            )

        def parse_layout_only(self, image: str | bytes | Path) -> Any:
            pages, original_images, layout_results, vis_images = self._collect_layout_stage(image)
            grouped_results = [
                [
                    {
                        **dict(region),
                        "content": str(region.get("content", "")),
                    }
                    for region in (layout_results[page_idx] if page_idx < len(layout_results) else [])
                ]
                for page_idx in range(len(pages))
            ]
            json_result, markdown_result, image_files = self._result_formatter.process(grouped_results)
            raw_json_result = _build_raw_json(grouped_results)
            return self._pipeline_result_cls(
                json_result=json_result,
                markdown_result=markdown_result,
                original_images=original_images,
                image_files=image_files or None,
                raw_json_result=raw_json_result,
                layout_vis_images=vis_images or None,
            )

        def _collect_layout_stage(
            self, image: str | bytes | Path
        ) -> tuple[list[tuple[Any, int]], list[str], list[list[dict[str, Any]]], dict[int, Any]]:
            sources = image if isinstance(image, bytes) else str(image)
            pages = list(self._page_loader.iter_pages_with_unit_indices(sources))
            if not pages:
                raise RuntimeError(f"GLM-OCR returned no results for {image!r}")
            original_images = [str(image)] if isinstance(image, (str, Path)) else []

            self._ensure_layout_started()
            page_images = [page for page, _unit_idx in pages]
            layout_results, vis_images = self._layout_detector.process(
                page_images,
                save_visualization=True,
                global_start_idx=0,
                use_polygon=self._config_model.pipeline.layout.use_polygon,
            )
            return pages, original_images, layout_results, vis_images

        def _ensure_layout_started(self) -> None:
            if self._started_layout:
                return
            self._layout_detector.start()
            self._started_layout = True

        def _ensure_ocr_started(self) -> None:
            if self._ocr_client is None:
                self._ocr_client = ocr_client_cls(self._config_model.pipeline.ocr_api)
            ocr_client = self._ocr_client
            if ocr_client is None:
                raise RuntimeError("OCR client failed to initialize")
            if self._started_ocr:
                return
            ocr_client.start()
            self._started_ocr = True

        def close(self) -> None:
            if self._started_ocr and self._ocr_client is not None:
                self._ocr_client.stop()
                self._started_ocr = False
            if self._started_layout:
                self._layout_detector.stop()
                self._started_layout = False

    return LazyGlmOcrParser


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
    try:
        return getattr(parser, method_name)(page_path)
    except Exception as exc:
        if not _should_retry_parse_on_cpu(exc, layout_device):
            raise

    close = getattr(parser, "close", None)
    if callable(close):
        with suppress(Exception):
            close()

    _cleanup_after_cuda_oom()
    with parser_cls(config_path=config_path, layout_device="cpu", _dotted=layout_dotted) as cpu_parser:
        return getattr(cpu_parser, method_name)(page_path)


def _should_retry_parse_on_cpu(exc: Exception, layout_device: str) -> bool:
    return layout_device.startswith("cuda") and _is_cuda_oom_error(exc)


def _is_cuda_oom_error(exc: BaseException) -> bool:
    if exc.__class__.__name__ == "OutOfMemoryError":
        module_name = getattr(exc.__class__, "__module__", "")
        if module_name.startswith("torch"):
            return True

    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _cleanup_after_cuda_oom() -> None:
    gc.collect()
    try:
        torch_module = import_module("torch")
    except ImportError:
        return

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not hasattr(cuda, "empty_cache"):
        return
    cuda.empty_cache()


def _build_raw_json(grouped_results: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    raw = []
    for page_results in grouped_results:
        sorted_results = sorted(page_results, key=lambda x: x.get("index", 0))
        raw.append(
            [
                {
                    "index": i,
                    "label": region.get("label", "text"),
                    "content": region.get("content", ""),
                    "bbox_2d": region.get("bbox_2d"),
                    "polygon": region.get("polygon"),
                }
                for i, region in enumerate(sorted_results)
            ]
        )
    return raw


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
    raw_root = Path(raw_dir)
    raw_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".page-{page_number:04d}-", dir=raw_root) as save_root:
        result.save(output_dir=save_root)
        return _publish_saved_model_json_path(save_root, raw_root, page_path, page_number)


def _publish_saved_model_json_path(
    save_root: str | Path,
    raw_root: str | Path,
    page_path: str,
    page_number: int,
) -> Path:
    page_stem = Path(page_path).stem
    source_dir = Path(save_root) / page_stem
    model_path = source_dir / f"{page_stem}_model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing saved GLM-OCR model JSON: {model_path}")

    target_dir = Path(raw_root) / f"page-{page_number:04d}"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_dir), str(target_dir))
    return target_dir / model_path.name


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
