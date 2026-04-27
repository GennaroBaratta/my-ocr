from __future__ import annotations

import copy
from importlib import import_module
from pathlib import Path
from typing import Any


def load_glmocr_parser() -> type:
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
    return build_lazy_glmocr_parser(
        load_config=config_module.load_config,
        page_loader_cls=dataloader_module.PageLoader,
        layout_detector_cls=layout_module.PPDocLayoutDetector,
        ocr_client_cls=ocr_client_module.OCRClient,
        pipeline_result_cls=result_module.PipelineResult,
        result_formatter_cls=formatter_module.ResultFormatter,
        crop_image_region=image_utils_module.crop_image_region,
    )


def _copy_ocr_api_config(ocr_api: Any, updates: dict[str, Any]) -> Any:
    if hasattr(ocr_api, "model_copy"):
        return ocr_api.model_copy(update=updates)

    cloned = copy.copy(ocr_api)
    for field_name, value in updates.items():
        setattr(cloned, field_name, value)
    return cloned


def _resolve_sdk_ocr_api_config(config_model: Any, config_path: str | Path) -> Any:
    pipeline = getattr(config_model, "pipeline", None)
    ocr_api = getattr(pipeline, "ocr_api", None)
    if ocr_api is None:
        raise RuntimeError(
            "GLM-OCR SDK config is missing pipeline.ocr_api; cannot start SDK OCR client"
        )

    inference_marker = getattr(pipeline, "inference", None)
    if inference_marker is None:
        return ocr_api

    from my_ocr.settings import resolve_inference_provider_config

    inference_config = resolve_inference_provider_config(config_path)
    if inference_config.provider == "ollama":
        api_mode = "ollama_generate"
    elif inference_config.provider == "openai_compatible":
        api_mode = "openai"
    else:
        raise RuntimeError(
            f"Unsupported inference provider for GLM-OCR SDK OCR: {inference_config.provider}"
        )

    return _copy_ocr_api_config(
        ocr_api,
        {
            "api_mode": api_mode,
            "api_url": inference_config.endpoint,
            "model": inference_config.model,
            "api_key": inference_config.api_key,
            "request_timeout": inference_config.timeout_seconds,
        },
    )


def build_lazy_glmocr_parser(
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
            raw_json_result = build_raw_json(grouped_results)
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
            raw_json_result = build_raw_json(grouped_results)
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
                self._ocr_client = ocr_client_cls(
                    _resolve_sdk_ocr_api_config(self._config_model, self._config_path)
                )
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


def build_raw_json(grouped_results: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
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
