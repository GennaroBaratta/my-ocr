from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from my_ocr.ocr import glmocr as ocr
from my_ocr.ocr import fallback as ocr_fallback
from my_ocr.ocr.glmocr import (
    prepare_review_artifacts as _prepare_review_artifacts,
)
from my_ocr.ocr.glmocr import run_ocr as _run_ocr
from my_ocr.ocr.fallback import run_crop_fallback_for_page
from my_ocr.ocr.planning import (
    FORMULA_RECOGNITION_PROMPT,
    TABLE_RECOGNITION_PROMPT,
    TEXT_RECOGNITION_PROMPT,
    build_ocr_chunks,
)
from my_ocr.support.text import normalize_table_html
from my_ocr.domain import LayoutBlock, PageRef, ReviewLayout, ReviewPage
from my_ocr.domain import OcrRuntimeOptions


class FakeResult:
    def __init__(
        self,
        *,
        markdown_result: str,
        json_result,
        saved_model_json=None,
        artifact_stem: str = "page-0001",
        error=None,
    ) -> None:
        self.markdown_result = markdown_result
        self.json_result = json_result
        self.saved_model_json = json_result if saved_model_json is None else saved_model_json
        self.artifact_stem = artifact_stem
        self._error = error
        self.saved_to: str | None = None

    def to_dict(self):
        payload = {"markdown": self.markdown_result, "json": self.json_result}
        if self._error is not None:
            payload["error"] = self._error
        return payload

    def save(self, output_dir: str) -> None:
        self.saved_to = output_dir
        page_dir = Path(output_dir) / self.artifact_stem
        page_dir.mkdir(parents=True, exist_ok=True)
        (page_dir / "artifact.txt").write_text("saved", encoding="utf-8")
        (page_dir / f"{self.artifact_stem}_model.json").write_text(
            json.dumps(self.saved_model_json),
            encoding="utf-8",
        )


class FakeGlmOcr:
    calls: list[str] = []

    def __init__(self, *, config_path: str, layout_device: str, **kwargs: object) -> None:
        self.config_path = config_path
        self.layout_device = layout_device

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return None

    def parse(self, input_path: str):
        self.calls.append(input_path)
        return FakeResult(
            markdown_result="# doc",
            json_result={"doc": Path(input_path).name},
            artifact_stem=Path(input_path).stem,
        )


def _patch_parser_cls(monkeypatch: pytest.MonkeyPatch, parser_cls: type) -> None:
    if hasattr(parser_cls, "parse_layout_only"):
        monkeypatch.setattr(ocr, "_load_glmocr_parser", lambda: parser_cls)
        return

    class ParserWithLayoutOnly(parser_cls):
        def parse_layout_only(self, input_path: str):
            return self.parse(input_path)

    monkeypatch.setattr(ocr, "_load_glmocr_parser", lambda: ParserWithLayoutOnly)


def _write_test_image(path: Path) -> None:
    image_module = import_module("PIL.Image")
    image_module.new("RGB", (10, 10), color="white").save(path)


def _page_refs(
    page_paths,
    *,
    page_numbers: list[int] | None = None,
) -> list[PageRef]:
    candidates = [page_paths] if isinstance(page_paths, (str, Path)) else list(page_paths)
    refs: list[PageRef] = []
    image_module = import_module("PIL.Image")
    for index, raw_path in enumerate(candidates, start=1):
        path = Path(raw_path)
        width = 0
        height = 0
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}:
            with image_module.open(path) as image:
                width, height = image.size
        inferred_number = (
            int(path.stem.rsplit("-", 1)[-1])
            if "-" in path.stem and path.stem.rsplit("-", 1)[-1].isdigit()
            else index
        )
        refs.append(
            PageRef(
                page_number=page_numbers[index - 1] if page_numbers else inferred_number,
                image_path=f"pages/{path.name}",
                width=width,
                height=height,
                resolved_path=path,
            )
        )
    return refs


def run_ocr(
    page_paths,
    run_dir,
    *,
    config_path: str = "config/local.yaml",
    layout_device: str = "cuda",
    layout_profile: str | None = "auto",
    review_layout: ReviewLayout | None = None,
    page_numbers=None,
):
    result = _run_ocr(
        _page_refs(page_paths, page_numbers=page_numbers),
        run_dir,
        review=review_layout,
        options=OcrRuntimeOptions(
            config_path=config_path,
            layout_device=layout_device,
            layout_profile=layout_profile,
        ),
    )
    pages = []
    for page in result.result.pages:
        payload = page.model_dump(mode="json")
        payload.update(page.raw_payload)
        pages.append(payload)
    summary = result.result.summary
    if review_layout is not None:
        summary["review_layout"] = {"page_count": len(review_layout.pages)}
    return {
        "markdown": result.result.markdown,
        "json": {"pages": pages, "summary": summary},
        "raw_dir": str(Path(run_dir) / "ocr_raw"),
        "config_path": config_path,
        "layout_device": layout_device,
        "layout_diagnostics": result.result.diagnostics,
        "_dto": result,
    }


def prepare_review_artifacts(
    page_paths,
    run_dir,
    *,
    config_path: str = "config/local.yaml",
    layout_device: str = "cuda",
    layout_profile: str | None = "auto",
    page_numbers=None,
):
    result = _prepare_review_artifacts(
        _page_refs(page_paths, page_numbers=page_numbers),
        run_dir,
        options=OcrRuntimeOptions(
            config_path=config_path,
            layout_device=layout_device,
            layout_profile=layout_profile,
        ),
    )
    payload = result.layout.model_dump(mode="json")
    payload["summary"] = result.layout.summary
    return {
        "review_layout": payload,
        "raw_dir": str(Path(run_dir) / "ocr_raw"),
        "config_path": config_path,
        "layout_device": layout_device,
        "layout_diagnostics": result.diagnostics.payload,
        "_dto": result,
    }


def _review_layout(*, blocks: list[LayoutBlock]) -> ReviewLayout:
    return ReviewLayout(
        pages=[
            ReviewPage(
                page_number=1,
                image_path="pages/page-0001.png",
                image_width=100,
                image_height=100,
                coord_space="pixel",
                blocks=blocks,
            )
        ],
        status="reviewed",
    )


def _review_block(
    *,
    block_id: str = "p0-b0",
    index: int = 0,
    label: str = "text",
    content: str = "",
    bbox: list[int] | None = None,
    confidence: float = 1.0,
) -> LayoutBlock:
    return LayoutBlock(
        id=block_id,
        index=index,
        label=label,
        content=content,
        confidence=confidence,
        bbox=bbox or [1, 2, 10, 20],
    )


def test_lazy_parser_wrapper_defers_pipeline_start_until_first_parse() -> None:
    events: list[str] = []

    class FakePipelineResult:
        def __init__(self, *, json_result, markdown_result, original_images, image_files=None, raw_json_result=None, layout_vis_images=None) -> None:
            self.json_result = json_result
            self.markdown_result = markdown_result
            self.original_images = original_images
            self.image_files = image_files
            self.raw_json_result = raw_json_result
            self.layout_vis_images = layout_vis_images

    class FakePageLoader:
        def __init__(self, config: object) -> None:
            self.config = config

        def iter_pages_with_unit_indices(self, source: str):
            events.append(f"load:{source}")
            image = import_module("PIL.Image").new("RGB", (10, 10), color="white")
            yield image, 0

        def build_request_from_image(self, image, task_type: str = "text") -> dict[str, object]:
            _ = image
            events.append(f"build-request:{task_type}")
            return {"messages": [{"role": "user", "content": [{"type": "text", "text": task_type}]}]}

    class FakeLayoutDetector:
        def __init__(self, config: object) -> None:
            self.config = config

        def start(self) -> None:
            events.append("layout-start")

        def stop(self) -> None:
            events.append("layout-stop")

        def process(self, images, **kwargs):
            _ = (images, kwargs)
            events.append("layout-process")
            return [[{"index": 0, "label": "text", "bbox_2d": [0, 0, 1000, 1000], "task_type": "text"}]], {}

    class FakeOcrClient:
        def __init__(self, config: object) -> None:
            self.config = config
            events.append("ocr-init")

        def start(self) -> None:
            events.append("ocr-start")

        def stop(self) -> None:
            events.append("ocr-stop")

        def process(self, request_data: dict[str, object]):
            events.append("ocr-process")
            return {"choices": [{"message": {"content": "recognized text"}}]}, 200

    class FakeResultFormatter:
        def __init__(self, config: object) -> None:
            self.config = config

        def process(self, grouped_results, cropped_images=None, image_prefix: str = "cropped"):
            _ = (cropped_images, image_prefix)
            events.append("format")
            return grouped_results, "# lazy doc", {}

    parser_cls = ocr._build_lazy_glmocr_parser(
        load_config=lambda *_args, **_kwargs: SimpleNamespace(
            pipeline=SimpleNamespace(
                page_loader="page-loader",
                layout=SimpleNamespace(use_polygon=False),
                ocr_api="ocr-api",
                result_formatter="formatter",
            )
        ),
        page_loader_cls=FakePageLoader,
        layout_detector_cls=FakeLayoutDetector,
        ocr_client_cls=FakeOcrClient,
        pipeline_result_cls=FakePipelineResult,
        result_formatter_cls=FakeResultFormatter,
        crop_image_region=lambda image, bbox_2d, polygon=None: image,
    )

    with parser_cls(config_path="config/local.yaml", layout_device="cuda") as parser:
        assert events == []
        first = parser.parse("/tmp/page-0001.png")
        second = parser.parse("/tmp/page-0002.png")

    assert events == [
        "load:/tmp/page-0001.png",
        "layout-start",
        "layout-process",
        "ocr-init",
        "ocr-start",
        "build-request:text",
        "ocr-process",
        "format",
        "load:/tmp/page-0002.png",
        "layout-process",
        "build-request:text",
        "ocr-process",
        "format",
        "ocr-stop",
        "layout-stop",
    ]
    assert first.markdown_result == "# lazy doc"
    assert second.markdown_result == "# lazy doc"


def test_lazy_parser_wrapper_returns_first_pipeline_result_only() -> None:
    class FakePipelineResult:
        def __init__(self, *, json_result, markdown_result, original_images, image_files=None, raw_json_result=None, layout_vis_images=None) -> None:
            self.json_result = json_result
            self.markdown_result = markdown_result
            self.original_images = original_images

    class FakePageLoader:
        def __init__(self, config: object) -> None:
            self.config = config

        def iter_pages_with_unit_indices(self, source: str):
            image = import_module("PIL.Image").new("RGB", (10, 10), color="white")
            yield image, 0

        def build_request_from_image(self, image, task_type: str = "text") -> dict[str, object]:
            _ = (image, task_type)
            return {"messages": []}

    class FakeLayoutDetector:
        def __init__(self, config: object) -> None:
            self.config = config

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def process(self, images, **kwargs):
            _ = (images, kwargs)
            return [[{"index": 0, "label": "text", "bbox_2d": [0, 0, 1000, 1000], "task_type": "text"}]], {}

    class FakeOcrClient:
        def __init__(self, config: object) -> None:
            self.config = config

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def process(self, request_data: dict[str, object]):
            _ = request_data
            return {"choices": [{"message": {"content": "only first"}}]}, 200

    class FakeResultFormatter:
        def __init__(self, config: object) -> None:
            self.config = config

        def process(self, grouped_results, cropped_images=None, image_prefix: str = "cropped"):
            _ = (grouped_results, cropped_images, image_prefix)
            return {"doc": "page-0001.png"}, "# first result", {}

    parser_cls = ocr._build_lazy_glmocr_parser(
        load_config=lambda *_args, **_kwargs: SimpleNamespace(
            pipeline=SimpleNamespace(
                page_loader="page-loader",
                layout=SimpleNamespace(use_polygon=False),
                ocr_api="ocr-api",
                result_formatter="formatter",
            )
        ),
        page_loader_cls=FakePageLoader,
        layout_detector_cls=FakeLayoutDetector,
        ocr_client_cls=FakeOcrClient,
        pipeline_result_cls=FakePipelineResult,
        result_formatter_cls=FakeResultFormatter,
        crop_image_region=lambda image, bbox_2d, polygon=None: image,
    )

    with parser_cls(config_path="config/local.yaml", layout_device="cuda") as parser:
        result = parser.parse("/tmp/page-0001.png")

    assert result.markdown_result == "# first result"


def test_lazy_parser_wrapper_layout_only_does_not_start_ocr() -> None:
    events: list[str] = []

    class FakePipelineResult:
        def __init__(self, *, json_result, markdown_result, original_images, image_files=None, raw_json_result=None, layout_vis_images=None) -> None:
            self.json_result = json_result
            self.markdown_result = markdown_result

    class FakePageLoader:
        def __init__(self, config: object) -> None:
            self.config = config

        def iter_pages_with_unit_indices(self, source: str):
            events.append(f"load:{source}")
            image = import_module("PIL.Image").new("RGB", (10, 10), color="white")
            yield image, 0

        def build_request_from_image(self, image, task_type: str = "text") -> dict[str, object]:
            raise AssertionError("layout-only path should not build OCR requests")

    class FakeLayoutDetector:
        def __init__(self, config: object) -> None:
            self.config = config

        def start(self) -> None:
            events.append("layout-start")

        def stop(self) -> None:
            events.append("layout-stop")

        def process(self, images, **kwargs):
            _ = (images, kwargs)
            events.append("layout-process")
            return [[{"index": 0, "label": "text", "bbox_2d": [0, 0, 1000, 1000], "task_type": "text"}]], {}

    class FakeOcrClient:
        def __init__(self, config: object) -> None:
            events.append("ocr-init")

        def start(self) -> None:
            events.append("ocr-start")

        def stop(self) -> None:
            events.append("ocr-stop")

    class FakeResultFormatter:
        def __init__(self, config: object) -> None:
            self.config = config

        def process(self, grouped_results, cropped_images=None, image_prefix: str = "cropped"):
            _ = (grouped_results, cropped_images, image_prefix)
            events.append("format")
            return {"doc": "page-0001.png"}, "# layout only", {}

    parser_cls = ocr._build_lazy_glmocr_parser(
        load_config=lambda *_args, **_kwargs: SimpleNamespace(
            pipeline=SimpleNamespace(
                page_loader="page-loader",
                layout=SimpleNamespace(use_polygon=False),
                ocr_api="ocr-api",
                result_formatter="formatter",
            )
        ),
        page_loader_cls=FakePageLoader,
        layout_detector_cls=FakeLayoutDetector,
        ocr_client_cls=FakeOcrClient,
        pipeline_result_cls=FakePipelineResult,
        result_formatter_cls=FakeResultFormatter,
        crop_image_region=lambda image, bbox_2d, polygon=None: image,
    )

    with parser_cls(config_path="config/local.yaml", layout_device="cuda") as parser:
        result = parser.parse_layout_only("/tmp/page-0001.png")

    assert result.markdown_result == "# layout only"
    assert events == ["load:/tmp/page-0001.png", "layout-start", "layout-process", "format", "layout-stop"]


def test_lazy_parser_wrapper_layout_only_result_save_writes_model_json(tmp_path) -> None:
    class FakePipelineResult:
        def __init__(self, *, json_result, markdown_result, original_images, image_files=None, raw_json_result=None, layout_vis_images=None) -> None:
            self.json_result = json_result
            self.markdown_result = markdown_result
            self.original_images = original_images
            self.image_files = image_files
            self.raw_json_result = raw_json_result
            self.layout_vis_images = layout_vis_images

        def save(self, output_dir: str) -> None:
            page_dir = Path(output_dir) / "page-0001"
            page_dir.mkdir(parents=True, exist_ok=True)
            (page_dir / "page-0001_model.json").write_text(
                json.dumps(self.raw_json_result),
                encoding="utf-8",
            )

    class FakePageLoader:
        def __init__(self, config: object) -> None:
            self.config = config

        def iter_pages_with_unit_indices(self, source: str):
            image = import_module("PIL.Image").new("RGB", (10, 10), color="white")
            yield image, 0

        def build_request_from_image(self, image, task_type: str = "text") -> dict[str, object]:
            raise AssertionError("layout-only path should not build OCR requests")

    class FakeLayoutDetector:
        def __init__(self, config: object) -> None:
            self.config = config

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def process(self, images, **kwargs):
            _ = (images, kwargs)
            return [[{"index": 0, "label": "text", "bbox_2d": [1, 2, 3, 4], "task_type": "text"}]], {}

    class FakeOcrClient:
        def __init__(self, config: object) -> None:
            self.config = config

    class FakeResultFormatter:
        def __init__(self, config: object) -> None:
            self.config = config

        def process(self, grouped_results, cropped_images=None, image_prefix: str = "cropped"):
            _ = (grouped_results, cropped_images, image_prefix)
            return {"doc": "page-0001.png"}, "# layout only", {}

    parser_cls = ocr._build_lazy_glmocr_parser(
        load_config=lambda *_args, **_kwargs: SimpleNamespace(
            pipeline=SimpleNamespace(
                page_loader="page-loader",
                layout=SimpleNamespace(use_polygon=False),
                ocr_api="ocr-api",
                result_formatter="formatter",
            )
        ),
        page_loader_cls=FakePageLoader,
        layout_detector_cls=FakeLayoutDetector,
        ocr_client_cls=FakeOcrClient,
        pipeline_result_cls=FakePipelineResult,
        result_formatter_cls=FakeResultFormatter,
        crop_image_region=lambda image, bbox_2d, polygon=None: image,
    )

    with parser_cls(config_path="config/local.yaml", layout_device="cuda") as parser:
        result = parser.parse_layout_only(str(tmp_path / "page-0001.png"))

    result.save(str(tmp_path / "out"))

    saved = json.loads((tmp_path / "out" / "page-0001" / "page-0001_model.json").read_text(encoding="utf-8"))
    assert saved == [[{"index": 0, "label": "text", "content": "", "bbox_2d": [1, 2, 3, 4], "polygon": None}]]


def test_run_ocr_uses_page_images_and_public_outputs(tmp_path, monkeypatch) -> None:
    FakeGlmOcr.calls = []
    _patch_parser_cls(monkeypatch, FakeGlmOcr)
    source = tmp_path / "page-0001.png"
    _write_test_image(source)

    result = run_ocr([str(source)], tmp_path / "run")

    assert FakeGlmOcr.calls == [str(source)]
    assert result["markdown"] == "# doc"
    assert result["json"]["summary"] == {"page_count": 1, "sources": {"sdk_markdown": 1}}
    page = result["json"]["pages"][0]
    expected_model_path = tmp_path / "run" / "ocr_raw" / "page-0001" / "page-0001_model.json"
    assert page["provider_path"] == "ocr/provider/page-0001"
    assert "sdk_json_path" not in page
    assert json.loads(expected_model_path.read_text(encoding="utf-8")) == {"doc": "page-0001.png"}
    assert result["config_path"] == "config/local.yaml"
    assert result["layout_device"] == "cuda"
    assert (tmp_path / "run" / "ocr_raw" / "page-0001" / "artifact.txt").exists()
    assert not (tmp_path / "run" / "ocr_raw" / "page-0001" / "page-0001").exists()


def test_run_ocr_emits_layout_profile_passthrough_warning_to_stderr(
    tmp_path, monkeypatch, capsys
) -> None:
    _patch_parser_cls(monkeypatch, FakeGlmOcr)
    monkeypatch.setattr(
        ocr._layout_profile_mod,
        "resolve_layout_profile",
        lambda *_args, **_kwargs: (
            {},
            {
                "layout_profile_requested": "auto",
                "layout_profile_source": "auto",
                "layout_profile_status": "passthrough_existing_config",
                "layout_profile_warning": "Proceeding with existing config/default mappings.",
            },
        ),
    )
    source = tmp_path / "page-0001.png"
    _write_test_image(source)

    run_ocr([str(source)], tmp_path / "run")

    captured = capsys.readouterr()
    assert "Warning: Proceeding with existing config/default mappings." in captured.err


def test_run_ocr_aggregates_pages_in_order_and_tracks_sources(tmp_path, monkeypatch) -> None:
    class MixedSourceGlmOcr(FakeGlmOcr):
        def parse(self, input_path: str):
            self.calls.append(input_path)
            page_name = Path(input_path).name
            if page_name == "page-0001.png":
                return FakeResult(
                    markdown_result="# first",
                    json_result={"doc": page_name},
                    artifact_stem=Path(input_path).stem,
                )
            return FakeResult(
                markdown_result="",
                json_result={
                    "blocks": [
                        {"label": "text", "content": "second page", "bbox_2d": [0, 0, 10, 10]}
                    ]
                },
                artifact_stem=Path(input_path).stem,
            )

    _patch_parser_cls(monkeypatch, MixedSourceGlmOcr)
    page_one = tmp_path / "page-0001.png"
    page_two = tmp_path / "page-0002.png"
    _write_test_image(page_one)
    _write_test_image(page_two)

    result = run_ocr([str(page_one), str(page_two)], tmp_path / "run")

    assert result["markdown"] == "# first\n\nsecond page"
    assert result["json"]["summary"] == {
        "page_count": 2,
        "sources": {"sdk_markdown": 1, "layout_json": 1},
    }
    assert [page["image_path"] for page in result["json"]["pages"]] == [
        "pages/page-0001.png",
        "pages/page-0002.png",
    ]


def test_run_ocr_raises_on_sdk_error_field(tmp_path, monkeypatch) -> None:
    class ErrorGlmOcr(FakeGlmOcr):
        def parse(self, input_path: str):
            self.calls.append(input_path)
            return FakeResult(
                markdown_result="",
                json_result={},
                artifact_stem=Path(input_path).stem,
                error="boom",
            )

    _patch_parser_cls(monkeypatch, ErrorGlmOcr)
    source = tmp_path / "page-0001.png"
    _write_test_image(source)

    with pytest.raises(RuntimeError, match="boom"):
        run_ocr([str(source)], tmp_path / "run")


def test_run_ocr_retries_same_page_once_on_cpu_after_cuda_oom(tmp_path, monkeypatch) -> None:
    empty_cache_calls: list[str] = []

    class CudaOomGlmOcr(FakeGlmOcr):
        init_devices: list[str] = []
        parse_devices: list[tuple[str, str]] = []

        def __init__(self, *, config_path: str, layout_device: str, **kwargs: object) -> None:
            super().__init__(config_path=config_path, layout_device=layout_device, **kwargs)
            self.init_devices.append(layout_device)

        def parse(self, input_path: str):
            self.parse_devices.append((self.layout_device, input_path))
            if self.layout_device == "cuda":
                raise RuntimeError("CUDA out of memory while running layout detection")
            return FakeResult(
                markdown_result="# cpu doc",
                json_result={"doc": Path(input_path).name},
                artifact_stem=Path(input_path).stem,
            )

    _patch_parser_cls(monkeypatch, CudaOomGlmOcr)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: empty_cache_calls.append("called"))),
    )
    source = tmp_path / "page-0001.png"
    _write_test_image(source)

    result = run_ocr([str(source)], tmp_path / "run")

    assert CudaOomGlmOcr.init_devices == ["cuda", "cpu"]
    assert CudaOomGlmOcr.parse_devices == [("cuda", str(source)), ("cpu", str(source))]
    assert empty_cache_calls == ["called"]
    assert result["markdown"] == "# cpu doc"
    assert result["layout_device"] == "cuda"
    page = result["json"]["pages"][0]
    assert page["provider_path"] == "ocr/provider/page-0001"
    assert "sdk_json_path" not in page
    assert (tmp_path / "run" / "ocr_raw" / "page-0001" / "page-0001_model.json").exists()


def test_run_ocr_does_not_retry_non_oom_errors(tmp_path, monkeypatch) -> None:
    class NonOomGlmOcr(FakeGlmOcr):
        init_devices: list[str] = []

        def __init__(self, *, config_path: str, layout_device: str, **kwargs: object) -> None:
            super().__init__(config_path=config_path, layout_device=layout_device, **kwargs)
            self.init_devices.append(layout_device)

        def parse(self, input_path: str):
            raise RuntimeError("unexpected parser failure")

    _patch_parser_cls(monkeypatch, NonOomGlmOcr)
    source = tmp_path / "page-0001.png"
    _write_test_image(source)

    with pytest.raises(RuntimeError, match="unexpected parser failure"):
        run_ocr([str(source)], tmp_path / "run")

    assert NonOomGlmOcr.init_devices == ["cuda"]


def test_run_ocr_rejects_empty_page_list(tmp_path, monkeypatch) -> None:
    _patch_parser_cls(monkeypatch, FakeGlmOcr)

    with pytest.raises(ValueError, match="At least one normalized page image is required"):
        run_ocr([], tmp_path / "run")


def test_run_ocr_rejects_directory_input(tmp_path, monkeypatch) -> None:
    _patch_parser_cls(monkeypatch, FakeGlmOcr)
    source = tmp_path / "images"
    source.mkdir()

    with pytest.raises(ValueError, match="page image files"):
        run_ocr([str(source)], tmp_path / "run")


def test_run_ocr_reconstructs_markdown_from_layout_json(tmp_path, monkeypatch) -> None:
    class LayoutGlmOcr(FakeGlmOcr):
        def parse(self, input_path: str):
            self.calls.append(input_path)
            return FakeResult(
                markdown_result="",
                json_result={},
                saved_model_json={
                    "blocks": [
                        {"label": "text", "content": "Recovered text", "bbox_2d": [0, 0, 10, 10]}
                    ]
                },
                artifact_stem=Path(input_path).stem,
            )

    _patch_parser_cls(monkeypatch, LayoutGlmOcr)
    source = tmp_path / "page-0001.png"
    _write_test_image(source)

    result = run_ocr([str(source)], tmp_path / "run")

    assert result["markdown"] == "Recovered text"
    assert result["json"]["pages"][0]["markdown_source"] == "layout_json"


def test_run_ocr_falls_back_to_full_page_when_sdk_and_layout_are_empty(
    tmp_path, monkeypatch
) -> None:
    class EmptyGlmOcr(FakeGlmOcr):
        def parse(self, input_path: str):
            self.calls.append(input_path)
            return FakeResult(
                markdown_result="",
                json_result={},
                saved_model_json={
                    "blocks": [
                        {"label": "text", "content": "", "bbox_2d": [0, 0, 10, 10], "index": 1}
                    ]
                },
                artifact_stem=Path(input_path).stem,
            )

    _patch_parser_cls(monkeypatch, EmptyGlmOcr)
    monkeypatch.setattr(ocr, "run_crop_fallback_for_page", lambda **_kwargs: ("", []))
    monkeypatch.setattr(ocr, "recognize_full_page", lambda *_args, **_kwargs: "Full page text")
    source = tmp_path / "page-0001.png"
    _write_test_image(source)

    result = run_ocr([str(source)], tmp_path / "run")

    assert result["markdown"] == "Full page text"
    assert result["json"]["pages"][0]["markdown_source"] == "full_page_fallback"
    assert result["json"]["pages"][0]["assessment"] == {
        "use_fallback": True,
        "reason": "empty_markdown_and_empty_layout_text",
        "structured_layout_payload": True,
        "layout_block_count": 1,
        "ocr_block_count": 1,
        "text_block_count": 1,
        "meaningful_text_block_count": 0,
        "bbox_coord_space": "normalized",
    }


def test_build_ocr_chunks_assigns_text_table_and_formula_prompts() -> None:
    chunks = build_ocr_chunks(
        {
            "blocks": [
                {"label": "text", "bbox_2d": [0, 0, 50, 50], "index": 1},
                {"label": "table", "bbox_2d": [60, 0, 120, 50], "index": 2},
                {"label": "formula", "bbox_2d": [0, 60, 50, 120], "index": 3},
            ]
        },
        width=200,
        height=200,
        coord_space="pixel",
    )

    assert [chunk["task"] for chunk in chunks] == ["text", "table", "formula"]
    assert [chunk["prompt"] for chunk in chunks] == [
        TEXT_RECOGNITION_PROMPT,
        TABLE_RECOGNITION_PROMPT,
        FORMULA_RECOGNITION_PROMPT,
    ]


def test_build_ocr_chunks_treats_display_and_inline_formula_labels_as_formula() -> None:
    chunks = build_ocr_chunks(
        {
            "blocks": [
                {"label": "display_formula", "bbox_2d": [0, 0, 50, 50], "index": 1},
                {"label": "inline_formula", "bbox_2d": [60, 0, 120, 50], "index": 2},
            ]
        },
        width=200,
        height=200,
        coord_space="pixel",
    )

    assert [chunk["task"] for chunk in chunks] == ["formula", "formula"]
    assert [chunk["prompt"] for chunk in chunks] == [
        FORMULA_RECOGNITION_PROMPT,
        FORMULA_RECOGNITION_PROMPT,
    ]


def test_run_ocr_always_loads_saved_model_json(tmp_path, monkeypatch) -> None:
    captured = {}

    class TableModelGlmOcr(FakeGlmOcr):
        def parse(self, input_path: str):
            self.calls.append(input_path)
            return FakeResult(
                markdown_result="",
                json_result={"blocks": []},
                saved_model_json={
                    "blocks": [
                        {
                            "label": "table",
                            "content": "",
                            "bbox_2d": [100, 100, 500, 500],
                            "index": 2,
                        }
                    ]
                },
                artifact_stem=Path(input_path).stem,
            )

    def fake_crop_fallback(**kwargs):
        captured["page_json"] = kwargs["page_json"]
        return "table text", []

    _patch_parser_cls(monkeypatch, TableModelGlmOcr)
    monkeypatch.setattr(ocr, "run_crop_fallback_for_page", fake_crop_fallback)
    source = tmp_path / "page-0001.png"
    _write_test_image(source)

    result = run_ocr([str(source)], tmp_path / "run")

    assert captured["page_json"] == {
        "blocks": [{"label": "table", "content": "", "bbox_2d": [100, 100, 500, 500], "index": 2}]
    }
    expected_model_path = tmp_path / "run" / "ocr_raw" / "page-0001" / "page-0001_model.json"
    assert json.loads(expected_model_path.read_text(encoding="utf-8")) == captured["page_json"]
    assert result["json"]["pages"][0]["markdown_source"] == "crop_fallback"


def test_build_ocr_chunks_scales_glmocr_normalized_offsets() -> None:
    chunks = build_ocr_chunks(
        {
            "blocks": [
                {"label": "text", "bbox_2d": [100, 100, 500, 500]},
                {"label": "text", "bbox_2d": [600, 200, 900, 400], "index": 7},
            ]
        },
        width=200,
        height=100,
    )

    assert [chunk["unpadded_bbox"] for chunk in chunks] == [[20, 10, 100, 50], [120, 20, 180, 40]]
    assert [chunk["source_indices"] for chunk in chunks] == [[0], [7]]


def test_normalize_table_html_uses_plain_rows() -> None:
    html = "<table><tr><th>head1</th><th>head2</th></tr><tr><td>a</td><td>b</td></tr></table>"

    assert normalize_table_html(html) == "head1 | head2\na | b"


def test_crop_fallback_normalizes_table_chunks_and_page_markdown(tmp_path, monkeypatch) -> None:
    page = tmp_path / "page-0001.png"
    _write_test_image(page)

    monkeypatch.setattr(
        ocr_fallback,
        "recognize_text_image",
        lambda *_args, **_kwargs: (
            "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        ),
    )

    page_markdown, chunks = run_crop_fallback_for_page(
        page_path=str(page),
        page_json={"blocks": [{"label": "table", "bbox_2d": [0, 0, 10, 10], "index": 1}]},
        coord_space="pixel",
        page_fallback_dir=tmp_path / "fallback",
        model="m",
        endpoint="e",
        num_ctx=1,
    )

    assert page_markdown == "A | B\n1 | 2"
    assert chunks[0]["text"] == "A | B\n1 | 2"
    assert (tmp_path / "fallback" / "chunk-0001.txt").read_text(encoding="utf-8") == "A | B\n1 | 2"


def test_run_ocr_persists_normalized_table_markdown(tmp_path, monkeypatch) -> None:
    class TableLayoutGlmOcr(FakeGlmOcr):
        def parse(self, input_path: str):
            self.calls.append(input_path)
            return FakeResult(
                markdown_result="",
                json_result={},
                saved_model_json={
                    "blocks": [
                        {"label": "text", "content": "Intro", "bbox_2d": [0, 0, 10, 10]},
                        {
                            "label": "table",
                            "content": "<table><tr><th>X</th><th>Y</th></tr><tr><td>1</td><td>2</td></tr></table>",
                            "bbox_2d": [0, 20, 10, 30],
                        },
                    ]
                },
                artifact_stem=Path(input_path).stem,
            )

    _patch_parser_cls(monkeypatch, TableLayoutGlmOcr)
    source = tmp_path / "page-0001.png"
    _write_test_image(source)

    result = run_ocr([str(source)], tmp_path / "run")

    assert result["markdown"] == "Intro\n\nX | Y\n1 | 2"


def test_run_ocr_consumes_review_layout_for_planning_and_fallback(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_crop_fallback(**kwargs):
        captured["page_json"] = kwargs["page_json"]
        captured["coord_space"] = kwargs["coord_space"]
        return "review-driven text", []

    def fail_parser_load():
        raise AssertionError("review-layout OCR should not load the GLM-OCR parser")

    monkeypatch.setattr(ocr, "_load_glmocr_parser", fail_parser_load)
    monkeypatch.setattr(ocr, "run_crop_fallback_for_page", fake_crop_fallback)
    source = tmp_path / "page-0001.png"
    _write_test_image(source)

    review_layout = _review_layout(
        blocks=[
            _review_block(
                label="table",
                bbox=[2, 3, 9, 10],
            )
        ],
    )

    result = run_ocr(
        [str(source)],
        tmp_path / "run",
        review_layout=review_layout,
    )

    assert captured["page_json"] == {
        "blocks": [{"index": 0, "label": "table", "content": "", "bbox_2d": [2, 3, 9, 10]}]
    }
    assert captured["coord_space"] == "pixel"
    assert result["markdown"] == "review-driven text"
    assert result["json"]["pages"][0]["layout_source"] == "review_layout"
    expected_model_path = tmp_path / "run" / "ocr_raw" / "page-0001" / "page-0001_model.json"
    assert "sdk_json_path" not in result["json"]["pages"][0]
    assert json.loads(expected_model_path.read_text(encoding="utf-8")) == {
        "blocks": [{"index": 0, "label": "table", "content": "", "bbox_2d": [2, 3, 9, 10]}]
    }
    assert result["json"]["summary"]["review_layout"] == {"page_count": 1}
    assert result["json"]["pages"][0]["markdown"] == "review-driven text"


def test_run_ocr_uses_nonempty_review_layout_as_canonical_layout(
    tmp_path, monkeypatch
) -> None:
    def fail_parser_load():
        raise AssertionError("review-layout OCR should not load the GLM-OCR parser")

    def fail_crop_fallback(**_kwargs):
        raise AssertionError("non-empty review layout should not use crop fallback")

    monkeypatch.setattr(ocr, "_load_glmocr_parser", fail_parser_load)
    monkeypatch.setattr(ocr, "run_crop_fallback_for_page", fail_crop_fallback)
    source = tmp_path / "page-0001.png"
    _write_test_image(source)

    review_layout = _review_layout(
        blocks=[
            _review_block(
                content="Reviewed block text",
                bbox=[2, 3, 9, 10],
            )
        ],
    )

    result = run_ocr(
        [str(source)],
        tmp_path / "run",
        review_layout=review_layout,
    )

    page = result["json"]["pages"][0]
    assert page["layout_source"] == "review_layout"
    assert page["sdk_markdown"] == ""
    assert page["markdown"] == "Reviewed block text"
    assert page["markdown_source"] == "layout_json"
    assert result["json"]["summary"]["review_layout"] == {"page_count": 1}
    expected_model_path = tmp_path / "run" / "ocr_raw" / "page-0001" / "page-0001_model.json"
    assert "sdk_json_path" not in page
    assert json.loads(expected_model_path.read_text(encoding="utf-8")) == {
        "blocks": [
            {
                "index": 0,
                "label": "text",
                "content": "Reviewed block text",
                "bbox_2d": [2, 3, 9, 10],
            }
        ]
    }


def test_run_ocr_uses_review_layout_without_loading_glm_parser(
    tmp_path, monkeypatch
) -> None:
    def fail_parser_load():
        raise AssertionError("review-layout OCR should not load the GLM-OCR parser")

    monkeypatch.setattr(ocr, "_load_glmocr_parser", fail_parser_load)
    monkeypatch.setattr(
        ocr._layout_profile_mod,
        "resolve_layout_profile",
        lambda *_args, **_kwargs: ({}, {}),
    )
    source = tmp_path / "page-0001.png"
    _write_test_image(source)

    review_layout = _review_layout(
        blocks=[
            _review_block(
                content="Reviewed text",
                bbox=[1, 2, 40, 30],
            )
        ],
    )

    result = run_ocr(
        [str(source)],
        tmp_path / "run",
        review_layout=review_layout,
    )

    assert result["markdown"] == "Reviewed text"
    assert result["json"]["pages"][0]["layout_source"] == "review_layout"
    assert result["json"]["pages"][0]["markdown_source"] == "layout_json"
    assert result["json"]["summary"]["review_layout"] == {"page_count": 1}


def test_subset_runs_preserve_original_page_numbers(tmp_path, monkeypatch) -> None:
    _patch_parser_cls(monkeypatch, FakeGlmOcr)
    source = tmp_path / "page-0005.png"
    _write_test_image(source)

    ocr_result = run_ocr([str(source)], tmp_path / "ocr-run")
    review_result = prepare_review_artifacts([str(source)], tmp_path / "review-run")

    assert ocr_result["json"]["pages"][0]["page_number"] == 5
    assert ocr_result["json"]["pages"][0]["provider_path"] == "ocr/provider/page-0005"
    assert "sdk_json_path" not in ocr_result["json"]["pages"][0]
    assert review_result["review_layout"]["pages"][0]["page_number"] == 5
    assert review_result["review_layout"]["pages"][0]["provider_path"] == "layout/provider/page-0005"


def test_run_ocr_canonicalizes_raw_dir_for_explicit_page_number(tmp_path, monkeypatch) -> None:
    _patch_parser_cls(monkeypatch, FakeGlmOcr)
    source = tmp_path / "scan.png"
    _write_test_image(source)

    result = run_ocr([str(source)], tmp_path / "run", page_numbers=[5])

    expected_model_path = tmp_path / "run" / "ocr_raw" / "page-0005" / "scan_model.json"
    page = result["json"]["pages"][0]
    assert page["page_number"] == 5
    assert page["provider_path"] == "ocr/provider/page-0005"
    assert "sdk_json_path" not in page
    assert expected_model_path.exists()
    assert not (tmp_path / "run" / "ocr_raw" / "scan").exists()


def test_run_ocr_keeps_colliding_page_stems_isolated(tmp_path, monkeypatch) -> None:
    class CollidingStemGlmOcr(FakeGlmOcr):
        def parse(self, input_path: str):
            return FakeResult(
                markdown_result=f"# {Path(input_path).parent.name}",
                json_result={"source_dir": Path(input_path).parent.name},
                artifact_stem=Path(input_path).stem,
            )

    _patch_parser_cls(monkeypatch, CollidingStemGlmOcr)
    first = tmp_path / "first" / "page-0001.png"
    second = tmp_path / "second" / "page-0001.png"
    first.parent.mkdir()
    second.parent.mkdir()
    _write_test_image(first)
    _write_test_image(second)

    result = run_ocr([str(first), str(second)], tmp_path / "run", page_numbers=[1, 5])

    first_model_path = tmp_path / "run" / "ocr_raw" / "page-0001" / "page-0001_model.json"
    second_model_path = tmp_path / "run" / "ocr_raw" / "page-0005" / "page-0001_model.json"
    assert result["json"]["pages"][0]["provider_path"] == "ocr/provider/page-0001"
    assert result["json"]["pages"][1]["provider_path"] == "ocr/provider/page-0005"
    assert "sdk_json_path" not in result["json"]["pages"][0]
    assert "sdk_json_path" not in result["json"]["pages"][1]
    assert json.loads(first_model_path.read_text(encoding="utf-8")) == {"source_dir": "first"}
    assert json.loads(second_model_path.read_text(encoding="utf-8")) == {"source_dir": "second"}


def test_prepare_review_canonicalizes_raw_dir_for_explicit_page_number(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_parser_cls(monkeypatch, FakeGlmOcr)
    source = tmp_path / "scan.png"
    _write_test_image(source)

    result = prepare_review_artifacts([str(source)], tmp_path / "review", page_numbers=[5])

    expected_model_path = tmp_path / "review" / "ocr_raw" / "page-0005" / "scan_model.json"
    page = result["review_layout"]["pages"][0]
    assert page["page_number"] == 5
    assert page["provider_path"] == "layout/provider/page-0005"
    assert expected_model_path.exists()
    assert not (tmp_path / "review" / "ocr_raw" / "scan").exists()


def test_prepare_review_artifacts_preserves_saved_block_text(tmp_path, monkeypatch) -> None:
    class ReviewPrepParser(FakeGlmOcr):
        parse_calls: list[str] = []
        layout_only_calls: list[str] = []

        def parse(self, input_path: str):
            self.parse_calls.append(input_path)
            raise AssertionError("review preparation should use layout-only parsing")

        def parse_layout_only(self, input_path: str):
            self.layout_only_calls.append(input_path)
            return FakeResult(
                markdown_result="",
                json_result={"doc": Path(input_path).name},
                saved_model_json={
                    "blocks": [
                        {"label": "text", "content": "Recovered text", "bbox_2d": [0, 0, 1000, 1000]}
                    ]
                },
                artifact_stem=Path(input_path).stem,
            )

    _patch_parser_cls(monkeypatch, ReviewPrepParser)
    source = tmp_path / "page-0001.png"
    _write_test_image(source)

    result = prepare_review_artifacts([str(source)], tmp_path / "review-run")

    assert ReviewPrepParser.parse_calls == []
    assert ReviewPrepParser.layout_only_calls == [str(source)]
    assert result["review_layout"]["pages"][0]["blocks"][0]["content"] == "Recovered text"


def test_prepare_review_artifacts_retries_same_page_once_on_cpu_after_cuda_oom(
    tmp_path, monkeypatch
) -> None:
    empty_cache_calls: list[str] = []

    class ReviewCudaOomGlmOcr(FakeGlmOcr):
        init_devices: list[str] = []
        layout_only_devices: list[tuple[str, str]] = []

        def __init__(self, *, config_path: str, layout_device: str, **kwargs: object) -> None:
            super().__init__(config_path=config_path, layout_device=layout_device, **kwargs)
            self.init_devices.append(layout_device)

        def parse(self, input_path: str):
            raise AssertionError("review preparation should use layout-only parsing")

        def parse_layout_only(self, input_path: str):
            self.layout_only_devices.append((self.layout_device, input_path))
            if self.layout_device == "cuda":
                raise RuntimeError("CUDA out of memory while parsing review artifacts")
            return FakeResult(
                markdown_result="",
                json_result={"doc": Path(input_path).name},
                saved_model_json={
                    "blocks": [
                        {"label": "text", "content": "Recovered text", "bbox_2d": [0, 0, 10, 10]}
                    ]
                },
                artifact_stem=Path(input_path).stem,
            )

    _patch_parser_cls(monkeypatch, ReviewCudaOomGlmOcr)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: empty_cache_calls.append("called"))),
    )
    source = tmp_path / "page-0001.png"
    _write_test_image(source)

    result = prepare_review_artifacts([str(source)], tmp_path / "review-run")

    assert ReviewCudaOomGlmOcr.init_devices == ["cuda", "cpu"]
    assert ReviewCudaOomGlmOcr.layout_only_devices == [("cuda", str(source)), ("cpu", str(source))]
    assert empty_cache_calls == ["called"]
    assert result["layout_device"] == "cuda"
    assert result["review_layout"]["summary"] == {"page_count": 1}
    assert result["review_layout"]["pages"][0]["provider_path"] == "layout/provider/page-0001"


def test_run_ocr_closes_failed_parser_before_cpu_retry(tmp_path, monkeypatch) -> None:
    empty_cache_calls: list[str] = []

    class CudaOomGlmOcr(FakeGlmOcr):
        close_devices: list[str] = []

        def parse(self, input_path: str):
            if self.layout_device == "cuda":
                raise RuntimeError("CUDA out of memory while running layout detection")
            return FakeResult(
                markdown_result="# cpu doc",
                json_result={"doc": Path(input_path).name},
                artifact_stem=Path(input_path).stem,
            )

        def close(self) -> None:
            self.close_devices.append(self.layout_device)

    _patch_parser_cls(monkeypatch, CudaOomGlmOcr)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: empty_cache_calls.append("called"))),
    )
    source = tmp_path / "page-0001.png"
    _write_test_image(source)

    result = run_ocr([str(source)], tmp_path / "run")

    assert CudaOomGlmOcr.close_devices == ["cuda"]
    assert empty_cache_calls == ["called"]
    assert result["markdown"] == "# cpu doc"


def test_run_ocr_reopens_cuda_parser_after_cpu_retry_for_later_pages(
    tmp_path, monkeypatch
) -> None:
    empty_cache_calls: list[str] = []

    class CudaOomGlmOcr(FakeGlmOcr):
        init_devices: list[str] = []
        parse_events: list[tuple[str, str]] = []

        def __init__(self, *, config_path: str, layout_device: str, **kwargs: object) -> None:
            super().__init__(config_path=config_path, layout_device=layout_device, **kwargs)
            self.closed = False
            self.init_devices.append(layout_device)

        def __exit__(self, _exc_type, _exc, _tb):
            self.close()

        def parse(self, input_path: str):
            if self.closed:
                raise AssertionError("closed parser was reused")
            self.parse_events.append((self.layout_device, input_path))
            page_stem = Path(input_path).stem
            if self.layout_device == "cuda" and page_stem == "page-0001":
                raise RuntimeError("CUDA out of memory while running layout detection")
            return FakeResult(
                markdown_result=f"# {self.layout_device} {page_stem}",
                json_result={"doc": Path(input_path).name},
                artifact_stem=page_stem,
            )

        def close(self) -> None:
            self.closed = True

    _patch_parser_cls(monkeypatch, CudaOomGlmOcr)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: empty_cache_calls.append("called"))),
    )
    page_one = tmp_path / "page-0001.png"
    page_two = tmp_path / "page-0002.png"
    _write_test_image(page_one)
    _write_test_image(page_two)

    result = run_ocr([str(page_one), str(page_two)], tmp_path / "run")

    assert CudaOomGlmOcr.init_devices == ["cuda", "cpu", "cuda"]
    assert CudaOomGlmOcr.parse_events == [
        ("cuda", str(page_one)),
        ("cpu", str(page_one)),
        ("cuda", str(page_two)),
    ]
    assert empty_cache_calls == ["called"]
    assert result["markdown"] == "# cpu page-0001\n\n# cuda page-0002"


def test_prepare_review_artifacts_closes_failed_parser_before_cpu_retry(tmp_path, monkeypatch) -> None:
    empty_cache_calls: list[str] = []

    class ReviewCudaOomGlmOcr(FakeGlmOcr):
        close_devices: list[str] = []

        def parse(self, input_path: str):
            raise AssertionError("review preparation should use layout-only parsing")

        def parse_layout_only(self, input_path: str):
            if self.layout_device == "cuda":
                raise RuntimeError("CUDA out of memory while parsing review artifacts")
            return FakeResult(
                markdown_result="",
                json_result={"doc": Path(input_path).name},
                saved_model_json={
                    "blocks": [
                        {"label": "text", "content": "Recovered text", "bbox_2d": [0, 0, 10, 10]}
                    ]
                },
                artifact_stem=Path(input_path).stem,
            )

        def close(self) -> None:
            self.close_devices.append(self.layout_device)

    _patch_parser_cls(monkeypatch, ReviewCudaOomGlmOcr)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: empty_cache_calls.append("called"))),
    )
    source = tmp_path / "page-0001.png"
    _write_test_image(source)

    result = prepare_review_artifacts([str(source)], tmp_path / "review-run")

    assert ReviewCudaOomGlmOcr.close_devices == ["cuda"]
    assert empty_cache_calls == ["called"]
    assert result["review_layout"]["summary"] == {"page_count": 1}


def test_prepare_review_artifacts_reopens_cuda_parser_after_cpu_retry_for_later_pages(
    tmp_path, monkeypatch
) -> None:
    empty_cache_calls: list[str] = []

    class ReviewCudaOomGlmOcr(FakeGlmOcr):
        init_devices: list[str] = []
        layout_only_events: list[tuple[str, str]] = []

        def __init__(self, *, config_path: str, layout_device: str, **kwargs: object) -> None:
            super().__init__(config_path=config_path, layout_device=layout_device, **kwargs)
            self.closed = False
            self.init_devices.append(layout_device)

        def __exit__(self, _exc_type, _exc, _tb):
            self.close()

        def parse(self, input_path: str):
            raise AssertionError("review preparation should use layout-only parsing")

        def parse_layout_only(self, input_path: str):
            if self.closed:
                raise AssertionError("closed parser was reused")
            self.layout_only_events.append((self.layout_device, input_path))
            page_stem = Path(input_path).stem
            if self.layout_device == "cuda" and page_stem == "page-0001":
                raise RuntimeError("CUDA out of memory while parsing review artifacts")
            return FakeResult(
                markdown_result="",
                json_result={"doc": Path(input_path).name},
                saved_model_json={
                    "blocks": [
                        {
                            "label": "text",
                            "content": f"{self.layout_device} {page_stem}",
                            "bbox_2d": [100, 100, 900, 900],
                        }
                    ]
                },
                artifact_stem=page_stem,
            )

        def close(self) -> None:
            self.closed = True

    _patch_parser_cls(monkeypatch, ReviewCudaOomGlmOcr)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: empty_cache_calls.append("called"))),
    )
    page_one = tmp_path / "page-0001.png"
    page_two = tmp_path / "page-0002.png"
    _write_test_image(page_one)
    _write_test_image(page_two)

    result = prepare_review_artifacts([str(page_one), str(page_two)], tmp_path / "review-run")

    assert ReviewCudaOomGlmOcr.init_devices == ["cuda", "cpu", "cuda"]
    assert ReviewCudaOomGlmOcr.layout_only_events == [
        ("cuda", str(page_one)),
        ("cpu", str(page_one)),
        ("cuda", str(page_two)),
    ]
    assert empty_cache_calls == ["called"]
    assert [page["blocks"][0]["content"] for page in result["review_layout"]["pages"]] == [
        "cpu page-0001",
        "cuda page-0002",
    ]
