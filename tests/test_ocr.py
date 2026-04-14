from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from free_doc_extract import ocr
from free_doc_extract.ocr import run_ocr


class FakeResult:
    def __init__(self, page_name: str, markdown_result: str, json_result: Any) -> None:
        self.page_name = page_name
        self.markdown_result = markdown_result
        self.json_result = json_result

    def save(self, output_dir: str) -> None:
        page_dir = Path(output_dir) / self.page_name
        page_dir.mkdir(parents=True, exist_ok=True)
        (page_dir / f"{self.page_name}.md").write_text(self.markdown_result, encoding="utf-8")


class FakeGlmOcr:
    def __init__(self, *, config_path: str, layout_device: str) -> None:
        self.config_path = config_path
        self.layout_device = layout_device

    def __enter__(self) -> FakeGlmOcr:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def parse(self, images: str | list[str]) -> FakeResult | list[FakeResult]:
        if isinstance(images, str):
            page_name = Path(images).stem
            return FakeResult(
                page_name=page_name,
                markdown_result=f"# {page_name}",
                json_result={"page": page_name},
            )

        return [
            FakeResult(
                page_name=Path(image).stem,
                markdown_result=f"# {Path(image).stem}",
                json_result={"page": Path(image).stem},
            )
            for image in images
        ]


def test_run_ocr_aggregates_multi_page_results(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "glmocr", SimpleNamespace(GlmOcr=FakeGlmOcr))

    page_paths = []
    for index in range(1, 3):
        page = tmp_path / f"page-{index:04d}.png"
        page.write_bytes(b"fake image")
        page_paths.append(str(page))

    result = run_ocr(page_paths, tmp_path / "run")

    assert result["markdown"] == "# page-0001\n\n---\n\n# page-0002"
    assert result["json"] == [
        {"page": "page-0001"},
        {"page": "page-0002"},
    ]
    assert (tmp_path / "run" / "ocr.md").read_text(encoding="utf-8") == result["markdown"]
    assert json.loads((tmp_path / "run" / "ocr.json").read_text(encoding="utf-8")) == result["json"]
    assert (tmp_path / "run" / "ocr_raw" / "page-0001" / "page-0001" / "page-0001.md").exists()
    assert (tmp_path / "run" / "ocr_raw" / "page-0002" / "page-0002" / "page-0002.md").exists()


def test_needs_crop_fallback_for_empty_block_content() -> None:
    page_json = [
        [
            {"index": 0, "label": "doc_title", "bbox_2d": [10, 10, 100, 40], "content": "# "},
            {"index": 1, "label": "text", "bbox_2d": [10, 60, 140, 90], "content": ""},
        ]
    ]

    assert ocr._needs_crop_fallback("# \n\n## ", page_json) is True


def test_run_ocr_uses_crop_fallback_when_page_text_is_empty(tmp_path, monkeypatch) -> None:
    class EmptyResult(FakeResult):
        def __init__(self, page_name: str) -> None:
            super().__init__(
                page_name=page_name,
                markdown_result="# \n\n## ",
                json_result=[
                    [
                        {
                            "index": 0,
                            "label": "doc_title",
                            "bbox_2d": [10, 10, 100, 40],
                            "content": "",
                        }
                    ]
                ],
            )

    class EmptyGlmOcr(FakeGlmOcr):
        def parse(self, images: str | list[str]) -> EmptyResult:
            assert isinstance(images, str)
            return EmptyResult(Path(images).stem)

    page_path = tmp_path / "page-0001.png"
    page_path.write_bytes(b"fake image")

    monkeypatch.setitem(sys.modules, "glmocr", SimpleNamespace(GlmOcr=EmptyGlmOcr))
    monkeypatch.setattr(
        ocr,
        "_run_crop_fallback_for_page",
        lambda **kwargs: (
            "Recovered text from crop",
            [{"chunk": 1, "text": "Recovered text from crop"}],
        ),
    )

    result = run_ocr([str(page_path)], tmp_path / "run")

    assert result["markdown"] == "Recovered text from crop"
    assert result["fallback_used"] is True
    assert (tmp_path / "run" / "ocr_fallback.json").exists()


def test_run_ocr_records_failed_fallback_attempt(tmp_path, monkeypatch) -> None:
    class EmptyResult(FakeResult):
        def __init__(self, page_name: str) -> None:
            super().__init__(
                page_name=page_name,
                markdown_result="# \n\n## ",
                json_result=[
                    [
                        {
                            "index": 0,
                            "label": "doc_title",
                            "bbox_2d": [10, 10, 100, 40],
                            "content": "",
                        }
                    ]
                ],
            )

    class EmptyGlmOcr(FakeGlmOcr):
        def parse(self, images: str | list[str]) -> EmptyResult:
            assert isinstance(images, str)
            return EmptyResult(Path(images).stem)

    page_path = tmp_path / "page-0001.png"
    page_path.write_bytes(b"fake image")

    monkeypatch.setitem(sys.modules, "glmocr", SimpleNamespace(GlmOcr=EmptyGlmOcr))
    monkeypatch.setattr(
        ocr,
        "_run_crop_fallback_for_page",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = run_ocr([str(page_path)], tmp_path / "run")
    fallback_payload = json.loads(
        (tmp_path / "run" / "ocr_fallback.json").read_text(encoding="utf-8")
    )

    assert result["fallback_used"] is True
    assert result["markdown"] == "# \n\n## "
    assert fallback_payload[0]["recovered_text"] is False
    assert fallback_payload[0]["chunks"][0]["error"] == "boom"


def test_build_text_chunks_preserves_each_layout_box() -> None:
    page_json = [
        [
            {"index": 0, "label": "doc_title", "bbox_2d": [100, 100, 300, 140], "content": ""},
            {"index": 1, "label": "text", "bbox_2d": [110, 150, 320, 180], "content": ""},
            {
                "index": 2,
                "label": "paragraph_title",
                "bbox_2d": [700, 900, 850, 940],
                "content": "",
            },
        ]
    ]

    chunks = ocr._build_text_chunks(page_json, width=1000, height=1200)

    assert len(chunks) == 3
    assert chunks[0]["source_indices"] == [0]
    assert chunks[1]["source_indices"] == [1]
    assert chunks[2]["source_indices"] == [2]


def test_clean_recognized_text_keeps_non_latin_text() -> None:
    assert ocr._clean_recognized_text("Привет мир") == "Привет мир"
