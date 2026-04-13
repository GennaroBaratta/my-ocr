from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from free_doc_extract.ocr import run_ocr


class FakeResult:
    def __init__(self, page_name: str, markdown_result: str, json_result: dict[str, str]) -> None:
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
