from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from my_ocr.adapters.outbound.filesystem.run_store import FilesystemRunStore
from my_ocr.application.dto import (
    ArtifactCopy,
    LayoutBlock,
    LayoutDetectionResult,
    OcrPageResult,
    OcrRecognitionResult,
    OcrRunResult,
    PageRef,
    ProviderArtifacts,
    ReviewLayout,
    ReviewPage,
    RunId,
    StructuredExtractionOptions,
)
from my_ocr.application.errors import UnsupportedRunSchema
from my_ocr.application.workflow import DocumentWorkflow


def test_prepare_layout_review_writes_v2_manifest_and_relative_payloads(tmp_path: Path) -> None:
    input_path = _image(tmp_path / "input.png")
    store = FilesystemRunStore(tmp_path / "runs")

    result = _workflow(store).prepare_review(str(input_path), run_id=RunId("demo"))

    run_dir = result.snapshot.run_dir
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    review = json.loads((run_dir / "layout" / "review.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["pages"][0]["image_path"] == "pages/page-0001.png"
    assert manifest["status"]["layout"] == "prepared"
    assert manifest["diagnostics"]["layout"]["layout_profile_warning"] == "profile warning"
    assert review["pages"][0]["image_path"] == "pages/page-0001.png"
    assert "tmp" not in json.dumps(review).lower()
    assert (run_dir / "layout" / "provider" / "page-0001" / "layout.json").exists()


def test_run_reviewed_ocr_writes_ocr_and_clears_extraction(tmp_path: Path) -> None:
    store, run_id = _prepared_run(tmp_path)
    tx = store.begin_update(run_id)
    tx.write_rules_extraction({"stale": True})
    tx.commit()

    result = _workflow(store).run_reviewed_ocr(RunId(run_id))

    run_dir = result.snapshot.run_dir
    assert (run_dir / "ocr" / "markdown.md").read_text(encoding="utf-8") == "# Page 1"
    assert not (run_dir / "extraction").exists()
    assert json.loads((run_dir / "ocr" / "pages.json").read_text(encoding="utf-8"))[
        "pages"
    ][0]["provider_path"] == "ocr/provider/page-0001"


def test_page_reruns_replace_only_target_page(tmp_path: Path) -> None:
    store, run_id = _prepared_run(tmp_path, page_count=2)
    workflow = _workflow(store)
    workflow.run_reviewed_ocr(RunId(run_id))

    _workflow(store, layout_detector=FakeLayoutDetector(label="table")).rerun_page_layout(
        RunId(run_id), page_number=2
    )
    snapshot = store.open_run(run_id)
    assert snapshot.review_layout is not None
    labels = {
        page.page_number: page.blocks[0].label for page in snapshot.review_layout.pages
    }
    assert labels == {1: "text", 2: "table"}
    assert not (snapshot.run_dir / "ocr").exists()

    store, run_id = _prepared_run(tmp_path / "ocr-rerun", page_count=2)
    _workflow(store).run_reviewed_ocr(RunId(run_id))
    _workflow(store, ocr_engine=FakeOcrEngine(prefix="Updated")).rerun_page_ocr(
        RunId(run_id), page_number=2
    )
    run_dir = tmp_path / "ocr-rerun" / "runs" / run_id
    ocr_payload = json.loads((run_dir / "ocr" / "pages.json").read_text(encoding="utf-8"))
    assert [page["markdown"] for page in ocr_payload["pages"]] == ["# Page 1", "Updated Page 2"]
    assert (run_dir / "ocr" / "markdown.md").read_text(encoding="utf-8") == (
        "# Page 1\n\nUpdated Page 2"
    )


def test_extract_rules_and_structured_write_v2_extraction_outputs(tmp_path: Path) -> None:
    store, run_id = _prepared_run(tmp_path)
    workflow = _workflow(store)
    workflow.run_reviewed_ocr(RunId(run_id))

    workflow.extract_rules(RunId(run_id))
    workflow.extract_structured(
        RunId(run_id),
        options=StructuredExtractionOptions(model="demo-model"),
    )

    run_dir = tmp_path / "runs" / run_id
    assert json.loads((run_dir / "extraction" / "rules.json").read_text()) == {
        "document_type": "rules"
    }
    assert json.loads((run_dir / "extraction" / "structured.json").read_text()) == {
        "document_type": "structured"
    }
    assert json.loads((run_dir / "extraction" / "canonical.json").read_text()) == {
        "document_type": "structured"
    }


def test_run_automatic_prepares_ocr_and_extracts_rules(tmp_path: Path) -> None:
    input_path = _image(tmp_path / "input.png")
    store = FilesystemRunStore(tmp_path / "runs")

    result = _workflow(store).run_automatic(str(input_path), run_id=RunId("demo"))

    run_dir = result.snapshot.run_dir
    assert result.snapshot.run_id == RunId("demo")
    assert (run_dir / "layout" / "review.json").exists()
    assert (run_dir / "ocr" / "markdown.md").read_text(encoding="utf-8") == "# Page 1"
    assert json.loads((run_dir / "extraction" / "rules.json").read_text()) == {
        "document_type": "rules"
    }


def test_v1_run_folders_are_explicitly_unsupported(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "old"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text("{}", encoding="utf-8")

    with pytest.raises(UnsupportedRunSchema, match="created before v2"):
        FilesystemRunStore(tmp_path / "runs").open_run("old")


def test_transaction_rollback_removes_work_dir_without_publishing(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path / "runs")
    tx = store.create_run("input.pdf", RunId("rollback"))
    work_dir = tx.work_dir

    tx.rollback()

    assert not work_dir.exists()
    assert not (tmp_path / "runs" / "rollback").exists()


class FakeNormalizer:
    def normalize(self, _input_path: str | Path, pages_dir: str | Path) -> list[PageRef]:
        pages_dir = Path(pages_dir)
        pages_dir.mkdir(parents=True, exist_ok=True)
        source = _image(pages_dir / "source.png")
        stem = Path(str(_input_path)).stem
        page_count = int(stem.rsplit("-", 1)[-1]) if "-" in stem else 1
        pages: list[PageRef] = []
        for page_number in range(1, page_count + 1):
            target = pages_dir / f"page-{page_number:04d}.png"
            target.write_bytes(source.read_bytes())
            pages.append(
                PageRef(
                    page_number=page_number,
                    image_path=f"pages/{target.name}",
                    width=10,
                    height=10,
                    resolved_path=target,
                )
            )
        return pages


class FakeLayoutDetector:
    def __init__(self, *, label: str = "text") -> None:
        self._label = label

    def detect_layout(self, pages: list[PageRef], _options: object) -> LayoutDetectionResult:
        copies: list[ArtifactCopy] = []
        review_pages: list[ReviewPage] = []
        for page in pages:
            provider_dir = page.path_for_io.parent / f"layout-provider-{page.page_number}"
            provider_dir.mkdir(exist_ok=True)
            (provider_dir / "layout.json").write_text("{}", encoding="utf-8")
            copies.append(
                ArtifactCopy(
                    source=provider_dir,
                    relative_target=f"layout/provider/page-{page.page_number:04d}",
                )
            )
            review_pages.append(
                ReviewPage(
                    page_number=page.page_number,
                    image_path=page.image_path,
                    image_width=page.width,
                    image_height=page.height,
                    provider_path=f"layout/provider/page-{page.page_number:04d}",
                    blocks=[
                        LayoutBlock(
                            id=f"p{page.page_number}-b0",
                            index=0,
                            label=self._label,
                            bbox=(1, 2, 8, 9),
                        )
                    ],
                )
            )
        return LayoutDetectionResult(
            layout=ReviewLayout(review_pages, status="prepared"),
            artifacts=ProviderArtifacts(tuple(copies)),
            diagnostics=type("Diagnostics", (), {
                "warning": "profile warning",
                "to_dict": lambda self: {"layout_profile_warning": "profile warning"},
            })(),
        )


class FakeOcrEngine:
    def __init__(self, *, prefix: str = "#") -> None:
        self._prefix = prefix

    def recognize(
        self, pages: list[PageRef], _review: ReviewLayout | None, _options: object
    ) -> OcrRecognitionResult:
        results = [
            OcrPageResult(
                page_number=page.page_number,
                image_path=page.image_path,
                markdown=f"{self._prefix} Page {page.page_number}",
                markdown_source="sdk_markdown",
                provider_path=f"ocr/provider/page-{page.page_number:04d}",
            )
            for page in pages
        ]
        markdown = "\n\n".join(page.markdown for page in results)
        return OcrRecognitionResult(OcrRunResult(results, markdown), ProviderArtifacts.empty())


class FakeRulesExtractor:
    def extract(self, _markdown: str) -> dict[str, Any]:
        return {"document_type": "rules"}


class FakeStructuredExtractor:
    def extract(
        self,
        _pages: list[PageRef],
        *,
        markdown_text: str | None,
        options: StructuredExtractionOptions,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert markdown_text
        return {"document_type": "structured"}, {"model": options.model}


def _prepared_run(tmp_path: Path, *, page_count: int = 1) -> tuple[FilesystemRunStore, str]:
    input_path = _image(tmp_path / f"input-{page_count}.png")
    store = FilesystemRunStore(tmp_path / "runs")
    _workflow(store).prepare_review(str(input_path), run_id=RunId("demo"))
    return store, "demo"


def _workflow(
    store: FilesystemRunStore,
    *,
    layout_detector: FakeLayoutDetector | None = None,
    ocr_engine: FakeOcrEngine | None = None,
) -> DocumentWorkflow:
    return DocumentWorkflow(
        store,
        FakeNormalizer(),
        layout_detector or FakeLayoutDetector(),
        ocr_engine or FakeOcrEngine(),
        FakeRulesExtractor(),
        FakeStructuredExtractor(),
    )


def _image(path: Path) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10), "white").save(path)
    return path
