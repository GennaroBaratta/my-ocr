from __future__ import annotations

from pathlib import Path

from my_ocr.storage import FilesystemRunReadModel
from my_ocr.storage import FilesystemRunStore
from my_ocr.models import ProviderArtifacts
from my_ocr.models import (
    LayoutBlock,
    PageRef,
    ReviewLayout,
    ReviewPage,
    RunId,
)
from my_ocr.ui.mappers import pages_from_snapshot, recent_run_summary


def test_read_model_lists_v3_and_unsupported_runs(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path)
    workspace = store.start_run("input.pdf", RunId("v3"))
    store.publish_prepared_run(
        workspace,
        [PageRef(page_number=1, image_path="pages/page-0001.png", width=10, height=10)],
        ReviewLayout(pages=[], status="prepared"),
        ProviderArtifacts.empty(),
    )
    (tmp_path / "v1").mkdir()
    (tmp_path / "v1" / "meta.json").write_text("{}", encoding="utf-8")

    records = FilesystemRunReadModel(tmp_path).list_recent_runs()
    statuses = {record.run_id: record.status for record in records}

    assert statuses["v3"] == "review_ready"
    assert statuses["v1"] == "unsupported"


def test_ui_mappers_convert_snapshot_pages_and_boxes(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path)
    workspace = store.start_run("input.pdf", RunId("demo"))
    snapshot = store.publish_prepared_run(
        workspace,
        [PageRef(page_number=1, image_path="pages/page-0001.png", width=100, height=200)],
        ReviewLayout(
            pages=[
                ReviewPage(
                    page_number=1,
                    image_path="pages/page-0001.png",
                    image_width=100,
                    image_height=200,
                    blocks=[
                        LayoutBlock(
                            id="b1",
                            index=0,
                            label="text",
                            bbox=[10, 20, 30, 40],
                            content="hello",
                        )
                    ],
                )
            ],
            status="reviewed",
        ),
        ProviderArtifacts.empty(),
    )

    pages = pages_from_snapshot(snapshot)
    record = recent_run_summary(FilesystemRunReadModel(tmp_path).list_recent_runs()[0])

    assert record.run_id == "demo"
    assert pages[0].page_number == 1
    assert pages[0].boxes[0].width == 20
    assert pages[0].boxes[0].content == "hello"

