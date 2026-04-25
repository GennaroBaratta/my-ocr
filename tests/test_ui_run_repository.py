from __future__ import annotations

from pathlib import Path

from my_ocr.adapters.outbound.filesystem.run_read_model import FilesystemRunReadModel
from my_ocr.adapters.outbound.filesystem.run_store import FilesystemRunStore
from my_ocr.application.dto import (
    LayoutBlock,
    PageRef,
    ProviderArtifacts,
    ReviewLayout,
    ReviewPage,
    RunId,
)
from my_ocr.ui.mappers import pages_from_snapshot, recent_run_summary


def test_read_model_lists_v2_and_unsupported_runs(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path)
    tx = store.create_run("input.pdf", RunId("v2"))
    tx.write_pages([PageRef(1, "pages/page-0001.png", 10, 10)])
    tx.commit()
    (tmp_path / "v1").mkdir()
    (tmp_path / "v1" / "meta.json").write_text("{}", encoding="utf-8")

    records = FilesystemRunReadModel(tmp_path).list_recent_runs()
    statuses = {record.run_id: record.status for record in records}

    assert statuses["v2"] == "pending"
    assert statuses["v1"] == "unsupported"


def test_ui_mappers_convert_snapshot_pages_and_boxes(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path)
    tx = store.create_run("input.pdf", RunId("demo"))
    tx.write_pages([PageRef(1, "pages/page-0001.png", 100, 200)])
    tx.write_review_layout(
        ReviewLayout(
            [
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
                            bbox=(10, 20, 30, 40),
                            content="hello",
                        )
                    ],
                )
            ],
            status="reviewed",
        ),
        ProviderArtifacts.empty(),
    )
    snapshot = tx.commit()

    pages = pages_from_snapshot(snapshot)
    record = recent_run_summary(FilesystemRunReadModel(tmp_path).list_recent_runs()[0])

    assert record.run_id == "demo"
    assert pages[0].page_number == 1
    assert pages[0].boxes[0].width == 20
    assert pages[0].boxes[0].content == "hello"

