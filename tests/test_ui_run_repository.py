from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from free_doc_extract.ui.run_repository import RunRepository
from free_doc_extract.ui.session import BoundingBox, PageData
from tests.support import (
    build_ocr_page,
    build_reviewed_layout_block,
    build_reviewed_layout_page,
)


def _write_image(path: Path, *, size: tuple[int, int] = (100, 120)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_module = import_module("PIL.Image")
    image_module.new("RGB", size, color="white").save(path)


def test_list_recent_runs_derives_status_from_predictions(tmp_path) -> None:
    run_root = tmp_path / "runs"
    pending_run = run_root / "pending-run"
    extracted_run = run_root / "extracted-run"
    pending_run.mkdir(parents=True)
    extracted_run.mkdir(parents=True)
    (pending_run / "meta.json").write_text(
        json.dumps({"input_path": "data/raw/pending.pdf"}),
        encoding="utf-8",
    )
    (extracted_run / "meta.json").write_text(
        json.dumps({"input_path": "data/raw/extracted.pdf"}),
        encoding="utf-8",
    )
    predictions_dir = extracted_run / "predictions"
    predictions_dir.mkdir()
    (predictions_dir / "demo.json").write_text("{}", encoding="utf-8")

    repository = RunRepository(str(run_root))

    recent_runs = {run.run_id: run for run in repository.list_recent_runs()}

    assert recent_runs["pending-run"].status == "pending"
    assert recent_runs["pending-run"].input_path == "data/raw/pending.pdf"
    assert recent_runs["extracted-run"].status == "extracted"


def test_load_run_prefers_reviewed_layout_over_ocr_payload(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    page_path = run_dir / "pages" / "page-0001.png"
    _write_image(page_path)

    source_sdk_json_path = run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json"
    source_sdk_json_path.parent.mkdir(parents=True)
    source_sdk_json_path.write_text(
        json.dumps({"blocks": [{"index": 0, "label": "ocr", "bbox_2d": [1, 2, 8, 9]}]}),
        encoding="utf-8",
    )
    (run_dir / "ocr.json").write_text(
        json.dumps(
            {
                "pages": [
                    build_ocr_page(
                        page_path=str(page_path),
                        sdk_json_path=str(source_sdk_json_path),
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "reviewed_layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "status": "reviewed",
                "pages": [
                    build_reviewed_layout_page(
                        page_path=str(page_path),
                        source_sdk_json_path=str(source_sdk_json_path),
                        blocks=[
                            build_reviewed_layout_block(
                                label="reviewed",
                                content="Reviewed",
                                bbox=[10, 20, 40, 60],
                            )
                        ],
                    )
                ],
                "summary": {"page_count": 1},
            }
        ),
        encoding="utf-8",
    )

    repository = RunRepository(str(run_root))

    loaded = repository.load_run("demo-ui")

    assert loaded.pages[0].boxes[0].label == "reviewed"
    assert loaded.pages[0].boxes[0].x == 10


def test_load_run_preserves_sparse_page_number_via_payload_lookup(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    page_path = run_dir / "pages" / "scan.png"
    _write_image(page_path)

    source_sdk_json_path = run_dir / "ocr_raw" / "page-0005" / "page-0005_model.json"
    source_sdk_json_path.parent.mkdir(parents=True)
    source_sdk_json_path.write_text(json.dumps({"blocks": []}), encoding="utf-8")
    (run_dir / "reviewed_layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "status": "reviewed",
                "pages": [
                    build_reviewed_layout_page(
                        page_number=5,
                        page_path="different-name.png",
                        source_sdk_json_path=str(source_sdk_json_path),
                        blocks=[build_reviewed_layout_block(content="Sparse page")],
                    )
                ],
                "summary": {"page_count": 1},
            }
        ),
        encoding="utf-8",
    )

    repository = RunRepository(str(run_root))

    loaded = repository.load_run("demo-ui")

    assert loaded.pages[0].page_number == 5


def test_load_run_falls_back_to_markdown_from_ocr_payload(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    page_path = run_dir / "pages" / "page-0001.png"
    _write_image(page_path)

    source_sdk_json_path = run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json"
    source_sdk_json_path.parent.mkdir(parents=True)
    source_sdk_json_path.write_text(json.dumps({"blocks": []}), encoding="utf-8")
    (run_dir / "ocr.md").write_text("", encoding="utf-8")
    (run_dir / "ocr.json").write_text(
        json.dumps(
            {
                "pages": [
                    build_ocr_page(
                        page_path=str(page_path),
                        markdown="# Page 1",
                        sdk_json_path=str(source_sdk_json_path),
                    )
                ]
            }
        ),
        encoding="utf-8",
    )

    repository = RunRepository(str(run_root))

    loaded = repository.load_run("demo-ui")

    assert loaded.ocr_markdown == "# Page 1"


def test_save_reviewed_layout_preserves_source_sdk_json_path_and_rounds_bbox(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-ui"
    page_path = run_dir / "pages" / "scan.png"
    _write_image(page_path)

    source_sdk_json_path = run_dir / "ocr_raw" / "page-0005" / "page-0005_model.json"
    source_sdk_json_path.parent.mkdir(parents=True)
    source_sdk_json_path.write_text(json.dumps({"blocks": []}), encoding="utf-8")
    (run_dir / "reviewed_layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "status": "reviewed",
                "pages": [
                    build_reviewed_layout_page(
                        page_number=5,
                        page_path="different-name.png",
                        source_sdk_json_path=str(source_sdk_json_path),
                        blocks=[build_reviewed_layout_block()],
                    )
                ],
                "summary": {"page_count": 1},
            }
        ),
        encoding="utf-8",
    )

    repository = RunRepository(str(run_root))
    run_paths = repository.load_run("demo-ui").run_paths
    pages = [
        PageData(
            index=0,
            page_number=5,
            image_path=str(page_path),
            boxes=[
                BoundingBox(
                    id="p0-b0",
                    page_index=0,
                    x=12.4,
                    y=20.2,
                    width=34.8,
                    height=39.9,
                    label="table",
                    content="Reviewed",
                )
            ],
        )
    ]

    repository.save_reviewed_layout(run_paths, pages)

    persisted = json.loads((run_dir / "reviewed_layout.json").read_text(encoding="utf-8"))

    assert persisted["status"] == "reviewed"
    assert persisted["pages"][0]["page_number"] == 5
    assert persisted["pages"][0]["source_sdk_json_path"] == str(source_sdk_json_path)
    assert persisted["pages"][0]["blocks"][0]["bbox"] == [12, 20, 47, 60]
