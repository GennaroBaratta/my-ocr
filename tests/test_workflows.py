from __future__ import annotations

from importlib import import_module
import json
from pathlib import Path

import pytest

from free_doc_extract import workflows
from free_doc_extract.workflows import (
    prepare_review_page_workflow,
    prepare_review_workflow,
    run_ocr_workflow,
    run_pipeline_workflow,
    run_reviewed_ocr_page_workflow,
    run_reviewed_ocr_workflow,
    run_structured_workflow,
)
from free_doc_extract.settings import DEFAULT_LAYOUT_DEVICE
from tests.support import (
    build_basic_ocr_result,
    build_fallback_chunk,
    build_fallback_page,
    build_ocr_page,
    build_reviewed_layout_block,
    build_reviewed_layout_page,
    normalize_to_single_page,
    seed_existing_run,
    write_basic_ocr_outputs,
    write_reviewed_layout,
)


@pytest.mark.parametrize("failure_phase", ["normalize", "ocr"])
def test_run_ocr_workflow_preserves_existing_run_on_failure(tmp_path, failure_phase: str) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo001"
    seed_existing_run(run_dir)
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    def failing_normalize(_input_path: str, run_dir_arg: str | Path) -> list[str]:
        if failure_phase == "normalize":
            raise ValueError("unsupported input")
        return normalize_to_single_page(_input_path, run_dir_arg)

    def failing_ocr(_page_paths, _run_dir_arg, *, config_path, layout_device, **_kw):
        _ = (config_path, layout_device)
        raise RuntimeError("ocr failed")

    with pytest.raises((ValueError, RuntimeError)):
        run_ocr_workflow(
            str(source),
            run="demo001",
            run_root=str(run_root),
            normalize_document_fn=failing_normalize,
            run_ocr_fn=failing_ocr,
        )

    assert (run_dir / "pages" / "page-0001.png").read_bytes() == b"existing page"
    assert (run_dir / "ocr.md").read_text(encoding="utf-8") == "existing markdown"
    assert json.loads((run_dir / "predictions" / "glmocr_structured.json").read_text()) == {
        "structured": "keep-me"
    }


def test_run_ocr_workflow_restores_existing_run_when_metadata_write_fails(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo013"
    seed_existing_run(run_dir)
    (run_dir / "meta.json").write_text('{"status": "existing"}', encoding="utf-8")
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    def fake_run_ocr(_page_paths, run_dir_arg, *, config_path, layout_device, **_kw):
        run_dir_path = Path(run_dir_arg)
        raw_dir = run_dir_path / "ocr_raw" / "page-0001"
        raw_dir.mkdir(parents=True, exist_ok=True)
        sdk_json_path = raw_dir / "page-0001_model.json"
        sdk_json_path.write_text("{}", encoding="utf-8")
        (run_dir_path / "ocr.md").write_text("replacement markdown", encoding="utf-8")
        (run_dir_path / "ocr.json").write_text(
            json.dumps(
                {
                    "pages": [
                        build_ocr_page(
                            page_path=str(run_dir_path / "pages" / "page-0001.png"),
                            sdk_json_path=str(sdk_json_path),
                        )
                    ],
                    "summary": {"page_count": 1, "sources": {"sdk_markdown": 1}},
                }
            ),
            encoding="utf-8",
        )
        return {
            "markdown": "replacement markdown",
            "json": {"replacement": True},
            "raw_dir": str(raw_dir.parent),
            "config_path": config_path,
            "layout_device": layout_device,
        }

    def failing_write_json(path: str | Path, payload) -> None:
        if Path(path).name == "meta.json":
            raise RuntimeError("metadata write failed")
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="metadata write failed"):
        run_ocr_workflow(
            str(source),
            run="demo013",
            run_root=str(run_root),
            normalize_document_fn=normalize_to_single_page,
            run_ocr_fn=fake_run_ocr,
            write_json_fn=failing_write_json,
        )

    assert (run_dir / "pages" / "page-0001.png").read_bytes() == b"existing page"
    assert (run_dir / "ocr.md").read_text(encoding="utf-8") == "existing markdown"
    assert json.loads((run_dir / "ocr.json").read_text(encoding="utf-8")) == {"existing": True}
    assert json.loads(
        (run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json").read_text(encoding="utf-8")
    ) == {"legacy": True}
    assert json.loads((run_dir / "meta.json").read_text(encoding="utf-8")) == {"status": "existing"}


def test_run_ocr_workflow_removes_new_run_dir_when_failure_happens_before_publish(tmp_path) -> None:
    run_root = tmp_path / "runs"
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    def failing_normalize(_input_path: str, _run_dir_arg: str | Path) -> list[str]:
        raise ValueError("unsupported input")

    with pytest.raises(ValueError, match="unsupported input"):
        run_ocr_workflow(
            str(source),
            run="demo014",
            run_root=str(run_root),
            normalize_document_fn=failing_normalize,
        )

    assert not (run_root / "demo014").exists()


def test_run_ocr_workflow_removes_new_run_dir_when_staging_setup_fails(
    tmp_path, monkeypatch
) -> None:
    run_root = tmp_path / "runs"
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    def failing_create_staging(_paths):
        raise RuntimeError("staging setup failed")

    monkeypatch.setattr(workflows, "_create_staged_ocr_paths", failing_create_staging)

    with pytest.raises(RuntimeError, match="staging setup failed"):
        run_ocr_workflow(
            str(source),
            run="demo015",
            run_root=str(run_root),
        )

    assert not (run_root / "demo015").exists()


def test_run_ocr_workflow_rerun_keeps_existing_predictions(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo001"
    seed_existing_run(run_dir)
    (run_dir / "ocr_fallback").mkdir(exist_ok=True)
    (run_dir / "ocr_fallback" / "stale.txt").write_text("stale", encoding="utf-8")
    (run_dir / "ocr_fallback.json").write_text('{"stale": true}', encoding="utf-8")
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    captured_page_paths = {}

    def fake_run_ocr(_page_paths, run_dir_arg, *, config_path, layout_device, **_kw):
        captured_page_paths["value"] = _page_paths
        return build_basic_ocr_result(
            run_dir_arg,
            config_path=config_path,
            layout_device=layout_device,
        )

    run_ocr_workflow(
        str(source),
        run="demo001",
        run_root=str(run_root),
        normalize_document_fn=normalize_to_single_page,
        run_ocr_fn=fake_run_ocr,
    )

    assert (run_dir / "pages" / "page-0001.png").read_bytes() == b"replacement image"
    assert len(captured_page_paths["value"]) == 1
    assert Path(captured_page_paths["value"][0]).name == "page-0001.png"
    assert Path(captured_page_paths["value"][0]).parent.name == "pages"
    assert (run_dir / "ocr.md").read_text(encoding="utf-8") == "replacement markdown"
    assert not (run_dir / "ocr_fallback").exists()
    assert not (run_dir / "ocr_fallback.json").exists()
    assert json.loads((run_dir / "predictions" / "glmocr_structured.json").read_text()) == {
        "structured": "keep-me"
    }
    assert json.loads((run_dir / "meta.json").read_text(encoding="utf-8")) == {
        "input_path": str(source),
        "page_paths": [str(run_dir / "pages" / "page-0001.png")],
        "ocr_raw_dir": str(run_dir / "ocr_raw"),
        "config_path": "config/local.yaml",
        "layout_device": DEFAULT_LAYOUT_DEVICE,
    }
    assert json.loads((run_dir / "predictions" / "demo001.json").read_text()) == {
        "canonical": "keep-me"
    }


def test_run_ocr_workflow_can_record_repo_relative_input_path(tmp_path) -> None:
    run_root = tmp_path / "runs"
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    def fake_run_ocr(_page_paths, run_dir_arg, *, config_path, layout_device, **_kw):
        return build_basic_ocr_result(
            run_dir_arg,
            config_path=config_path,
            layout_device=layout_device,
        )

    run_ocr_workflow(
        str(source),
        run="demo-relative",
        run_root=str(run_root),
        normalize_document_fn=normalize_to_single_page,
        run_ocr_fn=fake_run_ocr,
        recorded_input_path="data/raw/sample.pdf",
    )

    assert json.loads((run_root / "demo-relative" / "meta.json").read_text(encoding="utf-8")) == {
        "input_path": "data/raw/sample.pdf",
        "page_paths": [str(run_root / "demo-relative" / "pages" / "page-0001.png")],
        "ocr_raw_dir": str(run_root / "demo-relative" / "ocr_raw"),
        "config_path": "config/local.yaml",
        "layout_device": DEFAULT_LAYOUT_DEVICE,
    }


def test_run_ocr_workflow_accepts_directory_input_and_forwards_normalized_pages(tmp_path) -> None:
    run_root = tmp_path / "runs"
    source = tmp_path / "images"
    source.mkdir()
    captured = {}

    def fake_run_ocr(_page_paths, run_dir_arg, *, config_path, layout_device, **_kw):
        captured["page_paths"] = _page_paths
        return build_basic_ocr_result(
            run_dir_arg,
            config_path=config_path,
            layout_device=layout_device,
        )

    run_ocr_workflow(
        str(source),
        run="demo-dir",
        run_root=str(run_root),
        normalize_document_fn=normalize_to_single_page,
        run_ocr_fn=fake_run_ocr,
    )

    assert len(captured["page_paths"]) == 1
    assert Path(captured["page_paths"][0]).name == "page-0001.png"
    assert Path(captured["page_paths"][0]).parent.name == "pages"
    assert (run_root / "demo-dir" / "pages" / "page-0001.png").exists()


def test_run_ocr_workflow_publishes_fallback_artifacts(tmp_path) -> None:
    run_root = tmp_path / "runs"
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    def fake_run_ocr(_page_paths, run_dir_arg, *, config_path, layout_device, **_kw):
        run_dir_path = Path(run_dir_arg)
        result = build_basic_ocr_result(
            run_dir_arg,
            config_path=config_path,
            layout_device=layout_device,
        )
        fallback_dir = run_dir_path / "ocr_fallback" / "page-0001"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        (fallback_dir / "chunk-0001.txt").write_text("fallback text", encoding="utf-8")
        (run_dir_path / "ocr_fallback.json").write_text(
            '{"pages": [{"page_number": 1}], "summary": {"page_count": 1}}',
            encoding="utf-8",
        )
        return result

    run_dir = run_ocr_workflow(
        str(source),
        run="demo-fallback",
        run_root=str(run_root),
        normalize_document_fn=normalize_to_single_page,
        run_ocr_fn=fake_run_ocr,
    )

    assert (run_dir / "ocr_fallback" / "page-0001" / "chunk-0001.txt").read_text(
        encoding="utf-8"
    ) == "fallback text"
    assert json.loads((run_dir / "ocr_fallback.json").read_text(encoding="utf-8")) == {
        "pages": [{"page_number": 1}],
        "summary": {"page_count": 1},
    }


def test_run_ocr_workflow_rewrites_staged_paths_in_published_json(tmp_path) -> None:
    run_root = tmp_path / "runs"
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    def fake_run_ocr(_page_paths, run_dir_arg, *, config_path, layout_device, **_kw):
        run_dir_path = Path(run_dir_arg)
        staged_page = run_dir_path / "pages" / "page-0001.png"
        staged_page.parent.mkdir(parents=True, exist_ok=True)
        staged_page.write_bytes(b"replacement image")
        fallback_dir = run_dir_path / "ocr_fallback" / "page-0001"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        crop_path = fallback_dir / "chunk-0001.png"
        text_path = fallback_dir / "chunk-0001.txt"
        crop_path.write_bytes(b"PNG")
        text_path.write_text("fallback text", encoding="utf-8")
        raw_dir = run_dir_path / "ocr_raw" / "page-0001"
        raw_dir.mkdir(parents=True, exist_ok=True)
        sdk_json_path = raw_dir / "page-0001_model.json"
        sdk_json_path.write_text("{}", encoding="utf-8")
        (run_dir_path / "ocr.md").write_text("replacement markdown", encoding="utf-8")
        write_payload = {
            "pages": [
                build_ocr_page(
                    page_path=str(staged_page),
                    markdown_source="crop_fallback",
                    sdk_json_path=str(sdk_json_path),
                )
            ],
            "summary": {"page_count": 1, "sources": {"crop_fallback": 1}},
        }
        (run_dir_path / "ocr.json").write_text(json.dumps(write_payload), encoding="utf-8")
        (run_dir_path / "ocr_fallback.json").write_text(
            json.dumps(
                {
                    "pages": [
                        build_fallback_page(
                            page_path=str(staged_page),
                            chunks=[
                                build_fallback_chunk(
                                    crop_path=str(crop_path),
                                    text_path=str(text_path),
                                )
                            ],
                        )
                    ],
                    "summary": {"page_count": 1},
                }
            ),
            encoding="utf-8",
        )
        return {
            **write_basic_ocr_outputs(run_dir_path, json_payload=write_payload),
            "raw_dir": str(raw_dir.parent),
            "config_path": config_path,
            "layout_device": layout_device,
        }

    run_dir = run_ocr_workflow(
        str(source),
        run="demo-paths",
        run_root=str(run_root),
        normalize_document_fn=normalize_to_single_page,
        run_ocr_fn=fake_run_ocr,
    )

    published_ocr = json.loads((run_dir / "ocr.json").read_text(encoding="utf-8"))
    published_fallback = json.loads((run_dir / "ocr_fallback.json").read_text(encoding="utf-8"))

    assert published_ocr["pages"][0]["page_path"] == str(run_dir / "pages" / "page-0001.png")
    assert published_ocr["pages"][0]["sdk_json_path"] == str(
        run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json"
    )
    assert published_fallback["pages"][0]["page_path"] == str(run_dir / "pages" / "page-0001.png")
    assert published_fallback["pages"][0]["chunks"][0]["crop_path"] == str(
        run_dir / "ocr_fallback" / "page-0001" / "chunk-0001.png"
    )
    assert published_fallback["pages"][0]["chunks"][0]["text_path"] == str(
        run_dir / "ocr_fallback" / "page-0001" / "chunk-0001.txt"
    )


def test_prepare_review_workflow_persists_review_artifact_without_ocr_outputs(tmp_path) -> None:
    run_root = tmp_path / "runs"
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    def fake_prepare_review(_page_paths, run_dir_arg, *, config_path, layout_device, **_kw):
        run_dir_path = Path(run_dir_arg)
        raw_dir = run_dir_path / "ocr_raw" / "page-0001"
        raw_dir.mkdir(parents=True, exist_ok=True)
        sdk_json_path = raw_dir / "page-0001_model.json"
        sdk_json_path.write_text("{}", encoding="utf-8")
        reviewed_layout = {
            "version": 1,
            "status": "prepared",
            "pages": [
                build_reviewed_layout_page(
                    page_path=str(run_dir_path / "pages" / "page-0001.png"),
                    source_sdk_json_path=str(sdk_json_path),
                    blocks=[build_reviewed_layout_block(content="Prepared block")],
                )
            ],
            "summary": {"page_count": 1},
        }
        (run_dir_path / "reviewed_layout.json").write_text(
            json.dumps(reviewed_layout),
            encoding="utf-8",
        )
        return {
            "reviewed_layout": reviewed_layout,
            "raw_dir": str(raw_dir.parent),
            "config_path": config_path,
            "layout_device": layout_device,
            "reviewed_layout_path": str(run_dir_path / "reviewed_layout.json"),
        }

    run_dir = prepare_review_workflow(
        str(source),
        run="demo-review",
        run_root=str(run_root),
        normalize_document_fn=normalize_to_single_page,
        prepare_review_artifacts_fn=fake_prepare_review,
    )

    reviewed_layout = json.loads((run_dir / "reviewed_layout.json").read_text(encoding="utf-8"))
    assert reviewed_layout["status"] == "prepared"
    assert reviewed_layout["pages"][0]["page_path"] == str(run_dir / "pages" / "page-0001.png")
    assert reviewed_layout["pages"][0]["source_sdk_json_path"] == str(
        run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json"
    )
    assert not (run_dir / "ocr.md").exists()
    assert not (run_dir / "ocr.json").exists()
    assert json.loads((run_dir / "meta.json").read_text(encoding="utf-8")) == {
        "input_path": str(source),
        "page_paths": [str(run_dir / "pages" / "page-0001.png")],
        "ocr_raw_dir": str(run_dir / "ocr_raw"),
        "config_path": "config/local.yaml",
        "layout_device": DEFAULT_LAYOUT_DEVICE,
        "reviewed_layout_path": str(run_dir / "reviewed_layout.json"),
    }


def test_prepare_review_workflow_clears_stale_predictions(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-review"
    seed_existing_run(run_dir)
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    def fake_prepare_review(_page_paths, run_dir_arg, *, config_path, layout_device, **_kw):
        run_dir_path = Path(run_dir_arg)
        raw_dir = run_dir_path / "ocr_raw" / "page-0001"
        raw_dir.mkdir(parents=True, exist_ok=True)
        sdk_json_path = raw_dir / "page-0001_model.json"
        sdk_json_path.write_text("{}", encoding="utf-8")
        reviewed_layout = {
            "version": 1,
            "status": "prepared",
            "pages": [
                build_reviewed_layout_page(
                    page_path=str(run_dir_path / "pages" / "page-0001.png"),
                    source_sdk_json_path=str(sdk_json_path),
                    blocks=[build_reviewed_layout_block(content="Prepared block")],
                )
            ],
            "summary": {"page_count": 1},
        }
        (run_dir_path / "reviewed_layout.json").write_text(
            json.dumps(reviewed_layout),
            encoding="utf-8",
        )
        return {
            "reviewed_layout": reviewed_layout,
            "raw_dir": str(raw_dir.parent),
            "config_path": config_path,
            "layout_device": layout_device,
            "reviewed_layout_path": str(run_dir_path / "reviewed_layout.json"),
        }

    prepare_review_workflow(
        str(source),
        run="demo-review",
        run_root=str(run_root),
        normalize_document_fn=normalize_to_single_page,
        prepare_review_artifacts_fn=fake_prepare_review,
    )

    assert not (run_dir / "predictions" / "rules.json").exists()
    assert not (run_dir / "predictions" / "glmocr_structured.json").exists()
    assert not (run_dir / "predictions" / "glmocr_structured_meta.json").exists()
    assert not (run_dir / "predictions" / "demo-review.json").exists()


def test_prepare_review_workflow_restores_existing_ocr_outputs_when_metadata_write_fails(
    tmp_path,
) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-review"
    seed_existing_run(run_dir)
    (run_dir / "ocr_fallback").mkdir(parents=True, exist_ok=True)
    (run_dir / "ocr_fallback" / "page-0001.txt").write_text("legacy crop", encoding="utf-8")
    (run_dir / "ocr_fallback.json").write_text(
        json.dumps({"legacy": "fallback"}),
        encoding="utf-8",
    )
    (run_dir / "meta.json").write_text('{"status": "existing"}', encoding="utf-8")
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    def fake_prepare_review(_page_paths, run_dir_arg, *, config_path, layout_device, **_kw):
        run_dir_path = Path(run_dir_arg)
        raw_dir = run_dir_path / "ocr_raw" / "page-0001"
        raw_dir.mkdir(parents=True, exist_ok=True)
        sdk_json_path = raw_dir / "page-0001_model.json"
        sdk_json_path.write_text("{}", encoding="utf-8")
        reviewed_layout = {
            "version": 1,
            "status": "prepared",
            "pages": [
                build_reviewed_layout_page(
                    page_path=str(run_dir_path / "pages" / "page-0001.png"),
                    source_sdk_json_path=str(sdk_json_path),
                    blocks=[build_reviewed_layout_block(content="Prepared block")],
                )
            ],
            "summary": {"page_count": 1},
        }
        (run_dir_path / "reviewed_layout.json").write_text(
            json.dumps(reviewed_layout),
            encoding="utf-8",
        )
        return {
            "reviewed_layout": reviewed_layout,
            "raw_dir": str(raw_dir.parent),
            "config_path": config_path,
            "layout_device": layout_device,
            "reviewed_layout_path": str(run_dir_path / "reviewed_layout.json"),
        }

    def failing_write_json(path: str | Path, payload) -> None:  # noqa: ANN001
        if Path(path).name == "meta.json":
            raise RuntimeError("metadata write failed")
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="metadata write failed"):
        prepare_review_workflow(
            str(source),
            run="demo-review",
            run_root=str(run_root),
            normalize_document_fn=normalize_to_single_page,
            prepare_review_artifacts_fn=fake_prepare_review,
            write_json_fn=failing_write_json,
        )

    assert (run_dir / "pages" / "page-0001.png").read_bytes() == b"existing page"
    assert (run_dir / "ocr.md").read_text(encoding="utf-8") == "existing markdown"
    assert json.loads((run_dir / "ocr.json").read_text(encoding="utf-8")) == {"existing": True}
    assert json.loads((run_dir / "ocr_fallback.json").read_text(encoding="utf-8")) == {
        "legacy": "fallback"
    }
    assert (run_dir / "ocr_fallback" / "page-0001.txt").read_text(encoding="utf-8") == "legacy crop"
    assert not (run_dir / "reviewed_layout.json").exists()
    assert json.loads((run_dir / "predictions" / "glmocr_structured.json").read_text()) == {
        "structured": "keep-me"
    }
    assert json.loads((run_dir / "predictions" / "demo-review.json").read_text()) == {
        "canonical": "keep-me"
    }


def test_run_reviewed_ocr_workflow_passes_reviewed_layout_artifact_to_ocr(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-reviewed-ocr"
    page_path = Path(normalize_to_single_page("ignored", run_dir)[0])
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.4 stub")
    raw_dir = run_dir / "ocr_raw" / "page-0001"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "page-0001_model.json").write_text("{}", encoding="utf-8")
    write_reviewed_layout(run_dir, page_path=str(page_path))
    (run_dir / "meta.json").write_text(
        json.dumps({"input_path": str(source)}),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run_ocr(_page_paths, run_dir_arg, *, config_path, layout_device, reviewed_layout_path, **_kw):
        captured["page_paths"] = _page_paths
        captured["reviewed_layout_path"] = reviewed_layout_path
        return build_basic_ocr_result(
            run_dir_arg,
            config_path=config_path,
            layout_device=layout_device,
            json_payload={
                "pages": [
                    build_ocr_page(
                        page_path=str(page_path),
                        sdk_json_path=str(raw_dir / "page-0001_model.json"),
                    )
                ],
                "summary": {"page_count": 1, "sources": {"sdk_markdown": 1}},
            },
        )

    published_run_dir = run_reviewed_ocr_workflow(
        "demo-reviewed-ocr",
        run_root=str(run_root),
        run_ocr_fn=fake_run_ocr,
    )

    assert published_run_dir == run_dir
    assert captured["page_paths"] == [str(page_path)]
    assert captured["reviewed_layout_path"] == run_dir / "reviewed_layout.json"
    assert page_path.exists()
    assert (run_dir / "ocr.md").read_text(encoding="utf-8") == "replacement markdown"
    assert (run_dir / "reviewed_layout.json").exists()
    assert json.loads((run_dir / "meta.json").read_text(encoding="utf-8")) == {
        "input_path": str(source),
        "page_paths": [str(page_path)],
        "ocr_raw_dir": str(run_dir / "ocr_raw"),
        "config_path": "config/local.yaml",
        "layout_device": DEFAULT_LAYOUT_DEVICE,
        "reviewed_layout_path": str(run_dir / "reviewed_layout.json"),
    }


def test_run_reviewed_ocr_workflow_clears_stale_predictions(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-reviewed-ocr"
    seed_existing_run(run_dir)
    page_path = run_dir / "pages" / "page-0001.png"
    write_reviewed_layout(run_dir, page_path=str(page_path))

    def fake_run_ocr(_page_paths, run_dir_arg, *, config_path, layout_device, reviewed_layout_path, **_kw):
        _ = (reviewed_layout_path,)
        return build_basic_ocr_result(
            run_dir_arg,
            config_path=config_path,
            layout_device=layout_device,
            json_payload={
                "pages": [
                    build_ocr_page(
                        page_path=str(page_path),
                        sdk_json_path=str(
                            run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json"
                        ),
                    )
                ],
                "summary": {"page_count": 1, "sources": {"sdk_markdown": 1}},
            },
        )

    run_reviewed_ocr_workflow(
        "demo-reviewed-ocr",
        run_root=str(run_root),
        run_ocr_fn=fake_run_ocr,
    )

    assert not (run_dir / "predictions" / "rules.json").exists()
    assert not (run_dir / "predictions" / "glmocr_structured.json").exists()
    assert not (run_dir / "predictions" / "glmocr_structured_meta.json").exists()
    assert not (run_dir / "predictions" / "demo-reviewed-ocr.json").exists()


def test_prepare_review_page_workflow_replaces_only_selected_page(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-page-review"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)
    page_one = pages_dir / "page-0001.png"
    page_two = pages_dir / "page-0002.png"
    page_one.write_bytes(b"page-1")
    page_two.write_bytes(b"page-2")

    raw_page_one = run_dir / "ocr_raw" / "page-0001"
    raw_page_two = run_dir / "ocr_raw" / "page-0002"
    raw_page_one.mkdir(parents=True, exist_ok=True)
    raw_page_two.mkdir(parents=True, exist_ok=True)
    (raw_page_one / "page-0001_model.json").write_text(
        json.dumps({"page": 1, "status": "existing"}), encoding="utf-8"
    )
    (raw_page_two / "page-0002_model.json").write_text(
        json.dumps({"page": 2, "status": "existing"}), encoding="utf-8"
    )
    (run_dir / "reviewed_layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "status": "reviewed",
                "pages": [
                    build_reviewed_layout_page(
                        page_number=1,
                        page_path=str(page_one),
                        source_sdk_json_path=str(raw_page_one / "page-0001_model.json"),
                        blocks=[build_reviewed_layout_block(content="old page 1")],
                    ),
                    build_reviewed_layout_page(
                        page_number=2,
                        page_path=str(page_two),
                        source_sdk_json_path=str(raw_page_two / "page-0002_model.json"),
                        blocks=[build_reviewed_layout_block(content="old page 2")],
                    ),
                ],
                "summary": {"page_count": 2},
            }
        ),
        encoding="utf-8",
    )
    write_basic_ocr_outputs(
        run_dir,
        markdown="# Page 1\n\n# Page 2",
        json_payload={
            "pages": [
                build_ocr_page(
                    page_number=1,
                    page_path=str(page_one),
                    markdown="# Page 1",
                    sdk_json_path=str(raw_page_one / "page-0001_model.json"),
                ),
                build_ocr_page(
                    page_number=2,
                    page_path=str(page_two),
                    markdown="# Page 2",
                    sdk_json_path=str(raw_page_two / "page-0002_model.json"),
                ),
            ],
            "summary": {"page_count": 2, "sources": {"sdk_markdown": 2}},
        },
    )
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    (predictions_dir / "rules.json").write_text("{}", encoding="utf-8")
    (predictions_dir / "demo-page-review.json").write_text("{}", encoding="utf-8")
    (predictions_dir / "glmocr_structured.json").write_text("{}", encoding="utf-8")
    (predictions_dir / "glmocr_structured_meta.json").write_text("{}", encoding="utf-8")
    (run_dir / "meta.json").write_text(
        json.dumps({"input_path": "/tmp/source.pdf"}),
        encoding="utf-8",
    )

    def fake_prepare_review(_page_paths, run_dir_arg, *, config_path, layout_device, **_kw):
        assert _page_paths == [str(page_two)]
        raw_dir = Path(run_dir_arg) / "ocr_raw" / "page-0002"
        raw_dir.mkdir(parents=True, exist_ok=True)
        model_path = raw_dir / "page-0002_model.json"
        model_path.write_text(json.dumps({"page": 2, "status": "redetected"}), encoding="utf-8")
        reviewed_layout = {
            "version": 1,
            "status": "prepared",
            "pages": [
                build_reviewed_layout_page(
                    page_number=2,
                    page_path=str(Path(run_dir_arg) / "pages" / "page-0002.png"),
                    source_sdk_json_path=str(model_path),
                    blocks=[build_reviewed_layout_block(content="new page 2")],
                )
            ],
            "summary": {"page_count": 1},
        }
        (Path(run_dir_arg) / "reviewed_layout.json").write_text(
            json.dumps(reviewed_layout),
            encoding="utf-8",
        )
        return {
            "reviewed_layout": reviewed_layout,
            "raw_dir": str(Path(run_dir_arg) / "ocr_raw"),
            "config_path": config_path,
            "layout_device": layout_device,
        }

    prepare_review_page_workflow(
        "demo-page-review",
        2,
        run_root=str(run_root),
        prepare_review_artifacts_fn=fake_prepare_review,
    )

    reviewed_layout = json.loads((run_dir / "reviewed_layout.json").read_text(encoding="utf-8"))
    assert [page["page_number"] for page in reviewed_layout["pages"]] == [1, 2]
    assert reviewed_layout["pages"][0]["blocks"][0]["content"] == "old page 1"
    assert reviewed_layout["pages"][1]["blocks"][0]["content"] == "new page 2"
    assert reviewed_layout["pages"][1]["source_sdk_json_path"] == str(
        run_dir / "ocr_raw" / "page-0002" / "page-0002_model.json"
    )
    assert json.loads((run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json").read_text()) == {
        "page": 1,
        "status": "existing",
    }
    assert json.loads((run_dir / "ocr_raw" / "page-0002" / "page-0002_model.json").read_text()) == {
        "page": 2,
        "status": "redetected",
    }
    assert not (run_dir / "ocr.md").exists()
    assert not (run_dir / "ocr.json").exists()
    assert not (run_dir / "predictions" / "rules.json").exists()
    assert not (run_dir / "predictions" / "glmocr_structured.json").exists()
    assert not (run_dir / "predictions" / "glmocr_structured_meta.json").exists()
    assert not (run_dir / "predictions" / "demo-page-review.json").exists()


def test_prepare_review_page_workflow_resolves_sparse_renamed_page(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-sparse-review"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)
    page_path = pages_dir / "scan.png"
    page_path.write_bytes(b"page-5")

    existing_raw_dir = run_dir / "ocr_raw" / "page-0005"
    existing_raw_dir.mkdir(parents=True)
    (existing_raw_dir / "page-0005_model.json").write_text(
        json.dumps({"page": 5, "status": "existing"}),
        encoding="utf-8",
    )
    (run_dir / "reviewed_layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "status": "reviewed",
                "pages": [
                    build_reviewed_layout_page(
                        page_number=5,
                        page_path="different-name.png",
                        source_sdk_json_path=str(existing_raw_dir / "page-0005_model.json"),
                        blocks=[build_reviewed_layout_block(content="old page 5")],
                    )
                ],
                "summary": {"page_count": 1},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "meta.json").write_text(json.dumps({"input_path": "/tmp/source.pdf"}))

    def fake_prepare_review(
        _page_paths,
        run_dir_arg,
        *,
        config_path,
        layout_device,
        page_numbers,
        **_kw,
    ):
        assert _page_paths == [str(page_path)]
        assert page_numbers == [5]
        raw_dir = Path(run_dir_arg) / "ocr_raw" / "page-0005"
        raw_dir.mkdir(parents=True, exist_ok=True)
        model_path = raw_dir / "page-0005_model.json"
        model_path.write_text(json.dumps({"page": 5, "status": "redetected"}), encoding="utf-8")
        reviewed_layout = {
            "version": 1,
            "status": "prepared",
            "pages": [
                build_reviewed_layout_page(
                    page_number=5,
                    page_path=str(page_path),
                    source_sdk_json_path=str(model_path),
                    blocks=[build_reviewed_layout_block(content="new page 5")],
                )
            ],
            "summary": {"page_count": 1},
        }
        (Path(run_dir_arg) / "reviewed_layout.json").write_text(
            json.dumps(reviewed_layout),
            encoding="utf-8",
        )
        return {
            "reviewed_layout": reviewed_layout,
            "raw_dir": str(Path(run_dir_arg) / "ocr_raw"),
            "config_path": config_path,
            "layout_device": layout_device,
        }

    prepare_review_page_workflow(
        "demo-sparse-review",
        5,
        run_root=str(run_root),
        prepare_review_artifacts_fn=fake_prepare_review,
    )

    reviewed_layout = json.loads((run_dir / "reviewed_layout.json").read_text(encoding="utf-8"))
    assert reviewed_layout["pages"][0]["page_number"] == 5
    assert reviewed_layout["pages"][0]["blocks"][0]["content"] == "new page 5"
    assert reviewed_layout["pages"][0]["source_sdk_json_path"] == str(
        run_dir / "ocr_raw" / "page-0005" / "page-0005_model.json"
    )


def test_prepare_review_page_workflow_preserves_ocr_only_pages_when_review_missing(tmp_path) -> None:
    image_module = import_module("PIL.Image")

    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-page-review-ocr-only"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)
    page_one = pages_dir / "page-0001.png"
    page_two = pages_dir / "page-0002.png"
    image_module.new("RGB", (100, 120), color="white").save(page_one)
    image_module.new("RGB", (100, 120), color="white").save(page_two)

    raw_page_one = run_dir / "ocr_raw" / "page-0001"
    raw_page_two = run_dir / "ocr_raw" / "page-0002"
    raw_page_one.mkdir(parents=True, exist_ok=True)
    raw_page_two.mkdir(parents=True, exist_ok=True)
    (raw_page_one / "page-0001_model.json").write_text(
        json.dumps(
            {
                "blocks": [
                    {"label": "text", "content": "ocr-only page 1", "bbox_2d": [1, 2, 20, 30]}
                ]
            }
        ),
        encoding="utf-8",
    )
    (raw_page_two / "page-0002_model.json").write_text(
        json.dumps(
            {
                "blocks": [
                    {"label": "text", "content": "old page 2", "bbox_2d": [2, 3, 21, 31]}
                ]
            }
        ),
        encoding="utf-8",
    )
    write_basic_ocr_outputs(
        run_dir,
        markdown="# Page 1\n\n# Page 2",
        json_payload={
            "pages": [
                build_ocr_page(
                    page_number=1,
                    page_path=str(page_one),
                    markdown="# Page 1",
                    sdk_json_path=str(raw_page_one / "page-0001_model.json"),
                ),
                build_ocr_page(
                    page_number=2,
                    page_path=str(page_two),
                    markdown="# Page 2",
                    sdk_json_path=str(raw_page_two / "page-0002_model.json"),
                ),
            ],
            "summary": {"page_count": 2, "sources": {"sdk_markdown": 2}},
        },
    )
    (run_dir / "meta.json").write_text(
        json.dumps({"input_path": "/tmp/source.pdf"}),
        encoding="utf-8",
    )

    def fake_prepare_review(_page_paths, run_dir_arg, *, config_path, layout_device, **_kw):
        assert _page_paths == [str(page_two)]
        raw_dir = Path(run_dir_arg) / "ocr_raw" / "page-0002"
        raw_dir.mkdir(parents=True, exist_ok=True)
        model_path = raw_dir / "page-0002_model.json"
        model_path.write_text(
            json.dumps(
                {
                    "blocks": [
                        {"label": "text", "content": "new page 2", "bbox_2d": [3, 4, 25, 35]}
                    ]
                }
            ),
            encoding="utf-8",
        )
        reviewed_layout = {
            "version": 1,
            "status": "prepared",
            "pages": [
                build_reviewed_layout_page(
                    page_number=2,
                    page_path=str(Path(run_dir_arg) / "pages" / "page-0002.png"),
                    source_sdk_json_path=str(model_path),
                    blocks=[build_reviewed_layout_block(content="new page 2")],
                )
            ],
            "summary": {"page_count": 1},
        }
        (Path(run_dir_arg) / "reviewed_layout.json").write_text(
            json.dumps(reviewed_layout),
            encoding="utf-8",
        )
        return {
            "reviewed_layout": reviewed_layout,
            "raw_dir": str(Path(run_dir_arg) / "ocr_raw"),
            "config_path": config_path,
            "layout_device": layout_device,
        }

    prepare_review_page_workflow(
        "demo-page-review-ocr-only",
        2,
        run_root=str(run_root),
        prepare_review_artifacts_fn=fake_prepare_review,
    )

    reviewed_layout = json.loads((run_dir / "reviewed_layout.json").read_text(encoding="utf-8"))
    assert [page["page_number"] for page in reviewed_layout["pages"]] == [1, 2]
    assert reviewed_layout["pages"][0]["blocks"][0]["content"] == "ocr-only page 1"
    assert reviewed_layout["pages"][1]["blocks"][0]["content"] == "new page 2"
    assert reviewed_layout["pages"][0]["source_sdk_json_path"] == str(
        run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json"
    )
    assert reviewed_layout["pages"][1]["source_sdk_json_path"] == str(
        run_dir / "ocr_raw" / "page-0002" / "page-0002_model.json"
    )
    assert not (run_dir / "ocr.md").exists()
    assert not (run_dir / "ocr.json").exists()


def test_prepare_review_page_workflow_raises_when_untouched_ocr_sdk_json_is_missing(tmp_path) -> None:
    image_module = import_module("PIL.Image")

    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-page-review-missing-sdk"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)
    page_one = pages_dir / "page-0001.png"
    page_two = pages_dir / "page-0002.png"
    image_module.new("RGB", (100, 120), color="white").save(page_one)
    image_module.new("RGB", (100, 120), color="white").save(page_two)

    raw_page_two = run_dir / "ocr_raw" / "page-0002"
    raw_page_two.mkdir(parents=True, exist_ok=True)
    (raw_page_two / "page-0002_model.json").write_text(
        json.dumps(
            {
                "blocks": [
                    {"label": "text", "content": "old page 2", "bbox_2d": [2, 3, 21, 31]}
                ]
            }
        ),
        encoding="utf-8",
    )
    write_basic_ocr_outputs(
        run_dir,
        markdown="# Page 1\n\n# Page 2",
        json_payload={
            "pages": [
                build_ocr_page(
                    page_number=1,
                    page_path=str(page_one),
                    markdown="# Page 1",
                    sdk_json_path=str(run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json"),
                ),
                build_ocr_page(
                    page_number=2,
                    page_path=str(page_two),
                    markdown="# Page 2",
                    sdk_json_path=str(raw_page_two / "page-0002_model.json"),
                ),
            ],
            "summary": {"page_count": 2, "sources": {"sdk_markdown": 2}},
        },
    )
    (run_dir / "meta.json").write_text(
        json.dumps({"input_path": "/tmp/source.pdf"}),
        encoding="utf-8",
    )

    def fake_prepare_review(_page_paths, run_dir_arg, *, config_path, layout_device, **_kw):
        raw_dir = Path(run_dir_arg) / "ocr_raw" / "page-0002"
        raw_dir.mkdir(parents=True, exist_ok=True)
        model_path = raw_dir / "page-0002_model.json"
        model_path.write_text(
            json.dumps(
                {
                    "blocks": [
                        {"label": "text", "content": "new page 2", "bbox_2d": [3, 4, 25, 35]}
                    ]
                }
            ),
            encoding="utf-8",
        )
        reviewed_layout = {
            "version": 1,
            "status": "prepared",
            "pages": [
                build_reviewed_layout_page(
                    page_number=2,
                    page_path=str(Path(run_dir_arg) / "pages" / "page-0002.png"),
                    source_sdk_json_path=str(model_path),
                    blocks=[build_reviewed_layout_block(content="new page 2")],
                )
            ],
            "summary": {"page_count": 1},
        }
        (Path(run_dir_arg) / "reviewed_layout.json").write_text(
            json.dumps(reviewed_layout),
            encoding="utf-8",
        )
        return {
            "reviewed_layout": reviewed_layout,
            "raw_dir": str(Path(run_dir_arg) / "ocr_raw"),
            "config_path": config_path,
            "layout_device": layout_device,
        }

    with pytest.raises(FileNotFoundError, match="page-0001_model.json"):
        prepare_review_page_workflow(
            "demo-page-review-missing-sdk",
            2,
            run_root=str(run_root),
            prepare_review_artifacts_fn=fake_prepare_review,
        )

    assert not (run_dir / "reviewed_layout.json").exists()
    assert (run_dir / "ocr.json").exists()


def test_run_reviewed_ocr_page_workflow_merges_only_selected_page_outputs(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-page-ocr"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)
    page_one = pages_dir / "page-0001.png"
    page_two = pages_dir / "page-0002.png"
    page_one.write_bytes(b"page-1")
    page_two.write_bytes(b"page-2")

    raw_page_one = run_dir / "ocr_raw" / "page-0001"
    raw_page_two = run_dir / "ocr_raw" / "page-0002"
    raw_page_one.mkdir(parents=True, exist_ok=True)
    raw_page_two.mkdir(parents=True, exist_ok=True)
    (raw_page_one / "page-0001_model.json").write_text(
        json.dumps({"page": 1, "status": "existing"}), encoding="utf-8"
    )
    (raw_page_two / "page-0002_model.json").write_text(
        json.dumps({"page": 2, "status": "existing"}), encoding="utf-8"
    )
    (run_dir / "ocr.md").write_text("# Old Page 1\n\n# Old Page 2", encoding="utf-8")
    (run_dir / "ocr.json").write_text(
        json.dumps(
            {
                "pages": [
                    build_ocr_page(
                        page_number=1,
                        page_path=str(page_one),
                        markdown="# Old Page 1",
                        sdk_json_path=str(raw_page_one / "page-0001_model.json"),
                    ),
                    build_ocr_page(
                        page_number=2,
                        page_path=str(page_two),
                        markdown="# Old Page 2",
                        sdk_json_path=str(raw_page_two / "page-0002_model.json"),
                        markdown_source="crop_fallback",
                    ),
                ],
                "summary": {
                    "page_count": 2,
                    "sources": {"sdk_markdown": 1, "crop_fallback": 1},
                    "reviewed_layout": {
                        "path": str(run_dir / "reviewed_layout.json"),
                        "page_count": 2,
                        "apply_mode": "planning_and_fallback_only",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "ocr_fallback.json").write_text(
        json.dumps(
            {
                "pages": [
                    build_fallback_page(
                        page_number=1,
                        page_path=str(page_one),
                        chunks=[
                            build_fallback_chunk(
                                crop_path=str(run_dir / "ocr_fallback" / "page-0001" / "crop.png"),
                                text_path=str(run_dir / "ocr_fallback" / "page-0001" / "crop.txt"),
                            )
                        ],
                    ),
                    build_fallback_page(
                        page_number=2,
                        page_path=str(page_two),
                        chunks=[
                            build_fallback_chunk(
                                crop_path=str(run_dir / "ocr_fallback" / "page-0002" / "crop.png"),
                                text_path=str(run_dir / "ocr_fallback" / "page-0002" / "crop.txt"),
                            )
                        ],
                    ),
                ],
                "summary": {"page_count": 2},
            }
        ),
        encoding="utf-8",
    )
    fallback_page_one = run_dir / "ocr_fallback" / "page-0001"
    fallback_page_two = run_dir / "ocr_fallback" / "page-0002"
    fallback_page_one.mkdir(parents=True, exist_ok=True)
    fallback_page_two.mkdir(parents=True, exist_ok=True)
    (fallback_page_one / "crop.txt").write_text("keep me", encoding="utf-8")
    (fallback_page_two / "crop.txt").write_text("replace me", encoding="utf-8")
    write_reviewed_layout(run_dir, page_path=str(page_one))
    (run_dir / "meta.json").write_text(
        json.dumps({"input_path": "/tmp/source.pdf"}),
        encoding="utf-8",
    )
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    (predictions_dir / "rules.json").write_text("{}", encoding="utf-8")
    (predictions_dir / "demo-page-ocr.json").write_text("{}", encoding="utf-8")
    (predictions_dir / "glmocr_structured.json").write_text("{}", encoding="utf-8")
    (predictions_dir / "glmocr_structured_meta.json").write_text("{}", encoding="utf-8")

    def fake_run_ocr(_page_paths, run_dir_arg, *, config_path, layout_device, reviewed_layout_path, **_kw):
        assert _page_paths == [str(page_two)]
        assert reviewed_layout_path == run_dir / "reviewed_layout.json"
        raw_dir = Path(run_dir_arg) / "ocr_raw" / "page-0002"
        raw_dir.mkdir(parents=True, exist_ok=True)
        model_path = raw_dir / "page-0002_model.json"
        model_path.write_text(json.dumps({"page": 2, "status": "rerun"}), encoding="utf-8")
        (Path(run_dir_arg) / "ocr.md").write_text("# New Page 2", encoding="utf-8")
        (Path(run_dir_arg) / "ocr.json").write_text(
            json.dumps(
                {
                    "pages": [
                        build_ocr_page(
                            page_number=2,
                            page_path=str(page_two),
                            markdown="# New Page 2",
                            sdk_json_path=str(model_path),
                        )
                    ],
                    "summary": {"page_count": 1, "sources": {"sdk_markdown": 1}},
                }
            ),
            encoding="utf-8",
        )
        return {
            "markdown": "# New Page 2",
            "json": {"pages": []},
            "raw_dir": str(Path(run_dir_arg) / "ocr_raw"),
            "config_path": config_path,
            "layout_device": layout_device,
        }

    run_reviewed_ocr_page_workflow(
        "demo-page-ocr",
        2,
        run_root=str(run_root),
        run_ocr_fn=fake_run_ocr,
    )

    ocr_payload = json.loads((run_dir / "ocr.json").read_text(encoding="utf-8"))
    assert [page["page_number"] for page in ocr_payload["pages"]] == [1, 2]
    assert ocr_payload["pages"][0]["markdown"] == "# Old Page 1"
    assert ocr_payload["pages"][1]["markdown"] == "# New Page 2"
    assert ocr_payload["pages"][1]["sdk_json_path"] == str(
        run_dir / "ocr_raw" / "page-0002" / "page-0002_model.json"
    )
    assert ocr_payload["summary"] == {
        "page_count": 2,
        "sources": {"sdk_markdown": 2},
        "reviewed_layout": {
            "path": str(run_dir / "reviewed_layout.json"),
            "page_count": 2,
            "apply_mode": "planning_and_fallback_only",
        },
    }
    assert (run_dir / "ocr.md").read_text(encoding="utf-8") == "# Old Page 1\n\n# New Page 2"
    assert json.loads((run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json").read_text()) == {
        "page": 1,
        "status": "existing",
    }
    assert json.loads((run_dir / "ocr_raw" / "page-0002" / "page-0002_model.json").read_text()) == {
        "page": 2,
        "status": "rerun",
    }
    fallback_payload = json.loads((run_dir / "ocr_fallback.json").read_text(encoding="utf-8"))
    assert [page["page_number"] for page in fallback_payload["pages"]] == [1]
    assert (run_dir / "ocr_fallback" / "page-0001" / "crop.txt").read_text(encoding="utf-8") == "keep me"
    assert not (run_dir / "ocr_fallback" / "page-0002").exists()
    assert not (run_dir / "predictions" / "rules.json").exists()
    assert not (run_dir / "predictions" / "glmocr_structured.json").exists()
    assert not (run_dir / "predictions" / "glmocr_structured_meta.json").exists()
    assert not (run_dir / "predictions" / "demo-page-ocr.json").exists()


def test_run_reviewed_ocr_page_workflow_resolves_sparse_renamed_page(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo-sparse-ocr"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True)
    page_path = pages_dir / "scan.png"
    page_path.write_bytes(b"page-5")

    raw_dir = run_dir / "ocr_raw" / "page-0005"
    raw_dir.mkdir(parents=True)
    existing_model_path = raw_dir / "page-0005_model.json"
    existing_model_path.write_text(json.dumps({"page": 5, "status": "existing"}), encoding="utf-8")
    (run_dir / "ocr.md").write_text("# Old Page 5", encoding="utf-8")
    (run_dir / "ocr.json").write_text(
        json.dumps(
            {
                "pages": [
                    build_ocr_page(
                        page_number=5,
                        page_path="different-name.png",
                        markdown="# Old Page 5",
                        sdk_json_path=str(existing_model_path),
                    )
                ],
                "summary": {"page_count": 1, "sources": {"sdk_markdown": 1}},
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
                        page_number=5,
                        page_path="different-name.png",
                        source_sdk_json_path=str(existing_model_path),
                        blocks=[build_reviewed_layout_block(content="old page 5")],
                    )
                ],
                "summary": {"page_count": 1},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "meta.json").write_text(json.dumps({"input_path": "/tmp/source.pdf"}))

    def fake_run_ocr(
        _page_paths,
        run_dir_arg,
        *,
        config_path,
        layout_device,
        reviewed_layout_path,
        page_numbers,
        **_kw,
    ):
        assert _page_paths == [str(page_path)]
        assert reviewed_layout_path == run_dir / "reviewed_layout.json"
        assert page_numbers == [5]
        partial_raw_dir = Path(run_dir_arg) / "ocr_raw" / "page-0005"
        partial_raw_dir.mkdir(parents=True, exist_ok=True)
        model_path = partial_raw_dir / "page-0005_model.json"
        model_path.write_text(json.dumps({"page": 5, "status": "rerun"}), encoding="utf-8")
        (Path(run_dir_arg) / "ocr.md").write_text("# New Page 5", encoding="utf-8")
        (Path(run_dir_arg) / "ocr.json").write_text(
            json.dumps(
                {
                    "pages": [
                        build_ocr_page(
                            page_number=5,
                            page_path=str(page_path),
                            markdown="# New Page 5",
                            sdk_json_path=str(model_path),
                        )
                    ],
                    "summary": {"page_count": 1, "sources": {"sdk_markdown": 1}},
                }
            ),
            encoding="utf-8",
        )
        return {
            "markdown": "# New Page 5",
            "json": {"pages": []},
            "raw_dir": str(Path(run_dir_arg) / "ocr_raw"),
            "config_path": config_path,
            "layout_device": layout_device,
        }

    run_reviewed_ocr_page_workflow(
        "demo-sparse-ocr",
        5,
        run_root=str(run_root),
        run_ocr_fn=fake_run_ocr,
    )

    ocr_payload = json.loads((run_dir / "ocr.json").read_text(encoding="utf-8"))
    assert ocr_payload["pages"][0]["page_number"] == 5
    assert ocr_payload["pages"][0]["markdown"] == "# New Page 5"
    assert ocr_payload["pages"][0]["sdk_json_path"] == str(
        run_dir / "ocr_raw" / "page-0005" / "page-0005_model.json"
    )
    assert (run_dir / "ocr.md").read_text(encoding="utf-8") == "# New Page 5"


def test_run_pipeline_workflow_rerun_keeps_structured_outputs(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo001"
    seed_existing_run(run_dir)
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    def fake_run_ocr(_page_paths, run_dir_arg, *, config_path, layout_device, **_kw):
        return build_basic_ocr_result(
            run_dir_arg,
            config_path=config_path,
            layout_device=layout_device,
        )

    run_pipeline_workflow(
        str(source),
        run="demo001",
        run_root=str(run_root),
        normalize_document_fn=normalize_to_single_page,
        run_ocr_fn=fake_run_ocr,
        extract_from_markdown_fn=lambda markdown: {"rules": markdown},
    )

    assert json.loads((run_dir / "predictions" / "glmocr_structured.json").read_text()) == {
        "structured": "keep-me"
    }
    assert json.loads((run_dir / "predictions" / "glmocr_structured_meta.json").read_text()) == {
        "model": "keep-me"
    }
    assert json.loads((run_dir / "predictions" / "rules.json").read_text()) == {
        "rules": "replacement markdown"
    }
    assert json.loads((run_dir / "predictions" / "demo001.json").read_text()) == {
        "rules": "replacement markdown"
    }


def test_run_structured_workflow_uses_rules_as_canonical_when_structured_output_is_suspicious(
    tmp_path,
) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo003"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / "page-0001.png").write_bytes(b"fake image")

    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    rules_prediction = {"title": "Home Assignment 8", "document_type": "report"}
    (predictions_dir / "rules.json").write_text(json.dumps(rules_prediction), encoding="utf-8")

    def fake_extract_structured(_page_paths, *, markdown_text, config_path, model, endpoint):
        _ = (markdown_text, config_path, endpoint)
        return {
            "document_type": "document",
            "title": "document",
            "authors": ["[{"],
            "institution": "document",
            "date": "document",
            "language": "document",
            "summary_line": "document",
        }, {
            "model": model or "glm-ocr:latest",
            "source": "page_images_first_page",
            "_raw_body": {"response": '{"document_type":"document"}'},
        }

    run_structured_workflow(
        "demo003",
        run_root=str(run_root),
        extract_structured_fn=fake_extract_structured,
    )

    assert json.loads((predictions_dir / "glmocr_structured.json").read_text()) == {
        "document_type": "document",
        "title": "document",
        "authors": ["[{"],
        "institution": "document",
        "date": "document",
        "language": "document",
        "summary_line": "document",
    }
    assert json.loads((predictions_dir / "demo003.json").read_text()) == rules_prediction
    assert json.loads((predictions_dir / "glmocr_structured_meta.json").read_text()) == {
        "model": "glm-ocr:latest",
        "source": "page_images_first_page",
        "canonical_source": "rules",
        "validation": {
            "ok": False,
            "reasons": [
                "all scalar fields collapsed to placeholder value 'document'",
                "authors field contains JSON fence or bracket fragments",
            ],
        },
    }
    assert json.loads((predictions_dir / "glmocr_structured_raw.json").read_text()) == {
        "response": '{"document_type":"document"}'
    }


def test_run_structured_workflow_rejects_hallucinated_fields_not_in_ocr_text(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo009"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / "page-0001.png").write_bytes(b"fake image")
    (run_dir / "ocr.md").write_text(
        "Home Assignment 8\n\nOnline and Reinforcement Learning, 2025-2026\n\n"
        "Solution report and experiment summary",
        encoding="utf-8",
    )

    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    rules_prediction = {"title": "Home Assignment 8", "document_type": "report"}
    (predictions_dir / "rules.json").write_text(json.dumps(rules_prediction), encoding="utf-8")

    def fake_extract_structured(_page_paths, *, markdown_text, config_path, model, endpoint):
        _ = (markdown_text, config_path, endpoint)
        return {
            "document_type": "report",
            "title": "Online and Reinforcement Learning, 2025-2026 Solution report and experiment summary",
            "authors": ["Scheffler", "Hoeffding"],
            "institution": "University of California, Berkeley",
            "date": "2025-02-15",
            "language": "English",
            "summary_line": "Required: document_type, title, authors, institution, date, language, summary_line",
        }, {
            "model": model or "glm-ocr:latest",
            "source": "ocr_markdown",
            "_raw_body": {"response": '{"document_type":"report"}'},
        }

    run_structured_workflow(
        "demo009",
        run_root=str(run_root),
        extract_structured_fn=fake_extract_structured,
    )

    assert json.loads((predictions_dir / "demo009.json").read_text()) == rules_prediction
    assert json.loads((predictions_dir / "glmocr_structured_meta.json").read_text()) == {
        "model": "glm-ocr:latest",
        "source": "ocr_markdown",
        "canonical_source": "rules",
        "validation": {
            "ok": False,
            "reasons": [
                "summary_line appears to echo extraction instructions",
                "author 'Scheffler' not found in OCR text",
                "author 'Hoeffding' not found in OCR text",
                "institution value 'University of California, Berkeley' not found in OCR text",
                "date value '2025-02-15' not found in OCR text",
            ],
        },
    }


def test_run_structured_workflow_forwards_config_and_overrides(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo004"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / "page-0001.png").write_bytes(b"fake image")

    captured = {}

    def fake_extract_structured(_page_paths, *, markdown_text, config_path, model, endpoint):
        captured["page_paths"] = _page_paths
        captured["markdown_text"] = markdown_text
        captured["config_path"] = config_path
        captured["model"] = model
        captured["endpoint"] = endpoint
        return {"title": "Demo"}, {"model": model, "_raw_body": {"response": "{}"}}

    run_structured_workflow(
        "demo004",
        run_root=str(run_root),
        config_path="config/custom.yaml",
        model="manual-model",
        endpoint="http://manual.example/api/generate",
        extract_structured_fn=fake_extract_structured,
    )

    assert captured == {
        "page_paths": [str(pages_dir / "page-0001.png")],
        "markdown_text": None,
        "config_path": "config/custom.yaml",
        "model": "manual-model",
        "endpoint": "http://manual.example/api/generate",
    }


def test_run_structured_workflow_prefers_ocr_markdown_when_available(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo005"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / "page-0001.png").write_bytes(b"fake image")
    (run_dir / "ocr.md").write_text("# OCR\nMarkdown text", encoding="utf-8")

    captured = {}

    def fake_extract_structured(_page_paths, *, markdown_text, config_path, model, endpoint):
        captured["page_paths"] = _page_paths
        captured["markdown_text"] = markdown_text
        _ = (config_path, endpoint)
        return {"title": "Demo"}, {
            "model": model or "glm-ocr:latest",
            "source": "ocr_markdown",
            "_raw_body": {"response": "{}"},
        }

    run_structured_workflow(
        "demo005",
        run_root=str(run_root),
        extract_structured_fn=fake_extract_structured,
    )

    assert captured["page_paths"] == [str(pages_dir / "page-0001.png")]
    assert captured["markdown_text"] == "# OCR\nMarkdown text"
    assert (
        json.loads((run_dir / "predictions" / "glmocr_structured_meta.json").read_text())["source"]
        == "ocr_markdown"
    )


def test_run_structured_workflow_falls_back_to_rules_when_structured_call_fails(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo006"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / "page-0001.png").write_bytes(b"fake image")

    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    rules_prediction = {"title": "Rules title", "document_type": "report"}
    (predictions_dir / "rules.json").write_text(json.dumps(rules_prediction), encoding="utf-8")

    def failing_extract_structured(*_args, **_kwargs):
        raise RuntimeError("model unavailable")

    run_structured_workflow(
        "demo006",
        run_root=str(run_root),
        extract_structured_fn=failing_extract_structured,
    )

    assert json.loads((predictions_dir / "demo006.json").read_text()) == rules_prediction
    assert json.loads((predictions_dir / "glmocr_structured_meta.json").read_text()) == {
        "status": "failed",
        "error": "model unavailable",
        "canonical_source": "rules",
    }
    assert not (predictions_dir / "glmocr_structured.json").exists()
    assert not (predictions_dir / "glmocr_structured_raw.json").exists()


def test_run_structured_workflow_clears_stale_structured_outputs_on_failure(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo008"
    seed_existing_run(run_dir)

    def failing_extract_structured(*_args, **_kwargs):
        raise RuntimeError("model unavailable")

    run_structured_workflow(
        "demo008",
        run_root=str(run_root),
        extract_structured_fn=failing_extract_structured,
    )

    predictions_dir = run_dir / "predictions"
    assert not (predictions_dir / "glmocr_structured.json").exists()
    assert not (predictions_dir / "glmocr_structured_raw.json").exists()
    assert json.loads((predictions_dir / "glmocr_structured_meta.json").read_text()) == {
        "status": "failed",
        "error": "model unavailable",
        "canonical_source": "rules",
    }


def test_run_structured_workflow_fails_cleanly_without_rules_prediction(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo007"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / "page-0001.png").write_bytes(b"fake image")

    def failing_extract_structured(*_args, **_kwargs):
        raise RuntimeError("model unavailable")

    with pytest.raises(RuntimeError, match="Structured extraction failed"):
        run_structured_workflow(
            "demo007",
            run_root=str(run_root),
            extract_structured_fn=failing_extract_structured,
        )


def test_run_ocr_workflow_restores_existing_run_when_publish_fails(tmp_path, monkeypatch) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo010"
    seed_existing_run(run_dir)
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    def fake_run_ocr(_page_paths, run_dir_arg, *, config_path, layout_device, **_kw):
        run_dir_path = Path(run_dir_arg)
        raw_dir = run_dir_path / "ocr_raw" / "page-0001"
        raw_dir.mkdir(parents=True, exist_ok=True)
        sdk_json_path = raw_dir / "page-0001_model.json"
        sdk_json_path.write_text("{}", encoding="utf-8")
        (run_dir_path / "ocr.md").write_text("replacement markdown", encoding="utf-8")
        (run_dir_path / "ocr.json").write_text(
            json.dumps(
                {
                    "pages": [
                        build_ocr_page(
                            page_path=str(run_dir_path / "pages" / "page-0001.png"),
                            sdk_json_path=str(sdk_json_path),
                        )
                    ],
                    "summary": {"page_count": 1, "sources": {"sdk_markdown": 1}},
                }
            ),
            encoding="utf-8",
        )
        return {
            "markdown": "replacement markdown",
            "json": {"replacement": True},
            "raw_dir": str(raw_dir.parent),
            "config_path": config_path,
            "layout_device": layout_device,
        }

    def failing_normalize_publish(*_args, **_kwargs) -> None:
        raise RuntimeError("publish normalize failed")

    monkeypatch.setattr(workflows, "_normalize_published_ocr_artifacts", failing_normalize_publish)

    with pytest.raises(RuntimeError, match="publish normalize failed"):
        run_ocr_workflow(
            str(source),
            run="demo010",
            run_root=str(run_root),
            normalize_document_fn=normalize_to_single_page,
            run_ocr_fn=fake_run_ocr,
        )

    assert (run_dir / "pages" / "page-0001.png").read_bytes() == b"existing page"
    assert (run_dir / "ocr.md").read_text(encoding="utf-8") == "existing markdown"
    assert json.loads((run_dir / "ocr.json").read_text(encoding="utf-8")) == {"existing": True}
    assert json.loads(
        (run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json").read_text(encoding="utf-8")
    ) == {"legacy": True}
    assert json.loads((run_dir / "predictions" / "demo010.json").read_text(encoding="utf-8")) == {
        "canonical": "keep-me"
    }


def test_run_ocr_workflow_restores_existing_run_when_backup_move_fails(
    tmp_path, monkeypatch
) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo011"
    seed_existing_run(run_dir)
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    def fake_run_ocr(_page_paths, run_dir_arg, *, config_path, layout_device, **_kw):
        run_dir_path = Path(run_dir_arg)
        raw_dir = run_dir_path / "ocr_raw" / "page-0001"
        raw_dir.mkdir(parents=True, exist_ok=True)
        sdk_json_path = raw_dir / "page-0001_model.json"
        sdk_json_path.write_text("{}", encoding="utf-8")
        (run_dir_path / "ocr.md").write_text("replacement markdown", encoding="utf-8")
        (run_dir_path / "ocr.json").write_text(
            json.dumps(
                {
                    "pages": [
                        build_ocr_page(
                            page_path=str(run_dir_path / "pages" / "page-0001.png"),
                            sdk_json_path=str(sdk_json_path),
                        )
                    ],
                    "summary": {"page_count": 1, "sources": {"sdk_markdown": 1}},
                }
            ),
            encoding="utf-8",
        )
        return {
            "markdown": "replacement markdown",
            "json": {"replacement": True},
            "raw_dir": str(raw_dir.parent),
            "config_path": config_path,
            "layout_device": layout_device,
        }

    real_move_path = workflows._move_path
    backup_move_calls = 0

    def flaky_move_path(source_path: Path, target_path: Path) -> None:
        nonlocal backup_move_calls
        if ".demo011-ocr-backup-" in str(target_path):
            backup_move_calls += 1
            if backup_move_calls == 2:
                raise RuntimeError("mid-backup failure")
        real_move_path(source_path, target_path)

    monkeypatch.setattr(workflows, "_move_path", flaky_move_path)

    with pytest.raises(RuntimeError, match="mid-backup failure"):
        run_ocr_workflow(
            str(source),
            run="demo011",
            run_root=str(run_root),
            normalize_document_fn=normalize_to_single_page,
            run_ocr_fn=fake_run_ocr,
        )

    assert (run_dir / "pages" / "page-0001.png").read_bytes() == b"existing page"
    assert (run_dir / "ocr.md").read_text(encoding="utf-8") == "existing markdown"
    assert json.loads((run_dir / "ocr.json").read_text(encoding="utf-8")) == {"existing": True}
    assert json.loads(
        (run_dir / "ocr_raw" / "page-0001" / "page-0001_model.json").read_text(encoding="utf-8")
    ) == {"legacy": True}
    assert json.loads((run_dir / "predictions" / "demo011.json").read_text(encoding="utf-8")) == {
        "canonical": "keep-me"
    }


def test_run_ocr_workflow_preserves_backup_when_restore_fails_midway(tmp_path, monkeypatch) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo012"
    seed_existing_run(run_dir)
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    def fake_run_ocr(_page_paths, run_dir_arg, *, config_path, layout_device, **_kw):
        run_dir_path = Path(run_dir_arg)
        raw_dir = run_dir_path / "ocr_raw" / "page-0001"
        raw_dir.mkdir(parents=True, exist_ok=True)
        sdk_json_path = raw_dir / "page-0001_model.json"
        sdk_json_path.write_text("{}", encoding="utf-8")
        (run_dir_path / "ocr.md").write_text("replacement markdown", encoding="utf-8")
        (run_dir_path / "ocr.json").write_text(
            json.dumps(
                {
                    "pages": [
                        build_ocr_page(
                            page_path=str(run_dir_path / "pages" / "page-0001.png"),
                            sdk_json_path=str(sdk_json_path),
                        )
                    ],
                    "summary": {"page_count": 1, "sources": {"sdk_markdown": 1}},
                }
            ),
            encoding="utf-8",
        )
        return {
            "markdown": "replacement markdown",
            "json": {"replacement": True},
            "raw_dir": str(raw_dir.parent),
            "config_path": config_path,
            "layout_device": layout_device,
        }

    def failing_normalize_publish(*_args, **_kwargs) -> None:
        raise RuntimeError("publish normalize failed")

    real_copy_path = workflows._copy_path
    restore_copy_calls = 0

    def flaky_copy_path(source_path: Path, target_path: Path) -> None:
        nonlocal restore_copy_calls
        restore_copy_calls += 1
        if restore_copy_calls == 2:
            raise RuntimeError("restore failed")
        real_copy_path(source_path, target_path)

    monkeypatch.setattr(workflows, "_normalize_published_ocr_artifacts", failing_normalize_publish)
    monkeypatch.setattr(workflows, "_copy_path", flaky_copy_path)

    with pytest.raises(RuntimeError, match="Backup preserved at"):
        run_ocr_workflow(
            str(source),
            run="demo012",
            run_root=str(run_root),
            normalize_document_fn=normalize_to_single_page,
            run_ocr_fn=fake_run_ocr,
        )

    backup_dirs = [
        path for path in run_root.iterdir() if path.name.startswith(".demo012-ocr-backup-")
    ]
    assert len(backup_dirs) == 1
    backup_dir = backup_dirs[0]
    assert (backup_dir / "pages" / "page-0001.png").read_bytes() == b"existing page"
    assert (backup_dir / "ocr.md").read_text(encoding="utf-8") == "existing markdown"
    assert json.loads((backup_dir / "ocr.json").read_text(encoding="utf-8")) == {"existing": True}
    assert json.loads(
        (backup_dir / "ocr_raw" / "page-0001" / "page-0001_model.json").read_text(encoding="utf-8")
    ) == {"legacy": True}
