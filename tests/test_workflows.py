from __future__ import annotations

import json
from pathlib import Path

import pytest

from free_doc_extract.workflows import run_ocr_workflow, run_pipeline_workflow


@pytest.mark.parametrize("failure_phase", ["normalize", "ocr"])
def test_run_ocr_workflow_preserves_existing_run_on_failure(tmp_path, failure_phase: str) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo001"
    _seed_existing_run(run_dir)
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    def failing_normalize(input_path: str, run_dir_arg: str | Path) -> list[str]:
        if failure_phase == "normalize":
            raise ValueError("unsupported input")

        pages_dir = Path(run_dir_arg) / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        page = pages_dir / "page-0001.png"
        page.write_bytes(b"replacement image")
        return [str(page)]

    def failing_ocr(page_paths, run_dir_arg, *, config_path, layout_device):
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


def test_run_ocr_workflow_rerun_keeps_existing_predictions(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo001"
    _seed_existing_run(run_dir)
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    def fake_normalize(input_path: str, run_dir_arg: str | Path) -> list[str]:
        pages_dir = Path(run_dir_arg) / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        page = pages_dir / "page-0001.png"
        page.write_bytes(b"replacement image")
        return [str(page)]

    def fake_run_ocr(page_paths, run_dir_arg, *, config_path, layout_device):
        run_dir_path = Path(run_dir_arg)
        (run_dir_path / "ocr.md").write_text("replacement markdown", encoding="utf-8")
        (run_dir_path / "ocr.json").write_text('{"replacement": true}', encoding="utf-8")
        raw_dir = run_dir_path / "ocr_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        return {"markdown": "replacement markdown", "json": {"replacement": True}, "raw_dir": str(raw_dir)}

    run_ocr_workflow(
        str(source),
        run="demo001",
        run_root=str(run_root),
        normalize_document_fn=fake_normalize,
        run_ocr_fn=fake_run_ocr,
    )

    assert (run_dir / "pages" / "page-0001.png").read_bytes() == b"replacement image"
    assert (run_dir / "ocr.md").read_text(encoding="utf-8") == "replacement markdown"
    assert json.loads((run_dir / "predictions" / "glmocr_structured.json").read_text()) == {
        "structured": "keep-me"
    }
    assert json.loads((run_dir / "predictions" / "demo001.json").read_text()) == {
        "canonical": "keep-me"
    }


def test_run_pipeline_workflow_rerun_keeps_structured_outputs(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo001"
    _seed_existing_run(run_dir)
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    def fake_normalize(input_path: str, run_dir_arg: str | Path) -> list[str]:
        pages_dir = Path(run_dir_arg) / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        page = pages_dir / "page-0001.png"
        page.write_bytes(b"replacement image")
        return [str(page)]

    def fake_run_ocr(page_paths, run_dir_arg, *, config_path, layout_device):
        run_dir_path = Path(run_dir_arg)
        (run_dir_path / "ocr.md").write_text("replacement markdown", encoding="utf-8")
        (run_dir_path / "ocr.json").write_text('{"replacement": true}', encoding="utf-8")
        raw_dir = run_dir_path / "ocr_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        return {"markdown": "replacement markdown", "json": {"replacement": True}, "raw_dir": str(raw_dir)}

    run_pipeline_workflow(
        str(source),
        run="demo001",
        run_root=str(run_root),
        normalize_document_fn=fake_normalize,
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


def _seed_existing_run(run_dir: Path) -> None:
    (run_dir / "pages").mkdir(parents=True, exist_ok=True)
    (run_dir / "pages" / "page-0001.png").write_bytes(b"existing page")
    (run_dir / "ocr_raw").mkdir(exist_ok=True)
    (run_dir / "ocr.md").write_text("existing markdown", encoding="utf-8")
    (run_dir / "ocr.json").write_text('{"existing": true}', encoding="utf-8")

    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    (predictions_dir / "glmocr_structured.json").write_text(
        json.dumps({"structured": "keep-me"}),
        encoding="utf-8",
    )
    (predictions_dir / "glmocr_structured_meta.json").write_text(
        json.dumps({"model": "keep-me"}),
        encoding="utf-8",
    )
    (predictions_dir / "rules.json").write_text(
        json.dumps({"rules": "existing"}),
        encoding="utf-8",
    )
    (predictions_dir / "demo001.json").write_text(
        json.dumps({"canonical": "keep-me"}),
        encoding="utf-8",
    )
