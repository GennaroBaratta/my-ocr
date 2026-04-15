from __future__ import annotations

import json
from pathlib import Path

import pytest

from free_doc_extract.workflows import (
    run_ocr_workflow,
    run_pipeline_workflow,
    run_structured_workflow,
)
from free_doc_extract.settings import DEFAULT_LAYOUT_DEVICE


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
    (run_dir / "ocr_fallback").mkdir(exist_ok=True)
    (run_dir / "ocr_fallback" / "stale.txt").write_text("stale", encoding="utf-8")
    (run_dir / "ocr_fallback.json").write_text('{"stale": true}', encoding="utf-8")
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    def fake_normalize(input_path: str, run_dir_arg: str | Path) -> list[str]:
        pages_dir = Path(run_dir_arg) / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        page = pages_dir / "page-0001.png"
        page.write_bytes(b"replacement image")
        return [str(page)]

    captured_page_paths = {}

    def fake_run_ocr(page_paths, run_dir_arg, *, config_path, layout_device):
        captured_page_paths["value"] = page_paths
        run_dir_path = Path(run_dir_arg)
        (run_dir_path / "ocr.md").write_text("replacement markdown", encoding="utf-8")
        (run_dir_path / "ocr.json").write_text('{"replacement": true}', encoding="utf-8")
        raw_dir = run_dir_path / "ocr_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        return {
            "markdown": "replacement markdown",
            "json": {"replacement": True},
            "raw_dir": str(raw_dir),
            "config_path": config_path,
            "layout_device": layout_device,
        }

    run_ocr_workflow(
        str(source),
        run="demo001",
        run_root=str(run_root),
        normalize_document_fn=fake_normalize,
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


def test_run_ocr_workflow_accepts_directory_input_and_forwards_normalized_pages(tmp_path) -> None:
    run_root = tmp_path / "runs"
    source = tmp_path / "images"
    source.mkdir()
    captured = {}

    def fake_normalize(input_path: str, run_dir_arg: str | Path) -> list[str]:
        pages_dir = Path(run_dir_arg) / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        page = pages_dir / "page-0001.png"
        page.write_bytes(b"replacement image")
        return [str(page)]

    def fake_run_ocr(page_paths, run_dir_arg, *, config_path, layout_device):
        captured["page_paths"] = page_paths
        run_dir_path = Path(run_dir_arg)
        (run_dir_path / "ocr.md").write_text("replacement markdown", encoding="utf-8")
        (run_dir_path / "ocr.json").write_text('{"replacement": true}', encoding="utf-8")
        raw_dir = run_dir_path / "ocr_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        return {
            "markdown": "replacement markdown",
            "json": {"replacement": True},
            "raw_dir": str(raw_dir),
            "config_path": config_path,
            "layout_device": layout_device,
        }

    run_ocr_workflow(
        str(source),
        run="demo-dir",
        run_root=str(run_root),
        normalize_document_fn=fake_normalize,
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
        fallback_dir = run_dir_path / "ocr_fallback" / "page-0001"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        (fallback_dir / "chunk-0001.txt").write_text("fallback text", encoding="utf-8")
        (run_dir_path / "ocr_fallback.json").write_text(
            '{"pages": [{"page_number": 1}], "summary": {"page_count": 1}}',
            encoding="utf-8",
        )
        raw_dir = run_dir_path / "ocr_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        return {
            "markdown": "replacement markdown",
            "json": {"replacement": True},
            "raw_dir": str(raw_dir),
            "config_path": config_path,
            "layout_device": layout_device,
        }

    run_dir = run_ocr_workflow(
        str(source),
        run="demo-fallback",
        run_root=str(run_root),
        normalize_document_fn=fake_normalize,
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

    def fake_normalize(input_path: str, run_dir_arg: str | Path) -> list[str]:
        pages_dir = Path(run_dir_arg) / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        page = pages_dir / "page-0001.png"
        page.write_bytes(b"replacement image")
        return [str(page)]

    def fake_run_ocr(page_paths, run_dir_arg, *, config_path, layout_device):
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
        raw_dir = run_dir_path / "ocr_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (run_dir_path / "ocr.md").write_text("replacement markdown", encoding="utf-8")
        write_payload = {
            "pages": [
                {
                    "page_number": 1,
                    "page_path": str(staged_page),
                    "markdown": "replacement markdown",
                    "markdown_source": "crop_fallback",
                    "sdk_markdown": "",
                    "sdk_json": {},
                    "fallback_assessment": {},
                }
            ],
            "summary": {"page_count": 1, "sources": {"crop_fallback": 1}},
        }
        (run_dir_path / "ocr.json").write_text(json.dumps(write_payload), encoding="utf-8")
        (run_dir_path / "ocr_fallback.json").write_text(
            json.dumps(
                {
                    "pages": [
                        {
                            "page_number": 1,
                            "page_path": str(staged_page),
                            "chunks": [
                                {
                                    "crop_path": str(crop_path),
                                    "text_path": str(text_path),
                                }
                            ],
                        }
                    ],
                    "summary": {"page_count": 1},
                }
            ),
            encoding="utf-8",
        )
        return {
            "markdown": "replacement markdown",
            "json": write_payload,
            "raw_dir": str(raw_dir),
            "config_path": config_path,
            "layout_device": layout_device,
        }

    run_dir = run_ocr_workflow(
        str(source),
        run="demo-paths",
        run_root=str(run_root),
        normalize_document_fn=fake_normalize,
        run_ocr_fn=fake_run_ocr,
    )

    published_ocr = json.loads((run_dir / "ocr.json").read_text(encoding="utf-8"))
    published_fallback = json.loads((run_dir / "ocr_fallback.json").read_text(encoding="utf-8"))

    assert published_ocr["pages"][0]["page_path"] == str(run_dir / "pages" / "page-0001.png")
    assert published_fallback["pages"][0]["page_path"] == str(run_dir / "pages" / "page-0001.png")
    assert published_fallback["pages"][0]["chunks"][0]["crop_path"] == str(
        run_dir / "ocr_fallback" / "page-0001" / "chunk-0001.png"
    )
    assert published_fallback["pages"][0]["chunks"][0]["text_path"] == str(
        run_dir / "ocr_fallback" / "page-0001" / "chunk-0001.txt"
    )


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
        return {
            "markdown": "replacement markdown",
            "json": {"replacement": True},
            "raw_dir": str(raw_dir),
            "config_path": config_path,
            "layout_device": layout_device,
        }

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

    run_structured_workflow(
        "demo003",
        run_root=str(run_root),
        extract_structured_fn=lambda page_paths, *, markdown_text, config_path, model, endpoint: (
            {
                "document_type": "document",
                "title": "document",
                "authors": ["[{"],
                "institution": "document",
                "date": "document",
                "language": "document",
                "summary_line": "document",
            },
            {
                "model": model or "glm-ocr:latest",
                "source": "page_images_first_page",
                "_raw_body": {"response": '{"document_type":"document"}'},
            },
        ),
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

    run_structured_workflow(
        "demo009",
        run_root=str(run_root),
        extract_structured_fn=lambda page_paths, *, markdown_text, config_path, model, endpoint: (
            {
                "document_type": "report",
                "title": "Online and Reinforcement Learning, 2025-2026 Solution report and experiment summary",
                "authors": ["Scheffler", "Hoeffding"],
                "institution": "University of California, Berkeley",
                "date": "2025-02-15",
                "language": "English",
                "summary_line": "Required: document_type, title, authors, institution, date, language, summary_line",
            },
            {
                "model": model or "glm-ocr:latest",
                "source": "ocr_markdown",
                "_raw_body": {"response": '{"document_type":"report"}'},
            },
        ),
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

    def fake_extract_structured(page_paths, *, markdown_text, config_path, model, endpoint):
        captured["page_paths"] = page_paths
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

    def fake_extract_structured(page_paths, *, markdown_text, config_path, model, endpoint):
        captured["page_paths"] = page_paths
        captured["markdown_text"] = markdown_text
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

    def failing_extract_structured(*args, **kwargs):
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
    _seed_existing_run(run_dir)

    def failing_extract_structured(*args, **kwargs):
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

    def failing_extract_structured(*args, **kwargs):
        raise RuntimeError("model unavailable")

    with pytest.raises(RuntimeError, match="Structured extraction failed"):
        run_structured_workflow(
            "demo007",
            run_root=str(run_root),
            extract_structured_fn=failing_extract_structured,
        )


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
