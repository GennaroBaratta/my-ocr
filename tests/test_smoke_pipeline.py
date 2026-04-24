from __future__ import annotations

import json
from pathlib import Path

from my_ocr.adapters.inbound import cli
from my_ocr.adapters.outbound.llm import structured_extractor as extract_glmocr


def test_smoke_pipeline_with_stubbed_ocr(tmp_path, monkeypatch) -> None:
    run_root = tmp_path / "runs"
    gold_dir = tmp_path / "gold"
    reports_dir = tmp_path / "reports"
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    def fake_normalize_document(input_path: str, run_dir: str) -> list[str]:
        pages_dir = Path(run_dir) / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        page = pages_dir / "page-0001.png"
        page.write_bytes(b"fake image")
        return [str(page)]

    def fake_run_ocr(page_paths, run_dir, *, config_path, layout_device, **kwargs):
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "ocr.md").write_text(
            "# Sample Report\nAda Lovelace\nExample University\n2024-01-15\nA concise summary line.",
            encoding="utf-8",
        )
        (Path(run_dir) / "ocr.json").write_text("{}", encoding="utf-8")
        raw_dir = Path(run_dir) / "ocr_raw"
        raw_dir.mkdir(exist_ok=True)
        return {"markdown": "", "json": {}, "raw_dir": str(raw_dir)}

    monkeypatch.setattr(cli, "normalize_document", fake_normalize_document)
    monkeypatch.setattr(cli, "run_ocr", fake_run_ocr)

    cli.main(["run", str(source), "--run", "demo001", "--run-root", str(run_root)])

    prediction_path = run_root / "demo001" / "predictions" / "demo001.json"
    assert prediction_path.exists()

    gold_dir.mkdir(parents=True, exist_ok=True)
    (gold_dir / "demo001.json").write_text(
        json.dumps(
            {
                "document_type": "report",
                "title": "Sample Report",
                "authors": ["Ada Lovelace"],
                "institution": "Example University",
                "date": "2024-01-15",
                "language": "",
                "summary_line": "A concise summary line.",
            }
        ),
        encoding="utf-8",
    )

    output_report = reports_dir / "demo001.md"
    cli.main(
        [
            "eval",
            "--gold-dir",
            str(gold_dir),
            "--pred-dir",
            str(run_root / "demo001" / "predictions"),
            "--output",
            str(output_report),
        ]
    )

    assert output_report.exists()
    assert "Evaluation Report" in output_report.read_text(encoding="utf-8")


def test_run_generates_run_id_when_omitted(tmp_path, monkeypatch) -> None:
    run_root = tmp_path / "runs"
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 stub")

    def fake_normalize_document(input_path: str, run_dir: str) -> list[str]:
        pages_dir = Path(run_dir) / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        page = pages_dir / "page-0001.png"
        page.write_bytes(b"fake image")
        return [str(page)]

    def fake_run_ocr(page_paths, run_dir, *, config_path, layout_device, **kwargs):
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "ocr.md").write_text("# Sample Report\nAda Lovelace", encoding="utf-8")
        (Path(run_dir) / "ocr.json").write_text("{}", encoding="utf-8")
        raw_dir = Path(run_dir) / "ocr_raw"
        raw_dir.mkdir(exist_ok=True)
        return {"markdown": "", "json": {}, "raw_dir": str(raw_dir)}

    monkeypatch.setattr(cli, "normalize_document", fake_normalize_document)
    monkeypatch.setattr(cli, "run_ocr", fake_run_ocr)

    cli.main(["run", str(source), "--run-root", str(run_root)])

    created_runs = [path for path in run_root.iterdir() if path.is_dir()]
    assert len(created_runs) == 1
    assert (created_runs[0] / "predictions" / f"{created_runs[0].name}.json").exists()


def test_extract_glmocr_cli_writes_structured_prediction(tmp_path, monkeypatch) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo002"
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / "page-0001.png").write_bytes(b"fake image")

    monkeypatch.setattr(
        extract_glmocr,
        "extract_structured",
        lambda page_paths, *, markdown_text, config_path, model, endpoint: (
            {"title": "Sample"},
            {
                "model": model,
                "endpoint": endpoint,
                "config_path": config_path,
                "markdown_text": markdown_text,
            },
        ),
    )
    monkeypatch.setattr(
        extract_glmocr, "save_structured_result", lambda run_dir, prediction, metadata: None
    )
    monkeypatch.setattr(
        cli,
        "write_json",
        lambda path, payload: (
            Path(path).parent.mkdir(parents=True, exist_ok=True),
            Path(path).write_text(json.dumps(payload), encoding="utf-8"),
        ),
    )

    cli.main(["extract-glmocr", "--run", "demo002", "--run-root", str(run_root)])

    assert (run_dir / "predictions" / "demo002.json").exists()
