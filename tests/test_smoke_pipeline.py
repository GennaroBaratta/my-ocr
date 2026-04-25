from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from my_ocr.adapters.inbound import cli
from my_ocr.models import RunId, RunInput, RunManifest, RunSnapshot


def test_cli_run_uses_v3_workflow_service(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}
    run_dir = tmp_path / "runs" / "demo001"
    run_dir.mkdir(parents=True)

    def fake_run_automatic(*, input_path, run_id, layout_options, ocr_options):
        captured["input_path"] = input_path
        captured["run_id"] = run_id
        captured["layout_options"] = layout_options
        captured["ocr_options"] = ocr_options
        manifest = RunManifest(
            run_id=RunId("demo001"),
            input=RunInput.from_path(input_path),
            pages=[],
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )
        return SimpleNamespace(snapshot=RunSnapshot(run_dir=run_dir, manifest=manifest))

    monkeypatch.setattr(
        cli,
        "build_backend_services",
        lambda run_root: SimpleNamespace(
            workflow=SimpleNamespace(run_automatic=fake_run_automatic)
        ),
    )

    cli.main(["run", "input.pdf", "--run", "demo001", "--run-root", str(tmp_path / "runs")])

    assert captured["input_path"] == "input.pdf"
    assert str(captured["run_id"]) == "demo001"


def test_cli_prepare_review_command_is_available() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["prepare-review", "input.pdf", "--run", "demo"])

    assert args.command == "prepare-review"
    assert args.run == "demo"
