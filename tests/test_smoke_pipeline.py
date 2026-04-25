from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from my_ocr.adapters.inbound import cli
from my_ocr.application.dto import RunId, RunInput, RunManifest, RunSnapshot


def test_cli_run_uses_v2_pipeline_service(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}
    run_dir = tmp_path / "runs" / "demo001"
    run_dir.mkdir(parents=True)

    def fake_run_pipeline(command):
        captured["input_path"] = command.input_path
        captured["run_id"] = command.run_id
        manifest = RunManifest(
            run_id=RunId("demo001"),
            input=RunInput.from_path(command.input_path),
            pages=[],
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )
        return SimpleNamespace(snapshot=RunSnapshot(run_dir, manifest))

    monkeypatch.setattr(
        cli,
        "build_backend_services",
        lambda run_root: SimpleNamespace(run_pipeline=fake_run_pipeline),
    )

    cli.main(["run", "input.pdf", "--run", "demo001", "--run-root", str(tmp_path / "runs")])

    assert captured["input_path"] == "input.pdf"
    assert str(captured["run_id"]) == "demo001"


def test_cli_prepare_review_command_is_available() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["prepare-review", "input.pdf", "--run", "demo"])

    assert args.command == "prepare-review"
    assert args.run == "demo"

