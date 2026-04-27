from __future__ import annotations

from types import SimpleNamespace

from my_ocr import cli


class FakeWorkflow:
    def prepare_review(self, *, input_path, run_id, options):
        _ = (input_path, run_id, options)
        return SimpleNamespace(snapshot=SimpleNamespace(run_id="run-1", run_dir="run-dir"))

    def run_reviewed_ocr(self, run_id, *, options):
        _ = (run_id, options)
        return SimpleNamespace(snapshot=SimpleNamespace(run_dir="run-dir"))

    def extract_structured(self, run_id, *, options):
        _ = (run_id, options)

    def run_automatic(self, *, input_path, run_id, layout_options, ocr_options):
        _ = (input_path, run_id, layout_options, ocr_options)
        return SimpleNamespace(snapshot=SimpleNamespace(run_dir="run-dir"))


def test_cli_passes_config_path_to_bootstrap_for_configured_commands(monkeypatch) -> None:
    captured: list[tuple[str, str]] = []

    def fake_build_backend_services(run_root, *, config_path):
        captured.append((run_root, config_path))
        return SimpleNamespace(workflow=FakeWorkflow())

    monkeypatch.setattr(cli, "build_backend_services", fake_build_backend_services)

    commands = [
        ["prepare-review", "input.pdf", "--run-root", "runs", "--config", "custom.yaml"],
        ["run-reviewed-ocr", "--run", "demo", "--run-root", "runs", "--config", "custom.yaml"],
        ["ocr", "input.pdf", "--run-root", "runs", "--config", "custom.yaml"],
        ["extract-glmocr", "--run", "demo", "--run-root", "runs", "--config", "custom.yaml"],
        ["run", "input.pdf", "--run-root", "runs", "--config", "custom.yaml"],
    ]

    for command in commands:
        assert cli.main(command) == 0

    assert captured == [("runs", "custom.yaml")] * len(commands)
