from __future__ import annotations

import ast
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "my_ocr"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _py_files(package: str) -> list[Path]:
    return sorted((SRC_ROOT / package).rglob("*.py"))


def _violations(package: str, blocked_prefixes: tuple[str, ...]) -> list[str]:
    failures: list[str] = []
    for path in _py_files(package):
        for module in _imported_modules(path):
            if module.startswith(blocked_prefixes):
                failures.append(f"{path.relative_to(SRC_ROOT)} imports {module}")
    return failures


def test_domain_does_not_import_outer_layers() -> None:
    assert not _violations(
        "domain",
        (
            "my_ocr.application",
            "my_ocr.adapters",
            "my_ocr.ui",
        ),
    )


def test_application_does_not_import_adapters_or_ui() -> None:
    assert not _violations(
        "application",
        (
            "my_ocr.adapters",
            "my_ocr.ui",
        ),
    )


def test_ui_uses_only_application_use_cases_and_public_backend_dtos() -> None:
    allowed_prefixes = (
        "my_ocr.application.artifacts",
        "my_ocr.application.use_cases",
        "my_ocr.adapters.outbound.config.settings",
        "my_ocr.domain",
    )
    failures: list[str] = []
    for path in _py_files("ui"):
        for module in _imported_modules(path):
            if module.startswith("my_ocr.") and not module.startswith(allowed_prefixes):
                failures.append(f"{path.relative_to(SRC_ROOT)} imports {module}")
    assert not failures
