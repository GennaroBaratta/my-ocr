from __future__ import annotations

import ast
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "my_ocr"


def _py_files(package: str) -> list[Path]:
    return sorted((SRC_ROOT / package).rglob("*.py"))


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _dynamic_import_strings(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_import_module = (
            isinstance(func, ast.Name)
            and func.id == "import_module"
            or isinstance(func, ast.Attribute)
            and func.attr == "import_module"
        )
        if not is_import_module or not node.args:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            modules.add(arg.value)
    return modules


def _violations(package: str, blocked_prefixes: tuple[str, ...]) -> list[str]:
    failures: list[str] = []
    for path in _py_files(package):
        modules = _imported_modules(path) | _dynamic_import_strings(path)
        for module in modules:
            if module.startswith(blocked_prefixes):
                failures.append(f"{path.relative_to(SRC_ROOT)} references {module}")
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


def test_application_does_not_import_adapters_or_ui_even_dynamically() -> None:
    assert not _violations(
        "application",
        (
            "my_ocr.adapters",
            "my_ocr.ui",
        ),
    )


def test_ui_does_not_import_removed_application_artifacts_or_use_cases() -> None:
    assert not _violations(
        "ui",
        (
            "my_ocr.application.artifacts",
            "my_ocr.application.use_cases",
        ),
    )

