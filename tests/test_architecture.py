from __future__ import annotations

import ast
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "my_ocr"
REPO_ROOT = SRC_ROOT.parents[1]


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


def _attribute_call_lines(path: Path, attribute: str) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == attribute
        ):
            lines.append(node.lineno)
    return lines


def _attribute_call_sites(roots: tuple[Path, ...], attribute: str) -> list[tuple[Path, int]]:
    calls: list[tuple[Path, int]] = []
    for root in roots:
        for path in root.rglob("*.py"):
            calls.extend((path, line) for line in _attribute_call_lines(path, attribute))
    return calls


def _defined_function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


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
            "my_ocr.pipeline",
            "my_ocr.adapters",
            "my_ocr.ui",
        ),
    )


def test_pipeline_does_not_import_adapters_or_ui_even_dynamically() -> None:
    assert not _violations(
        "pipeline",
        (
            "my_ocr.adapters",
            "my_ocr.ui",
        ),
    )


def test_source_tests_and_tools_do_not_import_application_package() -> None:
    roots = (REPO_ROOT / "src", REPO_ROOT / "tests", REPO_ROOT / "tools")
    failures: list[str] = []
    blocked = "my_ocr.application"
    for root in roots:
        for path in root.rglob("*.py"):
            modules = _imported_modules(path) | _dynamic_import_strings(path)
            if any(module == blocked or module.startswith(f"{blocked}.") for module in modules):
                failures.append(f"{path.relative_to(REPO_ROOT)} imports application package")

    assert not failures


def test_ui_does_not_import_filesystem_run_store_directly() -> None:
    assert not _violations(
        "ui",
        ("my_ocr.adapters.outbound.filesystem.run_store",),
    )


def test_ui_state_does_not_define_magic_session_delegation() -> None:
    state_path = SRC_ROOT / "ui" / "state.py"
    defined = _defined_function_names(state_path)

    assert "__getattr__" not in defined
    assert "__setattr__" not in defined


def test_ui_does_not_open_run_transactions_directly() -> None:
    roots = (REPO_ROOT / "src", REPO_ROOT / "tests", REPO_ROOT / "tools")
    failures = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for path, line in _attribute_call_sites(roots, "begin_update")
    ]

    assert not failures


def test_review_layout_saves_are_coordinated_by_workflow() -> None:
    workflow_path = SRC_ROOT / "pipeline" / "workflow.py"
    roots = (
        SRC_ROOT / "pipeline",
        SRC_ROOT / "adapters" / "inbound",
        SRC_ROOT / "ui",
    )
    failures = [
        f"{path.relative_to(SRC_ROOT)}:{line}"
        for path, line in _attribute_call_sites(roots, "write_review_layout")
        if path != workflow_path
    ]

    assert not failures


def test_source_tests_and_tools_do_not_define_removed_ocr_legacy_helpers() -> None:
    roots = (REPO_ROOT / "src", REPO_ROOT / "tests", REPO_ROOT / "tools")
    failures: list[str] = []
    blocked = {
        "_run_dir_from_pages",
        "_legacy_review_payload",
        "_review_layout_from_legacy",
        "_ocr_result_from_legacy",
        "_legacy_layout_cleanup_paths",
        "_legacy_ocr_cleanup_paths",
    }
    for root in roots:
        for path in root.rglob("*.py"):
            overlap = blocked & _defined_function_names(path)
            if overlap:
                failures.append(
                    f"{path.relative_to(REPO_ROOT)} defines removed helpers: {sorted(overlap)}"
                )

    assert not failures


def test_source_tests_and_tools_do_not_call_removed_run_transaction_methods() -> None:
    roots = (REPO_ROOT / "src", REPO_ROOT / "tests", REPO_ROOT / "tools")
    failures = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for attribute in ("create_run", "write_pages", "commit_run", "rollback_run")
        for path, line in _attribute_call_sites(roots, attribute)
    ]

    assert not failures

