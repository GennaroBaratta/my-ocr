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
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
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


def _defined_class_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def _is_module_or_child(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def _matches_any_prefix(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(_is_module_or_child(module, prefix) for prefix in prefixes)


def _first_party_imports(path: Path) -> set[str]:
    modules = _imported_modules(path) | _dynamic_import_strings(path)
    return {module for module in modules if _is_module_or_child(module, "my_ocr")}


def _violations(package: str, blocked_prefixes: tuple[str, ...]) -> list[str]:
    failures: list[str] = []
    for path in _py_files(package):
        for module in _first_party_imports(path):
            if _matches_any_prefix(module, blocked_prefixes):
                failures.append(f"{path.relative_to(SRC_ROOT)} references {module}")
    return failures


def _unexpected_imports(paths: list[Path], allowed_prefixes: tuple[str, ...]) -> list[str]:
    failures: list[str] = []
    for path in paths:
        for module in sorted(_first_party_imports(path)):
            if not _matches_any_prefix(module, allowed_prefixes):
                failures.append(f"{path.relative_to(SRC_ROOT)} references {module}")
    return failures


def test_old_architecture_packages_and_root_modules_are_gone() -> None:
    assert not (SRC_ROOT / "adapters").exists()
    assert not (SRC_ROOT / "pipeline").exists()
    assert not (SRC_ROOT / "application").exists()
    assert not (SRC_ROOT / "domain" / "layout.py").exists()
    assert not (SRC_ROOT / "ocr" / "review_layout.py").exists()
    for filename in (
        "artifact_store.py",
        "filesystem.py",
        "layout_profile.py",
        "models.py",
        "normalize.py",
        "page_identity.py",
        "run_layout.py",
        "run_manifest.py",
        "storage.py",
        "text.py",
    ):
        assert not (SRC_ROOT / filename).exists()


def test_source_tests_and_tools_do_not_import_old_architecture_packages() -> None:
    roots = (REPO_ROOT / "src", REPO_ROOT / "tests", REPO_ROOT / "tools")
    failures: list[str] = []
    blocked_prefixes = (
        "my_ocr.adapters",
        "my_ocr.pipeline",
        "my_ocr.application",
        "my_ocr.domain.layout",
        "my_ocr.models",
        "my_ocr.storage",
        "my_ocr.normalize",
    )
    for root in roots:
        for path in root.rglob("*.py"):
            modules = _imported_modules(path) | _dynamic_import_strings(path)
            for module in modules:
                if module.startswith(blocked_prefixes):
                    failures.append(f"{path.relative_to(REPO_ROOT)} imports {module}")

    assert not failures


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
        ("my_ocr.runs.store.FilesystemRunStore",),
    )


def test_ui_does_not_import_runtime_stage_implementations_directly() -> None:
    assert not _violations(
        "ui",
        (
            "my_ocr.ingest",
            "my_ocr.ocr",
            "my_ocr.runs.store",
        ),
    )


def test_ui_imports_only_through_boundary_packages() -> None:
    assert not _unexpected_imports(
        _py_files("ui"),
        (
            "my_ocr.bootstrap",
            "my_ocr.domain",
            "my_ocr.ui",
            "my_ocr.workflow",
        ),
    )


def test_ocr_and_extraction_do_not_import_ui_or_run_store_details() -> None:
    failures = [
        *_violations("ocr", ("my_ocr.ui", "my_ocr.runs.store")),
        *_violations("extraction", ("my_ocr.ui", "my_ocr.runs.store")),
    ]

    assert not failures


def test_workflow_imports_only_application_boundary_packages() -> None:
    assert not _unexpected_imports(
        [SRC_ROOT / "workflow.py"],
        (
            "my_ocr.domain",
            "my_ocr.extraction.validation",
        ),
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
    workflow_path = SRC_ROOT / "workflow.py"
    storage_path = SRC_ROOT / "runs" / "store.py"
    review_controller_path = SRC_ROOT / "ui" / "review_controller.py"
    roots = (
        SRC_ROOT,
    )
    failures = [
        f"{path.relative_to(SRC_ROOT)}:{line}"
        for path, line in _attribute_call_sites(roots, "write_review_layout")
        if path not in {workflow_path, storage_path, review_controller_path}
    ]

    assert not failures


def test_workflow_defines_storage_boundary_without_importing_filesystem_store() -> None:
    workflow_path = SRC_ROOT / "workflow.py"
    modules = _imported_modules(workflow_path) | _dynamic_import_strings(workflow_path)

    assert "my_ocr.runs.store" not in modules
    assert "RunRepository" in _defined_class_names(workflow_path)


def test_app_state_does_not_persist_review_layout_directly() -> None:
    state_path = SRC_ROOT / "ui" / "state.py"
    defined = _defined_function_names(state_path)
    calls = _attribute_call_lines(state_path, "save_review_layout")

    assert "save_review_layout" not in defined
    assert not calls


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


def test_structured_extraction_has_single_persistence_path() -> None:
    structured_path = SRC_ROOT / "extraction" / "structured.py"
    storage_path = SRC_ROOT / "runs" / "store.py"

    assert "save_structured_result" not in _defined_function_names(structured_path)
    assert "write_structured_extraction" in _defined_function_names(storage_path)


def test_structured_validation_is_not_part_of_rules_extractor() -> None:
    rules_path = SRC_ROOT / "extraction" / "rules.py"
    validation_path = SRC_ROOT / "extraction" / "validation.py"

    assert "validate_structured_prediction" not in _defined_function_names(rules_path)
    assert "validate_structured_prediction" in _defined_function_names(validation_path)


def test_no_unsupported_old_run_path_remains() -> None:
    roots = (REPO_ROOT / "src",)
    failures: list[str] = []
    blocked = {
        "UnsupportedRunSchema",
        "unsupported_run_message",
        "UNSUPPORTED_V3_MESSAGE",
    }
    for root in roots:
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in blocked:
                if token in source:
                    failures.append(f"{path.relative_to(REPO_ROOT)} contains {token}")

    assert not failures


def test_runtime_options_do_not_keep_old_aliases() -> None:
    options_path = SRC_ROOT / "domain" / "options.py"
    source = options_path.read_text(encoding="utf-8")

    assert "class OcrRuntimeOptions" in source
    assert "LayoutOptions =" not in source
    assert "OcrOptions =" not in source


def test_run_store_does_not_keep_alias_or_page_rerun_methods() -> None:
    storage_path = SRC_ROOT / "runs" / "store.py"
    workflow_path = SRC_ROOT / "workflow.py"
    store_functions = _defined_function_names(storage_path)
    workflow_functions = _defined_function_names(workflow_path)
    blocked = {
        "write_review_layout",
        "write_ocr_result",
        "replace_page_layout",
        "replace_page_ocr",
    }

    assert not (blocked & store_functions)
    assert "RunStore =" not in workflow_path.read_text(encoding="utf-8")
    assert {"rerun_page_layout", "rerun_page_ocr"} <= workflow_functions


def test_normalization_exposes_only_page_ref_api() -> None:
    normalize_path = SRC_ROOT / "ingest" / "normalize.py"

    removed_functions = {
        "normalize_into_run",
        "render_pdf_into_images",
        "render_pdf_to_images",
    }
    assert not (removed_functions & _defined_function_names(normalize_path))
    assert "FilesystemDocumentNormalizer" not in _defined_class_names(normalize_path)


def test_source_tests_and_tools_do_not_call_removed_run_transaction_methods() -> None:
    roots = (REPO_ROOT / "src", REPO_ROOT / "tests", REPO_ROOT / "tools")
    failures = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for attribute in ("create_run", "write_pages", "commit_run", "rollback_run")
        for path, line in _attribute_call_sites(roots, attribute)
    ]

    assert not failures
