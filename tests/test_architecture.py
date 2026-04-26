from __future__ import annotations

import ast
from pathlib import Path

from my_ocr.cli import build_parser


SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "my_ocr"
REPO_ROOT = SRC_ROOT.parents[1]
USE_CASE_EXPORTS = {
    "DocumentNormalizer",
    "ExtractionUseCase",
    "LayoutDetector",
    "OcrEngine",
    "OcrUseCase",
    "ReviewUseCase",
    "RulesExtractor",
    "RunRepository",
    "StructuredExtractor",
}


def _py_files(package: str) -> list[Path]:
    return sorted((SRC_ROOT / package).rglob("*.py"))


def _module_name_for_path(path: Path) -> str | None:
    try:
        relative = path.relative_to(SRC_ROOT).with_suffix("")
    except ValueError:
        return None
    parts = ("my_ocr", *relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_import_from_module(path: Path, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module

    current_module = _module_name_for_path(path)
    if current_module is None:
        return None
    current_package = current_module if path.name == "__init__.py" else current_module.rsplit(".", 1)[0]
    package_parts = current_package.split(".")
    parent_count = node.level - 1
    if parent_count:
        package_parts = package_parts[:-parent_count]
    if not package_parts:
        return node.module
    if node.module:
        package_parts.extend(node.module.split("."))
    return ".".join(package_parts)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_import_from_module(path, node)
            if module:
                exported_use_cases = {
                    alias.name for alias in node.names if alias.name in USE_CASE_EXPORTS
                }
                imports_use_cases_package = any(alias.name == "use_cases" for alias in node.names)
                if module == "my_ocr" and (exported_use_cases or imports_use_cases_package):
                    modules.add("my_ocr.use_cases")
                    modules.update(f"my_ocr.use_cases.{name}" for name in exported_use_cases)
                    modules.update(
                        f"{module}.{alias.name}"
                        for alias in node.names
                        if alias.name not in exported_use_cases and alias.name != "use_cases"
                    )
                else:
                    modules.add(module)
                    modules.update(f"{module}.{alias.name}" for alias in node.names)
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


def _target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in target.elts:
            names.update(_target_names(element))
        return names
    return set()


def _assignment_lines(path: Path, name: str) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(name in _target_names(target) for target in node.targets):
                lines.append(node.lineno)
        elif isinstance(node, ast.AnnAssign):
            if name in _target_names(node.target):
                lines.append(node.lineno)
        elif isinstance(node, ast.NamedExpr):
            if name in _target_names(node.target):
                lines.append(node.lineno)
    return lines


def _session_field_assignment_lines(path: Path, field_names: set[str]) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets.extend(node.targets)
            line_number = node.lineno
        elif isinstance(node, ast.AnnAssign):
            targets.append(node.target)
            line_number = node.lineno
        elif isinstance(node, ast.AugAssign):
            targets.append(node.target)
            line_number = node.lineno
        else:
            continue
        for target in targets:
            if _is_session_field_target(target, field_names):
                lines.append(line_number)
    return lines


def _is_session_field_target(target: ast.expr, field_names: set[str]) -> bool:
    return (
        isinstance(target, ast.Attribute)
        and target.attr in field_names
        and isinstance(target.value, ast.Attribute)
        and target.value.attr == "session"
    )


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _constructor_call_lines(path: Path, name: str) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node.func) == name
    ]


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


def test_product_boundary_packages_are_present_without_enterprise_layers() -> None:
    for package in (
        "ui",
        "use_cases",
        "runs",
        "ocr",
        "extraction",
        "inference",
    ):
        assert (SRC_ROOT / package / "__init__.py").exists()

    for package in ("adapters", "pipeline", "application"):
        assert not (SRC_ROOT / package).exists()


def test_cli_subcommands_parse_without_running_workflows() -> None:
    parser = build_parser()
    command_args = {
        "prepare-review": ["prepare-review", "input.pdf"],
        "run-reviewed-ocr": ["run-reviewed-ocr", "--run", "demo"],
        "ocr": ["ocr", "input.pdf"],
        "extract-rules": ["extract-rules", "--run", "demo"],
        "extract-glmocr": ["extract-glmocr", "--run", "demo"],
        "eval": [
            "eval",
            "--gold-dir",
            "gold",
            "--pred-dir",
            "pred",
            "--output",
            "report.md",
        ],
        "run": ["run", "input.pdf"],
    }

    parsed_commands = {
        command: parser.parse_args(args).command for command, args in command_args.items()
    }

    assert parsed_commands == {command: command for command in command_args}


def test_source_tests_and_tools_do_not_import_forbidden_architecture_packages() -> None:
    roots = (REPO_ROOT / "src", REPO_ROOT / "tests", REPO_ROOT / "tools")
    failures: list[str] = []
    blocked_prefixes = (
        "my_ocr.adapters",
        "my_ocr.pipeline",
        "my_ocr.application",
    )
    for root in roots:
        for path in root.rglob("*.py"):
            modules = _imported_modules(path) | _dynamic_import_strings(path)
            for module in modules:
                if module.startswith(blocked_prefixes):
                    failures.append(f"{path.relative_to(REPO_ROOT)} imports {module}")

    assert not failures


def test_ui_does_not_import_runtime_stage_implementations_directly() -> None:
    assert not _violations(
        "ui",
        (
            "my_ocr.ingest",
            "my_ocr.ocr",
            "my_ocr.extraction",
            "my_ocr.inference",
            "my_ocr.runs",
            "my_ocr.runs.store",
            "my_ocr.use_cases",
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


def test_ui_feature_modules_import_only_feature_local_or_shared_ui_modules() -> None:
    failures: list[str] = []
    for feature_dir in sorted((SRC_ROOT / "ui" / "features").iterdir()):
        if not feature_dir.is_dir() or feature_dir.name == "__pycache__":
            continue
        feature_prefix = f"my_ocr.ui.features.{feature_dir.name}"
        for path in sorted(feature_dir.rglob("*.py")):
            allowed = (
                "my_ocr.bootstrap",
                "my_ocr.domain",
                "my_ocr.ui",
                "my_ocr.workflow",
            )
            for module in sorted(_first_party_imports(path)):
                if _is_module_or_child(module, "my_ocr.ui.screens"):
                    failures.append(
                        f"{path.relative_to(SRC_ROOT)} references legacy screen shim {module}"
                    )
                if not _matches_any_prefix(module, allowed):
                    failures.append(
                        f"{path.relative_to(SRC_ROOT)} references {module}; "
                        f"expected {feature_prefix} or shared UI/boundary modules"
                    )

    assert not failures


def test_ui_feature_folders_exist_for_workflow_screens() -> None:
    for feature in ("upload", "review", "results"):
        feature_dir = SRC_ROOT / "ui" / "features" / feature
        assert (feature_dir / "__init__.py").exists()
        assert (feature_dir / "screen.py").exists()

    assert (SRC_ROOT / "ui" / "features" / "review" / "actions.py").exists()
    assert (SRC_ROOT / "ui" / "features" / "results" / "actions.py").exists()
    assert (SRC_ROOT / "ui" / "features" / "results" / "toolbar.py").exists()


def test_app_imports_feature_screen_builders_instead_of_screen_modules() -> None:
    app_path = SRC_ROOT / "ui" / "app.py"
    imports = _first_party_imports(app_path)

    assert "my_ocr.ui.features.upload.build_upload_view" in imports
    assert "my_ocr.ui.features.review.build_review_view" in imports
    assert "my_ocr.ui.features.results.build_results_view" in imports
    assert not any(
        module.startswith("my_ocr.ui.screens.")
        for module in imports
        if module != "my_ocr.ui.screens"
    )


def test_legacy_ui_screen_modules_are_compatibility_shims() -> None:
    screen_dir = SRC_ROOT / "ui" / "screens"
    failures: list[str] = []
    for path in sorted(screen_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) for node in ast.walk(tree)):
            failures.append(f"{path.relative_to(SRC_ROOT)} defines layout/action logic")
        first_party = _first_party_imports(path)
        if not first_party or not all(
            module.startswith("my_ocr.ui.features.") for module in first_party
        ):
            failures.append(f"{path.relative_to(SRC_ROOT)} imports non-feature modules: {sorted(first_party)}")

    assert not failures


def test_ocr_and_extraction_do_not_import_ui_or_run_store_details() -> None:
    failures = [
        *_violations("ocr", ("my_ocr.ui", "my_ocr.runs.store")),
        *_violations("extraction", ("my_ocr.ui", "my_ocr.runs.store")),
    ]

    assert not failures


def test_inference_does_not_import_runtime_packages() -> None:
    assert (SRC_ROOT / "inference").exists()
    assert not _violations(
        "inference",
        (
            "my_ocr.ocr",
            "my_ocr.extraction",
            "my_ocr.ui",
            "my_ocr.runs",
        ),
    )


def test_ocr_and_extraction_may_depend_on_inference_but_not_reverse() -> None:
    ocr_imports = {
        module
        for path in _py_files("ocr")
        for module in _first_party_imports(path)
    }
    extraction_imports = {
        module
        for path in _py_files("extraction")
        for module in _first_party_imports(path)
    }

    assert any(_is_module_or_child(module, "my_ocr.inference") for module in ocr_imports)
    assert any(_is_module_or_child(module, "my_ocr.inference") for module in extraction_imports)
    assert not _violations(
        "inference",
        ("my_ocr.ocr", "my_ocr.extraction", "my_ocr.ui", "my_ocr.runs"),
    )


def test_ocr_source_selection_stays_in_policy_facade() -> None:
    policy_path = SRC_ROOT / "ocr" / "ocr_policy.py"
    failures: list[str] = []

    assert "plan_page_ocr" in _defined_function_names(policy_path)
    for path in _py_files("ocr"):
        if path == policy_path:
            continue
        if "plan_page_ocr" in _defined_function_names(path):
            failures.append(f"{path.relative_to(SRC_ROOT)} defines plan_page_ocr")
        failures.extend(
            f"{path.relative_to(SRC_ROOT)}:{line} assigns primary_source"
            for line in _assignment_lines(path, "primary_source")
        )
        failures.extend(
            f"{path.relative_to(SRC_ROOT)}:{line} constructs PageOcrPlan"
            for line in _constructor_call_lines(path, "PageOcrPlan")
        )

    assert not failures


def test_workflow_imports_only_application_boundary_packages() -> None:
    workflow_path = SRC_ROOT / "workflow.py"
    workflow_imports = _first_party_imports(workflow_path)

    assert any(_is_module_or_child(module, "my_ocr.use_cases") for module in workflow_imports)
    assert not _unexpected_imports(
        [workflow_path],
        (
            "my_ocr.domain",
            "my_ocr.use_cases",
        ),
    )


def test_use_cases_do_not_import_runtime_or_ui_implementations() -> None:
    assert not _unexpected_imports(
        _py_files("use_cases"),
        (
            "my_ocr.domain",
            "my_ocr.extraction.canonical",
            "my_ocr.use_cases",
        ),
    )


def test_use_cases_do_not_import_concrete_filesystem_or_runtime_helpers() -> None:
    assert not _violations(
        "use_cases",
        (
            "my_ocr.ingest",
            "my_ocr.ocr",
            "my_ocr.runs",
            "my_ocr.support.filesystem",
            "my_ocr.ui",
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
    ports_path = SRC_ROOT / "use_cases" / "ports.py"
    modules = _imported_modules(workflow_path) | _dynamic_import_strings(workflow_path)

    assert "my_ocr.runs.store" not in modules
    assert "RunRepository" in _defined_class_names(ports_path)


def test_run_invalidation_policy_has_no_filesystem_dependencies() -> None:
    invalidation_path = SRC_ROOT / "runs" / "invalidation.py"
    blocked_imports = {
        "pathlib",
        "shutil",
        "tempfile",
        "my_ocr.runs.artifact_io",
        "my_ocr.runs.manifest",
    }
    failures = [
        module
        for module in sorted(_imported_modules(invalidation_path))
        if _matches_any_prefix(module, tuple(blocked_imports))
    ]

    assert invalidation_path.exists()
    assert not failures


def test_run_filesystem_mechanics_stay_under_runs_store() -> None:
    paths_with_filesystem_mechanics = [
        path.relative_to(SRC_ROOT)
        for package in ("use_cases", "workflow")
        for path in (
            _py_files(package) if (SRC_ROOT / package).is_dir() else [SRC_ROOT / f"{package}.py"]
        )
        for module in _imported_modules(path)
        if module in {"shutil", "tempfile"}
        or _is_module_or_child(module, "my_ocr.runs.artifact_io")
        or _is_module_or_child(module, "my_ocr.runs.artifacts")
        or _is_module_or_child(module, "my_ocr.runs.manifest")
    ]

    assert not paths_with_filesystem_mechanics


def test_app_state_does_not_persist_review_layout_directly() -> None:
    state_path = SRC_ROOT / "ui" / "state.py"
    defined = _defined_function_names(state_path)
    calls = _attribute_call_lines(state_path, "save_review_layout")

    assert "save_review_layout" not in defined
    assert not calls


def test_ui_page_and_selection_commands_own_screen_mutations() -> None:
    command_owned_fields = {
        "current_page_index",
        "is_adding_box",
        "selected_box_id",
    }
    paths = [
        SRC_ROOT / "ui" / "features" / "results" / "actions.py",
        SRC_ROOT / "ui" / "features" / "review" / "actions.py",
        SRC_ROOT / "ui" / "review_controller.py",
        SRC_ROOT / "ui" / "components" / "bbox_editor.py",
    ]
    failures = [
        f"{path.relative_to(SRC_ROOT)}:{line}"
        for path in paths
        for line in _session_field_assignment_lines(path, command_owned_fields)
    ]

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


def test_source_tests_and_tools_do_not_call_removed_run_transaction_methods() -> None:
    roots = (REPO_ROOT / "src", REPO_ROOT / "tests", REPO_ROOT / "tools")
    failures = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for attribute in ("create_run", "write_pages", "commit_run", "rollback_run")
        for path, line in _attribute_call_sites(roots, attribute)
    ]

    assert not failures
