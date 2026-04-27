from __future__ import annotations

import ast
from pathlib import Path

from my_ocr.cli import build_parser
from my_ocr.settings import SUPPORTED_INFERENCE_PROVIDERS, resolve_inference_provider_config


SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "my_ocr"
REPO_ROOT = SRC_ROOT.parents[1]
CONFIG_ROOT = REPO_ROOT / "config"
README_PATH = REPO_ROOT / "README.md"
LEGACY_UI_SCREENS_MODULE = "my_ocr.ui." + "screens"
REMOVED_CONFIG_PATH = "pipeline." + "ocr_api"
REMOVED_STRUCTURED_MODULE = "ollama" + "_structured"
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


def _class_methods(path: Path, class_name: str) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name: child
                for child in node.body
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
            }
    return {}


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        if parent is None:
            return None
        return f"{parent}.{node.attr}"
    return None


def _method_call_targets(method: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    targets: set[str] = set()
    for node in ast.walk(method):
        if isinstance(node, ast.Call):
            target = _dotted_name(node.func)
            if target:
                targets.add(target)
    return targets


def _protocol_class_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    protocol_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name = _dotted_name(base)
            if base_name in {"Protocol", "typing.Protocol"}:
                protocol_names.add(node.name)
    return protocol_names


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


def _constructor_calls(path: Path, name: str) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node
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


def _keyword_names(call: ast.Call) -> set[str]:
    return {keyword.arg for keyword in call.keywords if keyword.arg is not None}


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


def test_config_samples_use_clean_break_inference_surface() -> None:
    config_paths = sorted(CONFIG_ROOT.glob("*.yml")) + sorted(CONFIG_ROOT.glob("*.yaml"))
    assert config_paths

    for path in config_paths:
        source = path.read_text(encoding="utf-8")
        resolved = resolve_inference_provider_config(path)
        assert REMOVED_CONFIG_PATH not in source
        assert resolved.provider in SUPPORTED_INFERENCE_PROVIDERS
        assert resolved.base_url
        assert resolved.model
        assert isinstance(resolved.extra, dict)
        if resolved.provider == "ollama":
            assert resolved.num_ctx


def test_readme_documents_current_supported_surfaces_only() -> None:
    readme = README_PATH.read_text(encoding="utf-8")

    assert "pipeline.inference" in readme
    assert REMOVED_CONFIG_PATH not in readme
    assert REMOVED_STRUCTURED_MODULE not in readme
    assert LEGACY_UI_SCREENS_MODULE not in readme
    assert "does not launch or manage vLLM" in readme
    assert "No artifact schemas changed" in readme


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
                if _is_module_or_child(module, LEGACY_UI_SCREENS_MODULE):
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
        module.startswith(f"{LEGACY_UI_SCREENS_MODULE}.")
        for module in imports
        if module != LEGACY_UI_SCREENS_MODULE
    )


def test_legacy_ui_screen_modules_are_removed() -> None:
    screen_dir = SRC_ROOT / "ui" / "screens"
    assert not list(screen_dir.glob("*.py"))


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


def test_provider_clients_are_isolated_behind_inference_adapters() -> None:
    bootstrap_imports = _first_party_imports(SRC_ROOT / "bootstrap.py")
    settings_source = (SRC_ROOT / "settings.py").read_text(encoding="utf-8")

    assert "my_ocr.inference.OllamaClient" in bootstrap_imports
    assert "my_ocr.inference.OpenAICompatibleClient" in bootstrap_imports
    assert "SUPPORTED_INFERENCE_PROVIDERS" in settings_source
    assert "ollama" in settings_source
    assert "openai_compatible" in settings_source
    assert not _violations(
        "inference",
        ("my_ocr.ocr", "my_ocr.extraction", "my_ocr.ui", "my_ocr.runs"),
    )


def test_default_provider_tests_use_fake_transports_not_live_servers() -> None:
    failures: list[str] = []
    for path in sorted((REPO_ROOT / "tests").glob("test_*.py")):
        for client_name in ("OllamaClient", "OpenAICompatibleClient"):
            for call in _constructor_calls(path, client_name):
                if "opener" not in _keyword_names(call):
                    failures.append(f"{path.relative_to(REPO_ROOT)}:{call.lineno} constructs {client_name}")

    assert not failures


def test_ocr_source_selection_stays_in_policy_facade() -> None:
    policy_path = SRC_ROOT / "ocr" / "ocr_policy.py"
    failures: list[str] = []

    assert "plan_page_ocr" in _defined_function_names(policy_path)
    assert "reconstruct_markdown_from_layout" not in _defined_function_names(policy_path)
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


def test_document_workflow_public_methods_delegate_to_focused_use_cases() -> None:
    workflow_path = SRC_ROOT / "workflow.py"
    methods = _class_methods(workflow_path, "DocumentWorkflow")
    expected_delegates = {
        "prepare_review": {"self._review.prepare_review"},
        "run_reviewed_ocr": {"self._ocr.run_reviewed_ocr"},
        "save_review_layout": {"self._review.save_review_layout"},
        "rerun_page_layout": {"self._review.rerun_page_layout"},
        "rerun_page_ocr": {"self._ocr.rerun_page_ocr"},
        "extract_rules": {"self._extraction.extract_rules"},
        "extract_structured": {"self._extraction.extract_structured"},
        "run_automatic": {
            "self.prepare_review",
            "self.run_reviewed_ocr",
            "self.extract_rules",
        },
    }

    assert expected_delegates.keys() <= methods.keys()
    for method_name, required_targets in expected_delegates.items():
        assert required_targets <= _method_call_targets(methods[method_name])


def test_workflow_and_use_case_protocols_are_centralized_in_ports() -> None:
    workflow_path = SRC_ROOT / "workflow.py"
    ports_path = SRC_ROOT / "use_cases" / "ports.py"
    checked_paths = [workflow_path, *_py_files("use_cases")]
    misplaced_protocols = {
        path.relative_to(SRC_ROOT): sorted(_protocol_class_names(path))
        for path in checked_paths
        if path != ports_path and _protocol_class_names(path)
    }

    assert not misplaced_protocols
    assert {
        "DocumentNormalizer",
        "LayoutDetector",
        "OcrEngine",
        "RulesExtractor",
        "RunRepository",
        "RunWorkspace",
        "StructuredExtractor",
    } <= _protocol_class_names(ports_path)


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
            "my_ocr.inference",
            "my_ocr.ocr",
            "my_ocr.runs",
            "my_ocr.settings",
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


def test_structured_extraction_uses_provider_neutral_inference_boundary() -> None:
    structured_source = (SRC_ROOT / "extraction" / "structured.py").read_text(encoding="utf-8")

    assert "InferenceRequest" in structured_source
    assert "StructuredOutputRequest" in structured_source
    assert "request_structured_response" not in structured_source
    assert "post_json" not in structured_source


def test_source_tests_and_tools_do_not_call_removed_run_transaction_methods() -> None:
    roots = (REPO_ROOT / "src", REPO_ROOT / "tests", REPO_ROOT / "tools")
    failures = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for attribute in ("create_run", "write_pages", "commit_run", "rollback_run")
        for path, line in _attribute_call_sites(roots, attribute)
    ]

    assert not failures
