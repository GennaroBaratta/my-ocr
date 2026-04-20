"""Preflight layout-profile resolver.

Inspects the configured layout model's label set using lightweight metadata
loading only (no model instantiation) and selects the matching built-in
label profile before any layout recognition starts.

For ``auto`` selection, this resolver is advisory: if it cannot positively
identify a known specialization, it preserves the existing config/default
behavior and returns a warning in diagnostics instead of silently rewriting
the mappings or hard-failing.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_KNOWN_PROFILES = ("pp_doclayout_formula", "pp_doclayout_split_formula")


def _load_model_labels(model_dir: str) -> list[str]:
    """Load the model label list from metadata only."""
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError(
            "transformers library is required for preflight layout profile resolution. "
            "Install it with: pip install transformers"
        ) from exc

    try:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model metadata from '{model_dir}': {exc}") from exc

    id2label: dict[int, str] = getattr(model_config, "id2label", {})
    if not id2label:
        raise RuntimeError(
            f"No id2label found in model metadata for '{model_dir}'. "
            "Cannot determine the layout label set for profile selection."
        )
    return list(id2label.values())


def _detect_profile(labels: list[str]) -> str | None:
    if "formula" in labels:
        return "pp_doclayout_formula"
    if "display_formula" in labels or "inline_formula" in labels:
        return "pp_doclayout_split_formula"
    return None


def _build_overlay(
    effective_profile: str,
    base_task_mapping: dict[str, Any],
    base_vis_mapping: dict[str, Any],
) -> dict[str, Any]:
    task_mapping = copy.deepcopy(base_task_mapping)
    vis_mapping = copy.deepcopy(base_vis_mapping)

    if effective_profile == "pp_doclayout_formula":
        task_mapping["formula"] = ["formula"]
        vis_mapping["formula"] = ["formula"]
    else:
        task_mapping["formula"] = ["display_formula", "inline_formula"]
        vis_mapping["formula"] = ["display_formula", "inline_formula"]

    return {
        "pipeline.layout.label_task_mapping": task_mapping,
        "pipeline.result_formatter.label_visualization_mapping": vis_mapping,
    }


def _passthrough_diagnostics(
    *,
    requested: str,
    status: str,
    warning: str,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "layout_profile_requested": requested,
        "layout_profile_source": "auto",
        "layout_profile_status": status,
        "layout_profile_warning": warning,
    }
    if labels is not None:
        diagnostics["layout_model_labels"] = labels
    return diagnostics


def resolve_layout_profile(
    config_path: str | Path,
    layout_profile: str | None = "auto",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(_dotted_overrides, diagnostics)`` for layout-profile preflight."""
    requested_profile = "auto" if layout_profile in (None, "auto") else str(layout_profile)
    auto_mode = requested_profile == "auto"

    if not auto_mode and requested_profile not in _KNOWN_PROFILES:
        raise ValueError(
            f"Unknown layout profile '{requested_profile}'. Expected one of "
            f"{['auto', *_KNOWN_PROFILES]}."
        )

    config_path = Path(config_path)
    if not config_path.exists():
        if auto_mode:
            return {}, _passthrough_diagnostics(
                requested=requested_profile,
                status="passthrough_missing_config",
                warning=(
                    f"Layout preflight could not inspect '{config_path}' because the config file "
                    "does not exist. Proceeding with existing runtime defaults."
                ),
            )
        raise FileNotFoundError(f"Config not found: {config_path}")

    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Failed to parse config YAML '{config_path}': {exc}") from exc

    pipeline = payload.get("pipeline", {}) if isinstance(payload, dict) else {}
    layout_cfg = pipeline.get("layout", {}) if isinstance(pipeline, dict) else {}
    result_cfg = pipeline.get("result_formatter", {}) if isinstance(pipeline, dict) else {}
    model_dir = layout_cfg.get("model_dir") if isinstance(layout_cfg, dict) else None

    if not model_dir:
        if auto_mode:
            return {}, _passthrough_diagnostics(
                requested=requested_profile,
                status="passthrough_existing_config",
                warning=(
                    "Layout preflight could not inspect model labels because "
                    "pipeline.layout.model_dir is missing or empty. "
                    "Proceeding with existing config/default mappings."
                ),
            )
        raise ValueError(
            "pipeline.layout.model_dir is missing or empty in config. "
            f"Cannot validate explicit layout profile override. Config: {config_path}"
        )

    try:
        detected_labels = _load_model_labels(str(model_dir))
    except RuntimeError as exc:
        if auto_mode:
            return {}, _passthrough_diagnostics(
                requested=requested_profile,
                status="passthrough_unreadable_model_metadata",
                warning=(
                    f"Layout preflight could not read model metadata for '{model_dir}': {exc}. "
                    "Proceeding with existing config/default mappings."
                ),
            )
        raise

    detected_profile = _detect_profile(detected_labels)
    if auto_mode and detected_profile is None:
        return {}, _passthrough_diagnostics(
            requested=requested_profile,
            status="passthrough_existing_config",
            warning=(
                "Layout preflight did not find a built-in specialization for the detected "
                f"model labels {detected_labels}. Proceeding with existing config/default "
                "mappings."
            ),
            labels=detected_labels,
        )

    if not auto_mode:
        if detected_profile is None:
            raise RuntimeError(
                f"Preflight failed: Explicit layout profile override '{requested_profile}' "
                "cannot be validated because the model labels do not match any built-in "
                f"profile. Detected labels: {detected_labels}."
            )
        if requested_profile != detected_profile:
            raise RuntimeError(
                f"Preflight failed: Explicit layout profile override '{requested_profile}' "
                f"is not compatible with the model labels. Auto-detected profile: "
                f"'{detected_profile}'. Detected labels: {detected_labels}."
            )

    effective_profile = detected_profile if auto_mode else requested_profile
    base_task = (
        layout_cfg.get("label_task_mapping", {}) if isinstance(layout_cfg, dict) else {}
    )
    base_vis = (
        result_cfg.get("label_visualization_mapping", {})
        if isinstance(result_cfg, dict)
        else {}
    )
    dotted = _build_overlay(effective_profile, base_task, base_vis)
    diagnostics: dict[str, Any] = {
        "layout_profile_requested": requested_profile,
        "layout_profile_selected": effective_profile,
        "layout_profile_source": "auto" if auto_mode else "override",
        "layout_profile_status": "applied",
        "layout_model_labels": detected_labels,
    }

    logger.info(
        "Layout profile '%s' selected (%s). Model labels: %s",
        effective_profile,
        diagnostics["layout_profile_source"],
        detected_labels,
    )

    return dotted, diagnostics
