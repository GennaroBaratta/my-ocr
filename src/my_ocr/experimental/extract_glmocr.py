from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from ..ollama_client import encode_image_file, post_json
from ..paths import RunPaths
from ..schema import JSON_SCHEMA, DocumentFields
from ..settings import (
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OLLAMA_KEEP_ALIVE,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_NUM_CTX,
    resolve_ocr_api_client,
)
from ..text_normalization import replace_html_tables
from ..utils import write_json

DEFAULT_MODEL = DEFAULT_OLLAMA_MODEL
RAW_BODY_METADATA_KEY = "_raw_body"


def extract_structured(
    image_paths: list[str],
    *,
    markdown_text: str | None = None,
    config_path: str | Path | None = None,
    model: str | None = None,
    endpoint: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if markdown_text is None and not image_paths:
        raise ValueError("image_paths cannot be empty")

    resolved_model = model
    resolved_endpoint = endpoint
    resolved_num_ctx = DEFAULT_OLLAMA_NUM_CTX
    if config_path is not None:
        config_resolution = resolve_ocr_api_client(config_path)
        config_model, config_endpoint, config_num_ctx = config_resolution
        if resolved_model is None:
            resolved_model = config_model
        if resolved_endpoint is None:
            resolved_endpoint = config_endpoint
        resolved_num_ctx = config_num_ctx

    if resolved_model is None:
        resolved_model = DEFAULT_MODEL
    if resolved_endpoint is None:
        resolved_endpoint = DEFAULT_OLLAMA_ENDPOINT

    body: dict[str, Any] = {}
    source = "page_images_first_page"
    input_length: int | None = None
    input_page_count = len(image_paths)
    used_page_count = 0
    image_paths_metadata: list[str] = []
    cleaned_markdown_text = _clean_structured_input_text(markdown_text or "")

    if _has_meaningful_markdown(cleaned_markdown_text):
        body = post_json(
            endpoint=resolved_endpoint,
            payload=_build_structured_markdown_payload(
                markdown_text or "", model=resolved_model, num_ctx=resolved_num_ctx
            ),
            error_prefix="Ollama request",
            opener=urlopen,
        )
        source = "ocr_markdown"
        input_length = len(cleaned_markdown_text)
        input_page_count = len(image_paths)
        used_page_count = 0
    else:
        request_image_paths = image_paths[:1]
        body = post_json(
            endpoint=resolved_endpoint,
            payload=_build_structured_image_payload(
                request_image_paths, model=resolved_model, num_ctx=resolved_num_ctx
            ),
            error_prefix="Ollama request",
            opener=urlopen,
        )
        source = "page_images_first_page"
        input_length = None
        input_page_count = len(image_paths)
        used_page_count = len(request_image_paths)
        image_paths_metadata = request_image_paths

    if "response" not in body:
        raise RuntimeError(f"Ollama response missing 'response' field: {body}")

    parsed = DocumentFields.from_mapping(_parse_response_json(body["response"])).to_dict()
    metadata = {
        "model": body.get("model", resolved_model),
        "created_at": body.get("created_at"),
        "total_duration": body.get("total_duration"),
        "load_duration": body.get("load_duration"),
        "prompt_eval_count": body.get("prompt_eval_count"),
        "eval_count": body.get("eval_count"),
        "done": body.get("done"),
        "source": source,
        "input_length": input_length,
        "image_paths": image_paths_metadata,
        "input_page_count": input_page_count,
        "used_page_count": used_page_count,
        RAW_BODY_METADATA_KEY: body,
    }
    return parsed, metadata


def save_structured_result(
    run_dir: str | Path, prediction: dict[str, Any], metadata: dict[str, Any]
) -> None:
    paths = RunPaths.from_run_dir(run_dir)
    metadata_payload = dict(metadata)
    raw_body = metadata_payload.pop(RAW_BODY_METADATA_KEY, None)
    write_json(paths.structured_prediction_path, prediction)
    write_json(paths.structured_metadata_path, metadata_payload)
    if raw_body is not None:
        write_json(paths.structured_raw_path, raw_body)


def build_structured_prompt() -> str:
    return (
        "Extract only values explicitly present in the OCR text or image. "
        "Do not infer, guess, normalize, or repair missing content. "
        "Return valid JSON only. "
        "Leave any field that is not explicitly present empty (empty string or empty list). "
        "If unsure, use an empty value."
    )


def _clean_structured_input_text(markdown_text: str) -> str:
    text = replace_html_tables(markdown_text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _build_structured_image_payload(
    image_paths: list[str], *, model: str, num_ctx: int
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": build_structured_prompt(),
        "images": [encode_image_file(Path(image_path)) for image_path in image_paths],
        "format": JSON_SCHEMA,
        "stream": False,
        "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
        "options": {"num_ctx": num_ctx},
    }


def _build_structured_markdown_payload(
    markdown_text: str, *, model: str, num_ctx: int = DEFAULT_OLLAMA_NUM_CTX
) -> dict[str, Any]:
    cleaned_text = _clean_structured_input_text(markdown_text)
    return {
        "model": model,
        "prompt": f"{build_structured_prompt()}\n\nOCR text:\n{cleaned_text}",
        "format": JSON_SCHEMA,
        "stream": False,
        "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
        "options": {"num_ctx": num_ctx},
    }


def _has_meaningful_markdown(markdown_text: str | None) -> bool:
    if markdown_text is None:
        return False
    return any(char.isalnum() for char in markdown_text)


def _parse_response_json(response_text: str) -> dict[str, Any]:
    candidate = response_text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`").strip()
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(f"Could not parse structured JSON response: {response_text[:400]}")
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Could not parse structured JSON response: {response_text[:400]}"
            ) from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(f"Structured response was not a JSON object: {parsed!r}")
    return parsed
