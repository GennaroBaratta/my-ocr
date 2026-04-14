from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from .ollama_client import encode_image_file, post_json
from .paths import RunPaths
from .schema import JSON_SCHEMA, DocumentFields
from .settings import (
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OLLAMA_KEEP_ALIVE,
    DEFAULT_OLLAMA_MODEL,
)
from .utils import write_json

DEFAULT_MODEL = DEFAULT_OLLAMA_MODEL


def extract_structured(
    image_paths: list[str],
    *,
    model: str = DEFAULT_MODEL,
    endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not image_paths:
        raise ValueError("image_paths cannot be empty")

    body = post_json(
        endpoint=endpoint,
        payload=_build_structured_payload(image_paths, model=model),
        error_prefix="Ollama request",
        opener=urlopen,
    )

    if "response" not in body:
        raise RuntimeError(f"Ollama response missing 'response' field: {body}")

    parsed = DocumentFields.from_mapping(_parse_response_json(body["response"])).to_dict()
    metadata = {
        "model": body.get("model", model),
        "created_at": body.get("created_at"),
        "total_duration": body.get("total_duration"),
        "load_duration": body.get("load_duration"),
        "prompt_eval_count": body.get("prompt_eval_count"),
        "eval_count": body.get("eval_count"),
        "done": body.get("done"),
    }
    return parsed, metadata


def save_structured_result(
    run_dir: str | Path, prediction: dict[str, Any], metadata: dict[str, Any]
) -> None:
    paths = RunPaths.from_run_dir(run_dir)
    write_json(paths.structured_prediction_path, prediction)
    write_json(paths.structured_metadata_path, metadata)


def build_structured_prompt() -> str:
    return (
        "Extract the requested fields from this document. "
        "Return valid JSON only. "
        "If a field is unknown, return an empty string or empty list. "
        f"Use this schema exactly: {json.dumps(JSON_SCHEMA, ensure_ascii=False)}"
    )


def _build_structured_payload(image_paths: list[str], *, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": build_structured_prompt(),
        "images": [encode_image_file(Path(image_path)) for image_path in image_paths],
        "format": JSON_SCHEMA,
        "stream": False,
        "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
    }


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
