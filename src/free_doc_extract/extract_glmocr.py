from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .schema import JSON_SCHEMA, DocumentFields
from .utils import ensure_dir, write_json

DEFAULT_MODEL = "glm-ocr:latest"
DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"


def extract_structured(
    image_paths: list[str],
    *,
    model: str = DEFAULT_MODEL,
    endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
    keep_alive: str = "15m",
    timeout: int = 300,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not image_paths:
        raise ValueError("image_paths cannot be empty")

    images = [
        base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
        for image_path in image_paths
    ]
    prompt = build_structured_prompt()

    payload = {
        "model": model,
        "prompt": prompt,
        "images": images,
        "format": JSON_SCHEMA,
        "stream": False,
        "keep_alive": keep_alive,
    }

    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama request failed: {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {endpoint}") from exc

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
    predictions_dir = ensure_dir(Path(run_dir) / "predictions")
    write_json(predictions_dir / "glmocr_structured.json", prediction)
    write_json(predictions_dir / "glmocr_structured_meta.json", metadata)


def build_structured_prompt() -> str:
    return (
        "Extract the requested fields from this document. "
        "Return valid JSON only. "
        "If a field is unknown, return an empty string or empty list. "
        f"Use this schema exactly: {json.dumps(JSON_SCHEMA, ensure_ascii=False)}"
    )


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
