from __future__ import annotations

import json
from typing import Any


def parse_response_json(response_text: str) -> dict[str, Any]:
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


__all__ = ["parse_response_json"]
