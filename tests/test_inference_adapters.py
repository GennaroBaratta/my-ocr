from __future__ import annotations

from email.message import Message
import json
import socket
from io import BytesIO
from typing import Any
from urllib.error import HTTPError

import pytest

from my_ocr.inference import (
    InferenceImage,
    InferenceRequest,
    OllamaClient,
    OpenAICompatibleClient,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
    StructuredOutputRequest,
)
from my_ocr.inference.ollama import build_generate_payload
from my_ocr.inference.openai_compatible import build_chat_completions_payload


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self) -> bytes:
        if isinstance(self._payload, bytes):
            return self._payload
        return json.dumps(self._payload).encode("utf-8")


def test_openai_compatible_multimodal_payload_is_vllm_safe(tmp_path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"image-bytes")

    request = InferenceRequest(
        prompt="Read this page",
        images=(InferenceImage(path=image),),
        temperature=0.1,
        max_tokens=128,
        extra={"top_k": 5},
    )

    payload = build_chat_completions_payload(request, default_model="vision-model")

    assert payload["model"] == "vision-model"
    assert payload["temperature"] == 0.1
    assert payload["max_tokens"] == 128
    assert payload["top_k"] == 5
    assert "extra_body" not in payload
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Read this page"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "detail" not in content[1]["image_url"]
    assert "guided_json" not in json.dumps(payload)


def test_openai_compatible_defaults_fill_payload_and_request_values_override() -> None:
    payload = build_chat_completions_payload(
        InferenceRequest(
            prompt="Read this page",
            max_tokens=128,
            extra={"top_k": 7},
        ),
        default_model="vision-model",
        default_max_tokens=1024,
        default_extra={"top_k": 5, "repetition_penalty": 1.05},
    )

    assert payload["max_tokens"] == 128
    assert payload["top_k"] == 7
    assert payload["repetition_penalty"] == 1.05
    assert "extra_body" not in payload


def test_openai_compatible_schema_maps_to_response_format_and_structured_extra() -> None:
    schema = {"type": "object", "properties": {"title": {"type": "string"}}}

    payload = build_chat_completions_payload(
        InferenceRequest(
            prompt="Return JSON",
            structured_output=StructuredOutputRequest(
                schema=schema,
                name="doc",
                strict=True,
                extra={"structured_outputs": {"json": schema}},
            ),
        ),
        default_model="structured-model",
    )

    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "doc", "schema": schema, "strict": True},
    }
    assert payload["structured_outputs"] == {"json": schema}
    assert "extra_body" not in payload


@pytest.mark.parametrize("conflicting_key", ["model", "messages"])
def test_openai_compatible_rejects_extra_payload_collisions(conflicting_key: str) -> None:
    with pytest.raises(ValueError, match=rf"{conflicting_key}"):
        build_chat_completions_payload(
            InferenceRequest(prompt="bad", extra={conflicting_key: "override"}),
            default_model="model",
        )


def test_openai_compatible_rejects_removed_guided_fields() -> None:
    with pytest.raises(ValueError, match=r"guided_\*"):
        build_chat_completions_payload(
            InferenceRequest(prompt="bad", extra={"guided_json": {"type": "object"}}),
            default_model="model",
        )


def test_openai_compatible_client_parses_response_with_fake_transport() -> None:
    captured: dict[str, Any] = {}

    def opener(request, timeout):
        captured["endpoint"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            {"model": "m", "choices": [{"message": {"content": "recognized text"}}]}
        )

    client = OpenAICompatibleClient(
        endpoint="http://provider.test/v1/chat/completions",
        model="m",
        timeout=7,
        opener=opener,
    )

    response = client.generate(InferenceRequest(prompt="hello"))

    assert captured["endpoint"] == "http://provider.test/v1/chat/completions"
    assert captured["timeout"] == 7
    assert captured["body"]["messages"] == [
        {"role": "user", "content": "hello"},
    ]
    assert response.text == "recognized text"
    assert response.model == "m"


def test_openai_compatible_provider_error_maps_to_project_error() -> None:
    def opener(request, timeout):
        headers = Message()
        raise HTTPError(
            request.full_url,
            500,
            "server error",
            headers,
            BytesIO(b'{"error":"boom"}'),
        )

    client = OpenAICompatibleClient(
        endpoint="http://provider.test/v1/chat/completions",
        model="m",
        opener=opener,
    )

    with pytest.raises(ProviderRequestError, match="500"):
        client.generate(InferenceRequest(prompt="hello"))


def test_openai_compatible_timeout_maps_to_project_error() -> None:
    def opener(request, timeout):
        raise socket.timeout("slow")

    client = OpenAICompatibleClient(
        endpoint="http://provider.test/v1/chat/completions",
        model="m",
        opener=opener,
    )

    with pytest.raises(ProviderTimeoutError, match="timed out"):
        client.generate(InferenceRequest(prompt="hello"))


def test_openai_compatible_malformed_response_maps_to_project_error() -> None:
    client = OpenAICompatibleClient(
        endpoint="http://provider.test/v1/chat/completions",
        model="m",
        opener=lambda request, timeout: FakeResponse({"choices": []}),
    )

    with pytest.raises(ProviderResponseError, match="missing choices"):
        client.generate(InferenceRequest(prompt="hello"))


def test_openai_compatible_invalid_json_maps_to_project_error() -> None:
    client = OpenAICompatibleClient(
        endpoint="http://provider.test/v1/chat/completions",
        model="m",
        opener=lambda request, timeout: FakeResponse(b"not json"),
    )

    with pytest.raises(ProviderResponseError, match="invalid JSON"):
        client.generate(InferenceRequest(prompt="hello"))


def test_ollama_payload_uses_native_generate_shape_with_raw_base64(tmp_path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"ollama-image")
    schema = {"type": "object"}

    payload = build_generate_payload(
        InferenceRequest(
            prompt="Recognize",
            images=(InferenceImage(path=image),),
            json_schema=schema,
            temperature=0.2,
            max_tokens=64,
            extra={"keep_alive": "5m"},
        ),
        default_model="glm-ocr",
    )

    assert payload == {
        "model": "glm-ocr",
        "prompt": "Recognize",
        "stream": False,
        "images": ["b2xsYW1hLWltYWdl"],
        "format": schema,
        "options": {"temperature": 0.2, "num_predict": 64},
        "keep_alive": "5m",
    }
    assert "messages" not in payload


def test_ollama_defaults_fill_options_and_request_values_override() -> None:
    payload = build_generate_payload(
        InferenceRequest(
            prompt="Recognize",
            max_tokens=64,
            extra={"keep_alive": "1m", "options": {"temperature": 0.2}},
        ),
        default_model="glm-ocr",
        default_max_tokens=256,
        default_num_ctx=4096,
        default_extra={"keep_alive": "5m", "options": {"top_k": 5}},
    )

    assert payload == {
        "model": "glm-ocr",
        "prompt": "Recognize",
        "stream": False,
        "options": {
            "top_k": 5,
            "num_ctx": 4096,
            "num_predict": 64,
            "temperature": 0.2,
        },
        "keep_alive": "1m",
    }


@pytest.mark.parametrize(
    "default_extra, expected",
    [
        (
            {"model": "override"},
            "model",
        ),
        (
            {"prompt": "override"},
            "prompt",
        ),
        (
            {"images": ["override"]},
            "images",
        ),
        (
            {"stream": True},
            "stream",
        ),
        (
            {"format": "json"},
            "format",
        ),
        (
            {"options": "oops"},
            "options",
        ),
    ],
)
def test_ollama_defaults_reject_adapter_owned_extra_collisions(
    default_extra: dict[str, Any], expected: str
) -> None:
    with pytest.raises(ValueError, match=rf"Ollama extra fields cannot override .*{expected}"):
        build_generate_payload(
            InferenceRequest(prompt="Recognize"),
            default_model="glm-ocr",
            default_extra=default_extra,
        )


def test_ollama_request_extras_reject_adapter_owned_collisions_in_sorted_order() -> None:
    with pytest.raises(
        ValueError,
        match=r"Ollama extra fields cannot override adapter-owned fields: format, images, model, options, prompt, stream",
    ):
        build_generate_payload(
            InferenceRequest(
                prompt="Recognize",
                images=(InferenceImage(data=b"page-bytes"),),
                json_schema={"type": "object"},
                extra={
                    "model": "override",
                    "stream": True,
                    "prompt": "override",
                    "images": ["not allowed"],
                    "format": "json",
                    "options": "oops",
                },
            ),
            default_model="glm-ocr",
        )


def test_ollama_client_maps_missing_response_to_project_error() -> None:
    client = OllamaClient(
        endpoint="http://ollama.test/api/generate",
        model="glm",
        opener=lambda request, timeout: FakeResponse({"model": "glm"}),
    )

    with pytest.raises(ProviderResponseError, match="missing 'response'"):
        client.generate(InferenceRequest(prompt="hello"))


def test_ollama_client_timeout_remains_timeout_error() -> None:
    def opener(request, timeout):
        raise socket.timeout("slow")

    client = OllamaClient(
        endpoint="http://ollama.test/api/generate",
        model="glm",
        opener=opener,
    )

    with pytest.raises(ProviderTimeoutError, match="timed out"):
        client.generate(InferenceRequest(prompt="hello"))


def test_ollama_client_preserves_post_json_encoding_and_project_response() -> None:
    captured: dict[str, Any] = {}

    def opener(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"model": "glm", "response": "text"})

    client = OllamaClient(
        endpoint="http://ollama.test/api/generate",
        model="glm",
        opener=opener,
    )

    response = client.generate(InferenceRequest(prompt="hello"))

    assert captured["body"] == {"model": "glm", "prompt": "hello", "stream": False}
    assert response.text == "text"
    assert response.raw == {"model": "glm", "response": "text"}
