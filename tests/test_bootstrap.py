from __future__ import annotations

from pathlib import Path
import json
from typing import Any, cast

from my_ocr.bootstrap import build_backend_services, build_inference_client
from my_ocr.domain import LayoutBlock, OcrRuntimeOptions, PageRef, ReviewLayout, ReviewPage
from my_ocr.inference import InferenceRequest, OllamaClient, OpenAICompatibleClient
from my_ocr.ocr import fallback as ocr_fallback
from my_ocr.ocr import glmocr_runtime
from my_ocr.settings import InferenceProviderConfig, resolve_inference_provider_config


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_build_inference_client_creates_ollama_client_without_network_call() -> None:
    config = InferenceProviderConfig(
        provider="ollama",
        model="glm",
        base_url="http://ollama.test",
        endpoint="http://ollama.test/api/generate",
        num_ctx=8192,
    )

    client = build_inference_client(config)

    assert isinstance(client, OllamaClient)
    assert client._endpoint == "http://ollama.test/api/generate"
    assert client._model == "glm"


def test_build_inference_client_passes_ollama_config_defaults_to_adapter_payload() -> None:
    captured: dict[str, Any] = {}

    def opener(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"model": "glm", "response": "text"})

    config = InferenceProviderConfig(
        provider="ollama",
        model="glm",
        base_url="http://ollama.test",
        endpoint="http://ollama.test/api/generate",
        max_tokens=256,
        num_ctx=4096,
        extra={"keep_alive": "5m", "options": {"top_k": 5}},
    )
    client = build_inference_client(config)
    cast(Any, client)._opener = opener

    client.generate(
        InferenceRequest(
            prompt="hello",
            extra={"keep_alive": "1m", "options": {"temperature": 0.2}},
        )
    )

    assert captured["body"] == {
        "model": "glm",
        "prompt": "hello",
        "stream": False,
        "options": {
            "top_k": 5,
            "num_ctx": 4096,
            "num_predict": 256,
            "temperature": 0.2,
        },
        "keep_alive": "1m",
    }


def test_build_inference_client_creates_openai_compatible_client_without_network_call() -> None:
    config = InferenceProviderConfig(
        provider="openai_compatible",
        model="vision-model",
        base_url="http://vllm.test/v1",
        endpoint="http://vllm.test/v1/chat/completions",
        api_key="secret",
        max_tokens=1024,
        extra={"top_k": 5},
    )

    client = build_inference_client(config)

    assert isinstance(client, OpenAICompatibleClient)
    assert client._endpoint == "http://vllm.test/v1/chat/completions"
    assert client._model == "vision-model"
    assert client._api_key == "secret"


def test_build_inference_client_passes_openai_config_defaults_to_adapter_payload() -> None:
    captured: dict[str, Any] = {}

    def opener(request, timeout):
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            {"model": "vision-model", "choices": [{"message": {"content": "text"}}]}
        )

    config = InferenceProviderConfig(
        provider="openai_compatible",
        model="vision-model",
        base_url="http://vllm.test/v1",
        endpoint="http://vllm.test/v1/chat/completions",
        api_key="secret",
        max_tokens=1024,
        extra={"top_k": 5, "repetition_penalty": 1.05},
    )
    client = build_inference_client(config)
    cast(Any, client)._opener = opener

    client.generate(InferenceRequest(prompt="hello", extra={"top_k": 7}))

    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["body"]["max_tokens"] == 1024
    assert captured["body"]["top_k"] == 7
    assert captured["body"]["repetition_penalty"] == 1.05
    assert "extra_body" not in captured["body"]


def test_backend_services_injects_configured_inference_client_into_runtime_paths(
    tmp_path,
) -> None:
    config_path = tmp_path / "local.yaml"
    config_path.write_text(
        "pipeline:\n"
        "  inference:\n"
        "    provider: openai_compatible\n"
        "    model: qwen2-vl\n"
        "    base_url: http://vllm.example:8000/v1\n"
        "    api_key: test-key\n"
        "    max_tokens: 1024\n"
        "    extra:\n"
        "      top_k: 5\n",
        encoding="utf-8",
    )

    services = build_backend_services(
        run_root=str(tmp_path / "runs"),
        config_path=str(config_path),
    )

    assert services.inference_config == resolve_inference_provider_config(config_path)
    assert isinstance(services.inference_client, OpenAICompatibleClient)
    ocr_engine = cast(Any, services.workflow._ocr._ocr_engine)
    structured_extractor = cast(Any, services.workflow._extraction._structured_extractor)

    assert ocr_engine._inference_client is services.inference_client
    assert ocr_engine._inference_config is services.inference_config
    assert structured_extractor._inference_client is services.inference_client
    assert structured_extractor._inference_config is services.inference_config


def test_bootstrap_ocr_engine_uses_configured_openai_model_with_fresh_runtime_options(
    tmp_path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "local.yaml"
    config_path.write_text(
        "pipeline:\n"
        "  inference:\n"
        "    provider: openai_compatible\n"
        "    model: qwen2-vl\n"
        "    base_url: http://vllm.example:8000/v1\n"
        "    api_key: test-key\n",
        encoding="utf-8",
    )
    services = build_backend_services(
        run_root=str(tmp_path / "runs"),
        config_path=str(config_path),
    )
    captured_recognizers: list[object] = []

    def fail_if_rebuilt(*_args, **_kwargs):
        raise AssertionError("fresh runtime options must reuse the configured client")

    def fake_crop_fallback(**kwargs):
        captured_recognizers.append(kwargs["recognizer"])
        return "recognized text", []

    monkeypatch.setattr(glmocr_runtime, "build_fallback_inference_client", fail_if_rebuilt)
    monkeypatch.setattr(ocr_fallback, "run_crop_fallback_for_page", fake_crop_fallback)
    ocr_engine = cast(Any, services.workflow._ocr._ocr_engine)

    result = ocr_engine.recognize(
        *_single_page_review_inputs(tmp_path),
        OcrRuntimeOptions(config_path=str(config_path), model=None, endpoint=None),
    )

    assert services.inference_config.model == "qwen2-vl"
    assert getattr(services.inference_client, "_model") == "qwen2-vl"
    assert captured_recognizers == [services.inference_client]
    assert result.result.markdown == "recognized text"


def test_bootstrap_ocr_engine_uses_configured_ollama_model_with_fresh_runtime_options(
    tmp_path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "local.yaml"
    config_path.write_text(
        "pipeline:\n"
        "  inference:\n"
        "    provider: ollama\n"
        "    model: custom-ocr:stable\n"
        "    base_url: http://ollama.example:11434\n",
        encoding="utf-8",
    )
    services = build_backend_services(
        run_root=str(tmp_path / "runs"),
        config_path=str(config_path),
    )
    captured_recognizers: list[object] = []

    def fail_if_rebuilt(*_args, **_kwargs):
        raise AssertionError("fresh runtime options must reuse the configured client")

    def fake_crop_fallback(**kwargs):
        captured_recognizers.append(kwargs["recognizer"])
        return "recognized text", []

    monkeypatch.setattr(glmocr_runtime, "build_fallback_inference_client", fail_if_rebuilt)
    monkeypatch.setattr(ocr_fallback, "run_crop_fallback_for_page", fake_crop_fallback)
    ocr_engine = cast(Any, services.workflow._ocr._ocr_engine)

    result = ocr_engine.recognize(
        *_single_page_review_inputs(tmp_path),
        OcrRuntimeOptions(config_path=str(config_path), model=None, endpoint=None),
    )

    assert services.inference_config.model == "custom-ocr:stable"
    assert getattr(services.inference_client, "_model") == "custom-ocr:stable"
    assert captured_recognizers == [services.inference_client]
    assert result.result.markdown == "recognized text"


def test_bootstrap_ocr_engine_rebuilds_client_for_endpoint_override_once(
    tmp_path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "local.yaml"
    config_path.write_text(
        "pipeline:\n"
        "  inference:\n"
        "    provider: openai_compatible\n"
        "    model: qwen2-vl-configured\n"
        "    base_url: http://vllm.example:8000/v1\n"
        "    api_key: test-key\n"
        "    max_tokens: 1024\n"
        "    extra:\n"
        "      top_k: 5\n",
        encoding="utf-8",
    )
    services = build_backend_services(
        run_root=str(tmp_path / "runs"),
        config_path=str(config_path),
    )
    built_clients: list[object] = []
    captured_recognizers: list[object] = []

    class FakeOpenAICompatibleClient:
        def __init__(
            self,
            *,
            endpoint: str,
            model: str,
            timeout: int,
            api_key: str | None,
            default_max_tokens: int | None,
            default_extra: dict[str, object],
        ) -> None:
            self.endpoint = endpoint
            self.model = model
            self.timeout = timeout
            self.api_key = api_key
            self.default_max_tokens = default_max_tokens
            self.default_extra = default_extra
            built_clients.append(self)

    def fake_crop_fallback(**kwargs):
        captured_recognizers.append(kwargs["recognizer"])
        return f"text for page {len(captured_recognizers)}", []

    monkeypatch.setattr(
        glmocr_runtime,
        "OpenAICompatibleClient",
        FakeOpenAICompatibleClient,
    )
    monkeypatch.setattr(ocr_fallback, "run_crop_fallback_for_page", fake_crop_fallback)
    page_one = tmp_path / "page-0001.png"
    page_two = tmp_path / "page-0002.png"
    page_one.write_bytes(b"image placeholder")
    page_two.write_bytes(b"image placeholder")
    pages = [
        PageRef(
            page_number=1,
            image_path="pages/page-0001.png",
            width=100,
            height=100,
            resolved_path=page_one,
        ),
        PageRef(
            page_number=2,
            image_path="pages/page-0002.png",
            width=100,
            height=100,
            resolved_path=page_two,
        ),
    ]
    review = ReviewLayout(
        pages=[
            ReviewPage(
                page_number=1,
                image_path="pages/page-0001.png",
                image_width=100,
                image_height=100,
                coord_space="pixel",
                blocks=[
                    LayoutBlock(
                        id="p1-b0",
                        index=0,
                        label="text",
                        content="",
                        confidence=1.0,
                        bbox=[1, 2, 10, 20],
                    )
                ],
            ),
            ReviewPage(
                page_number=2,
                image_path="pages/page-0002.png",
                image_width=100,
                image_height=100,
                coord_space="pixel",
                blocks=[
                    LayoutBlock(
                        id="p2-b0",
                        index=0,
                        label="text",
                        content="",
                        confidence=1.0,
                        bbox=[1, 2, 10, 20],
                    )
                ],
            ),
        ],
        status="reviewed",
    )
    ocr_engine = cast(Any, services.workflow._ocr._ocr_engine)

    result = ocr_engine.recognize(
        pages,
        Path(tmp_path / "runs" / "sample"),
        review,
        OcrRuntimeOptions(
            config_path=str(config_path),
            endpoint="http://override.example/v1/chat/completions",
        ),
    )

    assert len(built_clients) == 1
    rebuilt = built_clients[0]
    assert getattr(rebuilt, "endpoint") == "http://override.example/v1/chat/completions"
    assert getattr(rebuilt, "model") == "qwen2-vl-configured"
    assert getattr(rebuilt, "api_key") == "test-key"
    assert getattr(rebuilt, "default_max_tokens") == 1024
    assert getattr(rebuilt, "default_extra") == {"top_k": 5}
    assert captured_recognizers == [rebuilt, rebuilt]
    assert result.result.markdown == "text for page 1\n\ntext for page 2"


def _single_page_review_inputs(
    tmp_path: Path,
) -> tuple[list[PageRef], Path, ReviewLayout]:
    page_path = tmp_path / "page-0001.png"
    page_path.write_bytes(b"image placeholder")
    pages = [
        PageRef(
            page_number=1,
            image_path="pages/page-0001.png",
            width=100,
            height=100,
            resolved_path=page_path,
        )
    ]
    review = ReviewLayout(
        pages=[
            ReviewPage(
                page_number=1,
                image_path="pages/page-0001.png",
                image_width=100,
                image_height=100,
                coord_space="pixel",
                blocks=[
                    LayoutBlock(
                        id="p1-b0",
                        index=0,
                        label="text",
                        content="",
                        confidence=1.0,
                        bbox=[1, 2, 10, 20],
                    )
                ],
            )
        ],
        status="reviewed",
    )
    return pages, tmp_path / "runs" / "sample", review
