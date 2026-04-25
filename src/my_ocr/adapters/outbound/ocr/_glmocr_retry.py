from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import gc
from importlib import import_module
from typing import Any


@dataclass(frozen=True, slots=True)
class _ParseOutcome:
    result: Any
    parser_retired: bool = False


def parse_page_with_cpu_fallback(
    parser: Any,
    page_path: str,
    *,
    method_name: str = "parse",
    parser_cls: type,
    config_path: str,
    layout_device: str,
    layout_dotted: dict[str, Any],
) -> _ParseOutcome:
    try:
        return _ParseOutcome(getattr(parser, method_name)(page_path))
    except Exception as exc:
        if not should_retry_parse_on_cpu(exc, layout_device):
            raise

    close = getattr(parser, "close", None)
    if callable(close):
        with suppress(Exception):
            close()

    cleanup_after_cuda_oom()
    with parser_cls(
        config_path=config_path,
        layout_device="cpu",
        _dotted=layout_dotted,
    ) as cpu_parser:
        return _ParseOutcome(
            getattr(cpu_parser, method_name)(page_path),
            parser_retired=True,
        )


def should_retry_parse_on_cpu(exc: Exception, layout_device: str) -> bool:
    return layout_device.startswith("cuda") and is_cuda_oom_error(exc)


def is_cuda_oom_error(exc: BaseException) -> bool:
    if exc.__class__.__name__ == "OutOfMemoryError":
        module_name = getattr(exc.__class__, "__module__", "")
        if module_name.startswith("torch"):
            return True

    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def cleanup_after_cuda_oom() -> None:
    gc.collect()
    try:
        torch_module = import_module("torch")
    except ImportError:
        return

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not hasattr(cuda, "empty_cache"):
        return
    cuda.empty_cache()
