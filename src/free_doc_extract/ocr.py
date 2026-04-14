from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .utils import ensure_dir, write_json, write_text

DEFAULT_OCR_MODEL = "glm-ocr:latest"
DEFAULT_OCR_ENDPOINT = "http://localhost:11434/api/generate"
TEXT_RECOGNITION_PROMPT = "Text Recognition: [img-0]"
TEXT_LABELS = {
    "abstract",
    "algorithm",
    "content",
    "doc_title",
    "figure_title",
    "paragraph_title",
    "reference_content",
    "text",
    "vertical_text",
    "vision_footnote",
    "seal",
    "formula_number",
}
BOX_PADDING_X = 8
BOX_PADDING_Y = 6


def run_ocr(
    page_paths: list[str],
    run_dir: str | Path,
    *,
    config_path: str = "config/local.yaml",
    layout_device: str = "cpu",
) -> dict[str, Any]:
    if not page_paths:
        raise ValueError("page_paths cannot be empty")

    try:
        from glmocr import GlmOcr
    except ImportError as exc:
        raise RuntimeError(
            "GLM-OCR is not installed. Install with `pip install -e .[glmocr]`."
        ) from exc

    run_dir = ensure_dir(run_dir)
    raw_dir = ensure_dir(run_dir / "ocr_raw")
    fallback_dir = ensure_dir(run_dir / "ocr_fallback")

    with GlmOcr(config_path=config_path, layout_device=layout_device) as parser:
        if len(page_paths) == 1:
            results = [parser.parse(page_paths[0])]
        else:
            parse_inputs: list[str | bytes | Path] = list(page_paths)
            results = parser.parse(parse_inputs)

        markdown_parts: list[str] = []
        json_parts: list[Any] = []
        fallback_pages: list[dict[str, Any]] = []
        for index, result in enumerate(results, start=1):
            page_raw_dir = ensure_dir(raw_dir / f"page-{index:04d}")
            result.save(output_dir=str(page_raw_dir))

            page_markdown = getattr(result, "markdown_result", "") or ""
            page_json = getattr(result, "json_result", {}) or {}
            page_layout_json = getattr(result, "raw_json_result", None) or page_json
            if _needs_crop_fallback(page_markdown, page_layout_json):
                try:
                    fallback_markdown, fallback_meta = _run_crop_fallback_for_page(
                        page_path=page_paths[index - 1],
                        page_json=page_layout_json,
                        page_fallback_dir=ensure_dir(fallback_dir / f"page-{index:04d}"),
                    )
                except (RuntimeError, ValueError, TypeError) as exc:
                    fallback_markdown = ""
                    fallback_meta = [{"chunk": 0, "text": "", "error": str(exc)}]

                if fallback_markdown:
                    page_markdown = fallback_markdown

                fallback_pages.append(
                    {
                        "page": index,
                        "page_path": page_paths[index - 1],
                        "recovered_text": bool(fallback_markdown),
                        "chunks": fallback_meta,
                    }
                )

            markdown_parts.append(page_markdown)
            json_parts.append(page_json)

        markdown = "\n\n---\n\n".join(part for part in markdown_parts if part)
        json_result = json_parts[0] if len(json_parts) == 1 else json_parts

    write_text(run_dir / "ocr.md", markdown)
    write_json(run_dir / "ocr.json", json_result)
    if fallback_pages:
        write_json(run_dir / "ocr_fallback.json", fallback_pages)

    return {
        "markdown": markdown,
        "json": json_result,
        "raw_dir": str(raw_dir),
        "fallback_used": bool(fallback_pages),
    }


def _needs_crop_fallback(markdown: str, page_json: Any) -> bool:
    blocks = _extract_layout_blocks(page_json)
    if not blocks:
        return False

    meaningful_block_text = [
        _clean_recognized_text(str(block.get("content", "")))
        for block in blocks
        if block.get("label") in TEXT_LABELS
    ]
    if any(meaningful_block_text):
        return False

    return not _has_meaningful_text(markdown)


def _run_crop_fallback_for_page(
    *,
    page_path: str,
    page_json: Any,
    page_fallback_dir: str | Path,
    model: str = DEFAULT_OCR_MODEL,
    endpoint: str = DEFAULT_OCR_ENDPOINT,
) -> tuple[str, list[dict[str, Any]]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Crop-based OCR fallback requires Pillow.") from exc

    page_fallback_dir = ensure_dir(page_fallback_dir)
    with Image.open(page_path) as image:
        width, height = image.size
        chunks = _build_text_chunks(page_json, width=width, height=height)

        recognized_chunks: list[dict[str, Any]] = []
        for chunk_index, chunk in enumerate(chunks, start=1):
            crop_path = page_fallback_dir / f"chunk-{chunk_index:04d}.png"
            text_path = page_fallback_dir / f"chunk-{chunk_index:04d}.txt"
            cropped = image.crop(tuple(chunk["bbox"]))
            cropped.save(crop_path)

            try:
                recognized_text = _recognize_text_crop(
                    crop_path,
                    model=model,
                    endpoint=endpoint,
                )
            except RuntimeError as exc:
                recognized_text = ""
                error = str(exc)
            else:
                error = ""

            write_text(text_path, recognized_text)
            recognized_chunks.append(
                {
                    "chunk": chunk_index,
                    "bbox": chunk["bbox"],
                    "labels": chunk["labels"],
                    "source_indices": chunk["source_indices"],
                    "crop_path": str(crop_path),
                    "text_path": str(text_path),
                    "text": recognized_text,
                    "error": error,
                }
            )

    page_markdown = "\n\n".join(
        chunk["text"].strip() for chunk in recognized_chunks if chunk["text"].strip()
    )
    return page_markdown, recognized_chunks


def _recognize_text_crop(
    crop_path: str | Path,
    *,
    model: str = DEFAULT_OCR_MODEL,
    endpoint: str = DEFAULT_OCR_ENDPOINT,
    timeout: int = 300,
) -> str:
    encoded = base64.b64encode(Path(crop_path).read_bytes()).decode("utf-8")
    payload = json.dumps(
        {
            "model": model,
            "prompt": TEXT_RECOGNITION_PROMPT,
            "images": [encoded],
            "stream": False,
            "keep_alive": "15m",
        }
    ).encode("utf-8")
    request = Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama crop OCR failed: {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {endpoint}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama crop OCR returned invalid JSON") from exc

    return _clean_recognized_text(str(body.get("response", "")))


def _build_text_chunks(page_json: Any, *, width: int, height: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for block in _extract_layout_blocks(page_json):
        bbox = _normalize_bbox(block.get("bbox_2d"), width, height)
        if block.get("label") in TEXT_LABELS and bbox is not None:
            chunks.append(
                {
                    "bbox": _pad_bbox(bbox, width, height),
                    "labels": [str(block.get("label", ""))],
                    "source_indices": [_safe_int(block.get("index"), -1)],
                    "unpadded_bbox": bbox,
                }
            )

    chunks.sort(key=lambda chunk: (chunk["unpadded_bbox"][1], chunk["unpadded_bbox"][0]))
    return chunks


def _extract_layout_blocks(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if "bbox_2d" in payload and "label" in payload:
            return [payload]
        blocks: list[dict[str, Any]] = []
        for value in payload.values():
            blocks.extend(_extract_layout_blocks(value))
        return blocks
    if isinstance(payload, list):
        blocks: list[dict[str, Any]] = []
        for item in payload:
            blocks.extend(_extract_layout_blocks(item))
        return blocks
    return []


def _normalize_bbox(raw_bbox: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in raw_bbox]
    except (TypeError, ValueError):
        return None

    if max(x1, y1, x2, y2) <= 1000:
        x1 = round(x1 * width / 1000)
        x2 = round(x2 * width / 1000)
        y1 = round(y1 * height / 1000)
        y2 = round(y2 * height / 1000)
    else:
        x1, y1, x2, y2 = [round(value) for value in (x1, y1, x2, y2)]

    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _pad_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    return [
        max(0, bbox[0] - BOX_PADDING_X),
        max(0, bbox[1] - BOX_PADDING_Y),
        min(width, bbox[2] + BOX_PADDING_X),
        min(height, bbox[3] + BOX_PADDING_Y),
    ]


def _clean_recognized_text(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    text = text.strip()
    if not _has_meaningful_text(text):
        return ""
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line.strip()).strip()


def _has_meaningful_text(value: str) -> bool:
    return any(char.isalnum() for char in value)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
