from __future__ import annotations

from pathlib import Path
from typing import Any

from my_ocr.settings import (
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OLLAMA_KEEP_ALIVE,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_NUM_CTX,
)
from my_ocr.inference.ollama import encode_image_file, post_json
from my_ocr.ocr.ocr_policy import (
    TEXT_RECOGNITION_PROMPT,
    build_ocr_chunks,
    clean_recognized_text,
)
from my_ocr.support.filesystem import ensure_dir, write_text


def run_crop_fallback_for_page(
    *,
    page_path: str,
    page_json: Any,
    coord_space: str | None = None,
    page_fallback_dir: str | Path,
    model: str = DEFAULT_OLLAMA_MODEL,
    endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
    num_ctx: int = DEFAULT_OLLAMA_NUM_CTX,
) -> tuple[str, list[dict[str, Any]]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Crop-based OCR fallback requires Pillow.") from exc

    page_fallback_dir = ensure_dir(page_fallback_dir)
    with Image.open(page_path) as image:
        width, height = image.size
        chunks = build_ocr_chunks(page_json, width=width, height=height, coord_space=coord_space)

        recognized_chunks: list[dict[str, Any]] = []
        for chunk_index, chunk in enumerate(chunks, start=1):
            crop_path = page_fallback_dir / f"chunk-{chunk_index:04d}.png"
            text_path = page_fallback_dir / f"chunk-{chunk_index:04d}.txt"
            cropped = image.crop(tuple(chunk["bbox"]))
            cropped.save(crop_path)

            try:
                recognized_text = recognize_text_image(
                    crop_path,
                    prompt=chunk["prompt"],
                    model=model,
                    endpoint=endpoint,
                    num_ctx=num_ctx,
                )
            except RuntimeError as exc:
                recognized_text = ""
                error = str(exc)
            else:
                error = ""

            recognized_text = clean_recognized_text(recognized_text)
            write_text(text_path, recognized_text)
            recognized_chunks.append(
                {
                    "chunk": chunk_index,
                    "bbox": chunk["bbox"],
                    "labels": chunk["labels"],
                    "task": chunk["task"],
                    "prompt": chunk["prompt"],
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


def recognize_full_page(
    page_path: str | Path,
    *,
    model: str = DEFAULT_OLLAMA_MODEL,
    endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
    num_ctx: int = DEFAULT_OLLAMA_NUM_CTX,
) -> str:
    return recognize_text_image(page_path, model=model, endpoint=endpoint, num_ctx=num_ctx)


def recognize_text_image(
    image_path: str | Path,
    *,
    prompt: str = TEXT_RECOGNITION_PROMPT,
    model: str = DEFAULT_OLLAMA_MODEL,
    endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
    num_ctx: int = DEFAULT_OLLAMA_NUM_CTX,
) -> str:
    body = post_json(
        endpoint=endpoint,
        payload={
            "model": model,
            "prompt": prompt,
            "images": [encode_image_file(image_path)],
            "stream": False,
            "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
            "options": {"num_ctx": num_ctx},
        },
        error_prefix="Ollama crop OCR",
    )
    return clean_recognized_text(str(body.get("response", "")))
