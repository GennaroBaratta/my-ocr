from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from my_ocr.inference import InferenceClient, InferenceImage, InferenceRequest
from my_ocr.ocr.bbox import build_ocr_chunks
from my_ocr.ocr.labels import TEXT_RECOGNITION_PROMPT
from my_ocr.ocr.text_cleanup import clean_recognized_text
from my_ocr.support.filesystem import ensure_dir, write_text


def run_crop_fallback_for_page(
    *,
    page_path: str,
    page_json: Any,
    coord_space: str | None = None,
    page_fallback_dir: str | Path,
    recognizer: InferenceClient,
    model: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    try:
        image_module = import_module("PIL.Image")
    except ImportError as exc:
        raise RuntimeError("Crop-based OCR fallback requires Pillow.") from exc

    page_fallback_dir = ensure_dir(page_fallback_dir)
    with image_module.open(page_path) as image:
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
                    recognizer=recognizer,
                    prompt=chunk["prompt"],
                    model=model,
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
    recognizer: InferenceClient,
    model: str | None = None,
) -> str:
    return recognize_text_image(page_path, recognizer=recognizer, model=model)


def recognize_text_image(
    image_path: str | Path,
    *,
    recognizer: InferenceClient,
    prompt: str = TEXT_RECOGNITION_PROMPT,
    model: str | None = None,
) -> str:
    response = recognizer.generate(
        InferenceRequest(
            prompt=prompt,
            model=model,
            images=(InferenceImage(path=image_path),),
        )
    )
    return clean_recognized_text(response.text)
