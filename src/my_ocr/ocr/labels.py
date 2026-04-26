from __future__ import annotations

TEXT_RECOGNITION_PROMPT = "Text Recognition:"
TABLE_RECOGNITION_PROMPT = "Table Recognition:"
FORMULA_RECOGNITION_PROMPT = "Formula Recognition:"

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
TABLE_LABELS = {"table"}
FORMULA_LABELS = {"formula", "display_formula", "inline_formula"}
OCR_LABELS = TEXT_LABELS | TABLE_LABELS | FORMULA_LABELS


def resolve_ocr_task(label: str) -> str:
    if label in TABLE_LABELS:
        return "table"
    if label in FORMULA_LABELS:
        return "formula"
    return "text"


def resolve_prompt_for_label(label: str) -> str:
    task = resolve_ocr_task(label)
    if task == "table":
        return TABLE_RECOGNITION_PROMPT
    if task == "formula":
        return FORMULA_RECOGNITION_PROMPT
    return TEXT_RECOGNITION_PROMPT
