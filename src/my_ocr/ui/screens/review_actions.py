"""Compatibility shim for review feature actions."""

from ..features.review.actions import (
    ReviewScreenActions,
    current_page_image_width,
    current_zoom_scale,
    fit_width_icon_color,
    review_page_label_text,
    start_redetect_layout,
    start_reviewed_ocr,
)

__all__ = [
    "ReviewScreenActions",
    "current_page_image_width",
    "current_zoom_scale",
    "fit_width_icon_color",
    "review_page_label_text",
    "start_redetect_layout",
    "start_reviewed_ocr",
]
