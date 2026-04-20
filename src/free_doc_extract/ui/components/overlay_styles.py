"""Shared overlay styling for review and results document boxes."""

from __future__ import annotations

from .inspector import LABEL_TO_BLOCK_TYPE
from .. import theme


_BOX_KIND_COLORS = {
    "Text Block": theme.BOX_TEXT_BLOCK,
    "Table": theme.BOX_TABLE,
    "Figure/Image": theme.BOX_FIGURE_IMAGE,
    "Header": theme.BOX_HEADER,
    "Title": theme.BOX_HEADER,
    "Formula": theme.BOX_FORMULA,
}

_LIGHTER_BLOCK_TYPES = {"Figure/Image", "Header", "Title"}

_SELECTED_FILL_ALPHA = {
    False: "1A",
    True: "12",
}
_UNSELECTED_BORDER_ALPHA = {
    False: "14",
    True: "0E",
}
_UNSELECTED_FILL_ALPHA = {
    False: "02",
    True: "01",
}


def overlay_colors_for_label(label: str, is_selected: bool) -> tuple[str, str]:
    """Return border and fill colors for a document box overlay."""
    block_type = LABEL_TO_BLOCK_TYPE.get(label, "Text Block")
    base_color = _BOX_KIND_COLORS.get(block_type, theme.BOX_TEXT_BLOCK)
    use_lighter_alpha = block_type in _LIGHTER_BLOCK_TYPES

    if is_selected:
        return base_color, f"{base_color}{_SELECTED_FILL_ALPHA[use_lighter_alpha]}"

    return (
        f"{base_color}{_UNSELECTED_BORDER_ALPHA[use_lighter_alpha]}",
        f"{base_color}{_UNSELECTED_FILL_ALPHA[use_lighter_alpha]}",
    )
