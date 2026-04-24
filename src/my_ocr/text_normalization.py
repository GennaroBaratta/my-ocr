from __future__ import annotations

import re
from html.parser import HTMLParser


class _TableHtmlToTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        _ = attrs
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"}:
            self._current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._current_row is not None and self._current_cell is not None:
            cell_text = " ".join("".join(self._current_cell).split())
            self._current_row.append(cell_text)
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            if any(cell for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)


def normalize_table_html(text: str) -> str:
    parser = _TableHtmlToTextParser()
    parser.feed(text)
    parser.close()
    rows = [row for row in parser.rows if any(cell.strip() for cell in row)]
    return "\n".join(" | ".join(cell.strip() for cell in row) for row in rows).strip()


def replace_html_tables(text: str) -> str:
    if "<table" not in text.lower():
        return text
    return re.sub(
        r"<table[^>]*>.*?</table>",
        lambda match: normalize_table_html(match.group(0)),
        text,
        flags=re.I | re.S,
    )
