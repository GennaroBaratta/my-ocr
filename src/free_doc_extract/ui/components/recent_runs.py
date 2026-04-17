"""Recent runs list with status badges."""

from __future__ import annotations

import time
from pathlib import Path

import flet as ft

from .. import theme
from ..state import AppState


def build_recent_runs(page: ft.Page, state: AppState) -> ft.Column:
    if not state.recent_runs:
        return ft.Column()

    rows: list[ft.Control] = []
    for run in state.recent_runs[:10]:
        filename = Path(run["input_path"]).name if run["input_path"] else run["run_id"]
        mtime = run.get("mtime", 0)
        date_str = time.strftime("%b %d, %Y", time.localtime(mtime)) if mtime else ""
        status = run["status"]
        is_ok = status == "extracted"

        badge = ft.Container(
            content=ft.Text(
                "Extracted" if is_ok else "Pending",
                size=10,
                weight=ft.FontWeight.W_500,
                color=theme.SUCCESS if is_ok else theme.TEXT_MUTED,
            ),
            bgcolor=f"{theme.SUCCESS}20" if is_ok else f"{theme.TEXT_MUTED}15",
            padding=ft.padding.symmetric(horizontal=8, vertical=2),
            border_radius=4,
        )

        run_id = run["run_id"]
        row = ft.Container(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.DESCRIPTION_OUTLINED, size=18, color=theme.TEXT_MUTED),
                    ft.Text(
                        filename,
                        size=13,
                        color=theme.TEXT_PRIMARY,
                        expand=True,
                        max_lines=1,
                        overflow=ft.TextOverflow.ELLIPSIS,
                    ),
                    ft.Text(date_str, size=11, color=theme.TEXT_MUTED),
                    badge,
                ],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,
            ),
            padding=ft.padding.symmetric(horizontal=14, vertical=10),
            on_click=lambda e, rid=run_id: page.go(f"/results/{rid}"),
            ink=True,
            border=ft.border.only(bottom=ft.BorderSide(1, theme.BORDER)),
        )
        rows.append(row)

    return ft.Column(
        [
            ft.Text(
                "RECENT RUNS",
                size=11,
                weight=ft.FontWeight.W_600,
                color=theme.TEXT_MUTED,
                style=ft.TextStyle(letter_spacing=1.2),
            ),
            ft.Container(
                content=ft.Column(rows, spacing=0),
                border=ft.border.all(1, theme.BORDER),
                border_radius=8,
                clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            ),
        ],
        spacing=8,
    )
