"""Allow running the UI with: python -m free_doc_extract.ui"""

from __future__ import annotations

import argparse

from . import main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the free-doc-extract Flet UI.")
    parser.add_argument("--web", action="store_true", help="Run the UI in browser/web mode.")
    parser.add_argument("--host", default=None, help="Host for web mode.")
    parser.add_argument("--port", default=0, type=int, help="Port for web mode.")
    return parser


def run() -> None:
    args = build_parser().parse_args()
    main(web=args.web, host=args.host, port=args.port)


run()
