# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .app import build_mcp, create_app
from .config import DevMcpConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local dev-only MCP sidecar.")
    parser.add_argument(
        "--transport",
        choices=("http", "stdio"),
        default="http",
        help="MCP transport to serve. Use stdio for OpenCode local MCP entries.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Use 127.0.0.1 only.")
    parser.add_argument("--port", default=8765, type=int, help="Bind port for the MCP server.")
    parser.add_argument("--ui-host", default="127.0.0.1", help="Bind host for the Flet UI.")
    parser.add_argument("--ui-port", default=8550, type=int, help="Bind port for the Flet UI.")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root used to scope filesystem access.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.transport == "stdio":
        config = DevMcpConfig.from_repo_root(
            args.repo_root,
            host=args.host,
            port=args.port,
            ui_host=args.ui_host,
            ui_port=args.ui_port,
        )
        build_mcp(config).run("stdio")
        return
    app = create_app(
        args.repo_root,
        host=args.host,
        port=args.port,
        ui_host=args.ui_host,
        ui_port=args.ui_port,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
