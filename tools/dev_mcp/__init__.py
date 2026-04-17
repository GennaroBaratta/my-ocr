"""Dev-only MCP sidecar for the local UX feedback loop."""

from .app import create_app

__all__ = ["create_app"]
