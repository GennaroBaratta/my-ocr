from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_UI_HOST = "127.0.0.1"
DEFAULT_UI_PORT = 8550


@dataclass(frozen=True, slots=True)
class DevMcpConfig:
    repo_root: Path
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    ui_host: str = DEFAULT_UI_HOST
    ui_port: int = DEFAULT_UI_PORT
    ui_start_timeout_seconds: float = 20.0
    screenshot_timeout_seconds: float = 20.0
    screenshot_settle_delay_seconds: float = 1.0
    default_viewport_width: int = 1440
    default_viewport_height: int = 1024
    ui_command_override: tuple[str, ...] | None = None
    run_root: Path = field(init=False)
    runtime_root: Path = field(init=False)
    logs_root: Path = field(init=False)
    feedback_root: Path = field(init=False)
    ui_state_path: Path = field(init=False)
    ui_base_url: str = field(init=False)

    def __post_init__(self) -> None:
        repo_root = self.repo_root.resolve()
        object.__setattr__(self, "repo_root", repo_root)

        runtime_root = repo_root / ".dev-mcp"
        object.__setattr__(self, "runtime_root", runtime_root)
        object.__setattr__(self, "run_root", repo_root / "data" / "runs")
        object.__setattr__(self, "logs_root", runtime_root / "logs")
        object.__setattr__(self, "feedback_root", runtime_root / "feedback")
        object.__setattr__(self, "ui_state_path", runtime_root / "ui-process.json")
        object.__setattr__(self, "ui_base_url", f"http://{self.ui_host}:{self.ui_port}")

    def ensure_runtime_dirs(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)
        self.feedback_root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_repo_root(
        cls,
        repo_root: str | Path,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        ui_host: str = DEFAULT_UI_HOST,
        ui_port: int = DEFAULT_UI_PORT,
    ) -> "DevMcpConfig":
        return cls(
            repo_root=Path(repo_root), host=host, port=port, ui_host=ui_host, ui_port=ui_port
        )

    @property
    def ui_command(self) -> tuple[str, ...]:
        if self.ui_command_override is not None:
            return self.ui_command_override
        return (
            "uv",
            "run",
            "python",
            "-m",
            "free_doc_extract.ui",
            "--web",
            "--host",
            self.ui_host,
            "--port",
            str(self.ui_port),
        )
