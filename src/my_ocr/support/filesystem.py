from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    _atomic_write_text(target, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


load_json = read_json


def write_text(path: str | Path, content: str) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    _atomic_write_text(target, content)


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _atomic_write_text(target: Path, content: str) -> None:
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        temp_path.replace(target)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
