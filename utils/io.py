from __future__ import annotations
from pathlib import Path


def safe_read_text(path: Path, max_bytes: int = 200_000) -> str:
    """
    Read file content safely and decode as UTF-8 (replace errors).
    Truncates to max_bytes to avoid large state/prompt payload.
    """
    try:
        data = path.read_bytes()
        if len(data) > max_bytes:
            data = data[:max_bytes]
        return data.decode("utf-8", errors="replace")
    except Exception as e:
        return f"<<FAILED_TO_READ_FILE path={path} error={e}>>"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
