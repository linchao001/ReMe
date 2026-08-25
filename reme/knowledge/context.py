"""Runtime context for shared knowledge-base operations."""

from __future__ import annotations

import contextvars
from pathlib import Path

from .store import resolve_knowledge_bases_dir

_knowledge_bases_dir: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "knowledge_bases_dir",
    default=None,
)


def set_knowledge_bases_dir(value: str | Path | None) -> contextvars.Token:
    """Bind the active shared-KB root for the current async/task context."""
    resolved = resolve_knowledge_bases_dir(str(value) if value else None)
    return _knowledge_bases_dir.set(resolved)


def reset_knowledge_bases_dir(token: contextvars.Token) -> None:
    _knowledge_bases_dir.reset(token)


def get_knowledge_bases_dir() -> Path:
    current = _knowledge_bases_dir.get()
    if current is not None:
        return current
    return resolve_knowledge_bases_dir(None)
