"""Mount a shared knowledge base into the ReMe workspace as ``knowledge/``."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from .store import ensure_kb, kb_root, validate_kb_id

logger = logging.getLogger(__name__)


class KnowledgeMountError(RuntimeError):
    """Raised when workspace→knowledge mount cannot be established."""


def _is_junction_or_symlink(path: Path) -> bool:
    if path.is_symlink():
        return True
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            GetFileAttributesW = ctypes.windll.kernel32.GetFileAttributesW
            GetFileAttributesW.argtypes = [wintypes.LPCWSTR]
            GetFileAttributesW.restype = wintypes.DWORD
            attrs = GetFileAttributesW(str(path))
            file_attribute_reparse_point = 0x400
            if attrs != 0xFFFFFFFF and attrs & file_attribute_reparse_point:
                return True
        except Exception:
            return False
    return False


def _resolve_mount_target(path: Path) -> Path | None:
    try:
        if path.is_symlink():
            return path.resolve()
        if os.name == "nt" and path.exists():
            return path.resolve()
    except OSError:
        return None
    return None


def _create_windows_junction(link: Path, target: Path) -> None:
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise KnowledgeMountError(
            "Failed to create Windows junction from "
            f"{link} -> {target}: {completed.stderr or completed.stdout}",
        )


def _create_symlink(link: Path, target: Path) -> None:
    if os.name == "nt":
        _create_windows_junction(link, target)
        return
    os.symlink(target, link, target_is_directory=True)


def detect_dangling_mount(
    workspace_dir: str | Path,
    *,
    mount_name: str = "knowledge",
) -> Path | None:
    """Return the mount path when it points at a missing shared KB."""
    mount = Path(workspace_dir).expanduser().resolve() / (mount_name or "knowledge")
    if not (mount.is_symlink() or _is_junction_or_symlink(mount)):
        return None
    target = _resolve_mount_target(mount)
    if target is None:
        return mount
    if not target.exists():
        return mount
    return None


def ensure_knowledge_mount(
    workspace_dir: str | Path,
    kb_id: str,
    *,
    knowledge_bases_dir: str | Path | None = None,
    mount_name: str = "knowledge",
    domain: str = "business",
    create_if_missing: bool = False,
) -> Path:
    """Ensure ``workspace/{mount_name}`` points at the shared KB entity."""
    kb_id = validate_kb_id(kb_id)
    workspace = Path(workspace_dir).expanduser().resolve()
    mount = workspace / (mount_name or "knowledge")

    dangling = detect_dangling_mount(workspace, mount_name=mount_name or "knowledge")
    if dangling is not None:
        raise KnowledgeMountError(
            f"Knowledge mount {mount} points at a missing shared knowledge "
            f"base (kb_id={kb_id}). Restore the knowledge-base directory or "
            "choose a different knowledge_base_id.",
        )

    root = kb_root(kb_id, knowledge_bases_dir=knowledge_bases_dir)
    if not root.is_dir():
        if not create_if_missing:
            raise KnowledgeMountError(
                f"Knowledge base {kb_id!r} not found at {root}. "
                "Set create_knowledge_base=true to create a skeleton.",
            )
        ensure_kb(kb_id, knowledge_bases_dir=knowledge_bases_dir, domain=domain)
        root = kb_root(kb_id, knowledge_bases_dir=knowledge_bases_dir)

    target = root.resolve()

    if mount.exists() or mount.is_symlink() or _is_junction_or_symlink(mount):
        if _is_junction_or_symlink(mount) or mount.is_symlink():
            current = _resolve_mount_target(mount)
            if current is not None and current.resolve() == target:
                return mount
            if mount.is_symlink() or _is_junction_or_symlink(mount):
                if mount.is_dir() and not mount.is_symlink():
                    mount.rmdir()
                else:
                    mount.unlink()
            else:
                raise KnowledgeMountError(
                    f"Mount path {mount} exists and is not a link; "
                    "refusing to overwrite for knowledge sharing.",
                )
        elif mount.is_dir():
            if any(mount.iterdir()):
                raise KnowledgeMountError(
                    f"Mount path {mount} is a non-empty directory. "
                    "Shared knowledge requires a junction/symlink mount.",
                )
            mount.rmdir()
        else:
            raise KnowledgeMountError(
                f"Mount path {mount} exists and cannot be replaced.",
            )

    try:
        _create_symlink(mount, target)
    except KnowledgeMountError:
        raise
    except OSError as exc:
        raise KnowledgeMountError(
            f"Failed to mount knowledge base {kb_id} at {mount}: {exc}",
        ) from exc

    logger.info(
        "Mounted knowledge base %s at %s -> %s",
        kb_id,
        mount,
        target,
    )
    return mount
