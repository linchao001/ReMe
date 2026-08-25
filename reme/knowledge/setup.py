"""Startup helpers for shared business knowledge bases."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .mount import KnowledgeMountError, ensure_knowledge_mount
from .store import knowledge_watch_dirs, resolve_knowledge_bases_dir

logger = logging.getLogger(__name__)

_WATCH_JOB_NAMES = (
    "index_update_loop",
    "reindex",
    "index_sync",
    "digest_watch_loop",
)


def _append_watch_dirs(job: dict[str, Any], extra_dirs: list[str]) -> None:
    watch_dirs = list(job.get("watch_dirs") or [])
    for path in extra_dirs:
        if path not in watch_dirs:
            watch_dirs.append(path)
    job["watch_dirs"] = watch_dirs


def augment_jobs_for_knowledge(
    jobs: dict[str, Any],
    *,
    workspace_dir: str,
    knowledge_dir: str,
) -> None:
    """Append published knowledge bucket paths to index/watch jobs."""
    extra_dirs = knowledge_watch_dirs(workspace_dir, knowledge_dir)
    if not extra_dirs:
        return
    for job_name in _WATCH_JOB_NAMES:
        job = jobs.get(job_name)
        if isinstance(job, dict):
            _append_watch_dirs(job, extra_dirs)


def prepare_knowledge_startup(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Mount the configured shared KB and augment watch jobs before Application init."""
    kb_id = (kwargs.get("knowledge_base_id") or "").strip()
    if not kb_id:
        return kwargs

    workspace_dir = str(
        Path(kwargs.get("workspace_dir") or ".reme").expanduser().resolve(),
    )
    Path(workspace_dir).mkdir(parents=True, exist_ok=True)
    knowledge_dir = (kwargs.get("knowledge_dir") or "knowledge").strip() or "knowledge"
    knowledge_bases_dir = resolve_knowledge_bases_dir(kwargs.get("knowledge_bases_dir"))
    create_if_missing = bool(kwargs.get("create_knowledge_base", False))

    try:
        ensure_knowledge_mount(
            workspace_dir,
            kb_id,
            knowledge_bases_dir=knowledge_bases_dir,
            mount_name=knowledge_dir,
            create_if_missing=create_if_missing,
        )
    except KnowledgeMountError:
        logger.exception(
            "Failed to mount knowledge base %s from %s",
            kb_id,
            knowledge_bases_dir,
        )
        raise

    jobs = kwargs.get("jobs")
    if isinstance(jobs, dict):
        augment_jobs_for_knowledge(
            jobs,
            workspace_dir=workspace_dir,
            knowledge_dir=knowledge_dir,
        )
    return kwargs
