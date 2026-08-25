"""Shared helpers for knowledge-base ReMe steps."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from agentscope.message import Msg

from ...knowledge.context import get_knowledge_bases_dir, reset_knowledge_bases_dir, set_knowledge_bases_dir
from ...knowledge.dream import (
    MAX_SYNAPSE_LINKS,
    MergeCandidate,
    knowledge_claim_similarity,
)
from ...knowledge.store import (
    knowledge_published_path_prefixes,
    resolve_knowledge_bases_dir,
)
from ...schema import ApplicationConfig
from ..base_step import BaseStep

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KnowledgeRuntimeSettings:
    kb_id: str
    knowledge_dir: str
    knowledge_bases_dir: str
    write_mode: str
    domain: str
    scan_days: int
    max_units: int
    inbox_enabled: bool
    dedup_enabled: bool
    dedup_threshold: float
    merge_enabled: bool
    merge_threshold: float
    merge_margin: float
    related_threshold: float
    merge_max_updates: int
    language: str


def settings_from_config(cfg: ApplicationConfig) -> KnowledgeRuntimeSettings:
    kb_id = (cfg.knowledge_base_id or "").strip()
    if not kb_id:
        raise ValueError("knowledge_base_id is not configured")
    return KnowledgeRuntimeSettings(
        kb_id=kb_id,
        knowledge_dir=(cfg.knowledge_dir or "knowledge").strip() or "knowledge",
        knowledge_bases_dir=str(resolve_knowledge_bases_dir(cfg.knowledge_bases_dir or None)),
        write_mode=(cfg.knowledge_write_mode or "strict").strip().lower() or "strict",
        domain=(cfg.knowledge_domain or "business").strip().lower() or "business",
        scan_days=int(cfg.knowledge_scan_days),
        max_units=int(cfg.knowledge_max_units),
        inbox_enabled=bool(cfg.knowledge_inbox_enabled),
        dedup_enabled=bool(cfg.knowledge_dedup_enabled),
        dedup_threshold=float(cfg.knowledge_dedup_threshold),
        merge_enabled=bool(cfg.knowledge_merge_enabled),
        merge_threshold=float(cfg.knowledge_merge_threshold),
        merge_margin=float(cfg.knowledge_merge_margin),
        related_threshold=float(cfg.knowledge_related_threshold),
        merge_max_updates=int(cfg.knowledge_merge_max_updates),
        language=(cfg.language or "").strip(),
    )


@contextmanager
def knowledge_context(settings: KnowledgeRuntimeSettings):
    token = set_knowledge_bases_dir(settings.knowledge_bases_dir)
    try:
        yield
    finally:
        reset_knowledge_bases_dir(token)


async def llm_call_from_step(step: BaseStep, prompt: str) -> str:
    if step.as_llm is None:
        raise RuntimeError("LLM is not configured")
    messages = [Msg(name="system", role="system", content=prompt)]
    response = await step.as_llm(messages)
    if isinstance(response, Msg):
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
            return "\n".join(parts)
        return str(content or "")
    return str(response or "")


def _published_prefixes(settings: KnowledgeRuntimeSettings) -> list[str]:
    return knowledge_published_path_prefixes(settings.knowledge_dir)


async def _node_search_hits(step: BaseStep, settings: KnowledgeRuntimeSettings, query: str, *, limit: int = 20):
    if not query.strip():
        return []
    try:
        response = await step.run_job(
            "node_search",
            query=query,
            limit=limit,
            prefixes=_published_prefixes(settings),
        )
    except Exception:
        logger.debug("knowledge node_search failed query=%r", query, exc_info=True)
        return []
    if not response.success:
        return []
    hits = response.metadata.get("hits") if response.metadata else None
    return hits if isinstance(hits, list) else []


def build_dedup_search(step: BaseStep, settings: KnowledgeRuntimeSettings):
    prefixes = _published_prefixes(settings)
    threshold = settings.dedup_threshold

    def _is_duplicate(name: str, hits: list, summary: str = "") -> bool:
        target = (name or "").strip().lower()
        if not target:
            return False
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            path = str(hit.get("path") or "").replace("\\", "/").lstrip("./")
            if not any(path.startswith(prefix) for prefix in prefixes):
                continue
            candidate = str(hit.get("name") or "").strip()
            if not candidate:
                continue
            ratio = knowledge_claim_similarity(
                target,
                candidate,
                summary,
                str(hit.get("description") or ""),
            )
            if ratio >= threshold:
                return True
        return False

    async def _dedup_search(name: str, summary: str) -> bool:
        query = f"{name} {summary}".strip() or name
        hits = await _node_search_hits(step, settings, query)
        if not hits:
            return False
        try:
            return _is_duplicate(name, hits, summary)
        except Exception:
            logger.debug("knowledge dedup parse failed name=%r", name, exc_info=True)
            return False

    return _dedup_search


def build_merge_search(step: BaseStep, settings: KnowledgeRuntimeSettings):
    prefixes = _published_prefixes(settings)
    knowledge_prefix = settings.knowledge_dir.strip("/") + "/"
    threshold = settings.merge_threshold
    margin = settings.merge_margin
    related_threshold = settings.related_threshold
    dedup_threshold = settings.dedup_threshold

    def _rank(name: str, hits: list, summary: str = "") -> MergeCandidate | None:
        target = (name or "").strip().lower()
        if not target:
            return None
        scored: list[tuple[float, str, str]] = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            path = str(hit.get("path") or "").replace("\\", "/").lstrip("./")
            if not any(path.startswith(prefix) for prefix in prefixes):
                continue
            candidate = str(hit.get("name") or "").strip()
            if not candidate:
                continue
            ratio = knowledge_claim_similarity(
                target,
                candidate,
                summary,
                str(hit.get("description") or ""),
            )
            scored.append((ratio, candidate, path))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        top_ratio, top_name, top_path = scored[0]
        second_ratio = scored[1][0] if len(scored) > 1 else 0.0
        is_clear = top_ratio >= threshold and (
            (top_ratio - second_ratio) >= margin or len(scored) == 1
        )
        related: list[str] = []
        for ratio, cand_name, _path in scored:
            if ratio < related_threshold:
                continue
            if cand_name.strip().lower() == target:
                continue
            if is_clear and cand_name.strip().lower() == top_name.strip().lower():
                continue
            if cand_name not in related:
                related.append(cand_name)
            if len(related) >= MAX_SYNAPSE_LINKS:
                break
        rel = top_path
        if rel.startswith(knowledge_prefix):
            rel = rel[len(knowledge_prefix):]
        if top_ratio < threshold:
            if top_ratio >= dedup_threshold:
                return MergeCandidate(
                    name=top_name,
                    path=rel,
                    ratio=top_ratio,
                    is_clear=False,
                    related_names=related,
                )
            if not related:
                return None
            return MergeCandidate(
                name="",
                path="",
                ratio=top_ratio,
                is_clear=False,
                related_names=related,
            )
        return MergeCandidate(
            name=top_name,
            path=rel,
            ratio=top_ratio,
            is_clear=is_clear,
            related_names=related,
        )

    async def _merge_search(name: str, summary: str) -> MergeCandidate | None:
        query = f"{name} {summary}".strip() or name
        hits = await _node_search_hits(step, settings, query)
        if not hits:
            return None
        try:
            return _rank(name, hits, summary)
        except Exception:
            logger.debug("knowledge merge parse failed name=%r", name, exc_info=True)
            return None

    return _merge_search


def build_catalog_search(step: BaseStep, settings: KnowledgeRuntimeSettings):
    async def _catalog_search(query: str) -> list[tuple[str, str]]:
        hits = await _node_search_hits(step, settings, query, limit=10)
        rows: list[tuple[str, str]] = []
        seen: set[str] = set()
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            name = str(hit.get("name") or "").strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            path = str(hit.get("path") or "").replace("\\", "/")
            bucket = "wiki"
            marker = f"/{settings.knowledge_dir.strip('/')}/"
            if marker in path:
                tail = path.split(marker, 1)[1]
                bucket = tail.split("/", 1)[0] if tail else bucket
            rows.append((name, bucket))
        return rows

    return _catalog_search
