"""Filesystem layout for shared knowledge bases."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)

BUSINESS_BUCKETS = (
    "business/wiki",
    "business/procedure",
    "business/personal",
    "business/dbInfo",
    "business/openapi",
)
TEST_BUCKETS = (
    "test/test_design",
    "test/test_cases",
    "test/test_data",
    "test/defects",
    "test/ui_pages",
    "test/ui_locators",
)
INBOX_BUCKET = "_inbox"
KB_BUCKETS = (*BUSINESS_BUCKETS, *TEST_BUCKETS, INBOX_BUCKET)
PUBLISHED_BUCKETS = tuple(b for b in KB_BUCKETS if b != INBOX_BUCKET)
LEGACY_FLAT_BUCKETS = ("personal", "procedure", "wiki")
PUBLISHED_DOMAIN_PREFIXES = ("business", "test")
_PUBLISHED_BUCKETS_BY_LOWER = {b.lower(): b for b in PUBLISHED_BUCKETS}

_KB_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")
_DEFAULT_KB_ROOT_ENV = "REME_KNOWLEDGE_BASES_DIR"


class KnowledgeBaseMeta(BaseModel):
    """Metadata persisted as KB.md frontmatter + body."""

    id: str
    name: str = ""
    domain: str = "business"
    version: int = 1
    created_at: str = ""
    updated_at: str = ""
    description: str = ""


def default_knowledge_bases_dir() -> Path:
    """Return the default shared-KB root under the ReMe home directory."""
    override = (os.environ.get(_DEFAULT_KB_ROOT_ENV) or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".reme" / "knowledge_bases"


def resolve_knowledge_bases_dir(value: str | None = None) -> Path:
    """Normalize configured or default ``knowledge_bases_dir``."""
    raw = (value or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return default_knowledge_bases_dir()


def _join_bucket(root: Path, bucket: str) -> Path:
    parts = [p for p in str(bucket).replace("\\", "/").split("/") if p]
    return root.joinpath(*parts)


def knowledge_watch_dirs(workspace_dir: str | Path, knowledge_dir: str) -> list[str]:
    """Absolute published-bucket paths for ReMe watch/reindex."""
    root = Path(workspace_dir).expanduser() / (knowledge_dir or "knowledge")
    buckets = list(PUBLISHED_BUCKETS)
    buckets.extend(b for b in LEGACY_FLAT_BUCKETS if b not in buckets)
    return [str(_join_bucket(root, b)) for b in buckets]


def canonicalize_published_bucket(bucket: str) -> str | None:
    """Map a bucket string to its canonical ``PUBLISHED_BUCKETS`` form.

    Matching is case-insensitive so callers that lower-case arguments still
    resolve mixed-case on-disk buckets such as ``business/dbInfo``. Legacy
    flat names (``wiki`` / ``procedure`` / ``personal``) map to
    ``business/{flat}``.
    """
    raw = (bucket or "").strip().replace("\\", "/").strip("/")
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in _PUBLISHED_BUCKETS_BY_LOWER:
        return _PUBLISHED_BUCKETS_BY_LOWER[lowered]
    if lowered in LEGACY_FLAT_BUCKETS:
        return f"business/{lowered}"
    return None


def knowledge_published_path_prefixes(knowledge_dir: str) -> list[str]:
    """Workspace-relative prefixes covering published KB nodes."""
    return knowledge_scope_path_prefixes(knowledge_dir, bucket="")


def knowledge_scope_path_prefixes(
    knowledge_dir: str,
    bucket: str = "",
) -> list[str]:
    kd = (knowledge_dir or "knowledge").replace("\\", "/").strip("/")
    raw = (bucket or "").strip().replace("\\", "/").strip("/")
    lowered = raw.lower()
    if lowered in ("all", "*", ""):
        prefixes = [f"{kd}/{domain}/" for domain in PUBLISHED_DOMAIN_PREFIXES]
        prefixes.extend(f"{kd}/{flat}/" for flat in LEGACY_FLAT_BUCKETS)
        return prefixes
    if lowered in PUBLISHED_DOMAIN_PREFIXES:
        prefixes = [f"{kd}/{lowered}/"]
        if lowered == "business":
            prefixes.extend(f"{kd}/{flat}/" for flat in LEGACY_FLAT_BUCKETS)
        return prefixes
    if lowered in LEGACY_FLAT_BUCKETS:
        return [f"{kd}/business/{lowered}/", f"{kd}/{lowered}/"]
    canon = canonicalize_published_bucket(raw)
    if canon is not None:
        prefixes = [f"{kd}/{canon}/"]
        if canon.startswith("business/"):
            flat = canon.split("/", 1)[1]
            if flat in LEGACY_FLAT_BUCKETS:
                prefixes.append(f"{kd}/{flat}/")
        return prefixes
    return []


def knowledge_bucket_choices() -> tuple[str, ...]:
    return (
        "all",
        *PUBLISHED_DOMAIN_PREFIXES,
        *PUBLISHED_BUCKETS,
        *LEGACY_FLAT_BUCKETS,
    )


def knowledge_bases_root(knowledge_bases_dir: str | Path | None = None) -> Path:
    return resolve_knowledge_bases_dir(
        str(knowledge_bases_dir) if knowledge_bases_dir is not None else None,
    )


def kb_root(kb_id: str, *, knowledge_bases_dir: str | Path | None = None) -> Path:
    return knowledge_bases_root(knowledge_bases_dir) / validate_kb_id(kb_id)


def validate_kb_id(kb_id: str) -> str:
    value = (kb_id or "").strip()
    if not value or not _KB_ID_RE.match(value):
        raise ValueError(
            f"Invalid knowledge_base_id {kb_id!r}. "
            "Use 1-64 chars: letters, digits, '.', '_', '-'.",
        )
    return value


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _read_kb_md(path: Path) -> KnowledgeBaseMeta | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    meta: dict[str, Any] = {}
    body = text
    if text.startswith("---"):
        parts = re.split(r"^---\s*$", text, maxsplit=2, flags=re.MULTILINE)
        if len(parts) >= 3:
            body = parts[2].lstrip("\n")
            frontmatter_text = parts[1]
            try:
                parsed = yaml.safe_load(frontmatter_text)
                if isinstance(parsed, dict):
                    meta = parsed
            except Exception:
                for line in frontmatter_text.splitlines():
                    if ":" not in line:
                        continue
                    key, raw = line.split(":", 1)
                    meta[key.strip()] = raw.strip().strip("\"'")
    return KnowledgeBaseMeta(
        id=str(meta.get("id") or path.parent.name),
        name=str(meta.get("name") or path.parent.name),
        domain=str(meta.get("domain") or "business"),
        version=int(meta.get("version") or 1),
        created_at=str(meta.get("created_at") or ""),
        updated_at=str(meta.get("updated_at") or ""),
        description=body.strip(),
    )


def _write_kb_md(path: Path, meta: KnowledgeBaseMeta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = meta.description.strip()
    content = (
        "---\n"
        f"id: {meta.id}\n"
        f'name: "{meta.name}"\n'
        f"domain: {meta.domain}\n"
        f"version: {meta.version}\n"
        f"created_at: {meta.created_at}\n"
        f"updated_at: {meta.updated_at}\n"
        "---\n"
        f"{body}\n"
    )
    path.write_text(content, encoding="utf-8")


def ensure_kb(
    kb_id: str,
    *,
    knowledge_bases_dir: str | Path | None = None,
    name: str | None = None,
    domain: str = "business",
    description: str = "",
) -> KnowledgeBaseMeta:
    """Create knowledge-base skeleton if missing; return metadata."""
    kb_id = validate_kb_id(kb_id)
    root = kb_root(kb_id, knowledge_bases_dir=knowledge_bases_dir)
    root.mkdir(parents=True, exist_ok=True)
    for bucket in KB_BUCKETS:
        (root / bucket).mkdir(parents=True, exist_ok=True)
    (root / ".locks").mkdir(exist_ok=True)

    kb_md = root / "KB.md"
    existing = _read_kb_md(kb_md)
    if existing is not None:
        return existing

    now = _now_iso()
    meta = KnowledgeBaseMeta(
        id=kb_id,
        name=(name or kb_id).strip() or kb_id,
        domain=domain or "business",
        version=1,
        created_at=now,
        updated_at=now,
        description=description,
    )
    _write_kb_md(kb_md, meta)
    index_path = knowledge_bases_root(knowledge_bases_dir) / "_index.json"
    try:
        index: dict[str, Any] = {}
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
        index[kb_id] = {
            "id": meta.id,
            "name": meta.name,
            "domain": meta.domain,
            "updated_at": meta.updated_at,
        }
        _write_json_atomic(index_path, index)
    except Exception:
        logger.debug("Failed to update knowledge base index", exc_info=True)
    logger.info("Created knowledge base %s at %s", kb_id, root)
    return meta


def list_knowledge_bases(
    knowledge_bases_dir: str | Path | None = None,
) -> list[KnowledgeBaseMeta]:
    """List knowledge bases present under ``knowledge_bases_dir``."""
    root = knowledge_bases_root(knowledge_bases_dir)
    if not root.is_dir():
        return []
    items: list[KnowledgeBaseMeta] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith(("_", ".")):
            continue
        meta = _read_kb_md(child / "KB.md")
        if meta is None:
            meta = KnowledgeBaseMeta(id=child.name, name=child.name)
        items.append(meta)
    return items


def read_knowledge_base_meta(
    kb_id: str,
    *,
    knowledge_bases_dir: str | Path | None = None,
) -> KnowledgeBaseMeta | None:
    """Return metadata for one knowledge base, or None when missing."""
    root = kb_root(kb_id, knowledge_bases_dir=knowledge_bases_dir)
    return _read_kb_md(root / "KB.md")
