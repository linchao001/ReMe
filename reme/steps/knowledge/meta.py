"""Return metadata for the active shared knowledge base."""

from ...components import R
from ...knowledge.store import (
    kb_root,
    read_knowledge_base_meta,
    resolve_knowledge_bases_dir,
    validate_kb_id,
)
from ..base_step import BaseStep


@R.register("knowledge_base_meta_step")
class KnowledgeBaseMetaStep(BaseStep):
    """Return metadata for ``knowledge_base_id`` (defaults to the active config id)."""

    async def execute(self):
        assert self.context is not None
        cfg = self.app_context.app_config if self.app_context is not None else None
        if cfg is None:
            self.context.response.success = False
            self.context.response.answer = "Error: application context is unavailable"
            return self.context.response
        requested = (self.context.get("knowledge_base_id") or cfg.knowledge_base_id or "").strip()
        if not requested:
            self.context.response.success = False
            self.context.response.answer = (
                "Error: knowledge_base_id is not configured. "
                "Set knowledge_base_id=zhb_kb when starting ReMe."
            )
            return self.context.response

        try:
            kb_id = validate_kb_id(requested)
        except ValueError as exc:
            self.context.response.success = False
            self.context.response.answer = str(exc)
            return self.context.response

        bases_dir = resolve_knowledge_bases_dir(cfg.knowledge_bases_dir or None)
        meta = read_knowledge_base_meta(kb_id, knowledge_bases_dir=bases_dir)
        if meta is None:
            self.context.response.success = False
            self.context.response.answer = (
                f"Knowledge base {kb_id!r} not found under {bases_dir}"
            )
            return self.context.response

        mount_name = (cfg.knowledge_dir or "knowledge").strip() or "knowledge"
        payload = meta.model_dump()
        payload.update(
            {
                "knowledge_bases_dir": str(bases_dir),
                "root_path": str(kb_root(kb_id, knowledge_bases_dir=bases_dir)),
                "workspace_mount": mount_name,
                "published_prefixes": [
                    f"{mount_name}/{prefix.rstrip('/')}/"
                    for prefix in ("business", "test")
                ],
            },
        )
        self.context.response.success = True
        self.context.response.answer = f"Knowledge base {kb_id}: {meta.name or kb_id}"
        self.context.response.metadata = payload
        return self.context.response
