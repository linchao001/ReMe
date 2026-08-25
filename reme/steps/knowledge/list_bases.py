"""List shared knowledge bases available on disk."""

from ...components import R
from ...knowledge.store import list_knowledge_bases, resolve_knowledge_bases_dir
from ..base_step import BaseStep


@R.register("list_knowledge_bases_step")
class ListKnowledgeBasesStep(BaseStep):
    """Return metadata for every knowledge base under ``knowledge_bases_dir``."""

    async def execute(self):
        assert self.context is not None
        cfg = self.app_context.app_config if self.app_context is not None else None
        if cfg is None:
            self.context.response.success = False
            self.context.response.answer = "Error: application context is unavailable"
            return self.context.response
        bases_dir = resolve_knowledge_bases_dir(cfg.knowledge_bases_dir or None)
        items = list_knowledge_bases(bases_dir)
        active_id = (cfg.knowledge_base_id or "").strip()
        payload = {
            "knowledge_bases": [item.model_dump() for item in items],
            "knowledge_bases_dir": str(bases_dir),
            "active_knowledge_base_id": active_id or None,
        }
        self.context.response.success = True
        self.context.response.answer = (
            f"Found {len(items)} knowledge base(s) under {bases_dir}"
        )
        self.context.response.metadata = payload
        return self.context.response
