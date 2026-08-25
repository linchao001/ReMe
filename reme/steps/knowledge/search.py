"""Scoped hybrid search over the mounted shared knowledge base."""

from ...components import R
from ...knowledge.store import knowledge_scope_path_prefixes
from ..index.search import SearchStep


@R.register("knowledge_search_step")
class KnowledgeSearchStep(SearchStep):
    """Search only published nodes under workspace/{knowledge_dir}."""

    async def execute(self):
        assert self.context is not None
        cfg = self.app_context.app_config if self.app_context is not None else None
        if cfg is None:
            self.context.response.success = False
            self.context.response.answer = "Error: application context is unavailable"
            return self.context.response
        active_id = (cfg.knowledge_base_id or "").strip()
        if not active_id:
            self.context.response.success = False
            self.context.response.answer = (
                "Error: knowledge_base_id is not configured. "
                "Start ReMe with knowledge_base_id=<id> to enable knowledge search."
            )
            return self.context.response

        knowledge_dir = (cfg.knowledge_dir or "knowledge").strip() or "knowledge"
        bucket = str(self.context.get("bucket") or "all").strip().lower()
        prefixes = knowledge_scope_path_prefixes(knowledge_dir, bucket=bucket)
        if not prefixes:
            self.context.response.success = False
            self.context.response.answer = f"Error: unknown knowledge bucket {bucket!r}"
            return self.context.response

        search_filter = dict(self.context.get("search_filter", {}) or {})
        search_filter["prefixes"] = prefixes
        self.context["search_filter"] = search_filter
        self.context.response.metadata = self.context.response.metadata or {}
        self.context.response.metadata["knowledge_base_id"] = active_id
        self.context.response.metadata["bucket"] = bucket or "all"
        self.context.response.metadata["prefixes"] = prefixes
        return await super().execute()
