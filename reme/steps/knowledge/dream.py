"""Extract daily notes into the shared knowledge base."""

from ...components import R
from ...knowledge.dream import run_knowledge_dream
from ...knowledge.lock import KnowledgeLockTimeout
from ..base_step import BaseStep
from ..evolve.dream.utils import workspace_dir
from ._runtime import (
    build_catalog_search,
    build_dedup_search,
    build_merge_search,
    knowledge_context,
    llm_call_from_step,
    settings_from_config,
)


@R.register("knowledge_dream_step")
class KnowledgeDreamStep(BaseStep):
    """Run one knowledge-dream pass over recent daily notes."""

    async def execute(self):
        assert self.context is not None
        cfg = self.app_context.app_config if self.app_context is not None else None
        if cfg is None:
            self.context.response.success = False
            self.context.response.answer = "Error: application context is unavailable"
            return self.context.response

        try:
            settings = settings_from_config(cfg)
        except ValueError as exc:
            self.context.response.success = False
            self.context.response.answer = str(exc)
            return self.context.response

        hint = str(self.context.get("hint") or "").strip()
        scan_days = int(self.context.get("scan_days") or settings.scan_days)
        max_units = int(self.context.get("max_units") or settings.max_units)

        async def _llm_call(prompt: str) -> str:
            return await llm_call_from_step(self, prompt)

        dedup_search = (
            build_dedup_search(self, settings)
            if settings.dedup_enabled
            else None
        )
        merge_search = (
            build_merge_search(self, settings)
            if settings.merge_enabled or settings.dedup_enabled
            else None
        )
        catalog_search = (
            build_catalog_search(self, settings)
            if settings.merge_enabled or settings.dedup_enabled
            else None
        )

        with knowledge_context(settings):
            try:
                result = await run_knowledge_dream(
                    agent_id="reme",
                    workspace_dir=workspace_dir(self),
                    kb_id=settings.kb_id,
                    daily_dir_name=self.config_value("daily_dir"),
                    metadata_dir=self.config_value("metadata_dir"),
                    language=settings.language,
                    domain=settings.domain,
                    scan_days=scan_days,
                    max_units=max_units,
                    write_mode=settings.write_mode,
                    inbox_enabled=settings.inbox_enabled,
                    knowledge_dir=settings.knowledge_dir,
                    llm_call=_llm_call,
                    dedup_search=dedup_search,
                    merge_search=merge_search,
                    catalog_search=catalog_search,
                    merge_enabled=settings.merge_enabled,
                    merge_max_updates=settings.merge_max_updates,
                )
            except KnowledgeLockTimeout:
                self.context.response.success = False
                self.context.response.answer = (
                    "Knowledge base is locked by another writer; try again."
                )
                return self.context.response

        skipped = bool(result.get("skipped"))
        written = [str(item) for item in result.get("written") or []]
        self.context.response.success = not skipped or bool(written)
        self.context.response.answer = (
            f"knowledge_dream finished: written={len(written)} "
            f"units={result.get('units', 0)} skipped={skipped}"
        )
        self.context.response.metadata = {
            "knowledge_base_id": settings.kb_id,
            **result,
        }
        return self.context.response
