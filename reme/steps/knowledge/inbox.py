"""Inbox review operations for shared knowledge bases."""

from ...components import R
from ...knowledge.dream import (
    InboxActionError,
    get_inbox_item,
    list_inbox_items,
    merge_inbox_item,
    promote_inbox_item,
    reject_inbox_item,
)
from ..base_step import BaseStep
from ._runtime import knowledge_context, settings_from_config


def _active_settings(step: BaseStep):
    cfg = step.app_context.app_config if step.app_context is not None else None
    if cfg is None:
        raise ValueError("application context is unavailable")
    return settings_from_config(cfg)


@R.register("list_knowledge_inbox_step")
class ListKnowledgeInboxStep(BaseStep):
    async def execute(self):
        assert self.context is not None
        try:
            settings = _active_settings(self)
        except ValueError as exc:
            self.context.response.success = False
            self.context.response.answer = str(exc)
            return self.context.response

        with knowledge_context(settings):
            items = list_inbox_items(settings.kb_id)
        payload = [item.model_dump() for item in items]
        self.context.response.success = True
        self.context.response.answer = f"Found {len(items)} inbox item(s)"
        self.context.response.metadata = {
            "knowledge_base_id": settings.kb_id,
            "items": payload,
        }
        return self.context.response


@R.register("promote_knowledge_inbox_step")
class PromoteKnowledgeInboxStep(BaseStep):
    async def execute(self):
        assert self.context is not None
        stem = str(self.context.get("stem") or "").strip()
        if not stem:
            self.context.response.success = False
            self.context.response.answer = "Error: stem is required"
            return self.context.response
        try:
            settings = _active_settings(self)
        except ValueError as exc:
            self.context.response.success = False
            self.context.response.answer = str(exc)
            return self.context.response

        with knowledge_context(settings):
            try:
                path = promote_inbox_item(
                    settings.kb_id,
                    stem,
                    agent_id="reme",
                    knowledge_dir=settings.knowledge_dir,
                )
            except InboxActionError as exc:
                self.context.response.success = False
                self.context.response.answer = str(exc)
                return self.context.response
        rel = str(path).replace("\\", "/")
        self.context.response.success = True
        self.context.response.answer = f"Promoted inbox item to {rel}"
        self.context.response.metadata = {"written": rel}
        return self.context.response


@R.register("merge_knowledge_inbox_step")
class MergeKnowledgeInboxStep(BaseStep):
    async def execute(self):
        assert self.context is not None
        stem = str(self.context.get("stem") or "").strip()
        target_path = str(self.context.get("target_path") or self.context.get("target") or "").strip()
        mode = str(self.context.get("mode") or "REFINE").strip()
        if not stem:
            self.context.response.success = False
            self.context.response.answer = "Error: stem is required"
            return self.context.response
        try:
            settings = _active_settings(self)
        except ValueError as exc:
            self.context.response.success = False
            self.context.response.answer = str(exc)
            return self.context.response

        with knowledge_context(settings):
            try:
                path = merge_inbox_item(
                    settings.kb_id,
                    stem,
                    agent_id="reme",
                    target_path=target_path,
                    mode=mode,
                    merge_max_updates=settings.merge_max_updates,
                    knowledge_dir=settings.knowledge_dir,
                )
            except InboxActionError as exc:
                self.context.response.success = False
                self.context.response.answer = str(exc)
                return self.context.response
        rel = str(path).replace("\\", "/")
        self.context.response.success = True
        self.context.response.answer = f"Merged inbox item into {rel}"
        self.context.response.metadata = {"written": rel}
        return self.context.response


@R.register("reject_knowledge_inbox_step")
class RejectKnowledgeInboxStep(BaseStep):
    async def execute(self):
        assert self.context is not None
        stem = str(self.context.get("stem") or "").strip()
        if not stem:
            self.context.response.success = False
            self.context.response.answer = "Error: stem is required"
            return self.context.response
        try:
            settings = _active_settings(self)
        except ValueError as exc:
            self.context.response.success = False
            self.context.response.answer = str(exc)
            return self.context.response

        with knowledge_context(settings):
            path = reject_inbox_item(settings.kb_id, stem)
        rel = str(path).replace("\\", "/")
        self.context.response.success = True
        self.context.response.answer = f"Rejected inbox item -> {rel}"
        self.context.response.metadata = {"rejected": rel}
        return self.context.response
