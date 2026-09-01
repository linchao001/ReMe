"""Explicitly write one unit into the shared knowledge base."""

from ...components import R
from ...knowledge.dream import (
    KnowledgeUnit,
    MergePayload,
    _find_exact_published_node,
    _parse_frontmatter,
    _wikilink_title_index,
    format_wikilink,
    integrate_units,
    resolve_wikilink_target,
    structural_merge_body,
)
from ...knowledge.lock import KnowledgeLockTimeout
from ...knowledge.store import canonicalize_published_bucket, kb_root
from ..base_step import BaseStep
from ._runtime import knowledge_context, settings_from_config


@R.register("save_to_knowledge_step")
class SaveToKnowledgeStep(BaseStep):
    """Publish or refine one node in the active shared knowledge base."""

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

        title = str(self.context.get("title") or "").strip()
        content = str(self.context.get("content") or "").strip()
        bucket_raw = str(self.context.get("bucket") or "wiki").strip()
        if not title or not content:
            self.context.response.success = False
            self.context.response.answer = "Error: title and content are required"
            return self.context.response

        bucket = canonicalize_published_bucket(bucket_raw)
        if bucket is None:
            bucket = (
                "test/test_cases"
                if settings.domain.startswith("test")
                else "business/wiki"
            )

        steps = self.context.get("steps") or []
        if not isinstance(steps, list):
            steps = []
        links = self.context.get("links") or []
        if not isinstance(links, list):
            links = []

        unit = KnowledgeUnit(
            name=title,
            bucket=bucket,
            summary=content,
            confidence=1.0,
            signals=["save_to_knowledge"],
            preconditions=str(self.context.get("preconditions") or "").strip(),
            steps=[str(item).strip() for item in steps if str(item).strip()],
            expected=str(self.context.get("expected") or "").strip(),
            priority=str(self.context.get("priority") or "").strip(),
            requirement_id=str(self.context.get("requirement_id") or "").strip(),
            links=[str(item).strip() for item in links if str(item).strip()],
        )

        merge_candidates = {}
        merge_payloads = {}
        with knowledge_context(settings):
            exact = _find_exact_published_node(
                settings.kb_id,
                title,
                preferred_bucket=bucket,
            )
            if exact is not None:
                target = kb_root(settings.kb_id) / exact.path
                try:
                    text = target.read_text(encoding="utf-8")
                    fm, body = _parse_frontmatter(text)
                    title_index = _wikilink_title_index(settings.kb_id)
                    formatted_links = [
                        format_wikilink(
                            *resolve_wikilink_target(
                                link,
                                title_index=title_index,
                                knowledge_dir=settings.knowledge_dir,
                            ),
                        )
                        for link in unit.links
                    ]
                    merged = structural_merge_body(
                        body,
                        unit,
                        formatted_links=formatted_links or None,
                    )
                    merge_candidates[title.lower()] = exact
                    merge_payloads[title.lower()] = MergePayload(
                        target_path=target,
                        expected_updated_at=fm.get("updated_at", "").strip().strip('"'),
                        merged_body=merged,
                        llm_ok=True,
                    )
                except OSError:
                    self.logger.debug(
                        "save_to_knowledge unreadable target %s",
                        target,
                        exc_info=True,
                    )

            try:
                written = integrate_units(
                    kb_id=settings.kb_id,
                    agent_id="reme",
                    units=[unit],
                    derived_from=["job:save_to_knowledge"],
                    write_mode=settings.write_mode,
                    inbox_enabled=settings.inbox_enabled,
                    knowledge_dir=settings.knowledge_dir,
                    merge_enabled=True,
                    merge_candidates=merge_candidates,
                    merge_payloads=merge_payloads,
                )
            except KnowledgeLockTimeout:
                self.context.response.success = False
                self.context.response.answer = (
                    "Knowledge base is locked by another writer; try again."
                )
                return self.context.response

        if not written:
            self.context.response.success = True
            self.context.response.answer = (
                f"No new node written (possible duplicate of {title!r})."
            )
            return self.context.response

        rel = str(written[0]).replace("\\", "/")
        self.context.response.success = True
        self.context.response.answer = f"Saved to knowledge base {settings.kb_id}: {rel}"
        self.context.response.metadata = {
            "knowledge_base_id": settings.kb_id,
            "written": [rel],
        }
        return self.context.response
