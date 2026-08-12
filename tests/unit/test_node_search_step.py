"""Unit tests for NodeSearchStep prefix filtering and node-level aggregation."""

import asyncio

from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.enumeration import LinkScopeEnum
from reme.schema import FileChunk, FileFrontMatter, FileLink, FileNode
from reme.steps.index import NodeSearchStep


class FakeNodeSearchStore(BaseFileStore):
    """Minimal file_store for NodeSearchStep: static search results and nodes."""

    def __init__(
        self,
        vector_results: list[FileChunk] | None = None,
        keyword_results: list[FileChunk] | None = None,
        nodes: list[FileNode] | None = None,
    ):
        super().__init__(name="fake_node_search_store")
        self.vector_results = vector_results or []
        self.keyword_results = keyword_results or []
        self.nodes = {n.path: n for n in (nodes or [])}
        self.calls: list[tuple[str, str, int, dict]] = []

    async def upsert(self, files: list[tuple[FileNode, list[FileChunk]]]) -> None:
        raise NotImplementedError

    async def delete(self, path: str | list[str]) -> None:
        raise NotImplementedError

    async def clear(self) -> None:
        raise NotImplementedError

    async def get_nodes(self, paths: list[str] | None = None) -> list[FileNode]:
        return [self.nodes[p] for p in (paths or []) if p in self.nodes]

    async def get_outlinks(
        self,
        path: str,
        scope: LinkScopeEnum = LinkScopeEnum.REAL,
    ) -> list[FileLink]:
        return []

    async def get_inlinks(
        self,
        path: str,
        scope: LinkScopeEnum = LinkScopeEnum.REAL,
    ) -> list[FileLink]:
        return []

    async def vector_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        self.calls.append(("vector", query, limit, search_filter))
        return self.vector_results[:limit]

    async def keyword_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        self.calls.append(("keyword", query, limit, search_filter))
        return self.keyword_results[:limit]


def _chunk(chunk_id: str, path: str, score_key: str, score: float) -> FileChunk:
    return FileChunk(
        id=chunk_id,
        path=path,
        text="text",
        start_line=1,
        end_line=1,
        scores={score_key: score, "score": score},
    )


def _node(path: str, name: str, description: str) -> FileNode:
    return FileNode(
        path=path,
        st_mtime=0.0,
        front_matter=FileFrontMatter(name=name, description=description),
    )


def test_node_search_defaults_to_digest_prefix_only():
    """Without ``prefixes`` the step keeps the original digest-only behavior."""

    async def run():
        store = FakeNodeSearchStore(
            vector_results=[
                _chunk("d1", "digest/wiki/a.md", "vector", 0.9),
                _chunk("k1", "knowledge/b.md", "vector", 0.8),
            ],
            keyword_results=[
                _chunk("d2", "digest/personal/c.md", "keyword", 5.0),
            ],
            nodes=[_node("digest/wiki/a.md", "A", "alpha"), _node("digest/personal/c.md", "C", "gamma")],
        )
        step = NodeSearchStep(file_store=store, vector_weight=0.7, candidate_multiplier=2)

        resp = await step(RuntimeContext(query="alpha", limit=5))

        assert resp.success is True
        paths = [h["path"] for h in resp.metadata["hits"]]
        assert "knowledge/b.md" not in paths
        assert set(paths) == {"digest/wiki/a.md", "digest/personal/c.md"}
        assert resp.metadata["prefixes"] == ["digest/"]

    asyncio.run(run())


def test_node_search_prefixes_widen_recall_to_knowledge_subtree():
    """Passing ``prefixes`` admits non-digest subtrees while still excluding others."""

    async def run():
        store = FakeNodeSearchStore(
            vector_results=[
                _chunk("k1", "knowledge/foo.md", "vector", 0.9),
                _chunk("d1", "digest/wiki/bar.md", "vector", 0.8),
                _chunk("daily1", "daily/2026-01-01/x.md", "vector", 0.7),
                _chunk("res1", "resource/y.txt", "vector", 0.6),
            ],
            keyword_results=[],
            nodes=[_node("knowledge/foo.md", "Foo", "foo desc")],
        )
        step = NodeSearchStep(file_store=store, vector_weight=0.7, candidate_multiplier=2)

        resp = await step(RuntimeContext(query="foo", limit=10, prefixes=["knowledge/"]))

        assert resp.success is True
        paths = [h["path"] for h in resp.metadata["hits"]]
        assert paths == ["knowledge/foo.md"]
        assert resp.metadata["prefixes"] == ["knowledge/"]

    asyncio.run(run())


def test_node_search_prefixes_accepts_multiple_and_scalar_forms():
    """A list of prefixes and a single prefix string are both normalized to ``x/``."""

    async def run():
        store = FakeNodeSearchStore(
            vector_results=[
                _chunk("k1", "knowledge/a.md", "vector", 0.9),
                _chunk("d1", "digest/b.md", "vector", 0.8),
                _chunk("other", "daily/c.md", "vector", 0.7),
            ],
            keyword_results=[],
            nodes=[],
        )
        step = NodeSearchStep(file_store=store, vector_weight=0.7, candidate_multiplier=2)

        resp = await step(
            RuntimeContext(query="q", limit=10, prefixes=["digest", "knowledge/"])
        )

        assert resp.success is True
        paths = [h["path"] for h in resp.metadata["hits"]]
        assert set(paths) == {"knowledge/a.md", "digest/b.md"}
        assert "daily/c.md" not in paths
        assert resp.metadata["prefixes"] == ["digest/", "knowledge/"]

        # Scalar form is accepted too.
        store2 = FakeNodeSearchStore(
            vector_results=[_chunk("k1", "knowledge/a.md", "vector", 0.9)],
        )
        step2 = NodeSearchStep(file_store=store2, vector_weight=0.7, candidate_multiplier=2)
        resp2 = await step2(RuntimeContext(query="q", limit=10, prefixes="knowledge"))
        assert resp2.metadata["prefixes"] == ["knowledge/"]

    asyncio.run(run())


def test_node_search_empty_prefixes_falls_back_to_digest_only():
    """An empty list / whitespace-only prefixes is treated as omitted."""

    async def run():
        store = FakeNodeSearchStore(
            vector_results=[
                _chunk("d1", "digest/a.md", "vector", 0.9),
                _chunk("k1", "knowledge/b.md", "vector", 0.8),
            ],
            keyword_results=[],
        )
        step = NodeSearchStep(file_store=store, vector_weight=0.7, candidate_multiplier=2)

        resp = await step(RuntimeContext(query="q", limit=10, prefixes=["   "]))
        assert resp.metadata["prefixes"] == ["digest/"]
        assert [h["path"] for h in resp.metadata["hits"]] == ["digest/a.md"]

    asyncio.run(run())


def test_node_search_empty_query_fails_before_store_calls():
    """Empty queries fail fast and do not call file_store search methods."""

    async def run():
        store = FakeNodeSearchStore()
        step = NodeSearchStep(file_store=store)

        resp = await step(RuntimeContext(query="  ", limit=5))

        assert resp.success is False
        assert resp.answer == "Error: query cannot be empty"
        assert not store.calls

    asyncio.run(run())
