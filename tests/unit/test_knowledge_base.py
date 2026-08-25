"""Shared knowledge-base path and scope helpers."""

from pathlib import Path

import pytest

from reme.knowledge.mount import ensure_knowledge_mount
from reme.knowledge.setup import augment_jobs_for_knowledge, prepare_knowledge_startup
from reme.knowledge.store import (
    default_knowledge_bases_dir,
    knowledge_published_path_prefixes,
    knowledge_scope_path_prefixes,
    knowledge_watch_dirs,
    list_knowledge_bases,
    read_knowledge_base_meta,
)


def test_default_knowledge_bases_dir():
    assert default_knowledge_bases_dir() == Path.home() / ".reme" / "knowledge_bases"


def test_knowledge_scope_prefixes_cover_domains():
    all_prefixes = knowledge_scope_path_prefixes("knowledge", bucket="")
    assert "knowledge/business/" in all_prefixes
    assert "knowledge/test/" in all_prefixes
    assert "knowledge/wiki/" in all_prefixes

    business = knowledge_scope_path_prefixes("knowledge", bucket="business")
    assert set(business) == {
        "knowledge/business/",
        "knowledge/wiki/",
        "knowledge/procedure/",
        "knowledge/personal/",
    }

    cases = knowledge_scope_path_prefixes("knowledge", bucket="test/test_cases")
    assert cases == ["knowledge/test/test_cases/"]


def test_knowledge_watch_dirs_use_published_buckets_only(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    mount = workspace / "knowledge"
    mount.mkdir()
    (mount / "business" / "wiki").mkdir(parents=True)
    (mount / "_inbox").mkdir()

    watched = knowledge_watch_dirs(workspace, "knowledge")
    watched_names = {Path(p).name for p in watched}
    assert "wiki" in watched_names
    assert "_inbox" not in watched_names
    assert all(Path(p).is_absolute() for p in watched)


def test_prepare_knowledge_startup_mounts_existing_kb(tmp_path, monkeypatch):
    kb_root = tmp_path / "knowledge_bases" / "demo"
    (kb_root / "business" / "wiki").mkdir(parents=True)
    (kb_root / "KB.md").write_text(
        "---\nid: demo\nname: Demo\ndomain: business\nversion: 1\n---\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    kwargs = prepare_knowledge_startup(
        {
            "workspace_dir": str(workspace),
            "knowledge_bases_dir": str(tmp_path / "knowledge_bases"),
            "knowledge_base_id": "demo",
            "knowledge_dir": "knowledge",
            "jobs": {"index_update_loop": {"watch_dirs": ["daily_dir"]}},
        },
    )

    mount = workspace / "knowledge"
    assert mount.exists()
    assert mount.resolve() == kb_root.resolve()
    watch_dirs = kwargs["jobs"]["index_update_loop"]["watch_dirs"]
    normalized = [str(entry).replace("\\", "/") for entry in watch_dirs]
    assert any("business/wiki" in entry for entry in normalized)


def test_augment_jobs_for_knowledge_appends_once(tmp_path):
    workspace = tmp_path / "ws"
    (workspace / "knowledge" / "business" / "wiki").mkdir(parents=True)
    jobs = {"reindex": {"watch_dirs": ["daily_dir"]}}
    augment_jobs_for_knowledge(jobs, workspace_dir=str(workspace), knowledge_dir="knowledge")
    augment_jobs_for_knowledge(jobs, workspace_dir=str(workspace), knowledge_dir="knowledge")
    assert jobs["reindex"]["watch_dirs"].count("daily_dir") == 1
    normalized = [str(entry).replace("\\", "/") for entry in jobs["reindex"]["watch_dirs"]]
    assert any("business/wiki" in entry for entry in normalized)


def test_list_and_read_zhb_kb_if_present():
    candidates = (
        Path.home() / ".reme" / "knowledge_bases",
        Path.home() / ".qwenpaw" / "knowledge_bases",
    )
    bases_dir = next((path for path in candidates if (path / "zhb").is_dir()), None)
    if bases_dir is None:
        pytest.skip("local zhb knowledge base not present")

    items = list_knowledge_bases(bases_dir)
    ids = {item.id for item in items}
    assert "zhb" in ids

    meta = read_knowledge_base_meta("zhb", knowledge_bases_dir=bases_dir)
    assert meta is not None
    assert meta.id == "zhb"
    assert meta.name

    prefixes = knowledge_published_path_prefixes("knowledge")
    assert prefixes[0].startswith("knowledge/")


def test_ensure_knowledge_mount_refuses_missing_kb(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(Exception, match="not found"):
        ensure_knowledge_mount(
            workspace,
            "missing",
            knowledge_bases_dir=tmp_path / "knowledge_bases",
            create_if_missing=False,
        )


@pytest.mark.asyncio
async def test_save_to_knowledge_writes_published_node(tmp_path):
    from reme.application import Application

    kb_root = tmp_path / "knowledge_bases" / "demo"
    (kb_root / "business" / "wiki").mkdir(parents=True)
    (kb_root / "KB.md").write_text(
        "---\nid: demo\nname: Demo\ndomain: business\nversion: 1\n---\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    cfg = {
        "enable_logo": False,
        "log_to_file": False,
        "log_to_console": False,
        "workspace_dir": str(workspace),
        "knowledge_bases_dir": str(tmp_path / "knowledge_bases"),
        "knowledge_base_id": "demo",
        "knowledge_dir": "knowledge",
        "knowledge_write_mode": "open",
        "service": {"backend": "http", "web_enabled": False, "port": 8198},
        "jobs": {
            "save_to_knowledge": {
                "backend": "base",
                "steps": [{"backend": "save_to_knowledge_step"}],
            },
        },
    }

    app = Application(**cfg)
    await app.start()
    resp = await app.run_job(
        "save_to_knowledge",
        title="测试节点",
        content="这是一个测试知识节点。",
        bucket="business/wiki",
    )
    await app.close()

    assert resp.success
    written = (resp.metadata or {}).get("written") or []
    assert written
    rel_path = kb_root / "business" / "wiki" / "测试节点.md"
    assert rel_path.is_file()
    text = rel_path.read_text(encoding="utf-8")
    assert "测试知识节点" in text
