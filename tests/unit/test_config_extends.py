"""Config extends support."""

from reme.config.config_parser import _load_config


def test_business_kb_config_extends_default():
    cfg = _load_config("business_kb")
    assert cfg.get("knowledge_base_id") == "zhb_kb"
    jobs = cfg.get("jobs", {})
    assert "search" in jobs
    assert "knowledge_search" in jobs
    assert "save_to_knowledge" in jobs
    assert "knowledge_dream" in jobs
    assert "list_knowledge_inbox" in jobs
    assert jobs.get("list_knowledge_bases")


def test_personal_with_kb_config_extends_default():
    cfg = _load_config("personal_with_kb")
    assert cfg.get("workspace_dir") == ".reme"
    assert cfg.get("knowledge_base_id") == "zhb_kb"
    jobs = cfg.get("jobs", {})
    assert "dream_cron" in jobs
    assert "knowledge_dream" in jobs
    assert "knowledge_dream_cron" in jobs
    assert jobs["knowledge_dream_cron"]["backend"] == "cron"
