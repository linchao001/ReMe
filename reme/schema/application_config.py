"""Application configuration models."""

import os
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..enumeration import component_type_name


class ComponentConfig(BaseModel):
    """Base config for a component; extra fields allowed for backend-specific options."""

    model_config = ConfigDict(extra="allow")

    backend: str = Field(default="", description="Backend implementation class name")


class JobConfig(ComponentConfig):
    """Config for a job — an ordered sequence of step components. Keyed by name in ApplicationConfig.jobs."""

    description: str = Field(default="", description="Human-readable description")
    parameters: dict = Field(default_factory=dict, description="Job-level parameters")
    steps: list[ComponentConfig] = Field(default_factory=list, description="Ordered step configs")
    enable_serve: bool = Field(default=True, description="Whether to expose this job through the service layer")


class ApplicationConfig(BaseModel):
    """Root config for the ReMe application."""

    app_name: str = Field(default=os.getenv("APP_NAME", "ReMe"), description="Application display name")
    environment: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables loaded once at startup and passed to agent subprocesses",
    )
    workspace_dir: str = Field(
        default=".reme",
        description="Workspace root directory for runtime files",
        validate_default=True,
    )
    metadata_dir: str = Field(default="metadata", description="Subdirectory for ReMe persistent state")
    session_dir: str = Field(default="session", description="Subdirectory for persisted agent sessions")
    # dialog_dir was removed; standard transcripts are always derived as ``{session_dir}/dialog``.
    mem_session_dir: str = Field(default="mem_session", description="Subdirectory for persisted agent sessions")
    resource_dir: str = Field(default="resource", description="Subdirectory for external assets")
    daily_dir: str = Field(default="daily", description="Subdirectory for daily memory")
    digest_dir: str = Field(default="digest", description="Subdirectory for digest memory")
    knowledge_bases_dir: str = Field(
        default="",
        description=(
            "Root directory for shared knowledge-base entities. "
            "Empty uses REME_KNOWLEDGE_BASES_DIR or ~/.reme/knowledge_bases."
        ),
    )
    knowledge_base_id: str = Field(
        default="",
        description=(
            "Active shared knowledge-base id to mount at workspace/{knowledge_dir}. "
            "When set, ReMe indexes published buckets under the mount."
        ),
    )
    knowledge_dir: str = Field(
        default="knowledge",
        description="Workspace mount name for the active shared knowledge base.",
    )
    create_knowledge_base: bool = Field(
        default=False,
        description="Create a KB skeleton when knowledge_base_id is missing on disk.",
    )
    knowledge_domain: str = Field(
        default="business",
        description="Knowledge dream extract domain: business or testcase.",
    )
    knowledge_write_mode: str = Field(
        default="strict",
        description="strict routes uncertain units to _inbox; open publishes directly.",
    )
    knowledge_scan_days: int = Field(default=2, ge=1, le=30)
    knowledge_max_units: int = Field(default=8, ge=1, le=50)
    knowledge_inbox_enabled: bool = Field(default=True)
    knowledge_dedup_enabled: bool = Field(default=True)
    knowledge_dedup_threshold: float = Field(default=0.78, ge=0.0, le=1.0)
    knowledge_merge_enabled: bool = Field(default=True)
    knowledge_merge_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    knowledge_merge_margin: float = Field(default=0.15, ge=0.0, le=1.0)
    knowledge_related_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    knowledge_merge_max_updates: int = Field(default=5, ge=1)
    enable_logo: bool = Field(default=True, description="Show ASCII logo on startup")
    timezone: str | None = Field(default="Asia/Shanghai", description="IANA timezone; None uses local time")
    language: str = Field(default="", description="Default language for LLM interactions")
    log_to_console: bool = Field(default=True, description="Log to console")
    log_to_file: bool = Field(default=True, description="Log to file")
    mcp_servers: dict[str, dict] = Field(default_factory=dict, description="MCP server configs by name")
    plugins: list[str] = Field(default_factory=list, description="Installed plugins enabled for this app")
    service: ComponentConfig = Field(default_factory=ComponentConfig, description="Service endpoint config")
    jobs: dict[str, JobConfig] = Field(default_factory=dict, description="Job definitions keyed by job name")
    thread_pool_max_workers: int = Field(default=0, description="Max worker threads; 0 to disable")
    components: dict[str, dict[str, ComponentConfig]] = Field(
        default_factory=dict,
        description="Component registry keyed by type then name",
    )

    @field_validator("components", mode="before")
    @classmethod
    def normalize_component_types(cls, value):
        """Canonicalize built-in enums and allow plugin-defined component type names."""
        if value is None:
            return {}
        if not isinstance(value, dict):
            return value
        return {component_type_name(component_type): group for component_type, group in value.items()}

    @field_validator("workspace_dir", mode="before")
    @classmethod
    def normalize_workspace_dir(cls, value) -> str:
        """Expand home-relative paths once so every component sees the same absolute workspace."""
        return str(Path(value).expanduser().resolve(strict=False))

    @field_validator("session_dir")
    @classmethod
    def validate_session_dir(cls, value: str) -> str:
        """Keep standard session storage inside the configured workspace."""
        path = value.strip()
        if PurePosixPath(path.replace("\\", "/")).is_absolute() or PureWindowsPath(path).anchor:
            raise ValueError("session_dir must be a workspace-relative path")
        return value
