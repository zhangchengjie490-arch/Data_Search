from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Evidence(BaseModel):
    source: str
    text: str
    url: str | None = None


class Classification(BaseModel):
    evaluation_layers: list[str] = Field(default_factory=list)
    risk_domains: list[str] = Field(default_factory=list)
    safety_categories: list[str] = Field(default_factory=list)
    attack_vectors: list[str] = Field(default_factory=list)
    target_systems: list[str] = Field(default_factory=list)
    evaluation_objectives: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    asset_types: list[str] = Field(default_factory=list)
    usage_roles: list[str] = Field(default_factory=list)
    is_ai_safety: bool = False
    is_evaluation_asset: bool = False
    relevance_confidence: float = 0.0
    classification_confidence: float = 0.0
    evidence: list[Evidence] = Field(default_factory=list)
    classifier: str = "rules"
    taxonomy_version: str = ""


class ArtifactRecord(BaseModel):
    artifact_id: str
    repo_id: str
    repo_type: str = "dataset"
    name: str
    author: str | None = None
    canonical_url: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    gated: bool | str | None = None
    license: str | None = None
    sha: str | None = None
    last_modified: str | None = None
    source_queries: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class ProjectRecord(BaseModel):
    project_id: str
    name: str
    summary_zh: str = ""
    primary_url: str = ""
    variants: list[str] = Field(default_factory=list)
    classification: Classification = Field(default_factory=Classification)
    quality: dict[str, Any] = Field(default_factory=dict)
    status: str = "review_pending"
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class SampleRecord(BaseModel):
    project_id: str
    artifact_id: str
    config: str
    split: str
    row_index: int
    sample_role: str = "representative"
    risk_level: str = "unknown"
    safety_categories: list[str] = Field(default_factory=list)
    source_method: str = "dataset_viewer"
    sanitized: bool = True
    content: dict[str, Any]
