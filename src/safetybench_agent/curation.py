from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from typing import Iterable
from urllib.parse import urlparse

from .models import ArtifactRecord, Classification, ProjectRecord, utc_now


ZH_LABELS = {
    "model_behavior": "模型行为安全",
    "attack_resistance": "攻击抵抗能力",
    "system_agent_security": "系统与 Agent 安全",
    "harmful_content": "有害内容",
    "toxicity_hate": "毒性与仇恨言论",
    "bias_fairness": "偏见与公平性",
    "truthfulness": "真实性与幻觉",
    "privacy": "隐私泄露",
    "cyber_dual_use": "网络与双重用途风险",
    "high_stakes": "高风险领域",
    "deception_autonomy": "欺骗与自主性风险",
    "direct_jailbreak": "直接越狱",
    "prompt_injection": "提示注入",
    "indirect_prompt_injection": "间接提示注入",
    "adversarial_perturbation": "对抗扰动",
    "data_poisoning": "数据投毒",
    "secret_extraction": "秘密提取",
    "chat_model": "对话模型",
    "safety_classifier": "安全分类器",
    "rag_system": "RAG 系统",
    "tool_agent": "工具调用 Agent",
    "code_model": "代码模型",
    "multimodal_model": "多模态模型",
    "static_benchmark": "静态评测基准",
    "train_eval_dataset": "训练与评测混合数据集",
    "evaluation_suite": "评测套件",
    "interactive_environment": "交互式评测环境",
}


def slugify(value: str) -> str:
    value = value.strip().casefold()
    value = re.sub(r"^论文[:：]\s*", "", value)
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", value)
    value = value.strip("-")
    return value[:96] or hashlib.sha1(value.encode()).hexdigest()[:12]


def family_key(value: str) -> str:
    value = value.casefold()
    value = re.sub(r"^论文[:：]\s*", "", value)
    value = re.sub(r"\b(?:dataset|benchmark|evaluation|eval|test|train|official)\b", "", value)
    value = re.sub(r"(?:-|_|\s)?v(?:ersion)?\s*\d+(?:\.\d+)*", "", value)
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value)


def extract_urls(text: str) -> list[str]:
    candidates = re.findall(r"https?://[^\s｜|）)]+", text)
    return list(dict.fromkeys(url.rstrip(".,，。；;") for url in candidates))


def strip_urls(text: str) -> str:
    text = re.sub(r"https?://[^\s｜|）)]+", "", text)
    return re.sub(r"\s*[｜|]\s*", " ", text).strip(" 。;；")


def external_artifact(url: str, name: str, index: int = 0) -> ArtifactRecord:
    parsed = urlparse(url)
    repo_type = "external"
    repo_id = url
    if parsed.netloc == "huggingface.co":
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 3 and parts[0] in ("datasets", "spaces"):
            repo_type = "dataset" if parts[0] == "datasets" else "space"
            repo_id = "/".join(parts[1:3])
        elif len(parts) >= 2:
            repo_type = "model"
            repo_id = "/".join(parts[:2])
    elif parsed.netloc == "github.com":
        repo_type = "github"
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            repo_id = "/".join(parts[:2])
    elif "arxiv.org" in parsed.netloc:
        repo_type = "paper"
    digest = hashlib.sha1(f"{repo_type}:{repo_id}".encode("utf-8")).hexdigest()[:16]
    return ArtifactRecord(
        artifact_id=f"external:{digest}", repo_id=repo_id, repo_type=repo_type,
        name=name if index == 0 else f"{name} ({repo_type})", canonical_url=url,
        source_queries=["legacy_seed"], raw={"source": "legacy_seed"},
    )


def resolve_family(artifact: ArtifactRecord, projects: Iterable[dict]) -> str | None:
    candidates = [artifact.name, artifact.repo_id.split("/")[-1]]
    best: tuple[float, str] | None = None
    for project in projects:
        names = [project["name"], project["project_id"], *project.get("variants", [])]
        for left in candidates:
            for right in names:
                a, b = family_key(left), family_key(right)
                if not a or not b:
                    continue
                score = 1.0 if a == b else SequenceMatcher(None, a, b).ratio()
                if best is None or score > best[0]:
                    best = (score, project["project_id"])
    return best[1] if best and best[0] >= 0.86 else None


def quality_profile(artifact: ArtifactRecord, readme: str, classification: Classification,
                    has_tasks: bool, has_samples: bool) -> dict:
    raw_text = f"{artifact.description}\n{readme}".casefold()
    documentation = "A" if len(readme) > 2500 else ("B" if len(readme) > 500 else "C")
    reproducibility_signals = sum(term in raw_text for term in (
        "metric", "baseline", "evaluation", "github", "citation", "license"
    ))
    reproducibility = "A" if reproducibility_signals >= 5 else ("B" if reproducibility_signals >= 3 else "C")
    return {
        "documentation": documentation,
        "reproducibility": reproducibility,
        "license_clarity": "A" if artifact.license else "C",
        "data_preview": "available" if has_samples else "unavailable_or_gated",
        "task_structure": "available" if has_tasks else "unknown",
        "contamination_risk": "medium" if classification.is_evaluation_asset and has_samples else "unknown",
        "maintenance_status": "active" if artifact.last_modified else "unknown",
    }


def make_summary_zh(artifact: ArtifactRecord, classification: Classification) -> str:
    targets = [ZH_LABELS.get(item, item) for item in classification.target_systems[:2]]
    topics = [ZH_LABELS.get(item, item) for item in (
        classification.safety_categories + classification.risk_domains
        + classification.attack_vectors + classification.evaluation_layers
    )[:3]]
    assets = [ZH_LABELS.get(item, item) for item in classification.asset_types[:1]]
    target_text = "、".join(targets) or "AI 模型或系统"
    topic_text = "、".join(topics) or "安全性"
    asset_text = assets[0] if assets else "评测资产"
    return f"这是一个面向{target_text}的{asset_text}，主要评估{topic_text}。"


def build_project(artifact: ArtifactRecord, classification: Classification, *, project_id: str | None = None,
                  summary: str | None = None, variants: list[str] | None = None,
                  quality: dict | None = None, curated: bool = False,
                  publish_threshold: float = 0.9) -> ProjectRecord:
    pid = project_id or f"hf-{slugify(artifact.repo_id)}"
    if curated:
        status = "published"
    elif (
        classification.is_ai_safety
        and classification.is_evaluation_asset
        and classification.relevance_confidence >= publish_threshold
        and bool({"evaluation", "both_train_and_eval"} & set(classification.usage_roles))
        and bool(classification.asset_types)
    ):
        status = "published"
    elif classification.is_ai_safety:
        status = "review_pending"
    else:
        status = "excluded"
    return ProjectRecord(
        project_id=pid,
        name=artifact.name,
        summary_zh=summary or make_summary_zh(artifact, classification),
        primary_url=artifact.canonical_url,
        variants=variants or [],
        classification=classification,
        quality=quality or {},
        status=status,
        updated_at=utc_now(),
    )
