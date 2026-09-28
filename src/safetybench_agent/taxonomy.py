from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

from .models import Classification, Evidence


def load_taxonomy(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hits(text: str, terms: Iterable[str]) -> list[str]:
    lowered = text.casefold()
    matched: set[str] = set()
    for term in terms:
        needle = term.casefold()
        if len(needle) <= 3 and needle.isascii() and needle.isalnum():
            if re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", lowered):
                matched.add(term)
        elif needle in lowered:
            matched.add(term)
    return sorted(matched)


class RuleClassifier:
    """可审计的基线分类器；每个标签都来自显式词表命中。"""

    def __init__(self, taxonomy: dict):
        self.taxonomy = taxonomy

    def classify(self, text: str, *, source: str = "metadata") -> Classification:
        compact = re.sub(r"\s+", " ", text).strip()
        positive = _hits(compact, self.taxonomy["ai_safety_positive"])
        evaluation = _hits(compact, self.taxonomy["evaluation_positive"])
        negative = _hits(compact, self.taxonomy["scope_negative"])

        result = Classification(taxonomy_version=self.taxonomy["version"])
        for dimension, labels in self.taxonomy["dimensions"].items():
            selected: list[str] = []
            for label, terms in labels.items():
                if _hits(compact, terms):
                    selected.append(label)
            setattr(result, dimension, selected)

        for label, terms in self.taxonomy["asset_type_keywords"].items():
            if _hits(compact, terms):
                result.asset_types.append(label)
        for label, terms in self.taxonomy["usage_role_keywords"].items():
            if _hits(compact, terms):
                result.usage_roles.append(label)

        if not result.asset_types and evaluation:
            result.asset_types = ["static_benchmark"]
        if "train" in result.usage_roles and "evaluation" in result.usage_roles:
            result.usage_roles = ["both_train_and_eval"]

        safety_signal = len(positive) + min(3, sum(len(getattr(result, d)) for d in (
            "risk_domains", "attack_vectors", "evaluation_layers"
        )))
        eval_signal = len(evaluation) + len(result.asset_types)
        result.is_ai_safety = safety_signal > 0 and not (negative and safety_signal < 3)
        strong_eval_terms = {
            "benchmark", "evaluation", "test suite", "test set", "leaderboard",
            "评测", "基准", "测试集", "排行榜",
        }
        result.is_evaluation_asset = bool(set(evaluation) & strong_eval_terms) or bool(result.asset_types)
        result.relevance_confidence = round(
            max(0.0, min(0.99, 0.45 + 0.09 * safety_signal + 0.06 * eval_signal - 0.3 * len(negative))),
            2,
        ) if result.is_ai_safety else round(max(0.01, 0.25 - 0.08 * len(negative)), 2)

        populated = sum(bool(getattr(result, d)) for d in self.taxonomy["dimensions"])
        result.classification_confidence = round(min(0.94, 0.55 + 0.055 * populated + 0.02 * math.log1p(len(positive))), 2)
        matched = positive + evaluation
        if matched:
            result.evidence.append(Evidence(
                source=source,
                text="关键词证据：" + "、".join(matched[:12]),
            ))
        if negative:
            result.evidence.append(Evidence(
                source=source,
                text="范围排除信号：" + "、".join(negative[:8]),
            ))
        return result
