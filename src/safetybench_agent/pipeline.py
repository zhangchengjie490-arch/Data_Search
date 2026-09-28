from __future__ import annotations

import json
import os
import re
import traceback
from pathlib import Path
from typing import Any

from .curation import (
    build_project,
    external_artifact,
    extract_urls,
    quality_profile,
    resolve_family,
    slugify,
    strip_urls,
)
from .hf_client import HuggingFaceSource, content_hash, infer_risk_level, infer_sample_role, sanitize_sample
from .llm import CompatibleLLMClassifier, merge_classifications
from .models import Classification, Evidence, ProjectRecord, SampleRecord, utc_now
from .storage import Store
from .taxonomy import RuleClassifier, load_taxonomy


class SafetyBenchPipeline:
    def __init__(self, *, db_path: str | Path, config_dir: str | Path):
        self.config_dir = Path(config_dir)
        self.settings = json.loads((self.config_dir / "settings.json").read_text(encoding="utf-8"))
        self.taxonomy = load_taxonomy(self.config_dir / "taxonomy.json")
        self.store = Store(db_path)
        self.store.initialize()
        self.rules = RuleClassifier(self.taxonomy)
        self.llm = CompatibleLLMClassifier(
            self.taxonomy,
            settings=self.settings,
            timeout=float(self.settings.get("request_timeout_seconds", 30)) * 2,
        )

    def make_source(self) -> HuggingFaceSource:
        return HuggingFaceSource(
            token=os.getenv("HF_TOKEN") or None,
            timeout=float(self.settings.get("request_timeout_seconds", 30)),
        )

    @staticmethod
    def _metadata_text(artifact: Any, readme: str = "") -> str:
        raw = artifact.raw if hasattr(artifact, "raw") else artifact.get("raw", {})
        tags = artifact.tags if hasattr(artifact, "tags") else artifact.get("tags", [])
        name = artifact.name if hasattr(artifact, "name") else artifact.get("name", "")
        description = artifact.description if hasattr(artifact, "description") else artifact.get("description", "")
        return "\n".join([
            name, description, " ".join(tags), json.dumps(raw.get("cardData", raw.get("card_data", {})), ensure_ascii=False), readme,
        ])

    def _artifact_scope(self, artifact: Any) -> tuple[str, int, int]:
        raw = artifact.raw if hasattr(artifact, "raw") else artifact.get("raw", {})
        source_queries = (
            artifact.source_queries if hasattr(artifact, "source_queries")
            else artifact.get("source_queries", [])
        )
        repo_type = artifact.repo_type if hasattr(artifact, "repo_type") else artifact.get("repo_type")
        downloads = int(raw.get("downloads") or 0)
        likes = int(raw.get("likes") or 0)
        if repo_type != "dataset":
            return "out_of_scope", downloads, likes
        if any(query.startswith("priority:") for query in source_queries):
            return "priority", downloads, likes
        return "safety_dataset", downloads, likes

    def _reconcile_publication_scope(self) -> dict[str, int]:
        changes = {"promoted": 0, "demoted": 0}
        for project in self.store.list_projects():
            ranked = [self._artifact_scope(artifact) for artifact in project.get("artifacts", [])]
            if not ranked:
                continue
            best_scope, downloads, likes = max(
                ranked,
                key=lambda item: ({"priority": 2, "safety_dataset": 1}.get(item[0], 0), item[1], item[2]),
            )
            downloads = max(item[1] for item in ranked)
            likes = max(item[2] for item in ranked)
            quality = dict(project.get("quality", {}))
            quality.update({
                "catalog_scope": best_scope,
                "hf_downloads": downloads,
                "hf_likes": likes,
            })
            old_status = project.get("status", "review_pending")
            new_status = old_status
            curated = quality.get("curation") == "seed_verified"
            classification = project.get("classification", {})
            if not curated and best_scope in {"priority", "safety_dataset"} and classification.get("is_ai_safety"):
                new_status = "published"
            elif not curated and (
                best_scope == "out_of_scope" or not classification.get("is_ai_safety")
            ):
                new_status = "excluded"
            if old_status == "published" and new_status != "published":
                self.store.add_change(
                    "project_scope_demoted", project_id=project["project_id"],
                    details={"reason": "not_ai_safety_dataset"},
                )
            self.store.update_project_catalog_scope(project["project_id"], new_status, quality)
            if old_status != new_status:
                changes["promoted" if new_status == "published" else "demoted"] += 1
        return changes

    def _classify(
        self,
        text: str,
        source: str,
        *,
        attack_vectors_from_samples: bool = False,
    ) -> Classification:
        rules = self.rules.classify(text, source=source)
        semantic = None
        if self.llm.enabled and rules.relevance_confidence >= 0.45:
            try:
                semantic = self.llm.classify(
                    text, rules,
                    attack_vectors_from_samples=attack_vectors_from_samples,
                )
            except Exception as exc:
                rules.evidence.append(Evidence(source="llm_error", text=str(exc)[:500]))
        return merge_classifications(
            rules, semantic,
            llm_owns_attack_vectors=attack_vectors_from_samples,
            llm_owns_safety_categories=True,
        )

    def import_legacy(self, path: str | Path) -> dict[str, int]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        stats = {"projects": 0, "artifacts": 0, "skipped": 0}
        for item in payload.get("source", []):
            if not item.get("id") or item.get("id") == "-1" or not item.get("name"):
                stats["skipped"] += 1
                continue
            project_id = slugify(item["id"])
            was_known = self.store.get_project(project_id) is not None
            text = "\n".join([item.get("name", ""), item.get("desc", ""), " ".join(item.get("variants", []))])
            classification = self._classify(text, "legacy_seed")
            classification.is_ai_safety = True
            classification.is_evaluation_asset = True
            classification.relevance_confidence = 0.99
            classification.classification_confidence = max(0.9, classification.classification_confidence)
            classification.evidence.append(Evidence(source="legacy_seed", text="用户维护的已知测评基准种子"))
            urls = extract_urls(item.get("desc", ""))
            summary = strip_urls(item.get("desc", ""))
            project = ProjectRecord(
                project_id=project_id,
                name=item["name"],
                summary_zh=summary,
                primary_url=urls[0] if urls else "",
                variants=item.get("variants", []),
                classification=classification,
                quality={"curation": "seed_verified", "documentation": "unknown", "reproducibility": "unknown"},
                status="published",
                updated_at=utc_now(),
            )
            self.store.upsert_project(project)
            stats["projects"] += 1
            for index, url in enumerate(urls):
                artifact = external_artifact(url, item["name"], index)
                self.store.upsert_artifact(artifact)
                self.store.link_project_artifact(project_id, artifact.artifact_id, "primary" if index == 0 else "related")
                stats["artifacts"] += 1
            if not was_known:
                self.store.add_change("seed_imported", project_id=project_id, details={"source": str(path)})
        return stats

    def discover(self, *, limit_per_query: int | None = None) -> dict[str, Any]:
        run_id = self.store.begin_run("discover")
        stats: dict[str, Any] = {
            "found": 0, "new": 0, "changed": 0, "projects": 0, "demoted": 0, "errors": []
        }
        source = self.make_source()
        try:
            artifacts, errors = source.discover(self.settings, limit_per_query=limit_per_query)
            stats["found"] = len(artifacts)
            stats["errors"] = errors
            projects = self.store.list_projects()
            for artifact in artifacts:
                previous_artifact = self.store.get_artifact(artifact.artifact_id)
                is_new, changed = self.store.upsert_artifact(artifact)
                stats["new"] += int(is_new)
                stats["changed"] += int(changed and not is_new)
                classification = self._classify(self._metadata_text(artifact), "hf_metadata")
                # 发现阶段还没有示例样本，攻击方式不得使用关键词规则结果冒充模型结论。
                # 深度解析完成后会由 Qwen 基于 README 和脱敏样本重新写入。
                classification.attack_vectors = []
                existing_id = resolve_family(artifact, projects)
                existing = self.store.get_project(existing_id) if existing_id else None
                project_id = existing_id or f"hf-{slugify(artifact.repo_id)}"
                project = build_project(
                    artifact, classification, project_id=project_id,
                    summary=existing.get("summary_zh") if existing else None,
                    variants=existing.get("variants", []) if existing else [],
                    quality=existing.get("quality", {}) if existing else {},
                    curated=bool(existing and existing.get("quality", {}).get("curation") == "seed_verified"),
                    publish_threshold=float(self.settings.get("auto_publish_confidence", 0.9)),
                )
                if existing:
                    project.name = existing["name"]
                    project.primary_url = existing["primary_url"] or artifact.canonical_url
                    if existing["status"] == "published" and existing.get("quality", {}).get("curation") == "seed_verified":
                        project.status = "published"
                if not (existing and existing.get("quality", {}).get("curation") == "seed_verified"):
                    project.status = "published" if classification.is_ai_safety else "excluded"
                self.store.upsert_project(project)
                self.store.link_project_artifact(project_id, artifact.artifact_id)
                if classification.is_ai_safety:
                    if is_new or changed or not previous_artifact or previous_artifact.get("status") in {
                        "discovered", "enrichment_failed"
                    }:
                        self.store.set_artifact_status(artifact.artifact_id, "pending_enrichment")
                else:
                    self.store.set_artifact_status(artifact.artifact_id, "excluded")
                if is_new:
                    self.store.add_change("artifact_discovered", artifact_id=artifact.artifact_id,
                                          project_id=project_id, details={"queries": artifact.source_queries})
                elif changed:
                    self.store.add_change("artifact_changed", artifact_id=artifact.artifact_id,
                                          project_id=project_id, details={"sha": artifact.sha})
                if project.status == "review_pending":
                    self.store.enqueue_review(project_id, "自动分类置信度不足或资产类型待确认")
                stats["projects"] += 1
                if not existing:
                    projects.append(project.model_dump())
            scope_changes = self._reconcile_publication_scope()
            stats["demoted"] = scope_changes["demoted"]
            self.store.finish_run(run_id, stats)
            return stats
        except Exception as exc:
            stats["traceback"] = traceback.format_exc(limit=8)
            self.store.finish_run(run_id, stats, str(exc))
            raise
        finally:
            source.close()

    def enrich(self, *, limit: int | None = None) -> dict[str, Any]:
        run_id = self.store.begin_run("enrich")
        limit = limit or int(self.settings.get("max_enrich_per_run", 50))
        artifacts = self.store.list_artifacts(status="pending_enrichment")
        artifacts.sort(
            key=lambda artifact: (
                {"priority": 2, "safety_dataset": 1}.get(self._artifact_scope(artifact)[0], 0),
                self._artifact_scope(artifact)[1],
                self._artifact_scope(artifact)[2],
            ),
            reverse=True,
        )
        artifacts = artifacts[:limit]
        stats: dict[str, Any] = {"attempted": len(artifacts), "completed": 0, "failed": 0, "samples": 0, "errors": []}
        source = self.make_source()
        try:
            for stored in artifacts:
                try:
                    counts = self.enrich_artifact(stored["artifact_id"], source=source)
                    stats["completed"] += 1
                    stats["samples"] += counts.get("samples", 0)
                except Exception as exc:
                    stats["failed"] += 1
                    stats["errors"].append({"artifact_id": stored["artifact_id"], "error": str(exc)})
                    self.store.set_artifact_status(stored["artifact_id"], "enrichment_failed")
            stats["scope_changes"] = self._reconcile_publication_scope()
            self.store.finish_run(run_id, stats)
            return stats
        except Exception as exc:
            self.store.finish_run(run_id, stats, str(exc))
            raise
        finally:
            source.close()

    @staticmethod
    def _representative_positions(length: int, limit: int) -> list[int]:
        if length <= 0 or limit <= 0:
            return []
        if length <= limit:
            return list(range(length))
        if limit == 1:
            return [length // 2]
        return sorted({round(index * (length - 1) / (limit - 1)) for index in range(limit)})

    def _collect_sample_candidates(
        self,
        source: HuggingFaceSource,
        artifact: dict[str, Any] | Any,
        project_id: str,
        *,
        add_tasks: bool = True,
        target_candidates: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
        """从 Dataset Viewer 和仓库结构化文件收集去重后的脱敏候选样本。"""
        repo_id = artifact.repo_id if hasattr(artifact, "repo_id") else artifact["repo_id"]
        artifact_id = artifact.artifact_id if hasattr(artifact, "artifact_id") else artifact["artifact_id"]
        revision = artifact.sha if hasattr(artifact, "sha") else artifact.get("sha")
        max_candidates = max(3, int(self.settings.get("max_candidate_samples", 120)))
        raw_target = min(
            max_candidates,
            max(3, target_candidates or int(self.settings.get("raw_fallback_target_samples", 39))),
        )
        report: dict[str, Any] = {
            "viewer_candidates": 0,
            "raw_candidates": 0,
            "readme_candidates": 0,
            "source_methods": [],
            "errors": [],
        }
        candidates: list[dict[str, Any]] = []
        fingerprints: set[str] = set()
        task_count = 0

        def append_candidate(entry: dict[str, Any], source_method: str) -> bool:
            content = sanitize_sample(entry.get("row", {}))
            if not isinstance(content, dict):
                content = {"value": content}
            fingerprint = content_hash(json.dumps(content, ensure_ascii=False, sort_keys=True, default=str))
            if fingerprint in fingerprints or len(candidates) >= max_candidates:
                return False
            fingerprints.add(fingerprint)
            candidates.append({
                "artifact_id": artifact_id,
                "config": str(entry.get("config", "default")),
                "split": str(entry.get("split", "train")),
                "row_index": int(entry.get("row_idx", len(candidates))),
                "content": content,
                "source_method": source_method,
            })
            count_key = {
                "dataset_viewer": "viewer_candidates",
                "raw_file": "raw_candidates",
                "readme_example": "readme_candidates",
            }[source_method]
            report[count_key] += 1
            return True

        try:
            splits = source.dataset_splits(repo_id)[:3]
        except Exception as exc:
            splits = []
            report["errors"].append(f"dataset_viewer/splits: {exc}")
        per_split = max(3, max_candidates // max(1, len(splits)))
        for split_info in splits:
            config = str(split_info.get("config", "default"))
            split = str(split_info.get("split", "train"))
            try:
                preview = source.first_rows(repo_id, config, split)
            except Exception as exc:
                report["errors"].append(f"dataset_viewer/{config}/{split}: {exc}")
                break
            features = preview.get("features", {})
            if add_tasks:
                self.store.add_task(project_id, artifact_id, config, split, features)
                task_count += 1
            rows = preview.get("rows", [])
            for position in self._representative_positions(len(rows), min(per_split, max_candidates - len(candidates))):
                entry = dict(rows[position])
                entry.update({"config": config, "split": split})
                append_candidate(entry, "dataset_viewer")
            if len(candidates) >= max_candidates:
                break

        if len(candidates) < raw_target:
            max_file_bytes = int(float(self.settings.get("raw_fallback_max_file_mb", 50)) * 1024 * 1024)
            raw_rows, raw_errors = source.raw_file_rows(
                repo_id,
                revision=revision,
                limit=max_candidates,
                max_file_bytes=max_file_bytes,
            )
            report["errors"].extend(f"raw_file/{error}" for error in raw_errors)
            raw_tasks: set[tuple[str, str]] = set()
            for entry in raw_rows:
                config = str(entry.get("config", "raw"))
                split = str(entry.get("split", "data"))
                if add_tasks and (config, split) not in raw_tasks:
                    raw_tasks.add((config, split))
                    row = entry.get("row", {})
                    schema = {str(key): type(value).__name__ for key, value in row.items()} if isinstance(row, dict) else {}
                    self.store.add_task(project_id, artifact_id, config, split, schema)
                    task_count += 1
                append_candidate(entry, "raw_file")
                if len(candidates) >= max_candidates:
                    break

        if len(candidates) < raw_target:
            try:
                readme = source.readme(repo_id, "dataset", revision)
                for entry in self._readme_example_rows(readme, limit=max_candidates):
                    append_candidate(entry, "readme_example")
                    if len(candidates) >= raw_target:
                        break
            except Exception as exc:
                report["errors"].append(f"readme_examples: {exc}")

        report["source_methods"] = [
            method for method, count_key in (
                ("dataset_viewer", "viewer_candidates"),
                ("raw_file", "raw_candidates"),
                ("readme_example", "readme_candidates"),
            ) if report[count_key]
        ]
        if not candidates and not report["errors"]:
            report["errors"].append("未发现 Dataset Viewer 预览或可安全解析的结构化原始文件")
        return candidates, report, task_count

    @staticmethod
    def _readme_example_rows(readme: str, *, limit: int = 30) -> list[dict[str, Any]]:
        """只提取 README 中明确的 JSON 或 Example/Sample/示例片段。"""
        if not readme or limit <= 0:
            return []
        result: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(value: Any) -> None:
            if len(result) >= limit:
                return
            item = value if isinstance(value, dict) else {"readme_example": value}
            fingerprint = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            if fingerprint not in seen:
                seen.add(fingerprint)
                result.append({
                    "config": "readme",
                    "split": "documented_examples",
                    "row_idx": len(result),
                    "row": item,
                })

        for match in re.finditer(r"```(?:json|jsonl)\s*\n(.*?)```", readme, re.I | re.S):
            block = match.group(1).strip()
            try:
                parsed = json.loads(block)
                values = parsed if isinstance(parsed, list) else [parsed]
                for value in values:
                    add(value)
            except json.JSONDecodeError:
                for line in block.splitlines():
                    try:
                        add(json.loads(line.strip()))
                    except (json.JSONDecodeError, TypeError):
                        continue

        active = False
        for raw_line in readme.splitlines():
            line = raw_line.strip()
            if re.match(r"^#{1,6}\s+", line):
                active = bool(re.search(r"\b(?:examples?|samples?)\b|示例|样例", line, re.I))
                continue
            if not active or not line:
                continue
            if re.fullmatch(r"\|?[\s:|-]+\|?", line):
                continue
            if line.startswith("|") and line.endswith("|"):
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if any(cells) and not all(re.fullmatch(r"[-: ]+", cell or "-") for cell in cells):
                    add({f"column_{index + 1}": value for index, value in enumerate(cells)})
                continue
            bullet = re.match(r"^(?:[-*+] |\d+[.)]\s+)(.+)", line)
            if bullet and len(bullet.group(1).strip()) >= 12:
                add(bullet.group(1).strip())
        return result[:limit]

    def _select_category_samples(
        self,
        candidates: list[dict[str, Any]],
        categories: list[str],
        *,
        per_category: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """每个安全类别选择固定数量样本；类别内不重复，类别间优先不复用。"""
        quota = max(1, per_category or int(self.settings.get("samples_per_safety_category", 3)))
        allowed_categories = self.taxonomy["dimensions"]["safety_categories"]
        categories = list(dict.fromkeys(category for category in categories if category in allowed_categories))
        if not candidates or not categories:
            return [], {category: 0 for category in categories}

        searchable = [
            json.dumps(candidate.get("content", {}), ensure_ascii=False, default=str).casefold()
            for candidate in candidates
        ]
        used_globally: set[int] = set()
        assignments: dict[int, set[str]] = {}
        per_category_counts: dict[str, int] = {}

        for category_index, category in enumerate(categories):
            terms = [str(term).casefold() for term in allowed_categories[category] if len(str(term).strip()) >= 2]
            scores = [
                sum(max(1, len(term) // 4) for term in terms if term in text)
                for text in searchable
            ]
            ranked = sorted(range(len(candidates)), key=lambda index: (scores[index], -index), reverse=True)
            selected: list[int] = []

            for index in ranked:
                if scores[index] > 0 and index not in used_globally:
                    selected.append(index)
                    if len(selected) == quota:
                        break
            if len(selected) < quota:
                remaining = [index for index in range(len(candidates)) if index not in used_globally and index not in selected]
                if remaining:
                    shift = (category_index * quota) % len(remaining)
                    remaining = remaining[shift:] + remaining[:shift]
                    selected.extend(remaining[:quota - len(selected)])
            if len(selected) < quota:
                for index in ranked:
                    if index not in selected:
                        selected.append(index)
                        if len(selected) == quota:
                            break

            for index in selected:
                assignments.setdefault(index, set()).add(category)
                used_globally.add(index)
            per_category_counts[category] = len(selected)

        selected_samples: list[dict[str, Any]] = []
        for index in sorted(assignments):
            item = dict(candidates[index])
            item["safety_categories"] = sorted(assignments[index])
            selected_samples.append(item)
        return selected_samples, per_category_counts

    def _persist_selected_samples(
        self,
        project_id: str,
        selected: list[dict[str, Any]],
    ) -> int:
        for item in selected:
            content = item["content"]
            self.store.add_sample(SampleRecord(
                project_id=project_id,
                artifact_id=item["artifact_id"],
                config=item["config"],
                split=item["split"],
                row_index=item["row_index"],
                sample_role=infer_sample_role(content),
                risk_level=infer_risk_level(content),
                safety_categories=item.get("safety_categories", []),
                source_method=item.get("source_method", "dataset_viewer"),
                content=content,
            ))
        return len(selected)

    def enrich_artifact(self, artifact_id: str, *, source: HuggingFaceSource | None = None) -> dict[str, Any]:
        own_source = source is None
        source = source or self.make_source()
        counts = {"tasks": 0, "samples": 0}
        try:
            stored = self.store.get_artifact(artifact_id)
            if not stored:
                raise KeyError(f"未知 artifact: {artifact_id}")
            if stored["repo_type"] not in ("dataset", "space"):
                self.store.set_artifact_status(artifact_id, "external")
                return counts
            refreshed = source.refresh_artifact(stored["repo_id"], stored["repo_type"])
            refreshed.source_queries = sorted(set(stored.get("source_queries", []) + ["refresh"]))
            _, changed = self.store.upsert_artifact(refreshed)
            readme = source.readme(refreshed.repo_id, refreshed.repo_type, refreshed.sha)
            if readme:
                is_new_snapshot = self.store.add_snapshot(
                    refreshed.artifact_id, "readme", content_hash(readme), readme
                )
                if is_new_snapshot and changed:
                    project = self.store.get_project_for_artifact(artifact_id)
                    self.store.add_change("readme_changed", artifact_id=artifact_id,
                                          project_id=project["project_id"] if project else None)

            project_data = self.store.get_project_for_artifact(artifact_id)
            project_id = project_data["project_id"] if project_data else f"hf-{slugify(refreshed.repo_id)}"
            attack_sample_evidence: list[dict[str, Any]] = []
            attack_sample_limit = int(self.settings.get("max_samples_for_attack_classification", 8))
            sample_candidates: list[dict[str, Any]] = []
            sample_report: dict[str, Any] = {
                "viewer_candidates": 0, "raw_candidates": 0,
                "readme_candidates": 0, "source_methods": [], "errors": [],
            }

            if refreshed.repo_type == "dataset":
                known_categories = (project_data or {}).get("classification", {}).get("safety_categories", [])
                target_candidates = (
                    int(self.settings.get("samples_per_safety_category", 3))
                    if known_categories else None
                )
                sample_candidates, sample_report, counts["tasks"] = self._collect_sample_candidates(
                    source, refreshed, project_id, target_candidates=target_candidates
                )
                for candidate in sample_candidates[:attack_sample_limit]:
                    attack_sample_evidence.append({
                        "config": candidate["config"],
                        "split": candidate["split"],
                        "sample_role": infer_sample_role(candidate["content"]),
                        "content": candidate["content"],
                    })

            classification_input = self._metadata_text(refreshed, readme)
            classification_input += "\n\n脱敏代表性样本：\n" + json.dumps(
                attack_sample_evidence, ensure_ascii=False, default=str
            )
            classification = self._classify(
                classification_input,
                "hf_readme_and_samples",
                attack_vectors_from_samples=True,
            )

            selected, per_category = self._select_category_samples(
                sample_candidates, classification.safety_categories
            )
            if sample_candidates:
                self.store.clear_artifact_samples(artifact_id)
            counts["samples"] = self._persist_selected_samples(project_id, selected)
            quota = int(self.settings.get("samples_per_safety_category", 3))
            sample_report.update({
                "samples_per_category_target": quota,
                "per_category": per_category,
                "shortages": {
                    category: quota - count for category, count in per_category.items() if count < quota
                },
                "stored_unique_samples": counts["samples"],
            })

            quality = quality_profile(
                refreshed, readme, classification,
                has_tasks=counts["tasks"] > 0, has_samples=counts["samples"] > 0,
            )
            ranked_scope = [
                self._artifact_scope(item)
                for item in self.store.list_project_artifacts(project_id)
            ] or [self._artifact_scope(refreshed)]
            scope, downloads, likes = max(
                ranked_scope,
                key=lambda item: ({"priority": 2, "safety_dataset": 1}.get(item[0], 0), item[1], item[2]),
            )
            downloads = max(item[1] for item in ranked_scope)
            likes = max(item[2] for item in ranked_scope)
            quality.update({
                "catalog_scope": scope,
                "hf_downloads": downloads,
                "hf_likes": likes,
                "sample_collection": sample_report,
            })
            curated = bool(project_data and project_data.get("quality", {}).get("curation") == "seed_verified")
            if curated:
                quality["curation"] = "seed_verified"
            curated_summary = (project_data or {}).get("summary_zh") if curated else None
            project = build_project(
                refreshed, classification, project_id=project_id,
                summary=curated_summary,
                variants=(project_data or {}).get("variants", []), quality=quality, curated=curated,
                publish_threshold=float(self.settings.get("auto_publish_confidence", 0.9)),
            )
            if project_data:
                project.name = project_data["name"]
                project.primary_url = project_data["primary_url"] or refreshed.canonical_url
                if project_data["status"] == "published" and project_data.get("quality", {}).get("curation") == "seed_verified":
                    project.status = "published"
            if not curated:
                project.status = "published" if classification.is_ai_safety else "excluded"
            self.store.upsert_project(project)
            self.store.set_artifact_status(artifact_id, "enriched")
            if project.status == "review_pending":
                self.store.enqueue_review(project_id, "深度解析后仍需人工确认")
            return counts
        finally:
            if own_source:
                source.close()

    def resample_by_category(
        self, *, limit: int | None = None, only_incomplete: bool = False,
        retry_shortages: bool = False, project_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """为现有已发布 Hugging Face 数据集按安全类别重新采集示例。"""
        projects = [
            project for project in self.store.list_projects(published_only=True)
            if any(artifact.get("repo_type") == "dataset" for artifact in project.get("artifacts", []))
            and project.get("classification", {}).get("safety_categories")
        ]
        if only_incomplete:
            projects = [
                project for project in projects
                if int(project.get("quality", {}).get("sample_collection", {}).get(
                    "samples_per_category_target", 0
                )) != int(self.settings.get("samples_per_safety_category", 3))
            ]
        if retry_shortages:
            projects = [
                project for project in projects
                if project.get("quality", {}).get("sample_collection", {}).get("shortages")
            ]
        if project_ids:
            selected_ids = set(project_ids)
            projects = [project for project in projects if project["project_id"] in selected_ids]
        if limit is not None:
            projects = projects[:limit]
        quota = int(self.settings.get("samples_per_safety_category", 3))
        stats: dict[str, Any] = {
            "projects": len(projects), "completed": 0, "partial": 0, "inaccessible": 0,
            "samples": 0, "category_targets": 0, "category_slots_filled": 0,
            "viewer_candidates": 0, "raw_candidates": 0, "shortages": [], "errors": [],
        }
        source = self.make_source()
        try:
            for project in projects:
                categories = list(dict.fromkeys(project["classification"].get("safety_categories", [])))
                stats["category_targets"] += len(categories) * quota
                candidates: list[dict[str, Any]] = [
                    {
                        "artifact_id": sample["artifact_id"],
                        "config": sample["config"],
                        "split": sample["split"],
                        "row_index": sample["row_index"],
                        "content": sample.get("content", {}),
                        "source_method": sample.get("source_method", "dataset_viewer"),
                    }
                    for sample in project.get("samples", [])
                    if sample.get("content")
                ]
                reports: list[dict[str, Any]] = []
                try:
                    if len(candidates) < quota:
                        for artifact in project.get("artifacts", []):
                            if artifact.get("repo_type") != "dataset":
                                continue
                            artifact_candidates, report, _ = self._collect_sample_candidates(
                                source,
                                artifact,
                                project["project_id"],
                                target_candidates=quota,
                            )
                            candidates.extend(artifact_candidates)
                            reports.append(report)
                            if len(candidates) >= quota:
                                break
                    unique: list[dict[str, Any]] = []
                    fingerprints: set[str] = set()
                    for candidate in candidates:
                        fingerprint = content_hash(json.dumps(
                            candidate["content"], ensure_ascii=False, sort_keys=True, default=str
                        ))
                        if fingerprint not in fingerprints:
                            fingerprints.add(fingerprint)
                            unique.append(candidate)
                    candidates = unique
                    selected, per_category = self._select_category_samples(candidates, categories)
                    filled = sum(per_category.values())
                    stats["category_slots_filled"] += filled
                    stats["viewer_candidates"] += sum(report["viewer_candidates"] for report in reports)
                    stats["raw_candidates"] += sum(report["raw_candidates"] for report in reports)
                    errors = [error for report in reports for error in report.get("errors", [])]

                    if candidates:
                        self.store.clear_project_samples(project["project_id"])
                        stored = self._persist_selected_samples(project["project_id"], selected)
                        stats["samples"] += stored
                    else:
                        stored = 0
                    shortages = {
                        category: quota - per_category.get(category, 0)
                        for category in categories if per_category.get(category, 0) < quota
                    }
                    if not candidates:
                        stats["inaccessible"] += 1
                    elif shortages:
                        stats["partial"] += 1
                    else:
                        stats["completed"] += 1
                    if shortages or not candidates:
                        stats["shortages"].append({
                            "project_id": project["project_id"],
                            "gated": any(bool(a.get("gated")) for a in project.get("artifacts", [])),
                            "candidate_count": len(candidates),
                            "per_category": per_category,
                            "shortages": shortages,
                            "errors": errors[:8],
                        })

                    quality = dict(project.get("quality", {}))
                    quality["sample_collection"] = {
                        "viewer_candidates": sum(report["viewer_candidates"] for report in reports),
                        "raw_candidates": sum(report["raw_candidates"] for report in reports),
                        "readme_candidates": sum(report.get("readme_candidates", 0) for report in reports),
                        "source_methods": sorted({
                            *(
                                method for report in reports
                                for method in report.get("source_methods", [])
                            ),
                            *(candidate.get("source_method", "dataset_viewer") for candidate in candidates),
                        }),
                        "samples_per_category_target": quota,
                        "per_category": per_category,
                        "shortages": shortages,
                        "stored_unique_samples": stored,
                        "errors": errors[:20],
                    }
                    self.store.update_project_catalog_scope(
                        project["project_id"], project.get("status", "published"), quality
                    )
                except Exception as exc:
                    stats["errors"].append({"project_id": project["project_id"], "error": str(exc)})
            return stats
        finally:
            source.close()

    def reclassify_attack_vectors(
        self, *, batch_size: int = 8, only_model_mismatch: bool = False
    ) -> dict[str, Any]:
        if not self.llm.enabled:
            raise RuntimeError("未配置可用的大模型，无法执行样本驱动的攻击方式分类")
        projects = self.store.list_projects(published_only=True)
        candidates = [
            project for project in projects
            if any(artifact.get("repo_type") == "dataset" for artifact in project.get("artifacts", []))
        ]
        if only_model_mismatch:
            expected_classifier = f"llm:{self.llm.model}"
            candidates = [
                project for project in candidates
                if project.get("classification", {}).get("classifier") != expected_classifier
            ]
        stats: dict[str, Any] = {
            "projects": len(candidates), "classified": 0, "without_attack_vectors": 0,
            "without_safety_categories": 0, "metadata_only_fallbacks": 0,
            "failed": 0, "errors": [],
        }
        max_samples = int(self.settings.get("max_samples_for_attack_classification", 8))

        def build_items(group: list[dict[str, Any]], *, include_sample_content: bool) -> list[dict[str, str]]:
            items: list[dict[str, str]] = []
            for project in group:
                evidence: dict[str, Any] = {
                    "name": project["name"],
                    "summary": project.get("summary_zh", ""),
                    "artifact_descriptions": [
                        artifact.get("description", "")[:1800]
                        for artifact in project.get("artifacts", [])
                        if artifact.get("repo_type") == "dataset"
                    ][:3],
                    "repository_tags": sorted({
                        str(tag)[:80]
                        for artifact in project.get("artifacts", [])
                        for tag in artifact.get("tags", [])
                    })[:40],
                }
                samples = project.get("samples", [])[:max_samples]
                if include_sample_content:
                    evidence["sanitized_samples"] = [
                        {
                            "config": sample.get("config"),
                            "split": sample.get("split"),
                            "sample_role": sample.get("sample_role"),
                            "content": sample.get("content", {}),
                        }
                        for sample in samples
                    ]
                else:
                    evidence["sample_structure"] = [
                        {
                            "config": sample.get("config"),
                            "split": sample.get("split"),
                            "sample_role": sample.get("sample_role"),
                            "fields": sorted(sample.get("content", {}).keys()),
                        }
                        for sample in samples
                    ]
                    evidence["collection_note"] = (
                        "原始攻击文本被上游内容安全策略拒绝传输；"
                        "请仅依据项目描述、仓库标签和样本字段结构分类。"
                    )
                items.append({
                    "id": project["project_id"],
                    "untrusted_evidence": json.dumps(
                        evidence, ensure_ascii=False, default=str
                    )[:7000],
                })
            return items

        def persist_decisions(
            group: list[dict[str, Any]], decisions: dict[str, dict[str, Any]]
        ) -> None:
            missing = [
                project["project_id"] for project in group
                if project["project_id"] not in decisions
            ]
            if missing:
                raise ValueError(f"大模型缺少 {len(missing)} 个项目的攻击方式结果")
            for project in group:
                decision = decisions[project["project_id"]]
                classification = dict(project.get("classification", {}))
                classification["safety_categories"] = decision["safety_categories"]
                classification["attack_vectors"] = decision["attack_vectors"]
                classification["classifier"] = f"llm:{self.llm.model}"
                classification["taxonomy_version"] = self.taxonomy["version"]
                evidence = [
                    item for item in classification.get("evidence", [])
                    if item.get("source") != "llm_sample_attack_analysis"
                ]
                evidence.append({
                    "source": "llm_sample_attack_analysis",
                    "text": decision["evidence_text"],
                    "url": None,
                })
                classification["evidence"] = evidence
                self.store.update_project_classification(project["project_id"], classification)
                stats["classified"] += 1
                stats["without_safety_categories"] += int(not decision["safety_categories"])
                stats["without_attack_vectors"] += int(not decision["attack_vectors"])

        for offset in range(0, len(candidates), max(1, batch_size)):
            group = candidates[offset:offset + max(1, batch_size)]
            try:
                items = build_items(group, include_sample_content=True)
                decisions = self.llm.classify_safety_profile_batch(items)
                persist_decisions(group, decisions)
            except Exception as exc:
                if len(group) == 1:
                    try:
                        fallback_items = build_items(group, include_sample_content=False)
                        fallback = self.llm.classify_safety_profile_batch(fallback_items)
                        persist_decisions(group, fallback)
                        stats["metadata_only_fallbacks"] += 1
                        continue
                    except Exception as fallback_exc:
                        exc = fallback_exc
                stats["failed"] += len(group)
                stats["errors"].append({
                    "project_ids": [project["project_id"] for project in group],
                    "error": str(exc),
                })
        return stats

    def refresh_repo(self, repo_id: str, repo_type: str = "dataset") -> dict[str, Any]:
        source = self.make_source()
        try:
            artifact = source.refresh_artifact(repo_id, repo_type)
            is_new, changed = self.store.upsert_artifact(artifact)
            if is_new:
                classification = self._classify(self._metadata_text(artifact), "webhook_metadata")
                project = build_project(
                    artifact, classification,
                    publish_threshold=float(self.settings.get("auto_publish_confidence", 0.9)),
                )
                self.store.upsert_project(project)
                self.store.link_project_artifact(project.project_id, artifact.artifact_id)
            self.store.set_artifact_status(artifact.artifact_id, "pending_enrichment")
            counts = self.enrich_artifact(artifact.artifact_id, source=source)
            return {"new": is_new, "changed": changed, **counts}
        finally:
            source.close()

    def run(self, *, legacy_path: str | Path | None = None, limit_per_query: int | None = None,
            enrich_limit: int | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if legacy_path and Path(legacy_path).exists():
            result["legacy"] = self.import_legacy(legacy_path)
        result["discover"] = self.discover(limit_per_query=limit_per_query)
        result["enrich"] = self.enrich(limit=enrich_limit)
        return result
