from __future__ import annotations

import hashlib
import csv
import json
import re
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from huggingface_hub import HfApi

from .models import ArtifactRecord


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dict__"):
        return {k: _jsonable(v) for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def _extract_license(tags: list[str], card_data: Any) -> str | None:
    for tag in tags:
        if tag.startswith("license:"):
            return tag.split(":", 1)[1]
    card = _jsonable(card_data)
    if isinstance(card, dict):
        value = card.get("license")
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value) if value else None
    return None


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


class HuggingFaceSource:
    def __init__(self, *, token: str | None = None, timeout: float = 30):
        self.token = token
        self.api = HfApi(token=token)
        headers = {"User-Agent": "safetybench-agent/0.1"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.http = httpx.Client(timeout=timeout, headers=headers, follow_redirects=True)

    def close(self) -> None:
        self.http.close()

    def discover(self, settings: dict, *, limit_per_query: int | None = None) -> tuple[list[ArtifactRecord], list[str]]:
        limit = limit_per_query or int(settings.get("limit_per_query", 20))
        found: dict[str, ArtifactRecord] = {}
        errors: list[str] = []
        expand = [
            "author", "description", "tags", "gated", "sha", "lastModified",
            "cardData", "downloads", "likes",
        ]
        min_downloads = int(settings.get("dataset_min_downloads", 0))
        min_likes = int(settings.get("dataset_min_likes", 0))
        if not settings.get("require_popularity_for_discovery", False):
            min_downloads = 0
            min_likes = 0

        for repo_id in settings.get("priority_dataset_ids", []):
            label = f"priority:{repo_id}"
            try:
                item = self.api.dataset_info(repo_id, files_metadata=False)
                self._merge_results(found, [item], "dataset", label)
            except Exception as exc:
                errors.append(f"{label}: {exc}")

        for dataset_filter in settings.get("dataset_filters", []):
            label = f"filter:{dataset_filter}"
            try:
                items = self.api.list_datasets(
                    filter=dataset_filter, sort=settings.get("dataset_sort", "downloads"), limit=limit,
                    expand=expand,
                )
                self._merge_results(
                    found, items, "dataset", label,
                    min_downloads=min_downloads, min_likes=min_likes,
                )
            except Exception as exc:  # 网络/API 错误不应使整个扫描失败
                errors.append(f"{label}: {exc}")

        for term in settings.get("dataset_search_terms", []):
            label = f"search:{term}"
            try:
                items = self.api.list_datasets(
                    search=term, sort=settings.get("dataset_sort", "downloads"), limit=limit,
                    expand=expand,
                )
                self._merge_results(
                    found, items, "dataset", label,
                    min_downloads=min_downloads, min_likes=min_likes,
                )
            except Exception as exc:
                errors.append(f"{label}: {exc}")

        if settings.get("include_spaces", False):
            for term in settings.get("space_search_terms", []):
                label = f"space:{term}"
                try:
                    items = self.api.list_spaces(search=term, sort="last_modified", limit=limit, full=True)
                    self._merge_results(found, items, "space", label)
                except Exception as exc:
                    errors.append(f"{label}: {exc}")
        artifacts = list(found.values())
        artifacts.sort(key=self._discovery_rank, reverse=True)
        return artifacts, errors

    def _merge_results(self, found: dict[str, ArtifactRecord], items: Any,
                       repo_type: str, source_query: str, *,
                       min_downloads: int = 0, min_likes: int = 0) -> None:
        for info in items:
            artifact = self.to_artifact(info, repo_type, [source_query])
            if not source_query.startswith("priority:"):
                downloads, likes = self._popularity(artifact)
                if downloads < min_downloads and likes < min_likes:
                    continue
            if artifact.artifact_id in found:
                current = found[artifact.artifact_id]
                current.source_queries = sorted(set(current.source_queries + [source_query]))
            else:
                found[artifact.artifact_id] = artifact

    @staticmethod
    def _popularity(artifact: ArtifactRecord) -> tuple[int, int]:
        raw = artifact.raw
        return int(raw.get("downloads") or 0), int(raw.get("likes") or 0)

    @classmethod
    def _discovery_rank(cls, artifact: ArtifactRecord) -> tuple[int, int, int]:
        downloads, likes = cls._popularity(artifact)
        priority = int(any(query.startswith("priority:") for query in artifact.source_queries))
        return priority, downloads, likes

    def to_artifact(self, info: Any, repo_type: str, source_queries: list[str]) -> ArtifactRecord:
        raw = _jsonable(info)
        repo_id = getattr(info, "id", None) or raw.get("id") or raw.get("name")
        if not repo_id:
            raise ValueError("Hugging Face 响应缺少仓库 ID")
        tags = list(getattr(info, "tags", None) or raw.get("tags") or [])
        card_data = getattr(info, "card_data", None) or raw.get("cardData") or raw.get("card_data")
        description = getattr(info, "description", None) or raw.get("description") or ""
        description = re.sub(r"\s+", " ", description).strip()
        author = getattr(info, "author", None) or raw.get("author")
        if not author and "/" in repo_id:
            author = repo_id.split("/", 1)[0]
        last_modified = getattr(info, "last_modified", None) or raw.get("lastModified") or raw.get("last_modified")
        sha = getattr(info, "sha", None) or raw.get("sha")
        base = "datasets" if repo_type == "dataset" else "spaces"
        return ArtifactRecord(
            artifact_id=f"hf:{repo_type}:{repo_id}",
            repo_id=repo_id,
            repo_type=repo_type,
            name=repo_id.split("/")[-1],
            author=author,
            canonical_url=f"https://huggingface.co/{base}/{repo_id}",
            description=description,
            tags=tags,
            gated=getattr(info, "gated", None) if hasattr(info, "gated") else raw.get("gated"),
            license=_extract_license(tags, card_data),
            sha=str(sha) if sha else None,
            last_modified=last_modified.isoformat() if isinstance(last_modified, datetime) else (str(last_modified) if last_modified else None),
            source_queries=source_queries,
            raw=raw,
        )

    def refresh_artifact(self, repo_id: str, repo_type: str) -> ArtifactRecord:
        if repo_type == "dataset":
            info = self.api.dataset_info(repo_id, files_metadata=False)
        elif repo_type == "space":
            info = self.api.space_info(repo_id, files_metadata=False)
        else:
            raise ValueError(f"不支持刷新 repo_type={repo_type}")
        return self.to_artifact(info, repo_type, ["refresh"])

    def readme(self, repo_id: str, repo_type: str, revision: str | None = None) -> str:
        base = "datasets" if repo_type == "dataset" else "spaces"
        rev = quote(revision or "main", safe="")
        url = f"https://huggingface.co/{base}/{quote(repo_id, safe='/')}/resolve/{rev}/README.md"
        response = self.http.get(
            url, timeout=httpx.Timeout(connect=8, read=12, write=8, pool=8)
        )
        if response.status_code == 404:
            return ""
        response.raise_for_status()
        return response.text

    def dataset_splits(self, repo_id: str) -> list[dict]:
        response = self.http.get(
            "https://datasets-server.huggingface.co/splits",
            params={"dataset": repo_id},
            timeout=httpx.Timeout(connect=8, read=12, write=8, pool=8),
        )
        if response.status_code in (401, 403, 404, 422, 500, 501, 503):
            return []
        response.raise_for_status()
        return response.json().get("splits", [])

    def first_rows(self, repo_id: str, config: str, split: str) -> dict:
        response = self.http.get(
            "https://datasets-server.huggingface.co/first-rows",
            params={"dataset": repo_id, "config": config, "split": split},
            timeout=httpx.Timeout(connect=8, read=12, write=8, pool=8),
        )
        if response.status_code in (401, 403, 404, 422, 500, 501, 503):
            return {}
        response.raise_for_status()
        return response.json()

    def raw_file_rows(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        limit: int = 120,
        max_file_bytes: int = 50 * 1024 * 1024,
    ) -> tuple[list[dict], list[str]]:
        """安全读取仓库中的结构化文件；不执行远程加载脚本。"""
        rows: list[dict] = []
        errors: list[str] = []
        try:
            api_url = f"https://huggingface.co/api/datasets/{quote(repo_id, safe='/')}"
            if revision:
                api_url += f"/revision/{quote(revision, safe='')}"
            response = self.http.get(
                api_url,
                params={"blobs": "true"},
                timeout=httpx.Timeout(connect=8, read=12, write=8, pool=8),
            )
            response.raise_for_status()
            siblings = response.json().get("siblings", [])
        except Exception as exc:
            return [], [f"dataset_info: {exc}"]
        candidates: list[tuple[int, int, str, int]] = []
        for sibling in siblings:
            filename = str(sibling.get("rfilename", "") or "")
            suffix = Path(filename).suffix.casefold()
            if suffix not in {".parquet", ".jsonl", ".json", ".csv", ".tsv", ".xlsx"}:
                continue
            size = int(sibling.get("size", 0) or 0)
            if size and size > max_file_bytes and suffix != ".parquet":
                continue
            lowered = filename.casefold()
            split_rank = 0 if any(token in lowered for token in ("test", "eval", "valid", "dev")) else 1
            format_rank = {
                ".parquet": 0, ".jsonl": 1, ".json": 2,
                ".csv": 3, ".tsv": 4, ".xlsx": 5,
            }[suffix]
            candidates.append((split_rank, format_rank, filename, size))
        for _, _, filename, known_size in sorted(candidates)[:3]:
            if len(rows) >= limit:
                break
            temporary_path: Path | None = None
            try:
                rev = quote(revision or "main", safe="")
                url = (
                    f"https://huggingface.co/datasets/{quote(repo_id, safe='/')}"
                    f"/resolve/{rev}/{quote(filename, safe='/')}"
                )
                suffix = Path(filename).suffix
                if suffix.casefold() == ".parquet" and known_size > max_file_bytes:
                    parsed = self._read_remote_parquet(url, limit - len(rows), token=self.token)
                else:
                    raw_timeout = httpx.Timeout(connect=10, read=12, write=10, pool=10)
                    with self.http.stream("GET", url, timeout=raw_timeout) as response:
                        response.raise_for_status()
                        declared_size = int(response.headers.get("content-length") or 0)
                        if declared_size and declared_size > max_file_bytes:
                            raise ValueError(
                                f"文件大小 {declared_size} 超过上限 {max_file_bytes} 字节"
                            )
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                            temporary_path = Path(handle.name)
                            downloaded = 0
                            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                                downloaded += len(chunk)
                                if downloaded > max_file_bytes:
                                    raise ValueError(f"下载内容超过上限 {max_file_bytes} 字节")
                                handle.write(chunk)
                    parsed = self._read_structured_file(temporary_path, limit - len(rows))
                config = f"raw:{filename}"
                split = self._infer_split(filename)
                for index, content in enumerate(parsed):
                    rows.append({
                        "config": config,
                        "split": split,
                        "row_idx": index,
                        "row": content,
                        "source_method": "raw_file",
                        "source_file": filename,
                    })
            except httpx.TimeoutException as exc:
                errors.append(f"{filename}: 下载超时（{exc}）")
                break
            except Exception as exc:
                errors.append(f"{filename}: {exc}")
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        return rows[:limit], errors

    @staticmethod
    def _read_remote_parquet(url: str, limit: int, *, token: str | None = None) -> list[dict]:
        """通过 HTTP Range 只读取远程 Parquet 的文本型示例列和首批行。"""
        import fsspec
        import pyarrow.parquet as pq

        headers = {"Authorization": f"Bearer {token}"} if token else None
        options = {"block_size": 256 * 1024}
        if headers:
            options["headers"] = headers
        with fsspec.open(url, "rb", **options) as handle:
            parquet = pq.ParquetFile(handle)
            names = list(parquet.schema.names)
            preferred_tokens = (
                "category", "prompt_text", "instruction", "question", "input",
                "classification", "label", "verdict", "behavior", "goal", "text",
            )
            excluded_tokens = ("image", "audio", "video", "bytes", "embedding")
            columns: list[str] = []
            for token in preferred_tokens:
                for name in names:
                    if (
                        name not in columns
                        and token in name.casefold()
                        and not any(value in name.casefold() for value in excluded_tokens)
                    ):
                        columns.append(name)
                        if len(columns) == 4:
                            break
                if len(columns) == 4:
                    break
            if not columns:
                columns = [
                    name for name in names
                    if not any(token in name.casefold() for token in excluded_tokens)
                ][:4]
            result: list[dict] = []
            for batch in parquet.iter_batches(
                batch_size=min(64, max(1, limit)), columns=columns
            ):
                result.extend(batch.to_pylist())
                if len(result) >= limit:
                    break
            return [item if isinstance(item, dict) else {"value": item} for item in result[:limit]]

    @staticmethod
    def _infer_split(filename: str) -> str:
        lowered = filename.casefold()
        for split in ("test", "validation", "valid", "dev", "train"):
            if split in lowered:
                return "validation" if split in {"valid", "dev"} else split
        return "data"

    @staticmethod
    def _read_structured_file(path: Path, limit: int) -> list[dict]:
        suffix = path.suffix.casefold()
        if suffix == ".xlsx":
            from openpyxl import load_workbook

            workbook = load_workbook(path, read_only=True, data_only=True)
            try:
                result: list[dict] = []
                for sheet in workbook.worksheets:
                    rows = sheet.iter_rows(values_only=True)
                    header_row = next(rows, None)
                    if not header_row:
                        continue
                    headers = [
                        str(value).strip() if value not in (None, "") else f"column_{index + 1}"
                        for index, value in enumerate(header_row)
                    ]
                    for values in rows:
                        if len(result) >= limit:
                            return result
                        if not any(value not in (None, "") for value in values):
                            continue
                        result.append({
                            headers[index] if index < len(headers) else f"column_{index + 1}": value
                            for index, value in enumerate(values)
                        })
                return result
            finally:
                workbook.close()
        if suffix == ".parquet":
            import pyarrow.parquet as pq

            result: list[dict] = []
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=min(64, max(1, limit))):
                result.extend(batch.to_pylist())
                if len(result) >= limit:
                    break
            return [item if isinstance(item, dict) else {"value": item} for item in result[:limit]]
        if suffix == ".jsonl":
            result = []
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if len(result) >= limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    value = json.loads(line)
                    result.append(value if isinstance(value, dict) else {"value": value})
            return result
        if suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(value, dict):
                for key in ("data", "rows", "examples", "items"):
                    if isinstance(value.get(key), list):
                        value = value[key]
                        break
                else:
                    value = [value]
            if not isinstance(value, list):
                value = [value]
            return [item if isinstance(item, dict) else {"value": item} for item in value[:limit]]
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            return [dict(row) for _, row in zip(range(limit), csv.DictReader(handle, delimiter=delimiter))]


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d .()-]{7,}\d)(?!\d)")
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)")
HIGH_RISK_RE = re.compile(
    r"\b(?:malware|exploit|phishing|weapon|suicide|self-harm|kill|bomb)\b|"
    r"恶意软件|漏洞利用|武器|自杀|自残|炸弹|杀害",
    re.I,
)


def sanitize_sample(value: Any, *, max_text: int = 500) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).casefold()
            if any(token in lowered for token in ("image", "audio", "video", "bytes", "blob")):
                clean[str(key)] = "[媒体内容已省略]"
            else:
                clean[str(key)] = sanitize_sample(item, max_text=max_text)
        return clean
    if isinstance(value, list):
        return [sanitize_sample(item, max_text=max_text) for item in value[:20]]
    if isinstance(value, str):
        text = EMAIL_RE.sub("[EMAIL]", value)
        text = PHONE_RE.sub("[PHONE]", text)
        text = CARD_RE.sub("[NUMBER]", text)
        if len(text) > max_text:
            text = text[:max_text].rstrip() + "…"
        return text
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return str(value)[:max_text]


def infer_risk_level(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if HIGH_RISK_RE.search(text) or re.search(
        r'"(?:severity|label)"\s*:\s*"(?:critical|high|harmful|unsafe|injection|jailbreak)"',
        text,
        re.I,
    ):
        return "high"
    return "low"


def infer_sample_role(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if re.search(r'"(?:label|category)"\s*:\s*"[^"]*(?:injection|jailbreak|harmful|unsafe|attack)', text, re.I):
        return "attack_or_unsafe"
    if re.search(r'"(?:label|category)"\s*:\s*"[^"]*(?:benign|safe|normal)', text, re.I):
        return "benign_control"
    return "representative"
