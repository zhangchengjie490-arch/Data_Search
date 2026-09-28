from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from .models import ArtifactRecord, ProjectRecord, SampleRecord, utc_now


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    summary_zh TEXT NOT NULL DEFAULT '',
    primary_url TEXT NOT NULL DEFAULT '',
    variants_json TEXT NOT NULL DEFAULT '[]',
    classification_json TEXT NOT NULL DEFAULT '{}',
    quality_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'review_pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    repo_id TEXT NOT NULL,
    repo_type TEXT NOT NULL,
    name TEXT NOT NULL,
    author TEXT,
    canonical_url TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    tags_json TEXT NOT NULL DEFAULT '[]',
    gated_json TEXT,
    license TEXT,
    sha TEXT,
    last_modified TEXT,
    source_queries_json TEXT NOT NULL DEFAULT '[]',
    raw_json TEXT NOT NULL DEFAULT '{}',
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'discovered',
    UNIQUE(repo_type, repo_id)
);

CREATE TABLE IF NOT EXISTS project_artifacts (
    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
    relationship TEXT NOT NULL DEFAULT 'primary',
    PRIMARY KEY(project_id, artifact_id)
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
    config TEXT NOT NULL,
    split TEXT NOT NULL,
    schema_json TEXT NOT NULL DEFAULT '{}',
    num_rows INTEGER,
    UNIQUE(artifact_id, config, split)
);

CREATE TABLE IF NOT EXISTS samples (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
    config TEXT NOT NULL,
    split TEXT NOT NULL,
    row_index INTEGER NOT NULL,
    sample_role TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    safety_categories_json TEXT NOT NULL DEFAULT '[]',
    source_method TEXT NOT NULL DEFAULT 'dataset_viewer',
    sanitized INTEGER NOT NULL,
    content_json TEXT NOT NULL,
    UNIQUE(project_id, artifact_id, config, split, row_index)
);

CREATE TABLE IF NOT EXISTS source_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(artifact_id, source_type, source_hash)
);

CREATE TABLE IF NOT EXISTS change_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_id TEXT,
    project_id TEXT,
    event_type TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_queue (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    reason TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    UNIQUE(project_id, reason, status)
);

CREATE TABLE IF NOT EXISTS crawl_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    stats_json TEXT NOT NULL DEFAULT '{}',
    error TEXT
);

CREATE TABLE IF NOT EXISTS webhook_events (
    webhook_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'received',
    received_at TEXT NOT NULL,
    processed_at TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS system_state (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS update_lock (
    lock_name TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    acquired_at TEXT NOT NULL
);
"""


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _load(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            sample_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(samples)").fetchall()
            }
            if "safety_categories_json" not in sample_columns:
                conn.execute(
                    "ALTER TABLE samples ADD COLUMN safety_categories_json TEXT NOT NULL DEFAULT '[]'"
                )
            if "source_method" not in sample_columns:
                conn.execute(
                    "ALTER TABLE samples ADD COLUMN source_method TEXT NOT NULL DEFAULT 'dataset_viewer'"
                )
            sample_sql_row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='samples'"
            ).fetchone()
            sample_sql = (sample_sql_row["sql"] if sample_sql_row else "").replace(" ", "").lower()
            if "unique(artifact_id,config,split,row_index)" in sample_sql:
                conn.execute("ALTER TABLE samples RENAME TO samples_legacy")
                conn.execute(
                    """
                    CREATE TABLE samples (
                        sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
                        artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
                        config TEXT NOT NULL,
                        split TEXT NOT NULL,
                        row_index INTEGER NOT NULL,
                        sample_role TEXT NOT NULL,
                        risk_level TEXT NOT NULL,
                        safety_categories_json TEXT NOT NULL DEFAULT '[]',
                        source_method TEXT NOT NULL DEFAULT 'dataset_viewer',
                        sanitized INTEGER NOT NULL,
                        content_json TEXT NOT NULL,
                        UNIQUE(project_id, artifact_id, config, split, row_index)
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO samples(
                        sample_id,project_id,artifact_id,config,split,row_index,sample_role,
                        risk_level,safety_categories_json,source_method,sanitized,content_json
                    )
                    SELECT sample_id,project_id,artifact_id,config,split,row_index,sample_role,
                        risk_level,safety_categories_json,source_method,sanitized,content_json
                    FROM samples_legacy
                    """
                )
                conn.execute("DROP TABLE samples_legacy")

    def begin_run(self, run_type: str) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO crawl_runs(run_type,status,started_at) VALUES(?,?,?)",
                (run_type, "running", utc_now()),
            )
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, stats: dict, error: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE crawl_runs SET status=?,finished_at=?,stats_json=?,error=? WHERE run_id=?",
                ("failed" if error else "completed", utc_now(), _dump(stats), error, run_id),
            )

    def upsert_project(self, project: ProjectRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO projects(project_id,name,summary_zh,primary_url,variants_json,
                    classification_json,quality_json,status,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(project_id) DO UPDATE SET
                    name=excluded.name,
                    summary_zh=CASE WHEN excluded.summary_zh='' THEN projects.summary_zh ELSE excluded.summary_zh END,
                    primary_url=CASE WHEN excluded.primary_url='' THEN projects.primary_url ELSE excluded.primary_url END,
                    variants_json=excluded.variants_json,
                    classification_json=excluded.classification_json,
                    quality_json=excluded.quality_json,
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                (
                    project.project_id, project.name, project.summary_zh, project.primary_url,
                    _dump(project.variants), _dump(project.classification.model_dump()),
                    _dump(project.quality), project.status, project.created_at, project.updated_at,
                ),
            )

    def upsert_artifact(self, artifact: ArtifactRecord) -> tuple[bool, bool]:
        """返回 (是否新建, 内容版本是否变化)。"""
        with self.connect() as conn:
            same_repo = conn.execute(
                "SELECT artifact_id FROM artifacts WHERE repo_type=? AND repo_id=?",
                (artifact.repo_type, artifact.repo_id),
            ).fetchone()
            if same_repo and same_repo["artifact_id"] != artifact.artifact_id:
                artifact.artifact_id = same_repo["artifact_id"]
            old = conn.execute(
                "SELECT sha,last_modified,raw_json FROM artifacts WHERE artifact_id=?",
                (artifact.artifact_id,),
            ).fetchone()
            now = utc_now()
            conn.execute(
                """
                INSERT INTO artifacts(artifact_id,repo_id,repo_type,name,author,canonical_url,
                    description,tags_json,gated_json,license,sha,last_modified,source_queries_json,
                    raw_json,first_seen_at,last_seen_at,status)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    name=excluded.name,author=excluded.author,canonical_url=excluded.canonical_url,
                    description=excluded.description,tags_json=excluded.tags_json,gated_json=excluded.gated_json,
                    license=excluded.license,sha=excluded.sha,last_modified=excluded.last_modified,
                    source_queries_json=excluded.source_queries_json,raw_json=excluded.raw_json,
                    last_seen_at=excluded.last_seen_at
                """,
                (
                    artifact.artifact_id, artifact.repo_id, artifact.repo_type, artifact.name,
                    artifact.author, artifact.canonical_url, artifact.description, _dump(artifact.tags),
                    _dump(artifact.gated), artifact.license, artifact.sha, artifact.last_modified,
                    _dump(artifact.source_queries), _dump(artifact.raw), now, now, "discovered",
                ),
            )
            is_new = old is None
            changed = is_new or old["sha"] != artifact.sha or old["last_modified"] != artifact.last_modified
            return is_new, changed

    def link_project_artifact(self, project_id: str, artifact_id: str, relationship: str = "primary") -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO project_artifacts(project_id,artifact_id,relationship) VALUES(?,?,?)",
                (project_id, artifact_id, relationship),
            )

    def add_snapshot(self, artifact_id: str, source_type: str, source_hash: str, content: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO source_snapshots(artifact_id,source_type,source_hash,content,created_at) VALUES(?,?,?,?,?)",
                (artifact_id, source_type, source_hash, content, utc_now()),
            )
            return cur.rowcount > 0

    def add_task(self, project_id: str, artifact_id: str, config: str, split: str,
                 schema: dict | list | None = None, num_rows: int | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks(project_id,artifact_id,config,split,schema_json,num_rows)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(artifact_id,config,split) DO UPDATE SET
                    schema_json=excluded.schema_json,
                    num_rows=COALESCE(excluded.num_rows,tasks.num_rows)
                """,
                (project_id, artifact_id, config, split, _dump(schema or {}), num_rows),
            )

    def add_sample(self, sample: SampleRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO samples(project_id,artifact_id,config,split,row_index,
                    sample_role,risk_level,safety_categories_json,source_method,sanitized,content_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    sample.project_id, sample.artifact_id, sample.config, sample.split,
                    sample.row_index, sample.sample_role, sample.risk_level,
                    _dump(sample.safety_categories), sample.source_method,
                    int(sample.sanitized), _dump(sample.content),
                ),
            )

    def clear_project_samples(self, project_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM samples WHERE project_id=?", (project_id,))

    def clear_artifact_samples(self, artifact_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM samples WHERE artifact_id=?", (artifact_id,))

    def add_change(self, event_type: str, *, artifact_id: str | None = None,
                   project_id: str | None = None, details: dict | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO change_events(artifact_id,project_id,event_type,details_json,created_at) VALUES(?,?,?,?,?)",
                (artifact_id, project_id, event_type, _dump(details or {}), utc_now()),
            )

    def enqueue_review(self, project_id: str, reason: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO review_queue(project_id,reason,status,created_at) VALUES(?,?,?,?)",
                (project_id, reason, "open", utc_now()),
            )

    def list_artifacts(self, *, status: str | None = None, limit: int | None = None) -> list[dict]:
        sql = "SELECT * FROM artifacts"
        params: list[Any] = []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        sql += " ORDER BY COALESCE(last_modified,first_seen_at) DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._decode_artifact(dict(row)) for row in rows]

    def get_artifact(self, artifact_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
        return self._decode_artifact(dict(row)) if row else None

    def set_artifact_status(self, artifact_id: str, status: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE artifacts SET status=? WHERE artifact_id=?", (status, artifact_id))

    def get_project(self, project_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE project_id=?", (project_id,)).fetchone()
        return self._decode_project(dict(row)) if row else None

    def set_project_status(self, project_id: str, status: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE projects SET status=?,updated_at=? WHERE project_id=?",
                (status, utc_now(), project_id),
            )

    def get_project_for_artifact(self, artifact_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT p.* FROM projects p
                JOIN project_artifacts pa ON pa.project_id=p.project_id
                WHERE pa.artifact_id=? ORDER BY pa.relationship LIMIT 1
                """,
                (artifact_id,),
            ).fetchone()
        return self._decode_project(dict(row)) if row else None

    def list_project_artifacts(self, project_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT a.*,pa.relationship FROM artifacts a
                JOIN project_artifacts pa ON pa.artifact_id=a.artifact_id
                WHERE pa.project_id=? ORDER BY pa.relationship,a.repo_type
                """,
                (project_id,),
            ).fetchall()
        return [self._decode_artifact(dict(row)) for row in rows]

    def update_project_catalog_scope(self, project_id: str, status: str, quality: dict) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE projects SET status=?,quality_json=?,updated_at=? WHERE project_id=?",
                (status, _dump(quality), utc_now(), project_id),
            )

    def update_project_classification(self, project_id: str, classification: dict) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE projects SET classification_json=?,updated_at=? WHERE project_id=?",
                (_dump(classification), utc_now(), project_id),
            )

    def list_projects(self, *, published_only: bool = False) -> list[dict]:
        sql = "SELECT * FROM projects"
        params: tuple[Any, ...] = ()
        if published_only:
            sql += " WHERE status='published'"
        sql += " ORDER BY name COLLATE NOCASE"
        with self.connect() as conn:
            projects = [self._decode_project(dict(row)) for row in conn.execute(sql, params).fetchall()]
            for project in projects:
                artifacts = conn.execute(
                    """
                    SELECT a.*,pa.relationship FROM artifacts a
                    JOIN project_artifacts pa ON pa.artifact_id=a.artifact_id
                    WHERE pa.project_id=? ORDER BY pa.relationship,a.repo_type
                    """,
                    (project["project_id"],),
                ).fetchall()
                project["artifacts"] = [self._decode_artifact(dict(row)) for row in artifacts]
                tasks = conn.execute(
                    "SELECT * FROM tasks WHERE project_id=? ORDER BY config,split", (project["project_id"],)
                ).fetchall()
                project["tasks"] = [self._decode_task(dict(row)) for row in tasks]
                samples = conn.execute(
                    "SELECT * FROM samples WHERE project_id=? ORDER BY artifact_id,config,split,row_index",
                    (project["project_id"],),
                ).fetchall()
                project["samples"] = [self._decode_sample(dict(row)) for row in samples]
        return projects

    def latest_changes(self, limit: int = 50) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM change_events ORDER BY event_id DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["details"] = _load(item.pop("details_json"), {})
            result.append(item)
        return result

    def save_webhook(self, webhook_id: str, payload: dict) -> bool:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO webhook_events(webhook_id,payload_json,status,received_at) VALUES(?,?,?,?)",
                (webhook_id, _dump(payload), "received", utc_now()),
            )
            return cur.rowcount > 0

    def finish_webhook(self, webhook_id: str, error: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE webhook_events SET status=?,processed_at=?,error=? WHERE webhook_id=?",
                ("failed" if error else "processed", utc_now(), error, webhook_id),
            )

    def set_state(self, key: str, value: Any) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO system_state(key,value_json,updated_at) VALUES(?,?,?)
                ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at
                """,
                (key, _dump(value), utc_now()),
            )

    def get_state(self, key: str, fallback: Any = None) -> Any:
        with self.connect() as conn:
            row = conn.execute("SELECT value_json FROM system_state WHERE key=?", (key,)).fetchone()
        return _load(row["value_json"], fallback) if row else fallback

    def try_acquire_update_lock(self, owner: str, *, stale_after_hours: int = 6) -> bool:
        now = datetime.now(UTC)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT owner,acquired_at FROM update_lock WHERE lock_name='catalog_update'"
            ).fetchone()
            if row:
                try:
                    acquired = datetime.fromisoformat(row["acquired_at"])
                    if acquired.tzinfo is None:
                        acquired = acquired.replace(tzinfo=UTC)
                except ValueError:
                    acquired = now - timedelta(hours=stale_after_hours + 1)
                if now - acquired < timedelta(hours=stale_after_hours):
                    return False
                conn.execute("DELETE FROM update_lock WHERE lock_name='catalog_update'")
            conn.execute(
                "INSERT INTO update_lock(lock_name,owner,acquired_at) VALUES('catalog_update',?,?)",
                (owner, now.isoformat()),
            )
            return True

    def release_update_lock(self, owner: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM update_lock WHERE lock_name='catalog_update' AND owner=?", (owner,)
            )

    @staticmethod
    def _decode_project(row: dict) -> dict:
        row["variants"] = _load(row.pop("variants_json"), [])
        row["classification"] = _load(row.pop("classification_json"), {})
        row["quality"] = _load(row.pop("quality_json"), {})
        return row

    @staticmethod
    def _decode_artifact(row: dict) -> dict:
        row["tags"] = _load(row.pop("tags_json"), [])
        row["gated"] = _load(row.pop("gated_json"), None)
        row["source_queries"] = _load(row.pop("source_queries_json"), [])
        row["raw"] = _load(row.pop("raw_json"), {})
        return row

    @staticmethod
    def _decode_task(row: dict) -> dict:
        row["schema"] = _load(row.pop("schema_json"), {})
        return row

    @staticmethod
    def _decode_sample(row: dict) -> dict:
        row["sanitized"] = bool(row["sanitized"])
        row["safety_categories"] = _load(row.pop("safety_categories_json", None), [])
        row["source_method"] = row.get("source_method") or "dataset_viewer"
        row["content"] = _load(row.pop("content_json"), {})
        return row
