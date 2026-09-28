from __future__ import annotations

import threading
import traceback
import uuid
from pathlib import Path
from typing import Any

from .models import utc_now
from .pipeline import SafetyBenchPipeline
from .publisher import build_site


def _execute_update_cycle(
    pipeline: SafetyBenchPipeline,
    site_dir: str | Path,
    *,
    source: str,
    owner: str,
    limit_per_query: int | None = None,
    enrich_limit: int | None = None,
) -> dict[str, Any]:
    previous = pipeline.store.get_state("catalog_update", {})
    state: dict[str, Any] = {
        "state": "running",
        "source": source,
        "owner": owner,
        "started_at": utc_now(),
        "finished_at": None,
        "last_success_at": previous.get("last_success_at"),
        "stats": {},
        "error": None,
    }
    pipeline.store.set_state("catalog_update", state)
    try:
        stats = {
            "discover": pipeline.discover(limit_per_query=limit_per_query),
            "enrich": pipeline.enrich(limit=enrich_limit),
            "site": build_site(pipeline.store, site_dir),
        }
        state.update({
            "state": "success",
            "finished_at": utc_now(),
            "last_success_at": utc_now(),
            "stats": stats,
        })
        pipeline.store.set_state("catalog_update", state)
        return {"accepted": True, **state}
    except Exception as exc:
        state.update({
            "state": "failed",
            "finished_at": utc_now(),
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        })
        pipeline.store.set_state("catalog_update", state)
        return {"accepted": True, **state}
    finally:
        pipeline.store.release_update_lock(owner)


def run_update_cycle(
    pipeline: SafetyBenchPipeline,
    site_dir: str | Path,
    *,
    source: str,
    limit_per_query: int | None = None,
    enrich_limit: int | None = None,
) -> dict[str, Any]:
    owner = f"{source}:{uuid.uuid4().hex}"
    if not pipeline.store.try_acquire_update_lock(owner):
        return {
            "accepted": False,
            "state": "already_running",
            "current": pipeline.store.get_state("catalog_update", {}),
        }
    return _execute_update_cycle(
        pipeline,
        site_dir,
        source=source,
        owner=owner,
        limit_per_query=limit_per_query,
        enrich_limit=enrich_limit,
    )


class UpdateManager:
    def __init__(self, pipeline: SafetyBenchPipeline, site_dir: str | Path):
        self.pipeline = pipeline
        self.site_dir = Path(site_dir)
        self._thread: threading.Thread | None = None
        self._guard = threading.Lock()

    def trigger(self, source: str = "manual") -> bool:
        with self._guard:
            if self._thread and self._thread.is_alive():
                return False
            owner = f"{source}:{uuid.uuid4().hex}"
            if not self.pipeline.store.try_acquire_update_lock(owner):
                return False
            self._thread = threading.Thread(
                target=_execute_update_cycle,
                kwargs={
                    "pipeline": self.pipeline,
                    "site_dir": self.site_dir,
                    "source": source,
                    "owner": owner,
                },
                name="safetybench-catalog-update",
                daemon=True,
            )
            try:
                self._thread.start()
            except Exception:
                self.pipeline.store.release_update_lock(owner)
                raise
            return True

    def status(self) -> dict[str, Any]:
        state = self.pipeline.store.get_state("catalog_update", {
            "state": "never_run",
            "source": None,
            "started_at": None,
            "finished_at": None,
            "last_success_at": None,
            "stats": {},
            "error": None,
        })
        state["update_mode"] = "manual"
        state["schedule_enabled"] = False
        state["interval_hours"] = None
        state["daily_time"] = None
        state["next_scheduled_at"] = None
        state["running"] = state.get("state") == "running"
        return state

