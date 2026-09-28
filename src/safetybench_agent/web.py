from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import SafetyBenchPipeline
from .publisher import auto_hf_projects
from .updates import UpdateManager


def create_app(*, db_path: str | Path = "data/safetybench.db",
               config_dir: str | Path = "config", site_dir: str | Path = "site") -> FastAPI:
    pipeline = SafetyBenchPipeline(db_path=db_path, config_dir=config_dir)
    update_manager = UpdateManager(pipeline, site_dir)
    app = FastAPI(title="SafetyBench Agent", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/benchmarks")
    def benchmarks() -> list[dict[str, Any]]:
        return auto_hf_projects(pipeline.store, published_only=True)

    @app.get("/api/benchmarks/{project_id}")
    def benchmark(project_id: str) -> dict[str, Any]:
        project = next(
            (item for item in auto_hf_projects(pipeline.store, published_only=True)
             if item["project_id"] == project_id),
            None,
        )
        if project is None:
            raise HTTPException(status_code=404, detail="benchmark not found")
        return project

    @app.get("/api/update/status")
    def update_status() -> dict[str, Any]:
        return update_manager.status()

    @app.post("/api/update/run", status_code=202)
    def run_update() -> JSONResponse:
        if not update_manager.trigger("manual"):
            return JSONResponse(
                {"accepted": False, "reason": "already_running", "status": update_manager.status()},
                status_code=409,
            )
        return JSONResponse(
            {"accepted": True, "status": update_manager.status()}, status_code=202
        )

    site_path = Path(site_dir)
    if site_path.exists():
        app.mount("/", StaticFiles(directory=site_path, html=True), name="site")
    return app
