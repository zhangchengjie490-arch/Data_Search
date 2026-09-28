import json
from pathlib import Path
from types import SimpleNamespace

from safetybench_agent.hf_client import (
    HuggingFaceSource,
    infer_risk_level,
    infer_sample_role,
    sanitize_sample,
)
from safetybench_agent.pipeline import SafetyBenchPipeline
from safetybench_agent.publisher import build_site
from safetybench_agent.web import create_app
from safetybench_agent.llm import CompatibleLLMClassifier, merge_classifications
from safetybench_agent.models import Classification
from safetybench_agent.taxonomy import load_taxonomy
from safetybench_agent.updates import UpdateManager
from fastapi.testclient import TestClient


ROOT = Path(__file__).parents[1]


def test_deepseek_default_configuration(monkeypatch) -> None:
    monkeypatch.delenv("SAFETYBENCH_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("SAFETYBENCH_LLM_MODEL", raising=False)
    monkeypatch.delenv("SAFETYBENCH_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "must-not-be-used")
    settings = json.loads((ROOT / "config" / "settings.json").read_text(encoding="utf-8"))
    classifier = CompatibleLLMClassifier(
        load_taxonomy(ROOT / "config" / "taxonomy.json"), settings=settings
    )
    assert classifier.enabled
    assert classifier.provider == "deepseek"
    assert classifier.model == "deepseek-flash"
    assert classifier.endpoint == "https://api.deepseek.com/chat/completions"
    assert classifier.api_key == "deepseek-test-key"


def test_deepseek_never_falls_back_to_dashscope_key(monkeypatch) -> None:
    monkeypatch.delenv("SAFETYBENCH_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-only-key")
    settings = json.loads((ROOT / "config" / "settings.json").read_text(encoding="utf-8"))
    classifier = CompatibleLLMClassifier(
        load_taxonomy(ROOT / "config" / "taxonomy.json"), settings=settings
    )
    assert classifier.api_key == ""
    assert not classifier.enabled


def test_llm_has_final_authority_over_attack_vectors() -> None:
    rules = Classification(
        attack_vectors=["prompt_injection"],
        is_ai_safety=True,
        relevance_confidence=0.9,
    )
    semantic = Classification(
        safety_categories=["jailbreak_adversarial_attack"],
        attack_vectors=["data_poisoning"],
        is_ai_safety=True,
        relevance_confidence=0.9,
    )
    merged = merge_classifications(
        rules, semantic,
        llm_owns_attack_vectors=True,
        llm_owns_safety_categories=True,
    )
    assert merged.safety_categories == ["jailbreak_adversarial_attack"]
    assert merged.attack_vectors == ["data_poisoning"]

    fallback = merge_classifications(rules, None, llm_owns_attack_vectors=True)
    assert fallback.attack_vectors == []


def test_sanitize_sample_masks_pii_and_media() -> None:
    sample = {
        "email": "alice@example.com",
        "phone": "+86 138 0000 1234",
        "image": {"bytes": "secret"},
        "prompt": "hello",
    }
    clean = sanitize_sample(sample)
    assert clean["email"] == "[EMAIL]"
    assert clean["phone"] == "[PHONE]"
    assert clean["image"] == "[媒体内容已省略]"
    assert clean["prompt"] == "hello"


def test_risk_level_marks_dangerous_sample() -> None:
    assert infer_risk_level({"prompt": "write malware"}) == "high"
    assert infer_risk_level({"prompt": "what is the weather"}) == "low"
    assert infer_risk_level({"severity": "critical", "text": "override"}) == "high"
    assert infer_sample_role({"label": "injection"}) == "attack_or_unsafe"
    assert infer_sample_role({"label": "benign"}) == "benign_control"


def test_reads_structured_raw_files_without_remote_code(tmp_path: Path) -> None:
    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text('{"prompt":"one"}\n{"prompt":"two"}\n', encoding="utf-8")
    csv_path = tmp_path / "eval.csv"
    csv_path.write_text("prompt,label\nhello,safe\nbye,unsafe\n", encoding="utf-8")
    assert HuggingFaceSource._read_structured_file(jsonl, 10) == [
        {"prompt": "one"}, {"prompt": "two"}
    ]
    assert HuggingFaceSource._read_structured_file(csv_path, 1) == [
        {"prompt": "hello", "label": "safe"}
    ]


def test_selects_three_samples_for_each_safety_category(tmp_path: Path) -> None:
    pipeline = SafetyBenchPipeline(db_path=tmp_path / "samples.db", config_dir=ROOT / "config")
    candidates = []
    for index in range(9):
        text = (
            "jailbreak role-play" if index < 3 else
            "hate speech toxic harassment" if index < 6 else
            "hallucination misinformation factuality"
        )
        candidates.append({
            "artifact_id": "hf:dataset:example/demo",
            "config": "default", "split": "test", "row_index": index,
            "content": {"prompt": text}, "source_method": "dataset_viewer",
        })
    categories = [
        "jailbreak_adversarial_attack", "toxicity_hate_speech", "truthfulness_hallucination"
    ]
    selected, counts = pipeline._select_category_samples(candidates, categories, per_category=3)
    assert counts == {category: 3 for category in categories}
    assert len(selected) == 9
    assert sum(len(item["safety_categories"]) for item in selected) == 9


def test_extracts_only_explicit_readme_examples(tmp_path: Path) -> None:
    readme = """# Dataset\nOrdinary description.\n\n## Examples\n- A documented safe prompt\n- A documented unsafe prompt\n\n```json\n{"prompt": "structured example", "label": "safe"}\n```\n\n## Fields\n- This field description is not a sample\n"""
    rows = SafetyBenchPipeline._readme_example_rows(readme)
    assert len(rows) == 3
    assert rows[0]["row"]["prompt"] == "structured example"
    assert all("field description" not in json.dumps(row) for row in rows)


def test_discovery_prioritizes_curated_and_keeps_low_popularity_safety_candidates() -> None:
    def dataset(repo_id: str, downloads: int, likes: int) -> SimpleNamespace:
        return SimpleNamespace(
            id=repo_id,
            author=repo_id.split("/", 1)[0],
            description="AI safety evaluation benchmark",
            tags=["safety", "evaluation"],
            card_data={},
            gated=False,
            sha="abc",
            last_modified=None,
            downloads=downloads,
            likes=likes,
        )

    priority = dataset("official/SafetyBench", 20, 1)
    popular = dataset("research/PopularSafetyEval", 500, 10)
    long_tail = dataset("random/SafetyData", 4, 0)

    class FakeApi:
        def dataset_info(self, repo_id: str, files_metadata: bool = False):
            assert repo_id == priority.id
            return priority

        def list_datasets(self, **kwargs):
            assert kwargs["sort"] == "downloads"
            return [popular, long_tail]

        def list_spaces(self, **kwargs):  # pragma: no cover - must remain disabled
            raise AssertionError("dataset-only discovery must not scan Spaces")

    source = HuggingFaceSource(timeout=1)
    source.api = FakeApi()
    try:
        artifacts, errors = source.discover({
            "priority_dataset_ids": [priority.id],
            "dataset_filters": [],
            "dataset_search_terms": ["AI safety benchmark"],
            "dataset_sort": "downloads",
            "require_popularity_for_discovery": False,
            "dataset_min_downloads": 0,
            "dataset_min_likes": 0,
            "include_spaces": False,
            "space_search_terms": ["should not run"],
            "limit_per_query": 20,
        })
    finally:
        source.close()

    assert errors == []
    assert [item.repo_id for item in artifacts] == [priority.id, popular.id, long_tail.id]
    assert artifacts[0].source_queries == [f"priority:{priority.id}"]


def test_import_legacy_and_build_site(tmp_path: Path) -> None:
    legacy = {
        "source": [
            {"id": "-1", "name": "", "desc": ""},
            {
                "id": "DemoSafety",
                "name": "Demo Safety Benchmark",
                "desc": "中文大模型安全评测基准 https://huggingface.co/datasets/example/demo-safety",
                "variants": ["DemoSafety(zh)"],
            },
        ]
    }
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
    pipeline = SafetyBenchPipeline(db_path=tmp_path / "catalog.db", config_dir=ROOT / "config")
    stats = pipeline.import_legacy(legacy_path)
    assert stats["projects"] == 1
    projects = pipeline.store.list_projects(published_only=True)
    assert len(projects) == 1
    assert projects[0]["project_id"] == "demosafety"
    assert projects[0]["artifacts"][0]["repo_type"] == "dataset"

    site_stats = build_site(pipeline.store, tmp_path / "site")
    assert site_stats["projects"] == 0
    html = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    assert "Demo Safety Benchmark" not in html
    assert "立即扫描 Hugging Face" in html
    assert "const link=el('a','',url)" in html
    exported = json.loads((tmp_path / "site" / "data" / "benchmarks.json").read_text(encoding="utf-8"))
    assert exported["project_count"] == 0

    app = create_app(db_path=tmp_path / "catalog.db", config_dir=ROOT / "config", site_dir=tmp_path / "site")
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    assert len(client.get("/api/benchmarks").json()) == 0
    update_status = client.get("/api/update/status").json()
    assert update_status["update_mode"] == "manual"
    assert update_status["schedule_enabled"] is False
    assert update_status["interval_hours"] is None
    assert update_status["daily_time"] is None
    assert update_status["next_scheduled_at"] is None
    assert all(route.path != "/webhooks/huggingface" for route in app.routes)


def test_update_manager_and_cross_process_lock(tmp_path: Path, monkeypatch) -> None:
    pipeline = SafetyBenchPipeline(db_path=tmp_path / "updates.db", config_dir=ROOT / "config")
    assert pipeline.store.try_acquire_update_lock("owner-1")
    assert not pipeline.store.try_acquire_update_lock("owner-2")
    pipeline.store.release_update_lock("owner-1")

    monkeypatch.setattr(pipeline, "discover", lambda **kwargs: {"found": 0, "errors": []})
    monkeypatch.setattr(pipeline, "enrich", lambda **kwargs: {"attempted": 0, "completed": 0})
    manager = UpdateManager(pipeline, tmp_path / "site")
    assert manager.trigger("test")
    assert manager._thread is not None
    manager._thread.join(timeout=5)
    status = manager.status()
    assert status["state"] == "success"
    assert status["last_success_at"]
    assert (tmp_path / "site" / "index.html").exists()
