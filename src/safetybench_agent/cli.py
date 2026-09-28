from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import SafetyBenchPipeline
from .publisher import build_site, export_catalog


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safetybench-agent",
        description="自动发现、分类和发布 Hugging Face AI 安全测评基准",
    )
    parser.add_argument("--db", default="data/safetybench.db", help="SQLite 数据库路径")
    parser.add_argument("--config-dir", default="config", help="配置目录")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="初始化数据库")

    legacy = sub.add_parser("import-legacy", help="导入现有基准种子 JSON")
    legacy.add_argument("path", nargs="?", default="评测基准库元数据.json")

    discover = sub.add_parser("discover", help="从 Hugging Face 发现候选")
    discover.add_argument("--limit-per-query", type=int, default=None)

    enrich = sub.add_parser("enrich", help="解析候选的 README、任务和示例")
    enrich.add_argument("--limit", type=int, default=None)

    attacks = sub.add_parser("classify-attacks", help="使用大模型和脱敏示例重新判断全部攻击方式")
    attacks.add_argument("--batch-size", type=int, default=8)
    attacks.add_argument(
        "--only-model-mismatch", action="store_true",
        help="仅处理分类器不是当前配置模型的项目，用于模型迁移断点续跑",
    )
    attacks.add_argument("--site-dir", default="site")

    resample = sub.add_parser("resample", help="按安全测试类别为每个项目重新采集示例")
    resample.add_argument("--limit", type=int, default=None)
    resample.add_argument("--only-incomplete", action="store_true", help="跳过已按当前配额采集的项目")
    resample.add_argument("--retry-shortages", action="store_true", help="仅重试上次未补齐的项目")
    resample.add_argument("--project-id", action="append", default=[], help="只处理指定项目，可重复")
    resample.add_argument("--site-dir", default="site")

    refresh = sub.add_parser("refresh", help="刷新一个已知 Hugging Face 仓库")
    refresh.add_argument("repo_id", help="例如 allenai/wildguardmix")
    refresh.add_argument("--repo-type", choices=("dataset", "space"), default="dataset")

    run = sub.add_parser("run", help="运行导入、发现、解析并生成导航")
    run.add_argument("--legacy", default="评测基准库元数据.json")
    run.add_argument("--limit-per-query", type=int, default=None)
    run.add_argument("--enrich-limit", type=int, default=None)
    run.add_argument("--site-dir", default="site")

    site = sub.add_parser("build-site", help="从数据库生成静态导航")
    site.add_argument("--output", default="site")
    site.add_argument("--include-review", action="store_true", help="同时导出待审核项目")

    export = sub.add_parser("export", help="导出结构化 JSON")
    export.add_argument("--output", default="data/benchmarks.json")
    export.add_argument("--include-review", action="store_true")

    serve = sub.add_parser("serve", help="启动 API、导航页和手动更新接口")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--site-dir", default="site")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pipeline = SafetyBenchPipeline(db_path=args.db, config_dir=args.config_dir)

    if args.command == "init":
        _json({"initialized": str(Path(args.db).resolve())})
    elif args.command == "import-legacy":
        _json(pipeline.import_legacy(args.path))
    elif args.command == "discover":
        _json(pipeline.discover(limit_per_query=args.limit_per_query))
    elif args.command == "enrich":
        _json(pipeline.enrich(limit=args.limit))
    elif args.command == "classify-attacks":
        result = pipeline.reclassify_attack_vectors(
            batch_size=args.batch_size,
            only_model_mismatch=args.only_model_mismatch,
        )
        result["site"] = build_site(pipeline.store, args.site_dir)
        _json(result)
    elif args.command == "resample":
        result = pipeline.resample_by_category(
            limit=args.limit,
            only_incomplete=args.only_incomplete,
            retry_shortages=args.retry_shortages,
            project_ids=args.project_id,
        )
        result["site"] = build_site(pipeline.store, args.site_dir)
        _json(result)
    elif args.command == "refresh":
        _json(pipeline.refresh_repo(args.repo_id, args.repo_type))
    elif args.command == "run":
        result = pipeline.run(
            legacy_path=args.legacy,
            limit_per_query=args.limit_per_query,
            enrich_limit=args.enrich_limit,
        )
        result["site"] = build_site(pipeline.store, args.site_dir)
        _json(result)
    elif args.command == "build-site":
        _json(build_site(pipeline.store, args.output, published_only=not args.include_review))
    elif args.command == "export":
        payload = export_catalog(pipeline.store, args.output, published_only=not args.include_review)
        _json({"projects": payload["project_count"], "output": str(Path(args.output).resolve())})
    elif args.command == "serve":
        import uvicorn

        from .web import create_app

        app = create_app(db_path=args.db, config_dir=args.config_dir, site_dir=args.site_dir)
        uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
