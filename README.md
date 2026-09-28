# AI 安全测评基准导航 Agent

这是一个可运行的 MVP，用于持续发现 Hugging Face 上的 AI 安全测评资产，完成多维分类、项目族归并、任务与示例抽取、版本监控，并输出静态导航和 JSON API。

## 核心设计

- 三层主导航：模型行为安全、攻击抵抗能力、系统与 Agent 安全。
- 六维标签：风险领域、攻击向量、被测对象、评价目标、模态、语言。
- 收录 Hugging Face Dataset 上所有可识别的 AI 安全数据集；下载量和点赞数只用于排序，不作为准入条件。
- 区分静态基准、训练/评测混合数据、评测套件、交互环境、攻击库和排行榜。
- 项目、Artifact、Task、Config、Split 和 Sample 分层存储。
- 所有分类保留置信度和证据；AI 安全 Dataset 直接发布，非 Dataset 或非安全数据直接排除，不产生待确认状态。
- DeepSeek 结合 README 和脱敏代表性样本，从 13 个方向多选安全测试类别：内容安全、越狱、拒答、毒性、偏见、真实性、隐私、网络与双重用途、Prompt Injection/RAG、Agent/工具、多模态、多语言文化、可靠性鲁棒性。
- 具体攻击技术仍由 DeepSeek 单独判断，不与关键词规则结果求并集；普通内容安全、偏见或真实性数据不涉及攻击机制时显示“不涉及明确攻击技术”，不会进入待确认。
- Hugging Face 内容始终视为不可信数据，不执行仓库代码。
- 通过 commit SHA、README 哈希和变更事件进行增量监控。

## 快速开始

当前目录已经包含 `评测基准库元数据.json`，可直接作为人工维护的种子库导入。

```powershell
python -m pip install -e .
python -m safetybench_agent init
python -m safetybench_agent import-legacy
python -m safetybench_agent build-site
python -m http.server 8080 --directory site
```

浏览器访问 `http://127.0.0.1:8080`。

运行一次完整的在线采集：

```powershell
python -m safetybench_agent run --limit-per-query 10 --enrich-limit 20
```

使用已保存的脱敏样本重新判断全部公开数据集的攻击方式：

```powershell
python -m safetybench_agent classify-attacks --batch-size 8 --site-dir site
python -m safetybench_agent classify-attacks --batch-size 1 --only-model-mismatch --site-dir site
```

按大模型判定的安全测试类别重新采集示例，每个类别默认 3 条：

```powershell
python -m safetybench_agent resample --site-dir site
python -m safetybench_agent resample --retry-shortages --site-dir site
```

采集顺序为 Dataset Viewer、仓库原始 Parquet/JSON/JSONL/CSV/TSV/XLSX，最后尝试 README 中明确标注的 Example/Sample/示例。`--only-incomplete` 可断点续跑，`--project-id` 可定向重试一个项目。

刷新指定的已知项目：

```powershell
python -m safetybench_agent refresh allenai/wildguardmix
```

## 服务与手动更新

启动静态导航、查询 API 和手动更新接口：

```powershell
python -m safetybench_agent serve --host 0.0.0.0 --port 8000
```

接口：

- `GET /health`
- `GET /api/benchmarks`
- `GET /api/benchmarks/{project_id}`
- `GET /api/update/status`：查询当前手动扫描状态和上次结果
- `POST /api/update/run`：立即启动一次 Hugging Face 数据集扫描、分类与站点重建

导航页右上角的“立即扫描 Hugging Face”按钮调用手动更新接口。更新在后台线程中执行，页面会轮询进度，完成后自动刷新；如果已有手动任务正在扫描，接口返回 `409`，不会启动第二个任务。系统不会定时自动更新。

## DeepSeek-Flash 语义分类

项目默认使用 DeepSeek 开放平台的稳定模型名 `deepseek-flash` 进行语义分类，并在模型调用失败或未配置密钥时回退到可审计规则分类器。设置 API Key：

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
```

默认兼容端点为 `https://api.deepseek.com/chat/completions`。如需覆盖 Base URL：

```powershell
$env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
```

也可以继续使用通用覆盖变量 `SAFETYBENCH_LLM_ENDPOINT`、`SAFETYBENCH_LLM_API_KEY` 和 `SAFETYBENCH_LLM_MODEL`。LLM 只处理相关性较高的候选，仓库文本被明确标记为不可信数据；输出经过固定标签白名单校验后才会入库。

## 配置

- `config/settings.json`：重点数据集清单、安全检索词、抓取上限、置信度阈值和采样数量。
- `config/taxonomy.json`：六维分类词表、资产类型词表和范围排除词。
- `data/safetybench.db`：SQLite 数据库。
- `site/index.html`：生成的导航页。
- `site/data/benchmarks.json`：公开目录 JSON。

## 安全和数据治理

- 不运行 Hugging Face 仓库中的脚本或远程代码。
- 示例通过 Dataset Viewer、受大小和超时约束的结构化原始文件、明确标注的 README 示例逐级读取，并对邮件、电话号码和长数字做脱敏。
- 图片、音频、视频及二进制字段不复制到导航中。
- 高风险样本在页面中默认折叠。
- Gated 数据不会绕过权限；未配置 `HF_TOKEN` 时只记录可公开获得的元数据。
- 自动发布只适用于高置信度结果；其余项目进入 `review_queue`。

## 测试

```powershell
pytest -q
```
