from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import utc_now
from .storage import Store


LABELS = {
    "model_behavior": "模型行为安全",
    "attack_resistance": "攻击抵抗能力",
    "system_agent_security": "系统与 Agent 安全",
    "harmful_content": "内容安全与有害输出",
    "toxicity_hate": "毒性与仇恨言论",
    "bias_fairness": "偏见与公平性",
    "truthfulness": "真实性与幻觉",
    "privacy": "隐私与数据泄露",
    "cyber_dual_use": "网络与双重用途",
    "high_stakes": "高风险领域",
    "deception_autonomy": "欺骗与自主性",
    "direct_jailbreak": "直接越狱",
    "prompt_injection": "Prompt Injection",
    "indirect_prompt_injection": "间接 Prompt Injection",
    "adversarial_perturbation": "对抗扰动",
    "data_poisoning": "数据投毒",
    "secret_extraction": "秘密提取",
    "chat_model": "对话模型",
    "safety_classifier": "安全分类器",
    "rag_system": "RAG 系统",
    "tool_agent": "工具调用 Agent",
    "code_model": "代码模型",
    "multimodal_model": "多模态模型",
    "zh": "中文",
    "en": "英文",
    "multilingual": "多语言",
    "text": "文本",
    "image": "图像",
    "audio": "音频",
    "video": "视频",
    "static_benchmark": "静态基准",
    "train_eval_dataset": "训练/评测混合数据",
    "evaluation_suite": "评测套件",
    "interactive_environment": "交互式环境",
    "leaderboard": "排行榜",
    "attack_library": "攻击库",
    "content_safety_harmful_output": "内容安全与有害输出",
    "jailbreak_adversarial_attack": "越狱与对抗攻击",
    "refusal_overrefusal": "拒答与过度拒答",
    "toxicity_hate_speech": "毒性与仇恨言论",
    "truthfulness_hallucination": "真实性与幻觉",
    "privacy_data_leakage": "隐私与数据泄露",
    "prompt_injection_rag_security": "Prompt Injection 与 RAG 安全",
    "agent_tool_security": "Agent 与工具调用安全",
    "multimodal_safety": "多模态安全",
    "multilingual_cultural_safety": "多语言与文化安全",
    "reliability_robustness": "可靠性与鲁棒性",
}


_PUBLIC_SECRET_PATTERNS = (
    (re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"), "[REDACTED_API_TOKEN]"),
    (re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{20,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED_AWS_ACCESS_KEY]"),
    (
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "[REDACTED_PRIVATE_KEY]",
    ),
)


def _redact_public_secrets(value: Any) -> Any:
    """Remove credential-shaped values before writing public catalog artifacts."""
    if isinstance(value, str):
        for pattern, replacement in _PUBLIC_SECRET_PATTERNS:
            value = pattern.sub(replacement, value)
        return value
    if isinstance(value, dict):
        return {key: _redact_public_secrets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_public_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_public_secrets(item) for item in value)
    return value


def _is_auto_hf_dataset(artifact: dict[str, Any]) -> bool:
    return (
        artifact.get("repo_type") == "dataset"
        and str(artifact.get("canonical_url", "")).startswith("https://huggingface.co/datasets/")
        and any(query != "legacy_seed" for query in artifact.get("source_queries", []))
    )


def _public_project(project: dict[str, Any]) -> dict[str, Any]:
    clean = dict(project)
    auto_artifacts = [artifact for artifact in clean.get("artifacts", []) if _is_auto_hf_dataset(artifact)]
    auto_artifacts.sort(
        key=lambda artifact: (
            int(any(str(query).startswith("priority:") for query in artifact.get("source_queries", []))),
            int(artifact.get("raw", {}).get("downloads") or 0),
        ),
        reverse=True,
    )
    artifacts = []
    for artifact in auto_artifacts:
        artifacts.append({key: artifact.get(key) for key in (
            "artifact_id", "repo_id", "repo_type", "name", "canonical_url", "description",
            "tags", "gated", "license", "sha", "last_modified", "status", "relationship",
        )})
    clean["artifacts"] = artifacts
    if artifacts:
        clean["primary_url"] = artifacts[0]["canonical_url"]
    return clean


def auto_hf_projects(store: Store, *, published_only: bool = True) -> list[dict[str, Any]]:
    candidates = [
        project for project in store.list_projects(published_only=published_only)
        if any(_is_auto_hf_dataset(artifact) for artifact in project.get("artifacts", []))
    ]
    candidates.sort(
        key=lambda project: (
            project.get("quality", {}).get("curation") == "seed_verified",
            project.get("name", "").casefold(),
        )
    )
    result: list[dict[str, Any]] = []
    seen_signatures: set[tuple[str, ...]] = set()
    for project in candidates:
        clean = _public_project(project)
        signature = tuple(sorted(artifact["repo_id"] for artifact in clean["artifacts"]))
        if not signature or signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        result.append(clean)
    return result


def export_catalog(store: Store, output: str | Path, *, published_only: bool = True) -> dict[str, Any]:
    projects = auto_hf_projects(store, published_only=published_only)
    projects.sort(
        key=lambda project: (
            {"priority": 2, "safety_dataset": 1}.get(project.get("quality", {}).get("catalog_scope"), 0),
            int(project.get("quality", {}).get("hf_downloads") or 0),
            int(project.get("quality", {}).get("hf_likes") or 0),
        ),
        reverse=True,
    )
    payload = _redact_public_secrets({
        "generated_at": utc_now(),
        "project_count": len(projects),
        "labels": LABELS,
        "projects": projects,
        "recent_changes": store.latest_changes(30),
    })
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_site(store: Store, output_dir: str | Path, *, published_only: bool = True) -> dict[str, Any]:
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = export_catalog(store, data_dir / "benchmarks.json", published_only=published_only)
    embedded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    html = SITE_TEMPLATE.replace("__CATALOG_JSON__", embedded).replace(
        "__GENERATED_AT__", payload["generated_at"]
    )
    (output_dir / "index.html").write_text(html, encoding="utf-8")
    return {"projects": payload["project_count"], "output": str(output_dir / "index.html")}


SITE_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="可追溯、持续更新的 AI 安全测评基准导航">
  <title>AI 安全测评基准导航</title>
  <style>
    :root { color-scheme: light; --bg:#f5f7f8; --surface:#fff; --text:#172129; --muted:#5d6b75; --line:#dce3e7; --accent:#0d6b59; --accent-soft:#e1f2ed; --warn:#93651c; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:15px/1.6 system-ui,-apple-system,"Segoe UI","PingFang SC",sans-serif; }
    header { background:#102e2a; color:#f6fffc; padding:42px 24px 34px; }
    .wrap { max-width:1180px; margin:auto; }
    h1 { margin:0 0 8px; font-size:clamp(28px,4vw,44px); font-weight:650; letter-spacing:-.02em; }
    .lead { margin:0; max-width:760px; color:#cfe3de; }
    .meta { margin-top:16px; color:#9fc3ba; font-size:13px; }
    .update-row { display:flex; flex-wrap:wrap; align-items:center; gap:12px; margin-top:18px; }
    .update-button { appearance:none; border:1px solid #8ed9c7; border-radius:8px; padding:9px 14px; background:#e9fff8; color:#0d5548; font:inherit; font-size:14px; font-weight:600; line-height:1.2; cursor:pointer; }
    .update-button:hover { background:#d8f8ee; }
    .update-button:disabled { cursor:wait; opacity:.65; }
    .update-status { color:#b9d7cf; font-size:13px; }
    main { padding:24px; }
    .controls { display:grid; grid-template-columns:2fr repeat(4,minmax(140px,1fr)); gap:10px; margin-bottom:16px; }
    input,select { width:100%; padding:10px 12px; border:1px solid var(--line); border-radius:8px; background:var(--surface); color:var(--text); font:inherit; }
    .summary { display:flex; gap:12px; align-items:center; justify-content:space-between; margin:12px 0 18px; color:var(--muted); }
    .summary strong { color:var(--text); }
    #grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); grid-auto-rows:1fr; gap:14px; align-items:stretch; }
    article { height:100%; display:flex; flex-direction:column; background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:18px; box-shadow:0 3px 14px rgba(20,45,40,.04); }
    article h2 { min-height:2.7em; font-size:19px; line-height:1.35; margin:0 0 8px; display:-webkit-box; -webkit-box-orient:vertical; -webkit-line-clamp:2; line-clamp:2; overflow:hidden; }
    article p { margin:8px 0; color:var(--muted); }
    .card-summary { min-height:6.4em; display:-webkit-box; -webkit-box-orient:vertical; -webkit-line-clamp:4; line-clamp:4; overflow:hidden; }
    .badges { min-height:58px; max-height:58px; overflow:hidden; align-content:flex-start; display:flex; flex-wrap:wrap; gap:6px; margin:10px 0; }
    .badge { background:var(--accent-soft); color:var(--accent); border-radius:999px; padding:3px 9px; font-size:12px; }
    .badge.secondary { background:#eef1f3; color:#52616b; }
    .facts { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:6px 14px; padding:10px 0; font-size:13px; color:var(--muted); }
    .facts b { color:var(--text); font-weight:600; }
    details { border-top:1px solid var(--line); margin-top:10px; padding-top:10px; }
    summary { color:var(--accent); cursor:pointer; }
    .links { display:flex; flex-direction:column; align-items:flex-start; gap:8px; margin-top:auto; padding-top:14px; }
    .links a { max-width:100%; overflow-wrap:anywhere; word-break:break-word; }
    a { color:var(--accent); text-decoration:none; }
    a:hover { text-decoration:underline; }
    pre { white-space:pre-wrap; word-break:break-word; background:#f1f4f5; border-radius:8px; padding:10px; max-height:260px; overflow:auto; font-size:12px; }
    .sample-category { margin:12px 0 6px; color:var(--text); font-weight:650; }
    .sample-meta { margin:7px 0 4px; color:var(--muted); font-size:12px; }
    .empty { grid-column:1/-1; text-align:center; padding:60px 20px; color:var(--muted); }
    footer { padding:30px 24px 50px; color:var(--muted); text-align:center; font-size:13px; }
    @media (max-width:900px) { .controls { grid-template-columns:1fr 1fr; } .controls input { grid-column:1/-1; } }
    @media (max-width:560px) { header,main { padding-left:14px; padding-right:14px; } .controls { grid-template-columns:1fr; } .controls input { grid-column:auto; } #grid { grid-template-columns:1fr; } }
    @media (prefers-color-scheme:dark) { :root { color-scheme:dark; --bg:#101717; --surface:#172120; --text:#edf6f3; --muted:#a9bab5; --line:#2a3a37; --accent:#71cfb7; --accent-soft:#183f36; } header{background:#0a2420} pre{background:#101817}.badge.secondary{background:#26312f;color:#c4d1cd} }
  </style>
</head>
<body>
<header><div class="wrap">
  <h1>AI 安全测评基准导航</h1>
  <p class="lead">按模型行为安全、攻击抵抗能力和系统与 Agent 安全组织；每个项目保留来源、分类证据、任务结构和示例样本。</p>
  <div class="meta">数据生成时间：__GENERATED_AT__</div>
  <div class="update-row">
    <button class="update-button" id="updateNow" type="button">立即扫描 Hugging Face</button>
    <span class="update-status" id="updateStatus" aria-live="polite">正在读取更新状态…</span>
  </div>
</div></header>
<main class="wrap">
  <section class="controls" aria-label="筛选条件">
    <input id="search" type="search" placeholder="搜索项目、简介、标签……" aria-label="搜索">
    <select id="layer" aria-label="测试层级"><option value="">全部测试层级</option></select>
    <select id="risk" aria-label="风险领域"><option value="">全部风险领域</option></select>
    <select id="target" aria-label="被测对象"><option value="">全部被测对象</option></select>
    <select id="language" aria-label="语言"><option value="">全部语言</option></select>
  </section>
  <div class="summary"><span>当前显示 <strong id="count">0</strong> 个项目</span><span>高风险示例默认折叠展示</span></div>
  <section id="grid" aria-live="polite"></section>
</main>
<footer>目录结果由可审计规则与可选语义分类生成；正式采用前请核对项目原始说明、许可证和版本。</footer>
<script>
const CATALOG = __CATALOG_JSON__;
const $ = id => document.getElementById(id);
const dims = {
  layer:['evaluation_layers','测试层级'], risk:['risk_domains','风险领域'],
  target:['target_systems','被测对象'], language:['languages','语言']
};
const label = key => CATALOG.labels[key] || key.replaceAll('_',' ');
function options(selectId, field) {
  const values = new Set();
  CATALOG.projects.forEach(p => (p.classification[field]||[]).forEach(v=>values.add(v)));
  [...values].sort((a,b)=>label(a).localeCompare(label(b),'zh-CN')).forEach(value=>{
    const option=document.createElement('option'); option.value=value; option.textContent=label(value); $(selectId).append(option);
  });
}
Object.entries(dims).forEach(([id,[field]])=>options(id,field));
function el(tag, cls, text) { const node=document.createElement(tag); if(cls) node.className=cls; if(text!==undefined) node.textContent=text; return node; }
function badge(value, secondary=false) { return el('span','badge'+(secondary?' secondary':''),label(value)); }
function sampleUnavailableReason(p) {
  const artifacts=p.artifacts||[];
  const report=(p.quality||{}).sample_collection||{};
  const errors=(report.errors||[]).join('\n');
  if(artifacts.some(a=>a.gated===true||a.gated==='auto'||a.gated==='manual')) {
    return '原因：该 Hugging Face 数据集为 Gated，当前访问凭据没有读取样本的权限。';
  }
  if(/401|Unauthorized|restricted|gated repo/i.test(errors)) {
    return '原因：Hugging Face 返回 401，数据集需要授权或有效的 HF_TOKEN。';
  }
  if(/403|Forbidden/i.test(errors)) {
    return '原因：Hugging Face 拒绝访问该数据集文件（403）。';
  }
  if(/timeout|timed out|超时/i.test(errors)) {
    return '原因：访问 Hugging Face Dataset Viewer 或下载源文件时超时。';
  }
  if(/超过上限|file size|too large/i.test(errors)) {
    return '原因：可用源文件超过当前安全下载大小上限。';
  }
  if(Number(report.viewer_candidates||0)===0 && Number(report.raw_candidates||0)===0 && Number(report.readme_candidates||0)===0) {
    return '原因：Dataset Viewer 没有可用预览，仓库结构化文件为空、无数据行或格式暂不支持。';
  }
  return '原因：源仓库当前没有提供可读取并可安全展示的示例数据。';
}
function card(p) {
  const article=el('article');
  article.append(el('h2','',p.name));
  article.append(el('p','card-summary',p.summary_zh || '暂无内容介绍。'));
  const badges=el('div','badges');
  if ((p.quality||{}).catalog_scope==='priority') badges.append(badge('重点数据集'));
  else if ((p.quality||{}).catalog_scope==='safety_dataset') badges.append(badge('安全数据集'));
  ['safety_categories','evaluation_layers','risk_domains'].forEach(field=>(p.classification[field]||[]).forEach(v=>badges.append(badge(v))));
  (p.classification.asset_types||[]).forEach(v=>badges.append(badge(v,true)));
  article.append(badges);
  const facts=el('div','facts');
  const pairs=[
    ['被测对象',(p.classification.target_systems||[]).map(label).join('、')||'待确认'],
    ['安全测试类别',(p.classification.safety_categories||[]).map(label).join('、')||'综合安全数据'],
    ['攻击技术',(p.classification.attack_vectors||[]).map(label).join('、')||'不涉及明确攻击技术'],
    ['模态',(p.classification.modalities||[]).map(label).join('、')||'待确认'],
    ['语言',(p.classification.languages||[]).map(label).join('、')||'待确认'],
    ['访问状态',(p.artifacts||[]).some(a=>a.gated)?'Gated':'公开或待确认'],
    ['分类置信度',Math.round((p.classification.classification_confidence||0)*100)+'%'],
    ['HF 下载量',Number((p.quality||{}).hf_downloads||0).toLocaleString('zh-CN')],
    ['HF 点赞数',Number((p.quality||{}).hf_likes||0).toLocaleString('zh-CN')]
  ];
  pairs.forEach(([k,v])=>{const d=el('div'); d.append(el('b','',k+'：'),document.createTextNode(v)); facts.append(d);});
  article.append(facts);
  if ((p.samples||[]).length) {
    const detail=el('details'); detail.append(el('summary','','示例样本'));
    const categories=p.classification.safety_categories||[];
    categories.forEach(category=>{
      const samples=p.samples.filter(sample=>(sample.safety_categories||[]).includes(category)).slice(0,3);
      if(!samples.length) return;
      detail.append(el('div','sample-category',`${label(category)}（${samples.length}/3）`));
      samples.forEach(sample=>{
        const source=sample.source_method==='raw_file'?'仓库原始文件':(sample.source_method==='readme_example'?'README 示例':'Dataset Viewer');
        const title=el('div','sample-meta',`${sample.config}/${sample.split} · row ${sample.row_index} · ${source} · 风险 ${sample.risk_level}`);
        const pre=el('pre','',JSON.stringify(sample.content,null,2)); detail.append(title,pre);
      });
    }); article.append(detail);
  } else if ((p.classification.safety_categories||[]).length) {
    const detail=el('details'); detail.append(el('summary','','示例样本暂不可访问'));
    detail.append(el('p','',sampleUnavailableReason(p)));
    article.append(detail);
  }
  const links=el('div','links');
  (p.artifacts||[]).slice(0,6).forEach(a=>{const url=a.canonical_url||'';if(!url)return;const link=el('a','',url);link.href=url;link.target='_blank';link.rel='noopener noreferrer';links.append(link);});
  if (!links.children.length && p.primary_url) { const link=el('a','',p.primary_url);link.href=p.primary_url;link.target='_blank';link.rel='noopener noreferrer';links.append(link); }
  article.append(links); return article;
}
function render() {
  const query=$('search').value.trim().toLocaleLowerCase();
  const selected={layer:$('layer').value,risk:$('risk').value,target:$('target').value,language:$('language').value};
  const projects=CATALOG.projects.filter(p=>{
    const c=p.classification||{};
    const haystack=JSON.stringify([p.name,p.summary_zh,p.variants,c]).toLocaleLowerCase();
    return (!query||haystack.includes(query)) && Object.entries(selected).every(([id,value])=>!value||(c[dims[id][0]]||[]).includes(value));
  });
  $('count').textContent=projects.length; const grid=$('grid'); grid.replaceChildren();
  if(!projects.length){grid.append(el('div','empty','没有符合当前条件的项目。'));return;}
  projects.forEach(p=>grid.append(card(p)));
}
['search','layer','risk','target','language'].forEach(id=>$(id).addEventListener(id==='search'?'input':'change',render));
render();

let manualUpdateStarted=false;
let updateTimer;
function readableTime(value) {
  if (!value) return '暂无';
  const date=new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN',{hour12:false});
}
async function readUpdateStatus() {
  clearTimeout(updateTimer);
  try {
    const response=await fetch('/api/update/status',{cache:'no-store'});
    if(!response.ok) throw new Error('status unavailable');
    const status=await response.json();
    $('updateNow').disabled=Boolean(status.running);
    if(status.running) {
      $('updateStatus').textContent=`正在扫描与更新，开始于 ${readableTime(status.started_at)}`;
    } else if(status.state==='failed') {
      $('updateStatus').textContent=`上次手动更新失败：${status.error||'未知错误'}`;
    } else {
      $('updateStatus').textContent=`上次手动更新成功：${readableTime(status.last_success_at)}`;
    }
    if(manualUpdateStarted && !status.running && status.finished_at) {
      window.location.reload();
      return;
    }
    updateTimer=setTimeout(readUpdateStatus,status.running?3000:60000);
  } catch(error) {
    $('updateNow').disabled=true;
    $('updateStatus').textContent='更新接口不可用；请通过 SafetyBench Agent 服务打开此页面。';
  }
}
$('updateNow').addEventListener('click',async()=>{
  $('updateNow').disabled=true;
  $('updateStatus').textContent='正在启动 Hugging Face 数据集扫描…';
  try {
    const response=await fetch('/api/update/run',{method:'POST'});
    if(!response.ok && response.status!==409) throw new Error('start failed');
    manualUpdateStarted=true;
    setTimeout(readUpdateStatus,500);
  } catch(error) {
    $('updateNow').disabled=false;
    $('updateStatus').textContent='无法启动更新，请检查服务日志。';
  }
});
readUpdateStatus();
</script>
</body>
</html>'''
