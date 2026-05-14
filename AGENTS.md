# IntelligenceCrawler — Agent Guide

> 本文档供 AI 编码助手阅读。它记录 IntelligenceCrawler 模块在 **v4.0 扁平化改造** 后的设计原则、配置流和开发约定。先阅读本文件，再阅读 `README.md`。

---

## 1. 模块概述

IntelligenceCrawler 是 IIS（IntelligenceIntegrationSystem）的爬虫子模块，负责：

```
发现频道列表 → 抓取文章列表 → 提取正文与元数据
   Discoverer       Fetcher           Extractor
```

核心组件：

| 文件 | 职责 |
|------|------|
| `Fetcher.py` | 网络抓取：Requests / Playwright / Playwright+Stealth |
| `Discoverer.py` | 频道发现：Sitemap / RSS / 列表页智能分析 |
| `Extractor.py` | 内容提取：Trafilatura / Readability / Newspaper3k / GenericCSS / Crawl4AI |
| `CrawlPipeline.py` | 工作流编排，消费 `CRAWLER_CONFIG` 执行完整抓取流程 |
| `CrawlerPlayground.py` | **PyQt5 GUI**，用于可视化调试组件组合并生成配置 |
| `CrawlerCodeGenerator.py` | 将扁平配置序列化为 `CrawlerConfig.py` |
| `PlaywrightActionEngine_v2.py` | Playwright 页面交互引擎（8 种 action） |
| `ActionEditorDialog.py` | 独立的 action 序列编辑器对话框 |

---

## 2. 核心设计原则（v4.0 扁平化改造后）

### 2.1 三种角色，职责分明

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Widget/Panel   │ ──► │    Playground   │ ──► │  Worker / Pipe  │
│   (控件自治)     │     │   (纯透传组装)   │     │  (消费扁平配置)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**原则 1：控件自治**
- 每个 widget/panel 内部管理自己的 UI 状态和配置片段
- 对外只暴露 `get_config() → Dict[str, Any]` 和 `set_config(config)`
- 内部负责：UI name ↔ class name 映射、字段重命名、默认值填充
- Playground **不允许**直接读写 widget 的内部控件

**原则 2：Playground 纯透传**
- `_build_flat_config_dict()` 只做 `dict.update()` 合并，不做任何语义转换
- `load_config_from_file()` 只做配置分发，不做手动恢复
- Playground 对 `post_extra_action` 等具体字段零感知

**原则 3：统一扁平配置**
- 唯一真相源：`CRAWLER_CONFIG` 扁平字典
- Workers、Generator、CrawlPipeline 都消费同一套键名
- 消除旧版的"嵌套配置 → 扁平配置 → 运行时参数"多层转换

---

## 3. 扁平配置规范（CRAWLER_CONFIG）

### 3.1 完整键名表

```python
CRAWLER_CONFIG = {
    # --- Entry Points (实时解析) ---
    "entry_points": {"bbc": "https://bbc.com/news", ...},

    # --- Discoverer ---
    "discoverer_name": "SitemapDiscoverer",           # 或 RSSDiscoverer / ListPageDiscoverer
    "discoverer_init_param": {
        "verbose": True,
        "manual_specified_signature": "...",          # 仅 ListPageDiscoverer
        "scope_selector": "...",                      # 仅 ListPageDiscoverer
    },
    "date_filter_enabled": False,
    "date_filter_days": 7,

    # --- Discovery Fetcher ---
    "d_fetcher_name": "RequestsFetcher",              # 或 PlaywrightFetcher / PlaywrightStealthFetcher
    "d_fetcher_init_param": {
        "log_callback": print,
        "proxy": "socks5://127.0.0.1:10808",
        "timeout_s": 10,
        "stealth": False,
        "pause_browser": False,
        "render_page": False,
    },
    "d_fetcher_kwargs": {
        "wait_until": "domcontentloaded",
        "wait_for_selector": None,
        "wait_for_timeout_s": 10,
        "scroll_pages": 0,
        "post_extra_action": [                        # 可为 None
            {"action": "click", "selector": "#accept", "timeout": 3000},
            ...
        ],
    },

    # --- Extractor ---
    "extractor_name": "TrafilaturaExtractor",         # 或 Readability / Newspaper3k / GenericCSS / Crawl4AI / PassThrough
    "extractor_init_param": {
        "verbose": True,
    },
    "extractor_kwargs": {
        "selectors": ["article", ".content"],         # 仅 GenericCSSExtractor
    },

    # --- Article Fetcher ---
    "e_fetcher_name": "PlaywrightStealthFetcher",
    "e_fetcher_init_param": {
        "log_callback": print,
        "proxy": None,
        "timeout_s": 20,
        "stealth": True,
        "pause_browser": False,
        "render_page": True,
    },
    "e_fetcher_kwargs": {
        "wait_until": "networkidle",
        "wait_for_selector": None,
        "wait_for_timeout_s": 20,
        "scroll_pages": 0,
        "post_extra_action": None,
    },

    # --- Channel Filter (由 tree 勾选状态生成) ---
    "channel_filter": {
        "channel_list_filter": ["news/sitemap.xml", "blog/rss.xml"]
    },
    # 或 "channel_filter": None,

    # --- Runtime computed (兼容 CrawlPipeline) ---
    "period_filter": (datetime, datetime),            # 或 (None, None)

    # --- Placeholders (预留) ---
    "article_filter": None,
    "content_handler": None,
    "exception_handler": None,
}
```

### 3.2 键名前缀约定

| 前缀 | 含义 | 来源 widget |
|------|------|------------|
| `d_` | Discovery（频道发现阶段） | `FetcherConfigWidget(prefix='d_')` |
| `e_` | Extraction（文章提取阶段） | `FetcherConfigWidget(prefix='e_')` |
| 无前缀 | Discoverer / Extractor / 全局 | `DiscovererConfigPanel` / `ExtractorConfigPanel` |

### 3.3 `post_extra_action` 格式

`post_extra_action` 是 `FetcherConfigWidget` 内部管理的 action 序列，通过 `ActionEditorDialog` 编辑。

每个 step 是一个 dict，支持 8 种 action：

| action | needs_target | needs_value | 说明 |
|--------|-------------|-------------|------|
| `click` | ✅ | ❌ | 点击元素 |
| `fill` | ✅ | ✅ | 输入文字 |
| `press` | ✅ | ✅ | 按下按键（如 Enter） |
| `check` | ✅ | ❌ | 勾选复选框 |
| `uncheck` | ✅ | ❌ | 取消勾选 |
| `wait` | opt | ❌ | 等待元素出现（或固定超时） |
| `sleep` | ❌ | ❌ | 固定休眠（毫秒） |
| `scroll` | opt | opt | 滚动页面或元素 |

step 字段：
```python
{
    "action": "click",
    "selector": "#cookie-accept",   # 或 "text": "同意并继续"
    "value": "",                     # fill/press/scroll 时使用
    "timeout": 3000,
    "on_success": "next",            # "next" / "skip" / "stop" / int(jump)
    "on_fail": "stop",               # 同上
}
```

---

## 4. 关键类职责

### 4.1 自治控件

| 类 | 职责 | 输出键 |
|----|------|--------|
| `FetcherConfigWidget` | Fetcher 参数 + Action 序列 | `d_/e_fetcher_name`, `d_/e_fetcher_init_param`, `d_/e_fetcher_kwargs` |
| `DiscovererConfigPanel` | Discoverer 选择 + 日期过滤 | `discoverer_name`, `discoverer_init_param`, `date_filter_enabled`, `date_filter_days` |
| `ExtractorConfigPanel` | Extractor 选择 + CSS selectors | `extractor_name`, `extractor_init_param`, `extractor_kwargs` |

### 4.2 Playground 组装层

| 方法 | 职责 |
|------|------|
| `_build_flat_config_dict()` | 从自治控件聚合扁平配置；实时解析 URL 输入为 `entry_points`；计算 `period_filter` |
| `load_config_from_file()` | 加载 `.py` 配置 → 解析 `CRAWLER_CONFIG` → 分发给各 `set_config()` |
| `update_generated_code()` | 调用 `_build_flat_config_dict()` → `CrawlerCodeGenerator.generate_code_from_config()` |

### 4.3 Workers（全部消费扁平配置）

| Worker | 消费的关键键 |
|--------|-------------|
| `ChannelDiscoveryWorker` | `discoverer_name`, `discoverer_init_param`, `d_fetcher_name`, `d_fetcher_init_param`, `d_fetcher_kwargs` |
| `ArticleListWorker` | 同上 |
| `ExtractionWorker` | `extractor_name`, `extractor_init_param`, `extractor_kwargs`, `e_fetcher_name`, `e_fetcher_init_param`, `e_fetcher_kwargs` |
| `ChannelSourceWorker` | 同 `ChannelDiscoveryWorker` |
| `SignatureAnalysisWorker` | 同 `ChannelDiscoveryWorker` |

### 4.4 Generator

`CrawlerCodeGenerator.generate_code_from_config(config)`：
- 直接消费扁平 `CRAWLER_CONFIG`
- 使用 `pprint.pformat(config, width=120, sort_dicts=False)` 生成代码
- **零模板字符串**，零手动字段拼接
- 输出可直接 `exec()` 得到有效的 `CRAWLER_CONFIG`

---

## 5. 数据流转图

```
用户操作（调参、勾选 tree、编辑 action）
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 自治控件内部状态                                               │
│ • FetcherConfigWidget._post_extra_action                     │
│ • DiscovererConfigPanel.discoverer_combo                    │
│ • ExtractorConfigPanel.css_selector_input                   │
│ • tree_widget.checkState()                                  │
└─────────────────────────────────────────────────────────────┘
    │  get_config()
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Playground._build_flat_config_dict()                        │
│ • 合并各控件输出（纯 dict.update）                             │
│ • 实时解析 url_input → entry_points                          │
│ • tree 勾选 → channel_filter                                │
│ • date_filter_* → period_filter（兼容 CrawlPipeline）        │
└─────────────────────────────────────────────────────────────┘
    │  flat CRAWLER_CONFIG
    ├──► CrawlerCodeGenerator ──► 保存为 CrawlerConfig.py
    │
    ├──► ChannelDiscoveryWorker ──► discoverer.discover_channels()
    ├──► ArticleListWorker      ──► discoverer.get_articles_for_channel()
    ├──► ExtractionWorker       ──► fetcher.get_content() + extractor.extract()
    ├──► ChannelSourceWorker    ──► discoverer.get_content_str()
    └──► SignatureAnalysisWorker ──► discoverer.get_signature_groups()

加载配置（逆向）：
    CrawlerConfig.py ──► exec() ──► CRAWLER_CONFIG
        │
        ▼
    Playground.load_config_from_file()
        │  set_config() 分发到各控件
        ▼
    各控件恢复 UI 状态
```

---

## 6. 开发约定

### 6.1 添加新 Widget/Panel

如果新增一个配置控件：

1. **内部自治**：自己管理所有 UI 字段的读写
2. **暴露接口**：
   ```python
   def get_config(self) -> Dict[str, Any]: ...
   def set_config(self, config: Dict[str, Any]) -> None: ...
   ```
3. **Playground 只注册、不侵入**：
   ```python
   # Playground.__init__
   self.new_panel = NewPanel(self)
   
   # _build_flat_config_dict
   if self.new_panel:
       config.update(self.new_panel.get_config())
   
   # load_config_from_file
   if self.new_panel:
       self.new_panel.set_config(config)
   ```

### 6.2 添加新 Worker

Worker 必须消费扁平配置：

```python
class NewWorker(QRunnable):
    def __init__(self, config: Dict[str, Any], ...):
        self.config = config  # 直接存扁平配置
        ...

    def run(self):
        fetcher_name = self.config.get('d_fetcher_name')
        fetcher_params = self.config.get('d_fetcher_init_param', {}).copy()
        ...
```

**禁止**在 Worker 内部构造嵌套配置结构。

### 6.3 添加新 Fetcher/Discoverer/Extractor

1. 在 `Fetcher.py` / `Discoverer.py` / `Extractor.py` 中实现类
2. 在对应的 `*_factory()` 中注册
3. 在 Playground 的 `FetcherConfigWidget` / `DiscovererConfigPanel` / `ExtractorConfigPanel` 中：
   - 更新 combo box 选项
   - 更新 name ↔ class 映射
   - `get_config()` / `set_config()` 中处理新参数

### 6.4 配置兼容性

- **旧配置加载**：`load_config_from_file()` 目前**不再**包含 `period_filter` 兼容 shim（v4.0 已移除）。旧配置若含 `period_filter` 而无线 `date_filter_*`，需手动转换。
- **Worker 兼容性**：所有 Worker 已统一消费扁平配置，不再支持旧版嵌套配置传入。

---

## 7. 启动方式

```bash
# 直接启动 Playground GUI
python IntelligenceCrawler/CrawlerPlayground.py

# 使用生成的配置运行 Pipeline
python -c "
from IntelligenceCrawler.CrawlerConfig import CRAWLER_CONFIG
from IntelligenceCrawler.CrawlPipeline import create_pipeline, drive_pipeline_batch

pipeline = create_pipeline(CRAWLER_CONFIG, name='demo')
drive_pipeline_batch(pipeline, CRAWLER_CONFIG)
"
```

---

## 8. 常见问题

| 现象 | 原因 | 排查 |
|------|------|------|
| Playground 生成代码后保存，再加载时 Action 丢失 | `post_extra_action` 未被 `set_config` 恢复 | 检查 `FetcherConfigWidget.set_config()` 是否读取了 `kwargs.get('post_extra_action')` |
| Worker 报 "unsupported action" | `post_extra_action` 中的 action 不在 `PlaywrightActionEngine` 支持的 8 种内 | 检查 `ActionEditorDialog` 是否限制了选项 |
| Generator 输出格式错乱 | `pprint.pformat` 参数问题 | 确认 `sort_dicts=False` 和 `width=120` |
| load_config 后 URL 输入框为空 | `entry_points` 是 dict 但 `str()` 输出格式不被解析 | 检查 `parse_input_to_channel_dict` 是否支持 Python dict literal |

---

> **最后更新**：2026-05-13 — 根据 v4.0 扁平化改造整理。
