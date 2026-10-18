# Crawler Config 编写指南（IIS 情报采集）

> 面向 Agent 的操作手册：为 IIS 新增一个新闻数据源（爬虫任务）。
> 产物 = `CrawlTasks/news/crawler_config_<name>.py`（组件与目标配置）
> + `CrawlTasks/news/task_crawl_<name>.py`（插件包装，通常 15 行）。
> 调试工具：`IntelligenceCrawler/CrawlerPlayground.py`（UI，可人工生成/调参）。

## 1. 什么时候可以接、什么时候放弃

- **先测可达性**：`requests.get(列表页)` + 抽 1-2 篇文章页。列表页 200 且能抽出文章链接、
  文章页正文非空 → 可以接。403/401/连接重置 → **直接放弃**（不反爬）。
- **RSS 优先**：有 RSS/Atom 用 RSS（稳定、结构化）；RSS 已死或没有 → 列表页发现。
- **警惕冻结源**：feed/列表页 200 但条目日期永远不变（如 VOA 2025-03 停更）= 死源，不接。
- **paywall 检查**：文章页正文 < 2k 字符且出现 subscribe/paywall 字样 → 价值存疑，标注观察。

## 2. 组件选型决策树

```
数据源有可用 RSS/Atom？
├─ 是 → discoverer = RSSDiscoverer
│        d_fetcher = RequestsFetcher（被拦才换 PlaywrightFetcher render_page=False）
└─ 否（列表页）→ discoverer = ListPageDiscoverer
         列表页是静态 HTML（requests 能抽到链接）→ d_fetcher = RequestsFetcher
         列表页需 JS 渲染/懒加载 → d_fetcher = PlaywrightFetcher(stealth=True)，
                                   d_fetcher_kwargs.scroll_pages 触发滚动加载
文章页（e_fetcher）：
  静态 → RequestsFetcher(timeout_s=20)
  需渲染/反爬 → PlaywrightFetcher(stealth=True, render_page=True)
extractor：默认 Trafilatura（别名 Trafilatura/TrafilaturaExtractor 均可）；
  正文选择器明确的站点可用 GenericCSSExtractor(selectors=[...])
```

现成范式（抄最近的）：RSS 常规=`crawler_config_yna.py`；列表页静态=`crawler_config_news_cn.py`；
列表页重渲染=`crawler_config_tass.py`。

## 3. CRAWLER_CONFIG 字段速查

| 字段 | 语义 | 备注 |
|---|---|---|
| `d_fetcher_name/_init_param` | 发现阶段抓取器 | init_param: `log_callback, proxy, timeout_s[, stealth, render_page]` |
| `e_fetcher_name/_init_param` | 文章页抓取器 | 同上 |
| `discoverer_name/_init_param` | 发现器 | RSSDiscoverer / ListPageDiscoverer / SitemapDiscoverer |
| `extractor_name/_init_param` | 提取器 | 通常 `Trafilatura, {'verbose': False}` |
| `entry_points` | 入口 | **优先 dict**：`{'world': url, 'politics': url, 'business': url}`，key 成为治理分组名 |
| `period_filter` | `(start, end)` | 仅 SitemapDiscoverer 用；其他写 `(None, None)` |
| `channel_filter` | 频道白名单 | 通常 `None` |
| `d_fetcher_kwargs` | 透传 `get_content()` | `wait_until/wait_for_selector/wait_for_timeout_s/scroll_pages`；Requests 下基本无用 |
| `e_fetcher_kwargs` | 同上 | 同上 |
| `extractor_kwargs` | 透传 `extract()` | 通常 `{}` |
| `article_filter` | 文章 URL 过滤 | 可设（如排除视频页 `lambda url: '/video/' not in url`）；`run_common_flow` 会将它与本地去重过滤**组合**生效 |
| `content_handler / exception_handler` | 回调 | **保持 None**，由 `run_common_flow` 注入（提交 Hub） |

ListPageDiscoverer 的 `discoverer_init_param`：
- 默认 `{'verbose': False}` 纯启发式即可（签名聚类 + 文本唯一性淘汰赛）；
- 启发式失效时：`extraction_mode`（auto/plain_list/card/signature）、
  `scope_selector`（列表容器 CSS）、`manual_specified_signature`（文章链接 CSS 指纹，
  优先级最高；用 CrawlerPlayground 的「Inspect...」挑选）。

## 4. task_crawl_<name>.py 模板

```python
from typing import Optional

from CrawlerServiceEngine import ServiceContext
from crawler_config_<name> import CRAWLER_CONFIG          # 裸导入，勿加包前缀
from Workflow.IntelligenceCrawlFlow import CommonIntelligenceCrawlFlow

NAME = '<name>'                                           # 治理分组前缀
FLOW: Optional[CommonIntelligenceCrawlFlow] = None

def module_init(service_context: ServiceContext):
    global FLOW
    FLOW = CommonIntelligenceCrawlFlow(NAME, service_context)

def start_task(stop_event):
    if FLOW: FLOW.run_common_flow(CRAWLER_CONFIG.copy(), stop_event)
```

要点：
- **必须 `CRAWLER_CONFIG.copy()`**——`run_common_flow` 会就地覆写 proxy/handler；
- `global_site=True`（默认）注入 `collector.json` 的 `global_site_proxy`（海外站）；
  国内站传 `global_site=False`；配置里的 `proxy: None` 会被覆写，留 None 即可；
- 子系统归属按目录：`CrawlTasks/news/` → news。

## 5. 治理与提交语义（写配置前必须知道）

- **入口页 is_seed**：注册为组入口的列表页/feed URL 永远可重复抓；文章 URL 成功/
  永久失败/跳过后不再抓（TEMP_FAIL 最多重试 3 次）。
- **去重键 = informant（文章 URL）**：服务端 cache 按 informant 去重。
- **提交契约**（2026-10 修复后）：`/collect` 应答 `resp` ∈ `queued/rejected/error`；
  服务端明确拒绝（重复/校验失败）→ 该 URL 记 SKIPPED 不重试；传输错误 → 缓存下轮重试。
- 治理台可读：`entry_rounds`（每轮 发现/成功/失败/跳过/忽略）、`crawl_status`（URL 终态）。
- 轮次节奏固定 15 分钟（`DEFAULT_CRAWL_LOOP_DURATION`）。

## 6. 本地验证（对照执行）

环境用生产 conda 环境：`C:\Users\sleep\.conda\envs\iis\python.exe`（anaconda 缺 trafilatura/playwright）。

```python
# 最小验证脚本：发现 + 抽 2 篇正文，不提交、不走治理
from IntelligenceCrawler.CrawlPipeline import build_pipeline
config = dict(CRAWLER_CONFIG)
config['d_fetcher_init_param']['proxy'] = 'socks5://127.0.0.1:10808'   # 海外站
config['e_fetcher_init_param']['proxy'] = 'socks5://127.0.0.1:10808'
pipeline = build_pipeline(name='<name>', config=config)
# discover_channels → get_articles_for_channel → extractor.extract 逐段验证
```

**验收清单**：
1. 每个 entry_point 能发现 ≥ 5 个文章链接；
2. 抽 2 篇文章：HTTP 200、Trafilatura `success`、markdown 正文 ≥ 1k 字符、
   metadata 有 title 和 date；
3. 文章链接形如稳定详情页（非 `#`/JS 跳转）；
4. 条目日期是新的（防冻结源）；
5. 运行配置自带 demo `python CrawlTasks/news/crawler_config_<name>.py`
   能落盘 Markdown 到 `CRAWLER_OUTPUT/`（可选）。

部署：文件放进 `CrawlTasks/news/` 即被 CrawlerServiceEngine 热加载；
之后在治理台（默认 18000 端口）观察首轮的 发现/成功/跳过 是否符合预期。
