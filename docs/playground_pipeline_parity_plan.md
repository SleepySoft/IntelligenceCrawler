# Playground 与 Pipeline 一致性及产品化实施方案

## 1. 目标

CrawlerPlayground 不是另一套爬虫实现，而是 `CRAWLER_CONFIG` 与正式
`CrawlPipeline` 的可视化编辑、诊断和小样本试跑入口。最高优先级验收条件是：

1. 同一配置在 UI、生成文件和正式 Pipeline 中具有相同过滤与提取语义；
2. UI 无法编辑的字段必须无损往返；
3. 诊断数据来自框架公共能力，不在 UI 内复制业务判断；
4. 调试默认不得触发提交、写盘等外部 handler 副作用。

## 2. 已实施能力

### 2.1 无损配置往返

`PlaygroundConfigState` 保存加载基线和 UI 投影，只把用户实际改动递归合并回
基线。未知顶层字段、嵌套扩展参数、固定 `period_filter` 和 callable hook 均被
保留；切换 Fetcher/Discoverer/Extractor 类型时才整体替换对应组件参数。

“Changes...” 显示当前配置相对加载基线的语义差异，保存前可人工核对。

### 2.2 声明式 URL 过滤

配置键 `url_filter`：

```python
"url_filter": {
    "allowed_hosts": ["example.com"],
    "include_subdomains": False,
    "path_includes": ["/politics/", "/world/"],
    "path_excludes": ["/sports/", "/opinion/"],
    "query_policy": "reject",  # allow | reject
}
```

规则由 `URLFilter.py` 唯一实现。声明式规则先执行，原 `article_filter` callable
随后执行，两者为 AND 关系。稳定拒绝原因包括 `invalid_url`、
`host_not_allowed`、`path_not_included`、`path_excluded`、`query_rejected` 和
`custom_filter_rejected`。

Playground 提供规则编辑器；列表预览、单篇分析、限量试跑和正式 Pipeline 共用
上述实现。

### 2.3 过滤回归样本

配置可选携带：

```python
"url_filter_samples": [
    {"url": "https://example.com/world/a", "allowed": True},
    {"url": "https://example.com/sports/a", "allowed": False},
]
```

Playground 加载配置时立即执行样本，并在 Diagnostics 中显示通过/失败数量及失败
原因。样本属于配置测试数据，不参与正式抓取。

### 2.4 统一抓取漏斗

`CrawlTrace` 是无 UI 依赖的框架组件，记录入口、频道、文章、过滤与提取阶段的
数量；同时按“阶段.原因”聚合拒绝数并保留有限 URL 样本。正式 Pipeline 可注入
`CrawlTrace`，Playground Diagnostics 使用相同字段。采样有硬上限，避免把大段
URL 或异常文本塞入界面和日志。

### 2.5 项目配置目录

Project Configs 静态扫描 `CrawlTasks/**/crawler_config_*.py` 和任务包装器，不执行
配置模块。目录支持搜索并展示分组、Discoverer、Extractor、入口数量、自定义
过滤器、任务存在性和禁用状态。UTF-8 BOM 文件也可识别。

### 2.6 限量完整试跑

“Full Trial Run”调用正式 `build_pipeline` / `drive_pipeline_batch`，默认最多处理
5 篇、可在 1–50 间调整。试跑强制禁用 `content_handler` 和
`exception_handler`，避免提交、写盘或外部状态变化；其他配置和正式任务一致。

## 3. 测试分层

### 3.1 纯逻辑测试

- 声明式规则、拒绝原因、精确主机和子域边界；
- 声明式规则与 callable 组合及回归样本；
- CrawlTrace 计数、拒绝聚合和样本上限；
- 配置基线合并、单字段修改、组件切换及差异输出；
- 项目目录静态扫描、BOM、无效文件、任务映射和禁用状态。

### 3.2 Pipeline 等价性测试

使用无网络假组件运行正式 `CrawlPipeline`，断言实际内容集合与公共 URL 过滤器
结果相同，并校验漏斗计数。限量试跑必须在正式 driver 内裁剪，而不是 UI 截断。

### 3.3 真实 Playground 集成测试

使用 Qt offscreen 创建真实窗口，加载临时 Python 配置，覆盖文件加载、控件恢复、
隐藏字段/callable 保留、回归样本诊断、生成代码重新执行，以及无操作时差异为零。

## 4. 后续边界

- 正文级回归夹具应使用稳定离线 HTML；不把易变化的线上网页作为自动测试依赖。
- Source 健康度和定时监控属于 Crawler Governance；应消费 `CrawlTrace`，不在
  Playground 内再实现调度器。
- AI 自动生成选择器只能作为建议，不能绕过回归样本和人工确认直接写入生产配置。
