# 字段字典（Field Dictionary）

> 本节给出前后端交互中常用对象的**精确定义**（字段名、类型、是否必填、含义、取值约束）。  
> 说明：类型以 JSON/JS 视角描述；时间字段同时标注来源与格式。

***

## 0. 通用基础类型与约定

### 0.1 `Status`（任务状态枚举，int）

```text
0 PENDING   待处理/新发现
1 RUNNING   处理中
2 SUCCESS   成功
3 TEMP_FAIL 临时失败（可重试）
4 PERM_FAIL 永久失败（不可重试）
5 SKIPPED   跳过
6 STOPPED   中断/停止
7 CACHED    缓存命中（通常为“内存态”事件）
8 IGNORED   忽略（通常为“内存态”事件）
```

### 0.2 时间字段格式（可能出现两类）

*   **SQLite 字符串时间**：`"YYYY-MM-DD HH:MM:SS[.ffffff]"`（例如 `"2026-02-03 10:55:03.123456"`）
*   **ISO 字符串时间**：`"YYYY-MM-DDTHH:MM:SS.ssssss"`（例如 `"2026-02-03T10:55:03.123456"`）
*   **Unix 秒时间戳（number）**：例如 `1706935200`（用于趋势桶、查询参数等）

***

## 1. DashboardStats（全局统计对象）

**来源接口**：`GET /api/dashboard/stats`

```json
{
  "active_spiders": 0,
  "success_rate": 92.3,
  "total_requests": 1200,
  "network_errors": 35,
  "running_count": 8,
  "pending_count": 560,
  "session_start": "2026-02-03 10:12:33.123456"
}
```

字段定义：

*   `active_spiders` *(int, required)*：活跃 spider 数（当前实现可能为占位）
*   `success_rate` *(number, required)*：成功率百分比（0\~100，保留 1 位小数）
*   `total_requests` *(int, required)*：请求总数（口径：traffic.total 聚合）
*   `network_errors` *(int, required)*：失败总数（口径：failed 聚合，包含 TEMP\_FAIL/PERM\_FAIL/STOPPED）
*   `running_count` *(int, required)*：RUNNING 数
*   `pending_count` *(int, required)*：PENDING 数（来自 `crawl_status`）
*   `session_start` *(string, required)*：会话起始时间（SQLite 字符串时间或 datetime 的字符串化）

***

## 2. Group（分组摘要对象）

**来源接口**：`GET /api/groups`

```json
{
  "group_path": "spider/news/tech",
  "name": "Tech",
  "stats": { "...": "GroupStats" },
  "list_url_status": { "...": "AnchorStatus | null" },
  "round_id": 3
}
```

字段定义：

*   `group_path` *(string, required)*：分组路径（层级用 `/` 分隔）
*   `name` *(string, required)*：显示名称
*   `stats` *(object, required)*：分组统计（见 **GroupStats**）
*   `list_url_status` *(object|null, optional)*：该 group 的入口/种子 URL 状态（见 **AnchorStatus**）
*   `round_id` *(int, required)*：当前轮次 ID（无轮次时可能为 0）

> 注：当前 `Group` 对象中一般不直接携带 `spider` 字段；spider 可从 `group_path` 首段推导。

***

## 3. GroupStats（分组统计对象）

**嵌入对象**：`Group.stats`

```json
{
  "results": { "...": "CountStats" },
  "traffic": { "...": "CountStats" },
  "perf": { "...": "PerfStats" }
}
```

字段定义：

*   `results` *(object, required)*：结果统计（通常为“按 URL 去重后的结果”，内存统计更准确）
*   `traffic` *(object, required)*：流量统计（按事件/日志次数统计，受重试影响）
*   `perf` *(object, required)*：性能统计（耗时）

***

## 4. CountStats（计数统计对象）

**嵌入对象**：`GroupStats.results` / `GroupStats.traffic`

```json
{
  "total": 100,
  "success": 90,
  "failed": 10,
  "running": 0
}
```

字段定义：

*   `total` *(int, required)*：有效总数（规则：仅成功/失败计入 total；PENDING/SKIPPED/IGNORED/CACHED 不计入）
*   `success` *(int, required)*：成功数（Status.SUCCESS）
*   `failed` *(int, required)*：失败数（Status.TEMP\_FAIL / Status.PERM\_FAIL / Status.STOPPED）
*   `running` *(int, required)*：运行中数（Status.RUNNING）

***

## 5. PerfStats（性能统计对象）

**嵌入对象**：`GroupStats.perf`

```json
{
  "min": 0.12,
  "max": 8.91,
  "sum": 102.31,
  "count": 100,
  "avg": 1.023
}
```

字段定义：

*   `min` *(number, required)*：最小耗时（秒）
*   `max` *(number, required)*：最大耗时（秒）
*   `sum` *(number, optional)*：耗时总和（秒，聚合过程字段；有的输出可能不包含）
*   `count` *(int, required)*：统计样本数（duration > 0 且为有效终态）
*   `avg` *(number, required)*：平均耗时（秒，保留 3 位小数）

> 注：对外展示通常只需要 `min/max/avg/count`；`sum` 是否出现取决于聚合实现版本。

***

## 6. AnchorStatus（种子/入口 URL 状态对象）

**来源**：`Group.list_url_status`（来自 `crawl_status` 查询）

```json
{
  "url": "https://example.com/list",
  "status": 2,
  "last_run_at": "2026-02-03 10:55:00",
  "next_run_at": "2026-02-03 11:05:00",
  "http_code": 200,
  "state_msg": "OK"
}
```

字段定义：

*   `url` *(string, required)*：入口 URL
*   `status` *(int, required)*：状态（Status）
*   `last_run_at` *(string|null, optional)*：最近一次运行时间（SQLite 字符串时间）
*   `next_run_at` *(string|null, optional)*：下次计划运行时间（SQLite 字符串时间）
*   `http_code` *(int|null, optional)*：HTTP 状态码
*   `state_msg` *(string|null, optional)*：状态信息/错误摘要

***

## 7. RecentStatusItem（最新状态快照项）

**来源接口**：`GET /api/status/recent`（内存快照、按 URL 去重）

```json
{
  "url": "https://example.com/a",
  "status": 2,
  "spider_name": "spider",
  "group_path": "spider/news/tech",
  "last_run_at": "2026-02-03T10:55:03.123456",
  "duration": 0.83,
  "state_msg": "OK"
}
```

字段定义：

*   `url` *(string, required)*：目标 URL
*   `status` *(int, required)*：状态（Status）
*   `spider_name` *(string|null, optional)*：spider 名（通常等于 group\_path 首段）
*   `group_path` *(string, required)*：归属分组路径
*   `last_run_at` *(string, required)*：事件时间（ISO 字符串，由内存事件 ts 转换）
*   `duration` *(number, optional)*：耗时（秒）
*   `state_msg` *(string|null, optional)*：状态信息

***

## 8. LogEntry（日志流水对象）

**来源接口**：`GET /api/logs`（DB 查询，`crawl_log` LEFT JOIN `crawl_status`）

```json
{
  "id": 123,
  "url": "https://example.com/a",
  "group_path": "spider/news/tech",
  "spider_name": "spider",
  "status": 2,
  "http_code": 200,
  "duration": 0.83,
  "created_at": "2026-02-03 10:55:03",
  "url_hash": "e10adc3949ba59abbe56e057f20f883e"
}
```

字段定义：

*   `id` *(int, required)*：日志自增 ID（一次尝试）
*   `url` *(string, required)*：目标 URL
*   `group_path` *(string, required)*：归属分组
*   `spider_name` *(string, required)*：spider 名
*   `status` *(int, required)*：状态（Status）
*   `http_code` *(int|null, optional)*：HTTP 状态码
*   `duration` *(number|null, optional)*：耗时（秒）
*   `created_at` *(string, required)*：日志创建时间（SQLite 字符串时间）
*   `url_hash` *(string|null, optional)*：URL hash（来自 `crawl_status.url_hash`，若 status 表无该 URL 可能为空）

***

## 9. RoundStatus（轮次状态对象）

**来源接口**：`GET /api/group/round_status?group=...`

```json
{
  "group_path": "spider/news/tech",
  "phase": "RUNNING",
  "round_id": 3,
  "completed_rounds": 12,
  "progress_pct": 45.5,
  "expected": 200,
  "processed": 91,
  "stats": {
    "success": 80,
    "failed": 8,
    "skipped": 3,
    "other": 0
  },
  "current_duration": 33.2,
  "last_duration": 120.5,
  "seconds_until_next": 0
}
```

字段定义：

*   `group_path` *(string, required)*：分组路径
*   `phase` *(string, required)*：`"IDLE"` 或 `"RUNNING"`
*   `round_id` *(int, required)*：当前轮次 ID（未开始为 0）
*   `completed_rounds` *(int, required)*：累计完成轮次数
*   `progress_pct` *(number, required)*：进度百分比（0\~100，保留 1 位）
*   `expected` *(int, required)*：本轮预计处理数
*   `processed` *(int, required)*：本轮已处理数（通常不计入 IGNORED/CACHED）
*   `stats` *(object, required)*：本轮分类统计
    *   `success` *(int, required)*：成功数
    *   `failed` *(int, required)*：失败数（TEMP\_FAIL/PERM\_FAIL/STOPPED）
    *   `skipped` *(int, required)*：跳过数（SKIPPED）
    *   `other` *(int, required)*：其他状态计数
*   `current_duration` *(number, required)*：RUNNING 时当前已运行秒数；IDLE 时为 0
*   `last_duration` *(number, required)*：上一轮耗时（秒）
*   `seconds_until_next` *(number, required)*：距离下次运行倒计时（秒，IDLE 且 next\_run\_ts>0 时）

***

## 9.5 EntryRound（入口轮次对象）

**来源接口**：

*   `GET /api/entry/status?group=...`（内存快照）
*   `GET /api/entry/history?group=...`（DB 历史）

```json
{
  "db_id": 42,
  "group_path": "spider/news/tech",
  "round_id": 5,
  "list_url": "https://example.com/list",
  "status": 2,
  "http_code": 200,
  "state_msg": "OK",
  "started_at": 1706935200,
  "finished_at": 1706935260,
  "duration": 0.5,
  "total_duration": 60.0,
  "articles_expected": 100,
  "articles_success": 92,
  "articles_failed": 5,
  "articles_skipped": 3,
  "phase": "IDLE"
}
```

字段定义：

*   `db_id` *(int, required)*：`entry_rounds` 表自增 ID，用于关联文章明细
*   `group_path` *(string, required)*：分组路径
*   `round_id` *(int, required)*：该 group 下的轮次序号
*   `list_url` *(string, required)*：入口 URL
*   `status` *(int, required)*：入口自身最终状态（Status）
*   `http_code` *(int|null, optional)*：入口 HTTP 状态码
*   `state_msg` *(string|null, optional)*：入口状态/错误信息
*   `started_at` *(int, required)*：入口开始时间（Unix 秒）
*   `finished_at` *(int|null, optional)*：整轮结束时间（Unix 秒）
*   `duration` *(number|null, optional)*：入口自身耗时（秒）
*   `total_duration` *(number|null, optional)*：整轮总耗时（秒）
*   `articles_expected` *(int, required)*：预期文章数
*   `articles_success` *(int, required)*：成功文章数
*   `articles_failed` *(int, required)*：失败文章数
*   `articles_skipped` *(int, required)*：跳过文章数
*   `phase` *(string, required)*：`"RUNNING"` / `"IDLE"` / `"ENTRY_FAILED"`

***

## 10. TrendBucket（趋势图桶对象 / 柱状图点）

**来源接口**：`GET /api/dashboard/chart`

```json
{
  "ts": 1706935200,
  "time": "02-03 10:00",
  "success": 12,
  "fail": 3,
  "total": 15
}
```

字段定义：

*   `ts` *(int, required)*：桶起始时间（Unix 秒）
*   `time` *(string, required)*：展示标签（由后端按桶粒度格式化）
*   `success` *(int, required)*：成功数
*   `fail` *(int, required)*：失败数
*   `total` *(int, required)*：总数（通常 = success + fail；其他状态不计入）

***

## 11. HistoryStats（历史统计对象）

**来源接口**：`GET /api/history/stats?days=...`

```json
{
  "daily_trend": [
    {"date": "2026-02-01", "valid": 1200, "fail": 30, "total": 1230}
  ],
  "current_status_dist": {
    "0": 560,
    "1": 8,
    "2": 9000,
    "3": 100,
    "4": 20
  }
}
```

字段定义：

*   `daily_trend` *(array, required)*：按天趋势
    *   `date` *(string, required)*：日期 `YYYY-MM-DD`
    *   `valid` *(int, required)*：有效数（SUCCESS + 可选 CACHED，取决于实现）
    *   `fail` *(int, required)*：失败数（TEMP\_FAIL/PERM\_FAIL/STOPPED）
    *   `total` *(int, required)*：当天日志总数（按 `crawl_log` 聚合）
*   `current_status_dist` *(object, required)*：当前状态分布（来自 `crawl_status`）
    *   key 为状态码字符串（例如 `"2"`），value 为计数

***

## 12. RPC 请求/响应对象（供爬虫调用）

### 12.1 RegisterGroupRequest / Response

**接口**：`POST /rpc/register_group`

```json
{
  "group_path": "spider/news/tech",
  "list_url": "https://example.com/list",
  "name": "Tech"
}
```

*   `group_path` *(string, required)*：分组路径
*   `list_url` *(string|null, optional)*：入口 URL（也可用字段 `url` 兼容）
*   `name` *(string|null, optional)*：显示名

响应：

```json
{"status": "registered"}
```

***

### 12.2 ShouldCrawlRequest / Response

**接口**：`POST /rpc/should_crawl`

请求：

```json
{
  "url": "https://example.com/a",
  "max_retries": 3
}
```

*   `url` *(string, required)*
*   `max_retries` *(int, optional, default 3)*

响应：

```json
{"should_crawl": true}
```

***

### 12.3 ReportResultRequest / Response

**接口**：`POST /rpc/report_result`

请求：

```json
{
  "url": "https://example.com/a",
  "group_path": "spider/news/tech",
  "spider": "spider",
  "status": 2,
  "duration": 0.83,
  "http_code": 200,
  "state_msg": "OK",
  "error_msg": null,
  "file_path": "/abs/path/to/file.html"
}
```

字段定义：

*   `url` *(string, required)*
*   `group_path` *(string, required)*
*   `spider` *(string|null, optional)*：缺省时可由 `group_path` 首段推导
*   `status` *(int, required)*：Status
*   `duration` *(number, optional, default 0.0)*：秒
*   `http_code` *(int, optional, default 0)*
*   `state_msg` *(string|null, optional)*：状态消息
*   `error_msg` *(string|null, optional)*：错误消息（后端会优先用它）
*   `file_path` *(string|null, optional)*：落盘路径

响应：

```json
{"status": "acked"}
```

***

### 12.4 RoundLifecycleRequest / Response

**接口**：`POST /rpc/round/lifecycle`

请求（start）：

```json
{
  "action": "start",
  "group": "spider/news/tech",
  "expected_count": 200
}
```

请求（finish）：

```json
{
  "action": "finish",
  "group": "spider/news/tech",
  "next_run_delay": 60
}
```

字段定义：

*   `action` *(string, required)*：`"start"` | `"finish"`
*   `group` *(string, required)*：group\_path
*   `expected_count` *(int, required for start)*：预计数量
*   `next_run_delay` *(number, optional for finish, default 0)*：秒

响应：

```json
{"status": "started", "group": "spider/news/tech"}
```

或

```json
{"status": "finished", "group": "spider/news/tech"}
```
