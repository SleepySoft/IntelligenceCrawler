# 爬虫治理（Crawl Governance）模块审查与软件设计文档（SDD）

***

## 1. 背景与目标

该模块的目标是为爬虫系统提供一套“治理层（Governance Layer）”能力，覆盖：

*   **任务注册与分组（Group / Spider）**：把爬虫任务按 `group_path` 分层组织，提供 UI 可聚合的节点。
*   **状态管理（Dashboard Snapshot）**：对每个 URL 维护最新状态（`crawl_status`）。
*   **审计追踪（Audit Trail）**：记录每次抓取尝试的流水（`crawl_log`）。
*   **实时监控与统计**：内存 RingBuffer 提供近实时统计与最近事件查询。
*   **调度与控制信号**：支持暂停/恢复/立即执行等控制。
*   **轮次（Round）统计**：对某个 group 的一轮任务提供进度、分类统计、耗时与倒计时。

***

## 2. 总体架构

### 2.1 分层与组件

*   **GovernanceManager（核心控制器）**
    *   对外：注册 group、判断 should\_crawl、创建事务 session、提供统计/查询 API、控制信号、sleep/wait。
    *   对内：管理 DB 与内存缓存的一致性；维护 round context；修复异常 RUNNING。

*   **DatabaseHandler（持久化层）**
    *   SQLite + WAL，统一执行/查询接口，带锁（RLock）。
    *   管理表结构初始化与列演进（`updated_at`）。

*   **StorageHandler（文件存储层）**
    *   将抓取结果保存到本地文件系统，返回绝对路径。

*   **CrawlSession（一次 URL 抓取的事务上下文）**
    *   `with manager.transaction(url, group)`：进入即标记 RUNNING 并写日志；退出时根据结果落库/回滚。
    *   支持 `success/fail_temp/fail_perm/skip/cached/ignore` 等终态提交策略。

*   **GroupRoundContext（分组轮次上下文）**
    *   维护 round 生命周期：start → update(每条事件) → finish。
    *   输出 UI 快照：进度、分类计数、耗时、倒计时等。

***

## 3. 数据模型设计（SQLite）

### 3.1 表职责

#### 1）`task_groups`：分组元数据注册表（UI 结构 & 入口）

*   `group_path`：主键，形如 `spider/news/tech`
*   `list_url`：该 group 的列表页/种子入口 URL
*   `name`：展示名
*   `config_json`：扩展配置（预留）
*   作用：**定义 UI 的“树节点”**，并把某个入口 URL 绑定到该节点。

#### 2）`crawl_status`：URL 最新状态快照（Dashboard）

*   `url`：主键（唯一 URL）
*   `url_hash`：便于短标识/文件定位
*   `group_path / spider_name`：归属信息（用于聚合/过滤）
*   `status`：状态机数值
*   `retry_count`：可重试失败计数（TEMP\_FAIL）
*   `http_code / duration / state_msg / file_path`
*   `last_run_at / next_run_at / updated_at`
*   作用：**每个 URL 仅一行**，代表当前“最终视图”。

#### 3）`crawl_log`：抓取流水日志（Audit Trail）

*   `id`：自增主键（一次尝试）
*   `url / group_path / spider_name`
*   `status / http_code / duration / created_at`
*   作用：**每次尝试都写入**，用于追溯与历史统计（Traffic）。

#### 4）`sys_control`：控制信号持久化

*   `key='global'`：全局信号
*   `signal`：NORMAL/PAUSE/IMMEDIATE
*   用于跨进程/重启保持控制状态。

#### 5）`entry_rounds`：入口轮次快照表

*   `id`：自增主键，关联 `crawl_log.entry_round_id`
*   `group_path / round_id`：联合唯一，标识某个 group 的第 N 轮入口抓取
*   `list_url`：入口 URL
*   `status / http_code / state_msg`：入口自身抓取结果
*   `started_at / finished_at / duration / total_duration`：入口开始时间、整轮结束时间、入口耗时、整轮总耗时
*   `articles_expected / articles_success / articles_failed / articles_skipped`：该轮预期/成功/失败/跳过的文章数
*   作用：**持久化每个入口的抓取轮次**，包括入口失败的轮次，使历史可追溯。

***

## 4. 状态机设计与流转

### 4.1 状态枚举（Status）

*   `PENDING`：待抓取/新发现
*   `RUNNING`：正在抓取（用于实时可见与并发保护）
*   `SUCCESS`：成功
*   `TEMP_FAIL`：临时失败（可重试，如网络超时）
*   `PERM_FAIL`：永久失败（不可重试，如解析错误/404）
*   `SKIPPED`：被逻辑跳过
*   `STOPPED`：手动/中断（或崩溃恢复时标记）
*   `CACHED`：缓存命中（“内存态”，不影响 DB 最终态）
*   `IGNORED`：临时忽略（“内存态”，不影响 DB 最终态）

### 4.2 一次 URL 抓取的状态流（核心事务）

**进入 Session（`__enter__`）**：

1.  `crawl_log` 插入一条 RUNNING（返回 `log_id`）
2.  `crawl_status` 对该 URL Upsert：状态置 RUNNING，更新时间戳

**退出 Session（`__exit__` 或显式调用 finalize）**：

*   **正常终态（落库）**：`SUCCESS / TEMP_FAIL / PERM_FAIL / SKIPPED / STOPPED`
    *   更新 `crawl_log`（同一个 `log_id`）写入终态、duration、http\_code
    *   更新 `crawl_status` 写入终态、duration、http\_code、file\_path、state\_msg
    *   TEMP\_FAIL：`retry_count += 1`；其他终态：`retry_count = 0`

*   **内存终态（不影响最终态）**：`CACHED / IGNORED`
    *   仍更新 `crawl_log`（保留审计）
    *   **回滚 `crawl_status` 到 session 前快照**
        *   若 URL 原来不存在：删除 start 时插入的行
        *   若原来存在：恢复原字段（RUNNING 会被修正回 PENDING，防止卡死）

> 设计意图：
>
> *   `crawl_status` 表示“最终视图/调度依据”，不希望缓存命中或临时忽略污染这个视图。
> *   `crawl_log` 表示“流水/审计”，可以记录这些行为，便于观测与追踪。

***

## 5. 关键业务流程与关联

### 5.1 Group 注册流程（Metadata → Anchor）

`register_group_metadata(group_path, list_url, friendly_name)`：

1.  规范化 group\_path（支持字符串/数组输入）
2.  写入 `task_groups`（upsert）
3.  写入内存 `runtime_groups`（仅本进程 session 内有效）
4.  将 `list_url` 加入 `known_anchors`（用于 UI 标识种子）
5.  确保 `crawl_status` 中存在 `list_url`，状态为 PENDING（仅首次插入）

> 设计意图：UI/看板只展示“本次会话注册过的 group”，避免全库加载。

### 5.2 是否应抓取（should\_crawl）

逻辑优先级（代码注释很完整）：

1.  **新 URL**：不在 `crawl_status` → True
2.  **并发保护**：RUNNING → False
3.  **调度时间 next\_run\_at**：理论上应优先（但当前被临时禁用：`next_run_at = False`）
4.  **种子列表页（is\_seed）**：永远可重复抓（建议依赖 next\_run\_at 防止过频）
5.  **普通文章页（one-off）**：
    *   SUCCESS / PERM\_FAIL / SKIPPED / STOPPED → False
    *   TEMP\_FAIL 且 `retry_count < max_retries` → True
    *   否则 False

> 重要现状：  
> **调度逻辑被临时移除**（`next_run_at = False # Temporary remove this logic`），会导致：
>
> *   list\_url 可能在循环中被频繁抓取（除非业务侧自行 sleep）
> *   next\_run\_at 字段存在但决策层不生效

### 5.3 Round（轮次）统计流程

业务侧手动调用：

*   `start_round(group_path, expected_count)`
    *   标记 RUNNING、round\_id++、初始化计数器与分类统计

*   每条 URL 结束时，`_finalize_event_in_memory()` 内：
    *   `round_contexts[group_path].update(status)`
    *   其中 `IGNORED/CACHED` 不计入 processed\_count（符合“仅观测不影响产出”的含义）

*   `finish_round(group_path, next_run_delay=0)`
    *   标记 IDLE，计算 last\_duration
    *   可写入 `next_run_ts`（用于 UI 倒计时；也可由 wait\_interval 接管）

*   `wait_interval(seconds, group_path=...)`
    *   sleep 前将 `ctx.next_run_ts = now + seconds`，支持 UI 展示倒计时
    *   支持 PAUSE/IMMEDIATE/stop\_event 中断或跳过等待

### 5.4 Entry Round（入口轮次）流程

入口轮次表示“针对某个 group 的 `list_url` 的一次完整处理周期”，从入口 URL 开始抓取，到该入口下所有文章提取结束。

与现有 `start_round/finish_round`（文章提取轮次）不同，Entry Round 由 `transaction(url, group_path)` **自动识别**并管理，业务侧无需新增调用：

*   **自动识别 Entry**：当 `transaction()` 发现 `url == task_groups.list_url` 时，自动创建或复用当前 RUNNING 的 Entry Round。
*   **Entry 成功**：入口抓取成功后，Entry Round 保持 `RUNNING` 状态，等待文章提取阶段。
*   **Entry 失败**：入口抓取失败后（TEMP\_FAIL / PERM\_FAIL / STOPPED），Entry Round 立即标记为 `ENTRY_FAILED` 并持久化；后续不会进入文章提取。
*   **文章提取阶段**：`start_round(group_path, expected_count)` 自动更新 Entry Round 的 `articles_expected`；每篇文章结束时自动累加 `articles_success / failed / skipped`。
*   **整轮结束**：`finish_round(group_path)` 自动计算 `finished_at` 与 `total_duration`，并将最终统计写回 `entry_rounds`。

内存与 DB 双轨：

*   内存 `entry_round_snapshots[group_path]` 保存当前 RUNNING 的 Entry Round，供 UI 实时轮询。
*   DB `entry_rounds` 保存历史轮次，进程重启后仍可查询。

***

## 6. 统计口径与数据流（Traffic vs Results）

### 6.1 内存事件模型（event\_buffer + active\_events\_map）

*   `event_buffer`：RingBuffer（deque），保存最近 N 条事件（RUNNING 与 FINISH）
*   `active_events_map`：`log_id -> event_obj`，用于把 RUNNING 事件 O(1) 更新为终态

事件生成路径：

1.  `_handle_task_start()`：插入 RUNNING 事件到 buffer，同时放入 map
2.  `_finalize_event_in_memory()`：
    *   若在 map 中：更新该对象为终态并从 map 删除
    *   否则：追加一条“孤儿终态事件”

### 6.2 聚合统计 `_get_aggregated_stats(since_time)`

输出结构：按 group\_path 聚合：

*   `traffic`：按“事件条数”（log/流水）计数
*   `results`：按“去重后的 URL 最终状态”计数（内存模式有效）
*   `perf`：min/max/avg/count（过滤掉 PENDING/RUNNING/SKIPPED/IGNORED 等）

**核心计数规则**集中在 `accumulate_counts()`：

*   RUNNING → `running += count`
*   **忽略**：PENDING / SKIPPED / IGNORED / CACHED（不计 total，也不计 success/fail）
*   其他终态：
    *   `total += count`
    *   SUCCESS → success++
    *   TEMP\_FAIL / PERM\_FAIL / STOPPED → failed++

> 口径含义：
>
> *   **Traffic**：体现“系统吞吐/尝试次数”，受重试影响。
> *   **Results**：体现“URL 结果分布（去重）”，更接近产出质量。

### 6.3 数据源选择（内存优先，DB 回退）

*   不传 since\_time → 直接用内存全量 buffer（实时）
*   传 since\_time：
    *   如果 since\_time 覆盖在 buffer 时间范围内 → 过滤内存（快）
    *   否则 → 查询 DB（历史）

DB 模式下的妥协：

*   `crawl_log` 只能按 group+status 聚合，**无法做“URL 去重最终态”**
*   所以实现为：`results = traffic`（注释写明妥协）

***

## 7. 对外接口（可视为后端 API）

*   Dashboard：
    *   `get_dashboard_summary(spider_filter, since_time)`
    *   只展示本 session 注册过的 group（runtime\_groups）
    *   同时查询 list\_url 的 crawl\_status 作为 anchor 状态

*   最近状态（纯内存去重）：
    *   `get_recent_statuses(limit, spider, status)`
    *   倒序遍历 buffer，对 URL 去重，返回最新状态

*   趋势统计（来自 crawl\_status 快照）：
    *   `get_log_trend_stats(start_ts, end_ts, bucket_minutes, group_filter, use_updated_at, include_cached_as_success)`
    *   从 `crawl_status.updated_at/last_run_at` 做桶聚合
    *   排除 `task_groups.list_url`（只看文章/子任务）

*   历史统计：
    *   `get_db_history_stats(days)`
    *   daily trend 基于 crawl\_log；当前分布基于 crawl\_status

*   Entry Round（新增）：
    *   `get_entry_round_status(group_path)`：当前 Entry Round 内存快照（实时）
    *   `get_entry_round_history(group_path, limit, offset)`：历史入口轮次（DB）
    *   `get_entry_round_articles(entry_round_db_id, limit, offset)`：某轮次下的文章明细（DB）

*   导出：
    *   `get_export_csv(export_type, group_path)`：全局/组状态/组日志导出

***

## 8. 关键设计意图总结（你这份代码“想解决什么问题”）

1.  **“快照 + 流水”双表模型**
    *   `crawl_status`：用于调度与 UI 最新态
    *   `crawl_log`：用于审计、吞吐、历史趋势

2.  **“内存热数据”加速实时看板**
    *   ring buffer + active map，避免频繁 DB 扫描
    *   统计可按 since\_time 在内存与 DB 间切换

3.  **“种子页可重复、文章页一次性”治理规则**
    *   通过 task\_groups.list\_url 识别 seed
    *   should\_crawl 体现任务类型差异（但调度目前被关闭）

4.  **“内存态终结”避免污染最终结果**
    *   CACHED/IGNORED 写 log、不改 status（回滚）

5.  **“Round 轮次”面向业务的批处理观测**
    *   expected/processed/progress、success/failed 分类、倒计时

***

## 9. 风险点与改进建议（审查结论）

### 9.1 调度逻辑被临时禁用（高优先级）

代码中将 `next_run_at` 强制置 False，导致：

*   `set_next_run()` 写入 DB，但决策层不使用
*   seed/list\_url 可能出现“过频抓取”，只能靠外部 sleep 控制

**建议**：恢复调度逻辑，并明确优先级：

*   `next_run_at` 对 seed 与 article 均生效（你注释也这么写）
*   对 seed：如果没有 next\_run\_at，建议加最小间隔保护（例如默认 60s）

### 9.2 Storage 路径清洗逻辑可能破坏层级（潜在 Bug）

`StorageHandler.save()` 使用：

```python
full_path = self.base_path / sanitize_filename(relative_path)
```

但 `relative_path` 是 `spider/group/sub/filename`，其中包含 `/`。  
`sanitize_filename()` 会把 `/` 替换成 `_`，导致目录层级被“压扁”为一个文件名，失去按 spider/group 分类的目录结构。

**建议**：只 sanitize “文件名”，不要 sanitize “路径分隔符”；或者逐段 sanitize：

*   `for part in Path(relative_path).parts: sanitize(part)`
*   `full_path = base/Path(*sanitized_parts)`

### 9.3 SQLite 并发与一致性边界

*   你使用 `check_same_thread=False` + RLock 控制，基本可用
*   但 **跨进程** 并发仍要小心（WAL 有帮助，但不是万能）
*   `_recover_incomplete_running_tasks()` 启动就把所有 RUNNING 改 PENDING，适合单实例；多实例时可能误伤

**建议**：若未来多 worker/多实例：

*   给 RUNNING 增加 `owner_id/session_id`，恢复时只恢复“属于本实例”的 RUNNING
*   或增加 RUNNING 超时机制（last\_run\_at 超过阈值才恢复）

### 9.4 统计口径：SKIPPED 不进入 total（符合或不符合要明确）

当前统计把 SKIPPED 排除（不计 total），但导出 CSV 里写了 skipped 列，且 dashboard 的 results/traffic 并未直接提供 skipped 计数。

**建议**：明确产品定义：

*   如果“跳过”也算处理过：应计入 total，并单列 skipped
*   如果“跳过”不算产出：可不计 total，但导出需从 round\_context 或新增统计字段补齐

### 9.5 “Results 去重”只在内存视图成立

DB 模式 `results=traffic` 是妥协，用户查看历史时可能误解。

**建议**：若需要真正历史去重 results：

*   增加 `crawl_log` 的“url 去重最终态”查询（复杂但可做）
*   或定期把“每日最终态快照”写入单独表（OLAP 思路）

***

## 10. 需求设计（PRD/需求列表 → 实现映射）

### 10.1 功能需求

1.  **分组注册与入口绑定**
    *   输入 group\_path/list\_url/name
    *   UI 可按 group 展示
    *   list\_url 作为 anchor 可单独显示状态

2.  **URL 生命周期状态管理**
    *   支持 PENDING → RUNNING → {SUCCESS/TEMP\_FAIL/PERM\_FAIL/SKIPPED/STOPPED}
    *   支持缓存/忽略（CACHED/IGNORED）但不影响最终态

3.  **并发保护**
    *   RUNNING 状态不可重复执行（should\_crawl 返回 False）

4.  **失败重试**
    *   TEMP\_FAIL 可重试，受 retry\_count 与 max\_retries 控制
    *   PERM\_FAIL 不重试

5.  **可观测性：实时事件与看板**
    *   最近事件列表（内存）
    *   每组统计（traffic/results/perf）
    *   全局统计（session\_stats）

6.  **轮次观测（Round）**
    *   start\_round(expected\_count)
    *   实时 processed\_count、成功/失败/跳过分类
    *   finish\_round + next run 倒计时展示

7.  **控制信号**
    *   PAUSE：等待循环暂停
    *   IMMEDIATE：立即跳出等待进入下一轮
    *   状态可持久化，重启后延续

8.  **数据导出**
    *   全局统计 CSV
    *   组状态快照 CSV
    *   组日志 CSV

### 10.2 非功能需求

*   性能：实时统计走内存，历史统计走 DB
*   可恢复：崩溃后 RUNNING 自动恢复，避免看板卡死
*   可扩展：task\_groups.config\_json 预留扩展配置
*   线程安全：DB 操作与内存统计均有锁保护

***

## 11. 实现说明（关键实现点）

### 11.1 Session 事务封装（强一致写入顺序）

*   start：`crawl_log insert` → `crawl_status upsert` → `event_buffer append`
*   finish：`crawl_log update` → `crawl_status update/rollback` → `event finalize` → `round update`

这种顺序保证：

*   DB 中总能找到一次尝试的流水
*   status 表始终是“最后一次已知状态”（除内存态回滚）

### 11.2 内存态回滚（CACHED/IGNORED）

*   进入 session 已经把 status 写成 RUNNING
*   内存态结束需要撤销影响：
    *   恢复原快照或删除新行
    *   避免污染 last\_run\_at/updated\_at（你实现目标是“不影响 DB，连 last\_run\_at 都不变”）

> 现状：回滚恢复了 last\_run\_at/next\_run\_at 等字段，达成“不改变”的语义。

### 11.3 统计聚合（单一规则源）

`accumulate_counts()` 把计数规则集中管理：

*   统计口径一致性强
*   后续变更只改一处

***
