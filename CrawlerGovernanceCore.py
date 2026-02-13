import os
import re
import time
import json
import logging
import sqlite3
import hashlib
import datetime
import threading
import collections
from pathlib import Path
from enum import IntEnum, Enum
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager

try:
    from CrawlerFlowScheduler import FlowScheduler
except Exception as e:
    print(str(e))
    from .CrawlerFlowScheduler import FlowScheduler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CrawlGovernance")


# --- Enums ---

class Status(IntEnum):
    PENDING = 0  # Ready to be crawled (or newly discovered)
    RUNNING = 1  # Currently processing (Real-time visibility)
    SUCCESS = 2  # Finished successfully
    TEMP_FAIL = 3  # Network error, timeout (Retryable)
    PERM_FAIL = 4  # Parse error, 404 (Non-retryable)
    SKIPPED = 5  # Skipped by logic (e.g., filtered content)
    STOPPED = 6  # Manually stopped or interrupted
    CACHED = 7
    IGNORED = 8  # 用于临时忽略，不改变最终状态


class ControlSignal(Enum):
    NORMAL = "NORMAL"
    PAUSE = "PAUSE"  # Pause execution loop
    IMMEDIATE = "IMMEDIATE"  # Skip current wait interval


# --- Constants ---

DEFAULT_DB_PATH = "data/db/governance.db"
DEFAULT_FILES_PATH = "data/files"
MAX_MEMORY_EVENTS = 10000


def _normalize_group_path(raw_input: Union[str, List[str], None]) -> str:
    """
    Standardizes the group path.
    Accepts:
      - str: "spider/news//tech"
      - list: ["spider", "news", "tech"]
      - None/Empty
    Returns:
      - str: "spider/news/tech"
      - default: "default"
    """
    if not raw_input:
        return "default"

    parts = []

    # 1. Flatten inputs into a list of strings
    if isinstance(raw_input, str):
        parts = raw_input.split('/')
    elif isinstance(raw_input, (list, tuple)):
        for item in raw_input:
            if item:
                # Handle case where list item contains slashes: ['spider/v1', 'news']
                parts.extend(str(item).split('/'))
    else:
        parts = [str(raw_input)]

    # 2. Clean inputs (remove empty strings, whitespace)
    clean_parts = [p.strip() for p in parts if p and p.strip()]

    # 3. Join or Fallback
    if not clean_parts:
        return "default"

    return "/".join(clean_parts)


def _extract_spider_name(normalized_group_path: str) -> str:
    """
    Derives spider name from the first segment of the group path.
    e.g., "google_bot/news/tech" -> "google_bot"
    Assumes _normalize_group_path has already been called.
    """
    return normalized_group_path.split("/")[0]


# --- Database Handler ---

class DatabaseHandler:
    """
    Handles SQLite operations.
    Manages three core tables:
    1. task_groups: Metadata registry (UI Hierarchy & Key Entry Points).
    2. crawl_status: Dashboard (Latest state of every unique URL).
    3. crawl_log: Audit Trail (History of all transaction attempts).
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self._init_schema()

    def _init_schema(self):
        with self.lock:
            cur = self.conn.cursor()

            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")

            # 1. Task Groups
            cur.execute("""
                CREATE TABLE IF NOT EXISTS task_groups (
                    group_path TEXT PRIMARY KEY,
                    list_url TEXT,
                    name TEXT,
                    config_json TEXT DEFAULT '{}',
                    created_at INTEGER NOT NULL
                )
            """)

            # 2. Crawl Status
            cur.execute("""
                CREATE TABLE IF NOT EXISTS crawl_status (
                    url TEXT PRIMARY KEY,
                    url_hash TEXT NOT NULL,
                    group_path TEXT NOT NULL,
                    spider_name TEXT NOT NULL,
                    status INTEGER DEFAULT 0,
                    retry_count INTEGER DEFAULT 0,
                    http_code INTEGER,
                    file_path TEXT,
                    last_run_at INTEGER,     -- epoch seconds
                    next_run_at INTEGER,     -- epoch seconds
                    updated_at INTEGER,      -- epoch seconds
                    duration REAL,
                    state_msg TEXT
                )
            """)

            # 3. Crawl Log
            cur.execute("""
                CREATE TABLE IF NOT EXISTS crawl_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    group_path TEXT NOT NULL,
                    spider_name TEXT NOT NULL,
                    status INTEGER,
                    http_code INTEGER,
                    duration REAL,
                    created_at INTEGER NOT NULL  -- epoch seconds
                )
            """)

            # 4. System Control
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sys_control (
                    key TEXT PRIMARY KEY,
                    signal TEXT DEFAULT 'NORMAL',
                    updated_at INTEGER NOT NULL
                )
            """)

            # Initialize global control signal if not present
            now_ts = int(time.time())
            cur.execute(
                "INSERT OR IGNORE INTO sys_control (key, signal, updated_at) VALUES ('global', 'NORMAL', ?)",
                (now_ts,)
            )

            # Indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_status_group ON crawl_status(group_path)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_status_spider ON crawl_status(spider_name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_status_updated ON crawl_status(updated_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_status_sched ON crawl_status(status, next_run_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_log_url ON crawl_log(url)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_log_created ON crawl_log(created_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_log_group_created ON crawl_log(group_path, created_at)")

            self.conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> int:
        """
        Executes SQL and returns the lastrowid (safe within lock).
        """
        with self.lock:
            try:
                cur = self.conn.cursor()
                cur.execute(sql, params)
                self.conn.commit()
                # 在锁释放前获取 ID
                return cur.lastrowid
            except sqlite3.Error as e:
                logger.error(f"DB Error: {e} | SQL: {sql}")
                raise

    def fetch_one(self, sql: str, params: tuple = ()):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(sql, params)
            return cur.fetchone()

    def fetch_all(self, sql: str, params: tuple = ()):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(sql, params)
            return cur.fetchall()

    def fetch_one_dict(self, sql: str, params: tuple = ()) -> Optional[dict]:
        """
        Fetch one row and convert sqlite3.Row -> dict.
        Returns None if no row found.

        Note:
        - Keeps the original fetch_one() behavior intact.
        - This method is recommended for service/engine layers to avoid Row/.get() incompatibility.
        """
        row = self.fetch_one(sql, params)
        return dict(row) if row else None

    def fetch_all_dict(self, sql: str, params: tuple = ()) -> List[dict]:
        """
        Fetch all rows and convert sqlite3.Row -> dict for each row.

        Note:
        - Keeps the original fetch_all() behavior intact.
        """
        rows = self.fetch_all(sql, params)
        return [dict(r) for r in rows] if rows else []

    def executemany(self, sql: str, seq_params: List[tuple]) -> int:
        """
        Execute a parameterized SQL statement against all parameter sequences.

        Returns:
            int: cursor.rowcount (may be -1 for some statements in sqlite).

        Note:
        - Runs under lock and commits once for the whole batch.
        - Prefer this for backfill/migration to avoid per-row commit overhead.
        """
        with self.lock:
            try:
                cur = self.conn.cursor()
                cur.executemany(sql, seq_params)
                self.conn.commit()
                return cur.rowcount
            except sqlite3.Error as e:
                logger.error(f"DB Error: {e} | SQL: {sql}")
                raise

    def get_control_signal(self, key='global') -> str:
        row = self.fetch_one("SELECT signal FROM sys_control WHERE key = ?", (key,))
        return row['signal'] if row else "NORMAL"

    def set_control_signal(self, signal: str, key='global'):
        now_ts = int(time.time())
        self.execute(
            "INSERT OR REPLACE INTO sys_control (key, signal, updated_at) VALUES (?, ?, ?)",
            (key, signal, now_ts)
        )

    @contextmanager
    def transaction(self):
        """
        A simple transaction context manager.

        Usage:
            with db.transaction() as cur:
                cur.execute(...)
                cur.executemany(...)
                ...

        Guarantees:
        - BEGIN -> COMMIT on success
        - ROLLBACK on exception

        Note:
        - The lock is held for the whole transaction scope.
        - Avoid nested transactions unless you extend this to SAVEPOINT.
        """
        with self.lock:
            cur = self.conn.cursor()
            try:
                # Use a plain BEGIN for sqlite; WAL mode works fine with it.
                cur.execute("BEGIN")
                yield cur
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise


# --- File Storage Handler ---

def sanitize_filename(filename: str, replacement: str = "_", max_length: int = 200) -> str:
    """
    清洗字符串以用作文件名。

    Args:
        filename: 原始文件名字符串
        replacement: 非法字符的替换符，默认为下划线
        max_length: 文件名最大长度截断（Windows路径通常限制260，预留后缀和路径空间建议设为200）

    Returns:
        清洗后的合法文件名字符串
    """
    if not filename:
        return "untitled"

    # 1. 替换非法字符 (Windows/Linux/Mac 通用集)
    # < > : " / \ | ? * 以及 ASCII 控制字符 (0-31)
    # 网页标题中常见的竖线 | 和冒号 : 会被替换
    illegal_pattern = r'[<>:"/\\|?*\x00-\x1f]'
    clean_name = re.sub(illegal_pattern, replacement, filename)

    # 2. 去除首尾空白字符
    clean_name = clean_name.strip()

    # 3. 避免文件名以 . 或空格结尾 (Windows 可能会自动删除这些，导致找不到文件)
    clean_name = clean_name.rstrip(". ")

    # 4. 处理 Windows 保留文件名 (如 CON, PRN, AUX, NUL, COM1...LPT9)
    # 如果文件名是保留字，或者是保留字加扩展名 (如 con.txt)，在前面加个下划线
    base_name = clean_name.split('.')[0].upper()
    reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
    }
    if base_name in reserved_names:
        clean_name = f"{replacement}{clean_name}"

    # 5. 长度截断
    # 某些文件系统对文件名长度有限制 (通常 255 字节)，中文占 3 字节，保险起见按字符数截断
    if len(clean_name) > max_length:
        clean_name = clean_name[:max_length]
        # 截断后再次去除可能出现的末尾空格或点
        clean_name = clean_name.rstrip(". ")

    # 6. 兜底：如果清洗后为空 (例如原文件名全是 ???)，给个默认名
    if not clean_name:
        clean_name = "untitled_file"

    return clean_name


class StorageHandler:
    """
    Decoupled file storage.
    Accepts explicit relative paths from the caller (e.g., spider/group/filename.html).
    """

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

    def save(self, content: Union[bytes, str], relative_path: str) -> str:
        """
        Saves content to: base_path / relative_path
        Returns: Absolute path string
        """
        if not content:
            return ""

        if isinstance(content, str):
            content = content.encode('utf-8')

        full_path = self.base_path / sanitize_filename(relative_path)

        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, "wb") as f:
                f.write(content)
            return str(full_path.absolute())
        except Exception as e:
            logger.error(f"Storage Error: {e}")
            return ""


# --- Context Manager (Session) ---

class CrawlSession:
    """
    Manages the lifecycle of a single URL crawl and persists the final outcome.

    Contract / Usage Rules
    ----------------------
    This context manager only treats *control-flow outcomes* as valid ways to end a crawl.
    Control-flow outcomes MUST be produced explicitly by the caller from inside the `with`
    block, either by calling one of the terminal methods:

        - success(), skip(), fail_temp(), fail_perm(), cached(), ignore()

    or by raising the corresponding control-flow exception type (if provided by this class).

    IMPORTANT:
    - Any non-control-flow exception raised inside the `with` block is considered a
      programming error or an unclassified runtime failure.
    - Callers are required to catch such exceptions *inside the `with` block* and convert
      them into a standard control-flow outcome (e.g., map a TimeoutError to fail_temp()).
      Do NOT rely on raising exceptions outside the `with` block to set the session status:
      once the context exits, the session is finalized and cannot be updated.

    Default behavior for unhandled exceptions
    ----------------------------------------
    If a non-control-flow exception escapes the `with` block, the session will be finalized
    as PERM_FAIL (http_code=500) with a diagnostic message instructing the caller to handle
    and map the exception inside the `with` block. The original exception will still be
    re-raised (i.e., it is not suppressed) to preserve the stack trace for debugging.

    Notes
    -----
    - Nested CrawlSession usage in the same thread is not allowed.
    - KeyboardInterrupt/SystemExit are not treated as crawl failures and are allowed to
      propagate.
    """
    class Flow(Exception):
        """Base class for control-flow outcomes."""
        pass

    @dataclass
    class Success(Flow):
        reason: str = "OK"

    @dataclass
    class Skip(Flow):
        reason: str = "Skipped"

    @dataclass
    class Cached(Flow):
        reason: str = "Cached"

    @dataclass
    class Ignore(Flow):
        reason: str = "Ignored"

    @dataclass
    class FailTemp(Flow):
        reason: str = "Retryable Error"
        http_code: int = 0
        next_run_in: int = 0  # 可选：多少秒后再跑（0 表示不改）

    @dataclass
    class FailPerm(Flow):
        reason: str = "Permanent Error"
        http_code: int = 0


    def __init__(self, manager, url: str, spider: str, group_path: Union[str, List[str], None]):
        self.manager = manager
        self.url = url
        self.spider = spider
        self.group_path = _normalize_group_path(group_path)
        self.start_time = time.time()

        # 记录“原来的状态”
        # 在把状态改成 RUNNING 之前，先查一下它是啥
        # 默认是 PENDING (如果是新URL)
        self.original_status = Status.PENDING

        existing = self.manager.db.fetch_one("SELECT * FROM crawl_status WHERE url = ?", (url,))
        self._has_prev_row = bool(existing)
        self._prev_row_snapshot = dict(existing) if existing else None

        if existing and "status" in existing.keys():
            self.original_status = Status(int(existing["status"]))

        # Unique ID for the specific log entry of this session
        self.log_id: Optional[int] = None

        self.status = Status.RUNNING
        self.http_code = None
        self.state_msg = None
        self.file_path = None
        self._finished = False

    def __enter__(self):
        # 1) 阻止同线程嵌套
        cur = self.manager._get_tls_session()
        if cur is not None:
            raise RuntimeError(
                f"Nested CrawlSession is not allowed in the same thread. "
                f"Current={cur.url}, New={self.url}"
            )

        # 2) 先注册 TLS（让 should_crawl 在本 session 内可识别）
        self.manager._set_tls_session(self)

        # Notify manager to start transaction (Insert Log, Update Status)
        self.log_id = self.manager._handle_task_start(self.url, self.spider, self.group_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            # A) 正常退出（没有异常）
            if exc_type is None:
                if not self._finished:
                    # 你原本的策略：如果用户没显式 success/skip/... 则当成 temp fail
                    self.fail_temp(state_msg="Exited without explicit status")
                return False  # 没异常，返回值无所谓；False 更直观（不吞任何东西）

            # B) 不要把中断/退出当业务失败（可选但强烈建议）
            if exc_type in (KeyboardInterrupt, SystemExit):
                # 如果你希望 Ctrl+C 也记录成某种状态，可以在这里自定义
                return False

            # C) 如果是你的控制流异常：映射到 finalize，并吞掉异常
            if issubclass(exc_type, CrawlSession.Flow):
                e = exc_val  # type: ignore

                # 已经完成了就不要重复 finalize，但依然吞掉该控制流异常
                if self._finished:
                    return True

                if isinstance(e, CrawlSession.Success):
                    self.success(state_msg=e.reason)

                elif isinstance(e, CrawlSession.Skip):
                    self.skip(state_msg=e.reason)

                elif isinstance(e, CrawlSession.Cached):
                    self.cached(state_msg=e.reason)

                elif isinstance(e, CrawlSession.Ignore):
                    self.ignore(state_msg=e.reason)

                elif isinstance(e, CrawlSession.FailTemp):
                    if getattr(e, "next_run_in", 0):
                        self.set_next_run(e.next_run_in)
                    self.fail_temp(http_code=e.http_code, state_msg=e.reason)

                elif isinstance(e, CrawlSession.FailPerm):
                    self.fail_perm(http_code=e.http_code, state_msg=e.reason)

                else:
                    # 理论上不会走到这；兜底
                    self.fail_perm(http_code=500, state_msg=f"Unknown Flow: {e}")

                return True  # 吞掉控制流异常

            # D) 真实异常（bug / 未预期异常）
            #    规则：如果没 finished，则记录为 perm_fail；如果已 finished，不覆盖状态。
            if not self._finished:
                exc_name = getattr(exc_type, "__name__", str(exc_type))
                self.fail_perm(
                    http_code=500,
                    state_msg=(
                        f"UNHANDLED EXCEPTION ({exc_name}): {exc_val}. "
                        "This CrawlSession only accepts explicit control-flow outcomes. "
                        "Catch this exception INSIDE the 'with CrawlSession(...)' block and "
                        "convert it to a standard outcome (success/skip/fail_temp/fail_perm), "
                        "e.g. 'except TimeoutError: raise CrawlSession.FailTemp(...)'. "
                        "Exceptions raised after the context exits cannot update the session status."
                    )
                )
            logger.exception(f"Session crashed for {self.url}: {exc_val}"
                             f"You should catch inside the 'with' block and convert to a CrawlSession outcome.")

            return False

        finally:
            self.manager._clear_tls_session(self)

    def save_file(self, content: Union[bytes, str], filename: str, sub_folder: str = ""):
        """
        Save content to disk.
        Logic: base_dir / spider / group_path / sub_folder / filename
        """
        # Construct path preserving hierarchy
        rel_path = Path(self.spider) / self.group_path / sub_folder / filename
        self.file_path = self.manager.storage.save(content, str(rel_path))
        return self.file_path

    def success(self, state_msg="OK"):
        self.state_msg = state_msg
        self._finalize(Status.SUCCESS)

    def skip(self, state_msg="Skipped"):
        self.state_msg = state_msg
        self._finalize(Status.SKIPPED)

    def cached(self, state_msg="Cached"):
        self.state_msg = state_msg
        self._finalize_memory_only(Status.CACHED)

    def ignore(self, state_msg="Ignored"):
        self.state_msg = state_msg
        self._finalize_memory_only(Status.IGNORED)

    def fail_temp(self, http_code=0, state_msg="Retryable Error"):
        self.http_code = http_code
        self.state_msg = state_msg
        self._finalize(Status.TEMP_FAIL)

    def fail_perm(self, http_code=0, state_msg="Permanent Error"):
        self.http_code = http_code
        self.state_msg = state_msg
        self._finalize(Status.PERM_FAIL)

    def set_next_run(self, interval_seconds: int):
        """
        Helper for list pages to set the next scheduled run time in DB.
        This updates the 'next_run_at' field in crawl_status.
        """
        if interval_seconds > 0:
            next_run = int(time.time()) + int(interval_seconds)
            self.manager.db.execute(
                "UPDATE crawl_status SET next_run_at = ?, updated_at = ? WHERE url = ?",
                (next_run, int(time.time()), self.url)
            )

    def _finalize(self, status: Status):
        if self._finished: return
        self.status = status
        duration = round(time.time() - self.start_time, 3)

        # Commit changes to DB
        self.manager._handle_task_finish(
            log_id=self.log_id,
            url=self.url,
            spider=self.spider,
            group_path=self.group_path,
            status=status,
            duration=duration,
            http_code=self.http_code,
            state_msg=self.state_msg,
            file_path=self.file_path
        )
        self._finished = True


    def _finalize_memory_only(self, status: Status):
        """
        纯内存状态：不影响 DB（连 last_run_at 都不变）。
        要求：
        - 撤销 _handle_task_start 写入的 crawl_status/crawl_log
        - 内存 event_buffer 仍然记录该事件（用于 UI/监控）
        """
        if self._finished:
            return

        self.status = status
        duration = round(time.time() - self.start_time, 3)

        self.manager._handle_task_memory_only_finish(
            log_id=self.log_id,
            url=self.url,
            spider=self.spider,
            group_path=self.group_path,
            status=status,
            duration=duration,
            state_msg=self.state_msg,
            prev_exists=self._has_prev_row,
            prev_snapshot=self._prev_row_snapshot
        )

        self._finished = True


class GroupRoundContext:
    """
    管理单个 Group 的一轮抓取任务的上下文状态。
    """

    def __init__(self, group_path: str):
        self.group_path = group_path

        # --- 长期状态 (Session级) ---
        self.completed_rounds = 0  # 已完成总轮次 (需求4)
        self.total_items_processed_session = 0

        # --- 当前轮次状态 (Round级) ---
        self.phase = "IDLE"  # IDLE | RUNNING
        self.round_id = 0
        self.start_ts = 0.0
        self.expected_count = 0  # 本轮计划抓多少 (需求2)
        self.processed_count = 0  # 实时计数 (需求3)

        # 实时分类统计
        self.stats = {
            "success": 0, "failed": 0, "skipped": 0, "other": 0
        }

        # --- 调度信息 ---
        self.last_duration = 0.0  # 上一轮耗时 (需求5)
        self.last_end_ts = 0.0
        self.next_run_ts = 0.0  # 下一轮开始时间 (需求5)

    def start(self, expected_count: int):
        """开启新的一轮"""
        self.phase = "RUNNING"
        self.start_ts = time.time()
        self.round_id += 1
        self.expected_count = expected_count

        # 重置当前轮次计数
        self.processed_count = 0
        self.stats = {"success": 0, "failed": 0, "skipped": 0, "other": 0}
        self.next_run_ts = 0.0  # 清除之前的倒计时

        logger.info(f"[Round Start] {self.group_path} (Round #{self.round_id}, Plan: {expected_count})")

    def reduce_expected(self, count: int = 1):
        if self.expected_count >= count:
            self.expected_count -= count
        else:
            self.expected_count = 0
            logger.warning('Expected count is reduced under 0.')

    def increase_progressed(self, count: int = 1):
        self.processed_count += count
        if self.processed_count > self.expected_count:
            logger.warning('Progressed count is larger that expected.')

    def update(self, status: int):
        if self.phase != "RUNNING":
            return

        self.increase_progressed()
        self.total_items_processed_session += 1

        if status in [Status.SUCCESS]:
            self.stats["success"] += 1
        elif status in [Status.TEMP_FAIL, Status.PERM_FAIL, Status.STOPPED]:
            self.stats["failed"] += 1
        elif status in [Status.SKIPPED]:
            self.stats["skipped"] += 1
        elif status in [Status.IGNORED, Status.CACHED]:
            pass
        else:
            self.stats["other"] += 1

    def finish(self, next_run_delay: float = 0):
        """
        Transitions state to IDLE and finalizes duration.
        now: next_run_delay is optional, primarily used if set explicitly,
        otherwise wait_interval will update next_run_ts later.
        """
        now = time.time()
        if self.phase == "RUNNING":
            self.last_duration = round(now - self.start_ts, 2)
            self.completed_rounds += 1

        self.phase = "IDLE"
        self.last_end_ts = now

        # 只有当显式传入了 delay (大于0) 时才设置，否则保持为 0 或由 wait_interval 设置
        if next_run_delay > 0:
            self.next_run_ts = now + next_run_delay
        # 注意：这里不要强制设为 0，因为如果在 finish 之后立即调用 wait_interval，
        # 我们希望由 wait_interval 来接管这个字段。

        logger.info(
            f"[Round End] {self.group_path} finished. Duration: {self.last_duration}s. Next run in {next_run_delay}s")

    def get_snapshot(self) -> Dict:
        """返回给 UI/API 的只读快照"""
        now = time.time()

        # 计算进度百分比
        progress = 0.0
        if self.expected_count > 0:
            progress = round((self.processed_count / self.expected_count) * 100, 1)

        # 计算运行时长
        duration_current = 0.0
        if self.phase == "RUNNING":
            duration_current = round(now - self.start_ts, 1)

        # 计算倒计时
        ttl = 0
        if self.phase == "IDLE" and self.next_run_ts > 0:
            ttl = max(0, round(self.next_run_ts - now, 1))

        return {
            "group_path": self.group_path,
            "phase": self.phase,
            "round_id": self.round_id,
            "completed_rounds": self.completed_rounds,
            "progress_pct": progress,
            "expected": self.expected_count,
            "processed": self.processed_count,
            "stats": self.stats,  # success, failed, etc.
            "current_duration": duration_current,
            "last_duration": self.last_duration,
            "seconds_until_next": ttl
        }


# --- Main Governance Class ---

class GovernanceManager:
    """
    GovernanceManager
    -----------------
    Central controller for spider governance.

    Runtime Data Model:
    - LIVE endpoints read from in-memory hot data only (event_buffer, runtime_groups,
      round_contexts, anchor_status_cache). No DB access, no time windows.
    - QUERY endpoints read from database only, and only when the caller specifies
      a time range. QUERY never mixes memory and DB data.

    Rules:
    - If the caller does NOT specify a time range → LIVE mode (pure memory).
    - If the caller specifies a time range → QUERY mode (pure DB).
    - No hybrid/mixed strategy. Memory is NOT used to satisfy time-window queries.
    """

    def __init__(
            self,
            db_path: str = DEFAULT_DB_PATH,
            files_path: str = DEFAULT_FILES_PATH,
            scheduler: Optional[FlowScheduler] = None
    ):
        self.db = DatabaseHandler(db_path)
        self.storage = StorageHandler(files_path)
        self.scheduler = scheduler

        self._recover_incomplete_running_tasks()

        # Sync control signal from DB to memory on startup
        self._control_signal = self.db.get_control_signal(key='global')
        self._signal_lock = threading.RLock()

        self.stats_lock = threading.RLock()
        self.session_start_time = time.time()

        # --- Memory Cache (The "Hot" Subset of DB) ---

        # 1. Event Ring Buffer
        # Stores event dictionaries. This mirrors the 'crawl_log' table.
        # Contains BOTH Running and Finished events.
        self.event_buffer = collections.deque(maxlen=MAX_MEMORY_EVENTS)

        # 2. Active Event Lookup
        # Key: log_id (int) -> Value: Reference to the dict object inside self.event_buffer
        # This allows O(1) retrieval to update status from RUNNING -> SUCCESS/FAIL.
        self.active_events_map: Dict[int, dict] = {}

        # 3. Runtime Group Registry
        # Stores metadata for groups seen IN THIS SESSION.
        # Key: normalized_group_path (str)
        # Value: {'list_url': str, 'name': str}
        # Replaces the behavior of loading all groups from DB at startup.
        self.runtime_groups: Dict[str, Dict] = {}

        # RoundContext 容器
        # Key: group_path, Value: GroupRoundContext
        self.round_contexts: Dict[str, GroupRoundContext] = {}

        self.known_anchors = set()

        # --- Anchor status cache (LIVE, memory-only) ---
        # Key: list_url (str)
        # Value: {
        #   'url': str,
        #   'status': int,
        #   'http_code': int|None,
        #   'state_msg': str|None,
        #   'last_run_at': int|None,    # epoch seconds
        #   'next_run_at': int|None,    # epoch seconds (can be from round next_run)
        #   'updated_ts': float         # epoch seconds, memory timestamp
        # }
        self.anchor_status_cache: Dict[str, Dict[str, Any]] = {}

        # Use TLS to record current context
        self._tls = threading.local()

        logger.info(f"Governance Manager initialized. Signal: {self._control_signal}")

    # --- Helper: Path Normalization & Name Extraction ---

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _get_tls_session(self):
        return getattr(self._tls, "current_session", None)

    def _set_tls_session(self, sess):
        self._tls.current_session = sess

    def _clear_tls_session(self, sess):
        if getattr(self._tls, "current_session", None) is sess:
            self._tls.current_session = None

    def _recover_incomplete_running_tasks(self):
        """
        恢复上次异常退出留下的 RUNNING 状态，避免永久卡死。
        策略：
        - crawl_status.RUNNING -> PENDING
        - crawl_log.RUNNING    -> STOPPED (可选，但建议)
        """
        try:
            # 1) 恢复 crawl_status
            self.db.execute(
                "UPDATE crawl_status SET status = ? WHERE status = ?",
                (int(Status.PENDING), int(Status.RUNNING))
            )

            # 2) 恢复 crawl_log (审计更干净；不影响调度)
            self.db.execute(
                "UPDATE crawl_log SET status = ? WHERE status = ?",
                (int(Status.STOPPED), int(Status.RUNNING))
            )

            logger.info("Recovered incomplete RUNNING tasks from previous session.")
        except Exception as e:
            logger.error(f"Failed to recover RUNNING tasks: {e}")


    # --- 1. Metadata Registration (UI & Entry Points) ---

    def register_group_metadata(self, group_path: Union[str, List[str]], list_url: str, friendly_name: str = None):
        """
        Registers a group and its associated List/Index URL.
        group_path: Can be "spider/news" or ["spider", "news"]
        This establishes the node in the Dashboard.
        Also pre-fills the 'crawl_status' table with the list_url as PENDING.
        """
        # STEP 1: Normalize Input
        norm_group_path = _normalize_group_path(group_path)
        spider_name = _extract_spider_name(norm_group_path)

        # Default name if missing
        if not friendly_name:
            friendly_name = norm_group_path.split('/')[-1].capitalize()

        try:
            # 1. Update DB (Persistent Record)

            now_ts = int(time.time())
            self.db.execute("""
                INSERT INTO task_groups (group_path, list_url, name, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(group_path) DO UPDATE SET
                    list_url = excluded.list_url,
                    name = excluded.name
            """, (norm_group_path, list_url, friendly_name, now_ts))

            # 2. Update Runtime Registry (UI Visibility) <--- KEY CHANGE
            with self.stats_lock:
                self.runtime_groups[norm_group_path] = {
                    'list_url': list_url,
                    'name': friendly_name,
                    'spider': spider_name
                }

                if list_url:
                    self.known_anchors.add(list_url)

                    # Initialize anchor LIVE cache (memory-only)
                    # Note: This avoids DB reads in LIVE summary.
                    self.anchor_status_cache[list_url] = {
                        "url": list_url,
                        "status": int(Status.PENDING),
                        "http_code": None,
                        "state_msg": None,
                        "last_run_at": None,
                        "next_run_at": None,
                        "updated_ts": time.time(),
                    }

            # 3. Ensure Status Entry in DB
            if list_url:
                self.db.execute("""
                            INSERT OR IGNORE INTO crawl_status (url, url_hash, group_path, spider_name, status)
                            VALUES (?, ?, ?, ?, ?)
                        """, (list_url, self._hash(list_url), norm_group_path, spider_name, Status.PENDING))

        except Exception as e:
            logger.error(f"Failed to register group {norm_group_path}: {e}")

    # --- 2. Crawl Decision Logic ---

    def should_crawl(self, url: str, max_retries: int = 3) -> bool:
        """
        Determines whether the given URL is eligible for crawling based on its current state,
        scheduling constraints, and role (Recurrent List vs. One-off Article).

        The decision logic follows this order of precedence:

        1. **New URL**: If the URL is not found in the registry, return True.
        2. **Concurrency**: If the status is RUNNING, return False to prevent duplicate processing.
        3. **Scheduling**: If a `next_run_at` timestamp is set:
           - Return True if current time >= `next_run_at` (Scheduled runs override SUCCESS status).
           - Return False if the scheduled time has not yet arrived.
        4. **Seed/List Logic**: If the URL is registered as a group entry point (is_seed):
           - Return True. (Seeds are recurrent by definition; a previous SUCCESS status should not prevent future crawls).
        5. **Article/One-off Logic**:
           - Return False if status is SUCCESS (task completed), PERM_FAIL, SKIPPED, or STOPPED.
           - If status is TEMP_FAIL, return True only if `retry_count` < `max_retries`.

        Args:
            url (str): The target URL to check.
            max_retries (int): The maximum number of retries allowed for temporary failures. Defaults to 3.

        Returns:
            bool: True if the URL should be processed, False otherwise.
        """
        row = self.db.fetch_one("""
            SELECT 
                s.status, 
                s.retry_count, 
                s.next_run_at,
                CASE WHEN g.list_url IS NOT NULL THEN 1 ELSE 0 END as is_seed
            FROM crawl_status s
            LEFT JOIN task_groups g ON s.url = g.list_url
            WHERE s.url = ?
        """, (url,))

        if not row:
            return True

        status = row['status']
        retry_count = row['retry_count']
        next_run_at = row['next_run_at']
        is_seed = bool(row['is_seed'])

        # RUNNING: protect concurrency, but allow self-check inside current session
        if status == Status.RUNNING:
            cur_sess = self._get_tls_session()
            if cur_sess is None or cur_sess.url != url:
                return False

            # self-owned RUNNING: use snapshot before entering session
            if not cur_sess._has_prev_row or not cur_sess._prev_row_snapshot:
                return True

            snap = cur_sess._prev_row_snapshot
            status = int(snap.get("status", int(Status.PENDING)))
            retry_count = int(snap.get("retry_count", 0))
            next_run_at = snap.get("next_run_at")

        # Schedule check (disabled temporarily)
        # if next_run_at and int(time.time()) < int(next_run_at):
        #     return False

        if is_seed:
            return True

        if status == Status.SUCCESS:
            return False

        if status in [Status.PERM_FAIL, Status.SKIPPED, Status.STOPPED]:
            return False

        if status == Status.TEMP_FAIL:
            return retry_count < max_retries

        return True

    # --- 3. Session Factory ---

    def transaction(self, url: str, group_path: Union[str, List[str]]):
        """
        Starts a crawling session.
        group_path: Can be "spider/news" or ["spider", "news"]
        """
        # STEP 1: Normalize Input
        norm_group_path = _normalize_group_path(group_path)
        spider_name = _extract_spider_name(norm_group_path)

        return CrawlSession(self, url, spider_name, norm_group_path)

    # --- Round Management ---

    def _get_round_context(self, group_path: str) -> GroupRoundContext:
        """获取或创建 Context，非线程安全，需外部加锁"""
        if group_path not in self.round_contexts:
            self.round_contexts[group_path] = GroupRoundContext(group_path)
        return self.round_contexts[group_path]

    def start_round(self, group_path: Union[str, List[str]], expected_count: int):
        """业务层调用：告诉系统这组任务开始了一轮"""
        group_path = _normalize_group_path(group_path)
        with self.stats_lock:
            if ctx := self._get_round_context(group_path):
                ctx.start(expected_count)

    def skip_round_step(self, group_path: Union[str, List[str]], count: int = 1):
        group_path = _normalize_group_path(group_path)
        with self.stats_lock:
            ctx = self._get_round_context(group_path)
            ctx.increase_progressed(count)

    def reduce_round_step(self, group_path: Union[str, List[str]], count: int = 1):
        group_path = _normalize_group_path(group_path)
        with self.stats_lock:
            if ctx := self._get_round_context(group_path):
                ctx.reduce_expected(count)

    def finish_round(self, group_path: Union[str, List[str]], next_run_delay: int = 0):
        """业务层调用：告诉系统这组任务这一轮结束了"""
        group_path = _normalize_group_path(group_path)
        with self.stats_lock:
            if ctx := self._get_round_context(group_path):
                ctx.finish(next_run_delay=next_run_delay)

    def get_group_round_status(self, group_path: str) -> Dict:
        """API 调用：获取实时轮次状态"""
        group_path = _normalize_group_path(group_path)
        with self.stats_lock:
            if group_path in self.round_contexts:
                return self.round_contexts[group_path].get_snapshot()
            return {}  # 或者返回一个默认空对象

    def get_pending_count(self) -> int:
        row = self.db.fetch_one("SELECT COUNT(1) AS c FROM crawl_status WHERE status=?", (int(Status.PENDING),))
        return int(row["c"]) if row else 0

    def get_scheduler_snapshot(self, max_items_per_state: int = 50):
        """
        Unified scheduler snapshot access for backend/UI.
        """
        if not self.scheduler:
            return {}

        try:
            return self.scheduler.get_status_snapshot(max_items_per_state=max_items_per_state)
        except Exception as e:
            return {"error": f"scheduler snapshot failed: {e}"}

    def schedule_pace(
            self,
            key: str,
            interval: float = 0.0,
            stop_event: Optional[threading.Event] = None):
        return self.scheduler.pace(key, interval, stop_event)

    # --- 4. Internal State Management (Called by Session) ---

    def _handle_task_start(self, url: str, spider: str, group: str) -> int:
        """
        Called when transaction starts.
        Updated: Now stores 'url' and 'spider' in memory for rich monitoring.
        """

        now_ts = int(time.time())

        # 1. DB Insert (Log)
        log_id = self.db.execute("""
            INSERT INTO crawl_log (url, group_path, spider_name, status, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (url, group, spider, int(Status.RUNNING), now_ts))

        if log_id is None:
            log_id = 0

        # 2. DB Upsert (Status)
        self.db.execute("""
            INSERT INTO crawl_status (url, url_hash, group_path, spider_name, status, last_run_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                status = excluded.status,
                spider_name = excluded.spider_name,
                last_run_at = excluded.last_run_at,
                updated_at = excluded.updated_at
        """, (url, self._hash(url), group, spider, int(Status.RUNNING), now_ts, now_ts))

        # 3. Memory Update (Rich Object)
        with self.stats_lock:
            event_obj = {
                "id": log_id,
                "ts": time.time(),          # start time (seconds)
                "updated_ts": time.time(),  # last update time (seconds) - equals start at begin
                "url": url,
                "spider": spider,
                "group_path": group,
                "status": int(Status.RUNNING),
                "http_code": 0,          # will be filled on finish
                "duration": 0.0,
                "state_msg": None,          # will be filled on finish
                "is_anchor": url in self.known_anchors,
            }

            self.event_buffer.append(event_obj)

            # Update anchor LIVE cache (memory-only) on start
            if event_obj["is_anchor"]:
                # For anchor, we treat "last_run_at" as start time
                now_s = int(time.time())
                self.anchor_status_cache[url] = {
                    "url": url,
                    "status": int(Status.RUNNING),
                    "http_code": 0,
                    "state_msg": None,
                    "last_run_at": now_s,
                    # next_run_at can be updated when round finishes (see finish hook)
                    "next_run_at": self.anchor_status_cache.get(url, {}).get("next_run_at"),
                    "updated_ts": time.time(),
                }

            # 只有有效的 ID 才放入 Map
            if log_id > 0:
                self.active_events_map[log_id] = event_obj

        return log_id

    def _update_crawl_log_finish(self, log_id, url, spider, group_path, status, duration, http_code):
        if log_id:
            self.db.execute(
                "UPDATE crawl_log SET status=?, duration=?, http_code=? WHERE id=?",
                (int(status), duration, http_code, log_id)
            )
        else:
            now_ts = int(time.time())
            self.db.execute(
                "INSERT INTO crawl_log (url, group_path, spider_name, status, http_code, duration, created_at) VALUES (?,?,?,?,?,?,?)",
                (url, group_path, spider, int(status), http_code, duration, now_ts)
            )

    def _finalize_event_in_memory(self, log_id, url, spider, group_path, status, duration, state_msg):
        with self.stats_lock:
            if log_id and log_id not in self.active_events_map:
                logger.warning(f"Orphaned Finish Task: {url} (ID: {log_id}). Start event not found in active map.")

            if log_id and log_id in self.active_events_map:
                event_obj = self.active_events_map[log_id]
                event_obj["status"] = int(status)
                event_obj["duration"] = duration
                event_obj["state_msg"] = state_msg
                event_obj["http_code"] = 0
                event_obj["updated_ts"] = time.time()
                del self.active_events_map[log_id]
            else:
                self.event_buffer.append({
                    "id": 0,
                    "ts": time.time(),
                    "updated_ts": time.time(),
                    "url": url,
                    "spider": spider,
                    "group_path": group_path,
                    "status": int(status),
                    "http_code": 0,
                    "duration": duration,
                    "state_msg": state_msg,
                    "is_anchor": url in self.known_anchors
                })

            # Update anchor LIVE cache (memory-only) on finish
            if url in self.known_anchors:
                now_s = int(time.time())

                # Best-effort: use round_context next_run_ts if available for this group
                next_run_at = None
                ctx = self.round_contexts.get(group_path)
                if ctx and getattr(ctx, "next_run_ts", None):
                    try:
                        next_run_at = int(ctx.next_run_ts)
                    except Exception:
                        next_run_at = None

                self.anchor_status_cache[url] = {
                    "url": url,
                    "status": int(status),
                    "http_code": 0,
                    "state_msg": state_msg,
                    "last_run_at": now_s,
                    "next_run_at": next_run_at,
                    "updated_ts": time.time(),
                }

            if group_path in self.round_contexts:
                self.round_contexts[group_path].update(status)

    def _update_crawl_status_finish(self, url, spider, status, duration, http_code, state_msg, file_path):
        now_ts = int(time.time())
        retry_inc = 1 if status == Status.TEMP_FAIL else 0
        retry_reset = "retry_count = 0," if status != Status.TEMP_FAIL else ""
        self.db.execute(
            f"""
            UPDATE crawl_status
            SET status=?, duration=?, http_code=?, state_msg=?, file_path=?, spider_name=?,
                updated_at=?,
                {retry_reset} retry_count = retry_count + ?
            WHERE url=?
            """,
            (int(status), duration, http_code, state_msg, file_path, spider, now_ts, retry_inc, url)
        )

    def _rollback_crawl_status(self, url: str, prev_exists: bool, prev_snapshot: dict):
        if prev_exists and prev_snapshot:
            prev_status = prev_snapshot.get("status")
            if prev_status == int(Status.RUNNING):
                prev_status = int(Status.PENDING)  # 防止恢复卡死 RUNNING

            self.db.execute("""
                UPDATE crawl_status
                SET
                    group_path = ?,
                    spider_name = ?,
                    status = ?,
                    retry_count = ?,
                    http_code = ?,
                    file_path = ?,
                    last_run_at = ?,
                    next_run_at = ?,
                    duration = ?,
                    state_msg = ?
                WHERE url = ?
            """, (
                prev_snapshot.get("group_path"),
                prev_snapshot.get("spider_name"),
                prev_status,
                prev_snapshot.get("retry_count"),
                prev_snapshot.get("http_code"),
                prev_snapshot.get("file_path"),
                prev_snapshot.get("last_run_at"),
                prev_snapshot.get("next_run_at"),
                prev_snapshot.get("duration"),
                prev_snapshot.get("state_msg"),
                url
            ))
        else:
            # 原本没有该 URL：撤销 start 时插入的行
            self.db.execute("DELETE FROM crawl_status WHERE url = ?", (url,))

    def _handle_task_finish(self, log_id, url, spider, group_path, status, duration, http_code, state_msg, file_path):
        if status in (Status.IGNORED, Status.CACHED):
            raise RuntimeError("IGNORED/CACHED must use _handle_task_memory_only_finish")

        # 1) crawl_log：保留记录
        self._update_crawl_log_finish(log_id, url, spider, group_path, status, duration, http_code)

        # 2) crawl_status：更新最终状态
        self._update_crawl_status_finish(url, spider, status, duration, http_code, state_msg, file_path)

        # 3) 内存事件 + round context
        self._finalize_event_in_memory(log_id, url, spider, group_path, status, duration, state_msg)

    def _handle_task_memory_only_finish(
            self, log_id, url, spider, group_path,
            status: Status, duration: float, state_msg: str,
            prev_exists: bool, prev_snapshot: dict,
            http_code: int = None,
            file_path: str = None
    ):
        # 1) crawl_log：保留记录（更新为 IGNORED/CACHED）
        self._update_crawl_log_finish(log_id, url, spider, group_path, status, duration, http_code)

        # 2) crawl_status：回滚到 session 前（不更新时间、不改最终态）
        try:
            self._rollback_crawl_status(url, prev_exists, prev_snapshot)
        except Exception as e:
            logger.error(f"Memory-only rollback failed for {url}: {e}")

        # 3) 内存事件 + round context
        self._finalize_event_in_memory(log_id, url, spider, group_path, status, duration, state_msg)

    def reset_statistics(self):
        """
        Resets the session view.
        Clears both the timeline buffer and the active lookup map.
        Active tasks will continue to run, but their 'Finish' updates
        will be ignored by memory stats (since they are removed from map).
        """
        logger.info("Session statistics reset by user.")
        with self.stats_lock:
            self.session_start_time = time.time()
            self.event_buffer.clear()
            self.active_events_map.clear()

    # --- 5. Flow Control & Signals ---

    def pause(self):
        self._set_signal(ControlSignal.PAUSE.value)

    def resume(self):
        self._set_signal(ControlSignal.NORMAL.value)

    def trigger_immediate(self):
        self._set_signal(ControlSignal.IMMEDIATE.value)

    def _set_signal(self, signal_str: str):
        with self._signal_lock:
            self._control_signal = signal_str
            # Persist to DB for consistency across restarts
            self.db.set_control_signal(signal_str)
            logger.info(f"Control signal set to: {signal_str}")

    def wait_interval(
            self,
            seconds: float,
            group_path: Union[str, List[str], None] = None,
            stop_event: threading.Event = None):
        """
        Smart sleep. Reads memory signal (fast) for Pause/Immediate.
        """
        if seconds <= 0: return

        # === 如果有 group_path，自动计算并更新倒计时 ===
        if group_path:
            norm_path = _normalize_group_path(group_path)
            with self.stats_lock:
                # 更新 context 里的 next_run_ts
                if norm_path in self.round_contexts:
                    # 设定预期唤醒时间
                    target_ts = time.time() + seconds
                    self.round_contexts[norm_path].next_run_ts = target_ts
                    logger.info(
                        f"[{norm_path}] Sleeping for {seconds}s. Next run at {datetime.datetime.fromtimestamp(target_ts)}")

        # === 睡眠逻辑 ===

        end_time = time.time() + seconds
        while time.time() < end_time:
            # 1. Stop Event (Highest Priority - Immediate Exit)
            if stop_event and stop_event.is_set():
                break

            # 2. Control Signal (Memory Check)
            current_signal = self._control_signal

            if current_signal == "PAUSE":
                time.sleep(1)
                # While paused, we effectively extend the wait indefinitely
                # until resumed. We do not break the loop.
                continue

            elif current_signal == "IMMEDIATE":
                self._set_signal("NORMAL")  # Consume signal
                logger.info("Immediate execution triggered.")
                break

            # 3. Sleep Chunk
            remaining = end_time - time.time()
            time.sleep(min(0.1, remaining))
