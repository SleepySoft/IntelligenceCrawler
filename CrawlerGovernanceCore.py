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
from typing import Optional, Union, List, Dict

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
        self._ensure_column("crawl_status", "updated_at TIMESTAMP")

    def _init_schema(self):
        with self.lock:
            cur = self.conn.cursor()

            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")

            # 1. Task Groups (Metadata Registry)
            # Used for UI aggregation. linking a group to a specific entry URL (list_url).
            # 'list_url' serves as a logical foreign key to crawl_status.url
            cur.execute("""
                CREATE TABLE IF NOT EXISTS task_groups (
                    group_path TEXT PRIMARY KEY,
                    list_url TEXT, 
                    name TEXT,
                    config_json TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 2. Crawl Status (The "Dashboard" - Current State)
            # Stores the LATEST known state of a URL.
            # 'spider_name' is stored for fast filtering/stats, derived from group_path.
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
                    last_run_at TIMESTAMP,
                    next_run_at TIMESTAMP, 
                    duration REAL,
                    state_msg TEXT
                )
            """)

            # 3. Crawl Log (The "Flow" - History)
            # Records every attempt. Linked to Session via 'id'.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS crawl_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    group_path TEXT NOT NULL,
                    spider_name TEXT NOT NULL,
                    status INTEGER,
                    http_code INTEGER,
                    duration REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 4. System Control (For persistent signaling)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sys_control (
                    key TEXT PRIMARY KEY,
                    signal TEXT DEFAULT 'NORMAL',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Initialize global control signal if not present
            cur.execute("INSERT OR IGNORE INTO sys_control (key, signal) VALUES ('global', 'NORMAL')")

            # Indexes for performance
            cur.execute("CREATE INDEX IF NOT EXISTS idx_status_group ON crawl_status(group_path)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_status_spider ON crawl_status(spider_name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_log_url ON crawl_log(url)")

            self.conn.commit()

    def _ensure_column(self, table: str, column_def: str):
        # column_def 例如 "updated_at TIMESTAMP"
        try:
            self.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")
            logger.info(f"Added column {column_def} to {table}")
        except sqlite3.OperationalError as e:
            # 重复添加会报 duplicate column name，直接忽略
            if "duplicate column name" in str(e).lower():
                return
            raise

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

    def get_control_signal(self, key='global') -> str:
        row = self.fetch_one("SELECT signal FROM sys_control WHERE key = ?", (key,))
        return row['signal'] if row else "NORMAL"

    def set_control_signal(self, signal: str, key='global'):
        self.execute("INSERT OR REPLACE INTO sys_control (key, signal, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                     (key, signal))


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
    Manages the lifecycle of a single URL crawl.
    1. Start: Updates DB to RUNNING, creates a Log entry (gets ID).
    2. Execution: Allows saving files and marking intermediate states.
    3. End: Updates Log entry (by ID) and Status table (by URL).
    """

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

        # Unique ID for the specific log entry of this session
        self.log_id: Optional[int] = None

        self.status = Status.RUNNING
        self.http_code = None
        self.state_msg = None
        self.file_path = None
        self._finished = False

    def __enter__(self):
        # Notify manager to start transaction (Insert Log, Update Status)
        self.log_id = self.manager._handle_task_start(self.url, self.spider, self.group_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._finished:
            if exc_type:
                # Handle unhandled exceptions/crashes
                self.fail_perm(http_code=500, state_msg=f"Exception: {str(exc_val)}")
                logger.error(f"Session crashed for {self.url}: {exc_val}")
            else:
                # Handle context exit without explicit status
                self.fail_temp(state_msg="Exited without explicit status")

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
            next_run = datetime.datetime.now() + datetime.timedelta(seconds=interval_seconds)
            self.manager.db.execute(
                "UPDATE crawl_status SET next_run_at = ? WHERE url = ?",
                (next_run, self.url)
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
    Central Controller for Spider Governance.
    Manages State, Storage, and Flow Control.

    Refactored Logic:
    - Uses a Ring Buffer (deque) to cache the latest N task events in memory.
    - Queries use Memory if the time range is covered by the buffer.
    - Falls back to DB if the time range exceeds memory history.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH, files_path: str = DEFAULT_FILES_PATH):
        self.db = DatabaseHandler(db_path)
        self.storage = StorageHandler(files_path)

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

        logger.info(f"Governance Manager initialized. Signal: {self._control_signal}")

    # --- Helper: Path Normalization & Name Extraction ---

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()


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
            self.db.execute("""
                        INSERT INTO task_groups (group_path, list_url, name)
                        VALUES (?, ?, ?)
                        ON CONFLICT(group_path) DO UPDATE SET
                        list_url = excluded.list_url,
                        name = excluded.name
                    """, (norm_group_path, list_url, friendly_name))

            # 2. Update Runtime Registry (UI Visibility) <--- KEY CHANGE
            with self.stats_lock:
                self.runtime_groups[norm_group_path] = {
                    'list_url': list_url,
                    'name': friendly_name,
                    'spider': spider_name
                }

                if list_url:
                    self.known_anchors.add(list_url)

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
        # 我们通过 LEFT JOIN 检查这个 URL 是否是某个组的 list_url
        # 结果集多了一列 is_seed (1 or 0)
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

        # 1. New URL (Never seen) -> Crawl it
        if not row:
            return True

        status = row['status']
        retry_count = row['retry_count']
        next_run_at = row['next_run_at']
        is_seed = bool(row['is_seed'])

        # 2. Running State: Always protect against concurrency
        if status == Status.RUNNING:
            return False

        # 3. Schedule Check (Time-based Priority)
        # 无论是列表还是文章，只要设定了 next_run_at，就必须遵循时间调度
        next_run_at = False # Temporary remove this logic
        if next_run_at:
            if isinstance(next_run_at, str):
                # Handle varying SQLite timestamp formats
                try:
                    target_ts = datetime.datetime.fromisoformat(next_run_at)
                except ValueError:
                    # Fallback for simple space-separated DB timestamps if any
                    target_ts = datetime.datetime.strptime(next_run_at, "%Y-%m-%d %H:%M:%S.%f")
            else:
                target_ts = next_run_at

            now = datetime.datetime.now()

            # 如果时间没到，坚决不抓
            if now < target_ts:
                return False

            # 如果时间到了，允许抓取 (return True)
            # 注意：这里我们隐式允许了即便 status=SUCCESS 也可以抓，只要时间到了
            return True

        # 4. Logic for "Seed/List" URLs (Recurrent)
        # 如果它是种子，且没有设定 next_run_at (可能是初次运行或逻辑疏忽)
        # 我们不能因为它 SUCCESS 了就停止抓取。
        if is_seed:
            # 种子页只有在 "RUNNING" 时才不抓 (上面已处理)
            # 其他状态 (SUCCESS, FAIL) 都应该允许重试或下一轮
            # 但为了防止死循环狂抓，建议业务逻辑必须设置 next_run_at。
            # 这里作为兜底，允许抓取。
            return True

        # 5. Logic for "Article/One-off" URLs
        # 普通文章，一旦成功，就永久停止
        if status == Status.SUCCESS:
            return False

            # Dead End States
        if status in [Status.PERM_FAIL, Status.SKIPPED, Status.STOPPED]:
            return False

        # Retry Logic for Temp Fails
        if status == Status.TEMP_FAIL:
            if retry_count < max_retries:
                return True
            else:
                return False

        # Default (e.g. PENDING)
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

    # --- 4. Internal State Management (Called by Session) ---

    def _handle_task_start(self, url: str, spider: str, group: str) -> int:
        """
        Called when transaction starts.
        Updated: Now stores 'url' and 'spider' in memory for rich monitoring.
        """
        # 1. DB Insert (Log)
        log_id = self.db.execute("""
                    INSERT INTO crawl_log (url, group_path, spider_name, status, created_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (url, group, spider, Status.RUNNING))

        if log_id is None: log_id = 0

        # 2. DB Upsert (Status)
        self.db.execute("""
            INSERT INTO crawl_status (url, url_hash, group_path, spider_name, status, last_run_at, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(url) DO UPDATE SET
                status = ?, spider_name = ?, last_run_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
        """, (url, self._hash(url), group, spider, Status.RUNNING, Status.RUNNING, spider))

        # 3. Memory Update (Rich Object)
        with self.stats_lock:
            event_obj = {
                'id': log_id,           # Track ID
                'ts': time.time(),
                'url': url,             # <--- Added for display
                'spider': spider,       # <--- Added for filtering
                'group_path': group,
                'status': int(Status.RUNNING),
                'duration': 0.0,  # Placeholder
                'is_anchor': url in self.known_anchors
            }

            self.event_buffer.append(event_obj)
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
            self.db.execute(
                "INSERT INTO crawl_log (url, group_path, spider_name, status, http_code, duration) VALUES (?,?,?,?,?,?)",
                (url, group_path, spider, int(status), http_code, duration)
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
                del self.active_events_map[log_id]
            else:
                self.event_buffer.append({
                    "ts": time.time(),
                    "url": url,
                    "spider": spider,
                    "group_path": group_path,
                    "status": int(status),
                    "duration": duration,
                    "state_msg": state_msg,
                    "is_anchor": url in self.known_anchors
                })

            if group_path in self.round_contexts:
                self.round_contexts[group_path].update(status)

    def _update_crawl_status_finish(self, url, spider, status, duration, http_code, state_msg, file_path):
        retry_inc = 1 if status == Status.TEMP_FAIL else 0
        retry_reset = "retry_count = 0," if status != Status.TEMP_FAIL else ""
        self.db.execute(
            f"""
            UPDATE crawl_status 
            SET status=?, duration=?, http_code=?, state_msg=?, file_path=?, spider_name=?,
                updated_at=CURRENT_TIMESTAMP,
                {retry_reset} retry_count = retry_count + ?
            WHERE url=?
            """,
            (int(status), duration, http_code, state_msg, file_path, spider, retry_inc, url)
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

    # --- 6. Dashboard Statistics (Unified Logic) ---

    def _get_aggregated_stats(self, since_time: Optional[datetime.datetime]) -> Dict[str, Dict]:
        """
        聚合统计核心函数。
        Refactored: 提取了公共计数逻辑。
        """

        # --- 1. 内部 Helper：核心计数逻辑 (DRY Principle) ---
        def accumulate_counts(target_dict: Dict, status: int, count: int = 1):
            """
            统一处理 Traffic 和 Results 的计数规则。
            在此处修改规则，所有统计视图都会自动生效。
            """
            if status == Status.RUNNING:
                target_dict['running'] += count

            # === 核心过滤规则 ===
            elif status in [Status.PENDING, Status.SKIPPED, Status.IGNORED, Status.CACHED]:
                return


            else:
                # 只有明确的 Success 或 Fail 才计入 Total
                target_dict['total'] += count

                if status in [Status.SUCCESS]:
                    target_dict['success'] += count
                elif status in [Status.TEMP_FAIL, Status.PERM_FAIL, Status.STOPPED]:
                    target_dict['failed'] += count

        # --- 2. 数据结构初始化 ---
        stats_map = {}

        def get_group_stats(group_path):
            if group_path not in stats_map:
                stats_map[group_path] = {
                    'results': {'total': 0, 'success': 0, 'failed': 0, 'running': 0},
                    'traffic': {'total': 0, 'success': 0, 'failed': 0, 'running': 0},
                    'perf': {'min': 999999, 'max': 0, 'sum': 0, 'count': 0, 'avg': 0},
                    '_unique_urls': {}  # 仅内存模式使用
                }
            return stats_map[group_path]

        # --- 3. 确定数据源 ---
        use_db = False
        target_ts = since_time.timestamp() if since_time else 0
        events_source = []

        with self.stats_lock:
            if not since_time:
                events_source = list(self.event_buffer)
            elif len(self.event_buffer) == 0:
                use_db = True
            else:
                if target_ts >= self.event_buffer[0]['ts']:
                    events_source = [e for e in self.event_buffer if e['ts'] > target_ts]
                else:
                    use_db = True

        # --- 4. 执行聚合 ---

        if not use_db:
            # === Path A: In-Memory (Live View) ===
            for e in events_source:
                gp = e['group_path']
                st = e['status']
                dur = e.get('duration', 0) or 0

                s_dict = get_group_stats(gp)

                # A1. Traffic 统计 (流水账)
                accumulate_counts(s_dict['traffic'], st, count=1)

                # A2. 性能统计 (仅针对有效耗时)
                # 排除 Pending/Running/Skipped/Ignored 对平均耗时的影响
                if st not in [Status.PENDING, Status.RUNNING, Status.SKIPPED, Status.IGNORED] and dur > 0:
                    p = s_dict['perf']
                    if dur < p['min']: p['min'] = dur
                    if dur > p['max']: p['max'] = dur
                    p['sum'] += dur
                    p['count'] += 1

                # A3. 记录 Unique 状态 (覆盖写)
                if e.get('url'):
                    s_dict['_unique_urls'][e['url']] = st

            # A4. Finalize Unique Results (修正后的逻辑)
            for gp, data in stats_map.items():
                r = data['results']
                # 遍历去重后的 URL 字典
                for u_st in data['_unique_urls'].values():
                    # 调用同一个 helper，确保逻辑一致！
                    accumulate_counts(r, u_st, count=1)

                # 清理临时内存
                del data['_unique_urls']

        else:
            # === Path B: Database (Historical View) ===
            rows = self.db.fetch_all("""
                SELECT 
                    group_path, status, COUNT(*) as cnt,
                    MIN(duration) as min_dur, MAX(duration) as max_dur,
                    SUM(duration) as sum_dur, COUNT(CASE WHEN duration > 0 THEN 1 END) as dur_cnt
                FROM crawl_log
                WHERE created_at > ?
                GROUP BY group_path, status
            """, (since_time,))

            for r in rows:
                gp = r['group_path']
                st = r['status']
                count = r['cnt']

                s_dict = get_group_stats(gp)

                # B1. Traffic 统计
                accumulate_counts(s_dict['traffic'], st, count=count)

                # B2. 性能统计
                if st not in [Status.PENDING, Status.RUNNING, Status.SKIPPED, Status.IGNORED]:
                    p = s_dict['perf']
                    row_min = r['min_dur'] or 0
                    row_max = r['max_dur'] or 0
                    row_cnt = r['dur_cnt'] or 0

                    if row_cnt > 0:
                        if row_min < p['min'] and row_min > 0: p['min'] = row_min
                        if row_max > p['max']: p['max'] = row_max
                        p['sum'] += (r['sum_dur'] or 0)
                        p['count'] += row_cnt

            # B3. DB 模式下的 Results 妥协 (Results = Traffic)
            for data in stats_map.values():
                data['results'] = data['traffic'].copy()
                if '_unique_urls' in data: del data['_unique_urls']

        # --- 5. 计算平均值 (收尾) ---
        for data in stats_map.values():
            p = data['perf']
            if p['count'] > 0:
                p['avg'] = round(p['sum'] / p['count'], 3)
                if p['min'] == 999999: p['min'] = 0
            else:
                p['min'] = 0

        return stats_map

    def get_dashboard_summary(self, spider_filter: str = None, since_time: datetime.datetime = None) -> List[Dict]:
        """
        Constructs the dashboard view.
        Source of 'groups' is now self.runtime_groups (Memory), not DB.
        """

        # 1. Fetch Structure (FROM MEMORY) <--- CHANGED
        # Only groups explicitly registered in this session appear.
        with self.stats_lock:
            # Sort by path for consistent UI
            sorted_paths = sorted(self.runtime_groups.keys())
            active_groups = [
                {'group_path': p, **self.runtime_groups[p]}
                for p in sorted_paths
            ]

        # If no groups registered yet, return empty list immediately
        if not active_groups:
            return []

        # 2. Get Dynamic Statistics
        # This returns stats for ALL groups found in logs/buffer.
        # We will map them to our active_groups.
        stats_map = self._get_aggregated_stats(since_time)

        # 3. Fetch Anchor/List URL Status
        list_urls = [g['list_url'] for g in active_groups if g['list_url']]
        list_url_map = {}

        if list_urls:
            placeholders = ','.join(['?'] * len(list_urls))
            params = list(list_urls)

            sql = """
                SELECT url, status, last_run_at, next_run_at, http_code, state_msg 
                FROM crawl_status 
                WHERE url IN ({})
            """

            if since_time:
                sql += " AND last_run_at > ?"
                params.append(since_time)

            url_rows = self.db.fetch_all(sql.format(placeholders), tuple(params))
            list_url_map = {row['url']: dict(row) for row in url_rows}

        # 4. Assemble Final Result
        result = []
        for g in active_groups:
            g_path = g['group_path']

            if spider_filter and not g_path.startswith(spider_filter):
                continue

            # Retrieve stats
            # Note: Even if stats_map has data for old groups, we only grab the ones for active_groups
            default_stats = {
                'results': {'total': 0, 'success': 0, 'failed': 0, 'running': 0},
                'traffic': {'total': 0, 'success': 0, 'failed': 0, 'running': 0},
                'perf': {'min': 0, 'max': 0, 'avg': 0, 'count': 0}
            }
            current_stats = stats_map.get(g_path, default_stats)

            l_url = g['list_url']
            l_status = list_url_map.get(l_url)

            current_round_id = 0
            if g_path in self.round_contexts:
                current_round_id = self.round_contexts[g_path].round_id

            result.append({
                'group_path': g_path,
                'name': g['name'],
                'stats': current_stats,
                'list_url_status': l_status,
                'round_id': current_round_id
            })

        return result

    def get_log_trend_stats(
            self,
            start_ts: float,
            end_ts: float,
            bucket_minutes: int = 60,
            group_filter: str = None,
            use_updated_at: bool = True,
            include_cached_as_success: bool = False
    ) -> List[Dict]:
        """
        Trend stats from crawl_status (Outcome-oriented, unique URL snapshot)。
        Bucket by crawl_status.updated_at (default) or last_run_at。
        Excludes group list_url anchors (task_groups.list_url)。
        Returns buckets with success/fail/total for stacked bar chart.
        """

        bucket_seconds = max(1, bucket_minutes) * 60
        time_col = "updated_at" if use_updated_at else "last_run_at"

        # 注意：crawl_status 是快照表，一 URL 一行，所以统计是“唯一 URL 在该时间段发生更新”的结果分布
        sql = f"""
            SELECT 
                (CAST(strftime('%s', s.{time_col}) AS INTEGER) / ?) * ? as bucket_ts,
                s.status as status,
                COUNT(*) as cnt
            FROM crawl_status s
            LEFT JOIN task_groups g ON s.url = g.list_url
            WHERE s.{time_col} IS NOT NULL
              AND s.{time_col} BETWEEN datetime(?, 'unixepoch') AND datetime(?, 'unixepoch')
              AND g.list_url IS NULL
        """

        params = [bucket_seconds, bucket_seconds, start_ts, end_ts]

        if group_filter:
            sql += " AND s.group_path = ?"
            params.append(group_filter)

        sql += """
            GROUP BY bucket_ts, status
            ORDER BY bucket_ts ASC
        """

        rows = self.db.fetch_all(sql, tuple(params))

        # 1) 先准备完整桶，补齐空桶
        def floor_bucket(ts: float) -> int:
            return int(ts // bucket_seconds) * bucket_seconds

        start_bucket = floor_bucket(start_ts)
        end_bucket = floor_bucket(end_ts)
        timeline = {}

        # 生成所有桶
        cur = start_bucket
        while cur <= end_bucket:
            # label 你可按 bucket 粒度自定义显示
            dt = datetime.datetime.fromtimestamp(cur)
            if bucket_seconds >= 86400:
                label = dt.strftime("%Y-%m-%d")
            elif bucket_seconds >= 3600:
                label = dt.strftime("%m-%d %H:00")
            else:
                label = dt.strftime("%H:%M")

            timeline[cur] = {"ts": cur, "time": label, "success": 0, "fail": 0, "total": 0}
            cur += bucket_seconds

        # 2) 回填统计
        success_set = {Status.SUCCESS}
        if include_cached_as_success:
            success_set.add(Status.CACHED)

        fail_set = {Status.TEMP_FAIL, Status.PERM_FAIL, Status.STOPPED}

        for r in rows:
            ts = r["bucket_ts"]
            st = r["status"]
            cnt = r["cnt"]

            if ts not in timeline:
                # 极端情况下（边界/时区）保护一下
                timeline[ts] = {"ts": ts, "time": datetime.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M"),
                                "success": 0, "fail": 0, "total": 0}

            bucket = timeline[ts]

            # 只统计成功/失败进 total，其他状态忽略
            if st in success_set:
                bucket["success"] += cnt
                bucket["total"] += cnt
            elif st in fail_set:
                bucket["fail"] += cnt
                bucket["total"] += cnt
            else:
                # PENDING/RUNNING/SKIPPED/IGNORED 等不进入柱图
                pass

        return [timeline[k] for k in sorted(timeline.keys())]

    def get_session_stats(self, since_time: datetime.datetime = None):
        """
        Returns global statistics.
        UPDATED: Aggregates 'traffic' stats from the nested structure.
        """
        # Get the nested stats map: { group: { 'traffic': {...}, 'results': {...} } }
        stats_map = self._get_aggregated_stats(since_time)

        # Flatten the grouped map into global totals (using TRAFFIC data)
        total = 0
        success = 0
        failed = 0
        running = 0

        for gp_data in stats_map.values():
            # Extract traffic dict, defaulting to empty if missing
            t = gp_data.get('traffic', {})

            total += t.get('total', 0)
            success += t.get('success', 0)
            failed += t.get('failed', 0)
            running += t.get('running', 0)

        rate = round((success / total) * 100, 1) if total > 0 else 0

        # Determine the effective start time
        if since_time:
            ref_time = since_time
        else:
            with self.stats_lock:
                ref_time = self.session_start_time

        return {
            'total': total,
            'success': success,
            'failed': failed,
            'running': running,
            'success_rate': rate,
            'session_start': ref_time
        }

    # --- 7. Data Access Interfaces for Backend (New) ---

    def get_pending_count(self) -> int:
        """Get the current depth of the queue (Persistent State)."""
        row = self.db.fetch_one("SELECT count(*) as cnt FROM crawl_status WHERE status=?", (Status.PENDING,))
        return row['cnt'] if row else 0

    def get_logs(self, spider_name=None, status=None, limit=100, since_time=None, until_time=None):
        """
        Retrieves recent logs with filters.
        Fixes 'ambiguous column name' error by specifying table alias 'l.'.
        """
        # 基础 SQL 包含别名 l (log) 和 s (status)
        sql = """
            SELECT l.*, s.url_hash 
            FROM crawl_log l
            LEFT JOIN crawl_status s ON l.url = s.url
            WHERE 1=1
        """
        params = []

        if spider_name:
            # 这里必须写 l.group_path，不能只写 group_path
            sql += " AND l.group_path LIKE ?"
            params.append(f"{spider_name}%")

        if status is not None:
            # 这里的 status 虽然通常只有 log 表有，但为了规范也建议加 l.
            sql += " AND l.status = ?"
            params.append(status)

        if since_time:
            sql += " AND l.created_at >= datetime(?, 'unixepoch')"
            params.append(since_time)

        if until_time:
            sql += " AND l.created_at <= datetime(?, 'unixepoch')"
            params.append(until_time)

        sql += " ORDER BY l.id DESC LIMIT ?"
        params.append(limit)

        rows = self.db.fetch_all(sql, tuple(params))

        # 转为字典
        return [dict(row) for row in rows]

    def get_snapshot_path(self, url_hash: str) -> Optional[str]:
        """Retrieve file path by URL hash."""
        row = self.db.fetch_one("SELECT file_path FROM crawl_status WHERE url_hash = ?", (url_hash,))
        return row['file_path'] if row else None

    def get_recent_statuses(self, limit: int = 100, spider: Optional[str] = None, status: Optional[int] = None) -> List[
        Dict]:
        """
        Fetch 'Live' statuses directly from Memory Buffer.
        No DB access. No Time parameter needed.

        Logic:
        1. Iterate Memory Buffer in Reverse (Newest first).
        2. Deduplicate by URL (Show only the LATEST state of a URL).
        3. Apply Filters.
        """
        result = []
        seen_urls = set()

        with self.stats_lock:
            # Iterate backwards to get the most recent events first
            # list(reversed(deque)) is efficient enough for typical buffer sizes (e.g. 5k)
            for event in reversed(self.event_buffer):
                if len(result) >= limit:
                    break

                url = event.get('url')

                # Deduplication: Only show the latest status for a specific URL
                if url in seen_urls:
                    continue

                # Filters
                if spider and event.get('spider') != spider:
                    continue
                if status is not None and event.get('status') != status:
                    continue

                seen_urls.add(url)

                # Create a clean copy for the view
                view_item = {
                    'url': url,
                    'status': event['status'],
                    'spider_name': event.get('spider'),
                    'group_path': event['group_path'],
                    'last_run_at': datetime.datetime.fromtimestamp(event['ts']).isoformat(),
                    # Convert timestamp to ISO for Frontend
                    'duration': event.get('duration', 0),
                    'state_msg': event.get('state_msg')
                }
                result.append(view_item)

        return result

    def get_db_history_stats(self, days: int = 7) -> Dict:
        """
        NEW: Queries DB for long-term historical statistics.
        Returns data for:
        1. Daily Bar Chart (Date vs Valid/Fail)
        2. Status Breakdown Pie Chart
        """

        # 1. Daily Stats
        # SQLite 'date' function extracts YYYY-MM-DD
        rows_daily = self.db.fetch_all("""
            SELECT 
                date(created_at) as day, 
                status, 
                COUNT(*) as cnt
            FROM crawl_log 
            WHERE created_at >= date('now', ?)
            GROUP BY day, status
            ORDER BY day ASC
        """, (f'-{days} days',))

        daily_map = {}
        for r in rows_daily:
            day = r['day']
            st = r['status']
            cnt = r['cnt']

            if day not in daily_map:
                daily_map[day] = {'date': day, 'valid': 0, 'fail': 0, 'total': 0}

            daily_map[day]['total'] += cnt

            if st in [Status.SUCCESS, Status.CACHED]:
                daily_map[day]['valid'] += cnt
            elif st in [Status.TEMP_FAIL, Status.PERM_FAIL, Status.STOPPED]:
                daily_map[day]['fail'] += cnt

        # 2. Overall DB Status Distribution (Snapshot of crawl_status table)
        rows_status = self.db.fetch_all("""
            SELECT status, COUNT(*) as cnt FROM crawl_status GROUP BY status
        """)
        status_dist = {r['status']: r['cnt'] for r in rows_status}

        return {
            'daily_trend': list(daily_map.values()),
            'current_status_dist': status_dist
        }

    def get_export_csv(self, export_type: str, group_path: str = None) -> str:
        """生成 CSV 格式的字符串"""
        import io
        import csv

        output = io.StringIO()
        writer = csv.writer(output)

        if export_type == 'global_stats':
            # 导出全局统计
            # Header
            writer.writerow(['Group', 'Spider', 'Round ID', 'Phase', 'Total Items', 'Success', 'Failed', 'Skipped',
                             'Avg Duration (s)', 'List URL Status'])

            # Data
            summary = self.get_dashboard_summary()  # 复用现有的聚合逻辑
            for item in summary:
                # 获取更详细的 round context
                ctx = self.round_contexts.get(item['group_path'])
                phase = ctx.phase if ctx else "IDLE"

                stats = item['stats']['results']  # 使用结果统计
                perf = item['stats']['perf']
                list_st = item['list_url_status']['status'] if item.get('list_url_status') else -1

                writer.writerow([
                    item['group_path'],
                    item.get('spider', ''),  # summary里可能需要补充spider字段，或者从path解析
                    item.get('round_id', 0),
                    phase,
                    stats.get('total', 0),
                    stats.get('success', 0),
                    stats.get('failed', 0),
                    stats.get('skipped', 0),  # 需要确保 dashboard summary 的 stats 里有 skipped，如果没有需从 traffic 或 ctx 取
                    perf.get('avg', 0),
                    list_st
                ])

        elif export_type == 'group_status':
            # 导出当前组的状态快照 (crawl_status)
            writer.writerow(['URL', 'Status', 'HTTP Code', 'Retry Count', 'Last Run', 'Next Run', 'Error Msg'])
            if group_path:
                rows = self.db.fetch_all(
                    "SELECT url, status, http_code, retry_count, last_run_at, next_run_at, state_msg FROM crawl_status WHERE group_path = ?",
                    (group_path,))
                for r in rows:
                    writer.writerow(
                        [r['url'], r['status'], r['http_code'], r['retry_count'], r['last_run_at'], r['next_run_at'],
                         r['state_msg']])

        elif export_type == 'group_logs':
            # 导出当前组的日志历史 (crawl_log)
            writer.writerow(['ID', 'Time', 'URL', 'Status', 'HTTP Code', 'Duration'])
            if group_path:
                rows = self.db.fetch_all(
                    "SELECT id, created_at, url, status, http_code, duration FROM crawl_log WHERE group_path = ? ORDER BY id DESC LIMIT 10000",
                    (group_path,))
                for r in rows:
                    writer.writerow([r['id'], r['created_at'], r['url'], r['status'], r['http_code'], r['duration']])

        return output.getvalue()
