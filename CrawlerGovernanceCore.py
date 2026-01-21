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
        self._finalize(Status.CACHED)

    def ignore(self):
        self._finished = True

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

        self.known_anchors = set()

        logger.info(f"Governance Manager initialized. Signal: {self._control_signal}")

    # --- Helper: Path Normalization & Name Extraction ---

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

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
            INSERT INTO crawl_status (url, url_hash, group_path, spider_name, status, last_run_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(url) DO UPDATE SET
                status = ?, spider_name = ?, last_run_at = CURRENT_TIMESTAMP
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

    def _handle_task_finish(self, log_id, url, spider, group_path, status, duration, http_code, state_msg, file_path):
        """
        Called when transaction ends.
        1. Update DB.
        2. Retrieve event from Memory Map and perform In-Place Update.
        3. Clean up Map to prevent leaks.
        """
        # 1. DB Update
        if log_id:
            self.db.execute("UPDATE crawl_log SET status=?, duration=?, http_code=? WHERE id=?",
                            (status, duration, http_code, log_id))
        else:
            self.db.execute(
                "INSERT INTO crawl_log (url, group_path, spider_name, status, http_code, duration) VALUES (?,?,?,?,?,?)",
                (url, group_path, spider, status, http_code, duration))

        # 2. Update crawl_status...
        retry_inc = 1 if status == Status.TEMP_FAIL else 0
        retry_reset = "retry_count = 0," if status != Status.TEMP_FAIL else ""
        self.db.execute(
            f"UPDATE crawl_status SET status=?, duration=?, http_code=?, state_msg=?, file_path=?, spider_name=?, {retry_reset} retry_count = retry_count + ? WHERE url=?",
            (status, duration, http_code, state_msg, file_path, spider, retry_inc, url))

        # 3. Memory Update (In-Place)
        with self.stats_lock:
            if log_id and log_id not in self.active_events_map:
                # 明明有 ID，但 Map 里找不到，导致变成了僵尸
                logger.warning(f"Orphaned Finish Task: {url} (ID: {log_id}). Start event not found in active map.")

            if log_id in self.active_events_map:
                event_obj = self.active_events_map[log_id]
                event_obj['status'] = int(status)
                event_obj['duration'] = duration
                event_obj['state_msg'] = state_msg  # Optional: add error msg for UI

                # Cleanup reference from active map
                del self.active_events_map[log_id]
            else:
                # Edge case: Task started before reset, finished after reset.
                # Or stateless report. Add to buffer now.
                event_obj = {
                    'ts': time.time(),
                    'url': url,
                    'spider': spider,
                    'group_path': group_path,
                    'status': int(status),
                    'duration': duration,
                    'state_msg': state_msg,
                    'is_anchor': url in self.known_anchors
                }
                self.event_buffer.append(event_obj)

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

    def wait_interval(self, seconds: float, stop_event: threading.Event = None):
        """
        Smart sleep. Reads memory signal (fast) for Pause/Immediate.
        """
        if seconds <= 0: return

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
        Helper: Aggregates statistics for all groups.
        UPDATED: Calculates BOTH Traffic (Requests) and Results (Unique URLs).

        Returns structure per group:
        {
            'results': {'total': 0, 'success': 0, 'failed': 0, 'running': 0},
            'traffic': {'total': 0, 'success': 0, 'failed': 0, 'running': 0},
            'perf': {'min': 0, 'max': 0, 'avg': 0, 'sum': 0, 'count': 0}
        }
        """
        use_db = False
        target_ts = since_time.timestamp() if since_time else 0
        events_source = []

        # 1. Determine Source (Memory vs DB)
        with self.stats_lock:
            if not since_time:
                # Case A: Default View -> Memory
                events_source = list(self.event_buffer)
            elif len(self.event_buffer) == 0:
                # Case B: Buffer Empty -> DB
                use_db = True
            else:
                # Case C: Check if time is within buffer
                if target_ts >= self.event_buffer[0]['ts']:
                    events_source = [e for e in self.event_buffer if e['ts'] > target_ts]
                else:
                    use_db = True

        stats_map = {}

        # Helper to initialize the data structure
        def init_stats():
            return {
                # [A] Result Stats (Snapshot/Unique) - For Tree View
                'results': {'total': 0, 'success': 0, 'failed': 0, 'running': 0},

                # [B] Traffic Stats (Throughput/Log) - For Header & Details
                'traffic': {'total': 0, 'success': 0, 'failed': 0, 'running': 0},

                # Performance Metrics
                'perf': {'min': 999999, 'max': 0, 'sum': 0, 'count': 0, 'avg': 0},

                # Internal helper for unique tracking (URL -> Last Status)
                '_unique_urls': {}
            }

        if not use_db:
            # --- Path A: In-Memory Aggregation ---
            for e in events_source:
                gp = e['group_path']
                st = e['status']
                url = e.get('url')
                dur = e.get('duration', 0) or 0

                if gp not in stats_map:
                    stats_map[gp] = init_stats()

                s_dict = stats_map[gp]

                # 1. Update Traffic (Log Count)
                t = s_dict['traffic']

                if st == Status.RUNNING:
                    t['running'] += 1
                elif st in [Status.PENDING, Status.SKIPPED]:
                    # Not count in total
                    pass
                else:
                    # Exclude 'running' state from 'total'
                    t['total'] += 1
                    if st in [Status.SUCCESS, Status.CACHED]:
                        t['success'] += 1
                    elif st in [Status.TEMP_FAIL, Status.PERM_FAIL, Status.STOPPED]:
                        t['failed'] += 1

                # 2. Update Performance (Finished tasks only)
                if st not in [Status.PENDING, Status.RUNNING, Status.SKIPPED] and dur > 0:
                    p = s_dict['perf']
                    if dur < p['min']: p['min'] = dur
                    if dur > p['max']: p['max'] = dur
                    p['sum'] += dur
                    p['count'] += 1

                # 3. Track Unique State (Last Write Wins)
                # Since events are ordered by time, the last one we see for a URL
                # is its "Latest State" in this window.
                if url:
                    s_dict['_unique_urls'][url] = st

            # 4. Finalize Unique Results & Averages
            for gp, data in stats_map.items():
                # Process Unique URLs into Results counts
                r = data['results']
                for u_st in data['_unique_urls'].values():
                    if u_st == Status.RUNNING:
                        r['running'] += 1
                    elif st in [Status.PENDING, Status.SKIPPED]:
                        # Not count in total
                        pass
                    else:
                        # Exclude 'running' state from 'total'
                        r['total'] += 1
                        if u_st in [Status.SUCCESS, Status.CACHED]:
                            r['success'] += 1
                        elif u_st in [Status.TEMP_FAIL, Status.PERM_FAIL, Status.STOPPED]:
                            r['failed'] += 1

                # Cleanup internal memory
                del data['_unique_urls']

                # Finalize Perf Avg
                p = data['perf']
                if p['count'] > 0:
                    p['avg'] = round(p['sum'] / p['count'], 3)
                    if p['min'] == 999999: p['min'] = 0
                else:
                    p['min'] = 0

        else:
            # --- Path B: Database Aggregation ---
            # NOTE: For DB queries, calculating strictly "Unique" results over a time range
            # is expensive (requires subqueries/window functions).
            # We fallback 'results' to match 'traffic' or use basic counts.

            rows = self.db.fetch_all("""
                SELECT 
                    group_path, 
                    status, 
                    COUNT(*) as cnt,
                    MIN(duration) as min_dur,
                    MAX(duration) as max_dur,
                    SUM(duration) as sum_dur,
                    COUNT(CASE WHEN duration > 0 THEN 1 END) as dur_cnt
                FROM crawl_log
                WHERE created_at > ?
                GROUP BY group_path, status
            """, (since_time,))

            for r in rows:
                gp = r['group_path']
                st = r['status']
                count = r['cnt']

                if gp not in stats_map:
                    stats_map[gp] = init_stats()

                s_dict = stats_map[gp]

                # Update Traffic
                t = s_dict['traffic']
                t['total'] += count

                if st == Status.RUNNING:
                    t['running'] += count
                elif st in [Status.SUCCESS, Status.CACHED]:
                    t['success'] += count
                elif st in [Status.TEMP_FAIL, Status.PERM_FAIL, Status.STOPPED]:
                    t['failed'] += count

                # Update Performance
                if st not in [Status.PENDING, Status.RUNNING, Status.SKIPPED]:
                    p = s_dict['perf']
                    row_min = r['min_dur'] or 0
                    row_max = r['max_dur'] or 0
                    row_sum = r['sum_dur'] or 0
                    row_cnt = r['dur_cnt'] or 0

                    if row_cnt > 0:
                        if row_min < p['min'] and row_min > 0: p['min'] = row_min
                        if row_max > p['max']: p['max'] = row_max
                        p['sum'] += row_sum
                        p['count'] += row_cnt

            # Finalize DB Stats
            for gp, data in stats_map.items():
                # Fallback: In DB mode, Results ~= Traffic
                data['results'] = data['traffic'].copy()

                p = data['perf']
                if p['count'] > 0:
                    p['avg'] = round(p['sum'] / p['count'], 3)
                    if p['min'] == 999999: p['min'] = 0
                else:
                    p['min'] = 0

                # Cleanup unused
                del data['_unique_urls']

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

            result.append({
                'group_path': g_path,
                'name': g['name'],
                'stats': current_stats,
                'list_url_status': l_status
            })

        return result

    # def get_memory_chart_data(self, bucket_minutes: int = 1) -> List[Dict]:
    #     """
    #     NEW: Aggregates memory buffer events into time buckets for the frontend chart.
    #     Returns a time-series: [{time: '10:00', valid: 10, fail: 2}, ...]
    #     FIXED: Now correctly uses bucket_minutes to group events.
    #     """
    #     timeline = {}
    #
    #     # Ensure bucket_minutes is at least 1 to avoid division by zero
    #     bucket_minutes = max(1, bucket_minutes)
    #
    #     with self.stats_lock:
    #         for e in self.event_buffer:
    #             # 1. Convert timestamp to datetime
    #             dt = datetime.datetime.fromtimestamp(e['ts'])
    #
    #             # 2. Round down to the nearest bucket
    #             # Example: If bucket=5 and time is 10:07, discard=2, result=10:05
    #             discard = dt.minute % bucket_minutes
    #             dt_floored = dt - datetime.timedelta(minutes=discard, seconds=dt.second, microseconds=dt.microsecond)
    #
    #             # 3. Generate Key (HH:MM)
    #             time_key = dt_floored.strftime("%H:%M")
    #
    #             if time_key not in timeline:
    #                 # 'ts' is used for sorting later (store the timestamp of the bucket start)
    #                 timeline[time_key] = {'time': time_key, 'ts': dt_floored.timestamp(), 'valid': 0, 'fail': 0}
    #
    #             st = e['status']
    #
    #             # Logic: "Valid vs Failed" (Excluding Skipped/Pending/Running)
    #             if st in [Status.SUCCESS, Status.CACHED]:
    #                 timeline[time_key]['valid'] += 1
    #             elif st in [Status.TEMP_FAIL, Status.PERM_FAIL, Status.STOPPED]:
    #                 timeline[time_key]['fail'] += 1
    #
    #     # Convert dict to sorted list (Sort by timestamp to ensure correct time order across hours/days)
    #     sorted_data = sorted(timeline.values(), key=lambda x: x['ts'])
    #
    #     # Remove the internal 'ts' helper field before returning to frontend
    #     for item in sorted_data:
    #         del item['ts']
    #
    #     return sorted_data

    def get_log_trend_stats(self, start_ts: float, end_ts: float, bucket_minutes: int = 1, group_filter: str = None) -> List[Dict]:
        """
        Generates Trend Chart Data from DB Logs.
        """
        # Ensure bucket is valid
        bucket_seconds = max(1, bucket_minutes) * 60

        # SQL Logic:
        # P1, P2: used for bucket calculation
        # P3, P4: used for time range filtering

        # We need to fill in the SELECT part explicitly to match the params order
        sql = """
            SELECT 
                (CAST(strftime('%s', created_at) AS INTEGER) / ?) * ? as bucket_ts,
                status,
                COUNT(*) as cnt
            FROM crawl_log
            WHERE created_at BETWEEN datetime(?, 'unixepoch') AND datetime(?, 'unixepoch')
        """

        # Initial params matching the 4 placeholders above
        params = [bucket_seconds, bucket_seconds, start_ts, end_ts]

        # Dynamic Filter
        if group_filter:
            sql += " AND group_path = ?"
            params.append(group_filter)

        # Grouping and Ordering
        sql += """
            GROUP BY bucket_ts, status
            ORDER BY bucket_ts ASC
        """

        # Correctly pass the dynamic params list
        rows = self.db.fetch_all(sql, tuple(params))

        # Process raw rows into structured timeline
        timeline = {}

        for r in rows:
            ts = r['bucket_ts']
            if not ts: continue  # Skip invalid dates

            st = r['status']
            cnt = r['cnt']

            if ts not in timeline:
                # Initialize bucket
                time_str = datetime.datetime.fromtimestamp(ts).strftime("%H:%M")
                timeline[ts] = {
                    'ts': ts,
                    'time': time_str,
                    'valid': 0,
                    'fail': 0,
                    'total': 0
                }

            bucket = timeline[ts]
            bucket['total'] += cnt

            if st in [Status.SUCCESS, Status.CACHED]:
                bucket['valid'] += cnt
            elif st in [Status.TEMP_FAIL, Status.PERM_FAIL, Status.STOPPED]:
                bucket['fail'] += cnt

        # Convert to sorted list
        return sorted(timeline.values(), key=lambda x: x['ts'])

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
