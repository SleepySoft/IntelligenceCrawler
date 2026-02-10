import time
import threading
import logging
import collections
from contextlib import ContextDecorator
from typing import Dict, Optional, Deque, List
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger("FlowScheduler")


# --- 1. 数据结构定义 ---

class TaskState(Enum):
    SLEEPING = "SLEEPING"  # 正在冷却 (Interval wait)
    QUEUED = "QUEUED"  # 冷却完毕，正在排队 (Resource wait)
    RUNNING = "RUNNING"  # 正在运行 (Critical section)


@dataclass
class SchedulerEntry:
    key: str
    state: TaskState = TaskState.SLEEPING
    enter_sleep_time: float = 0.0
    wake_up_time: float = 0.0
    enter_queue_time: float = 0.0
    start_run_time: float = 0.0

    wait_reason: str = ""          # "FIFO" | "CAPACITY" | "STAGGER" | ""
    wait_until: float = 0.0        # meaningful mainly for STAGGER
    last_state_ts: float = 0.0     # state change timestamp


# --- 2. 调度器核心类 ---

class FlowScheduler:
    """
    中央流控调度器。
    实现：Sleep -> Queue(FIFO) -> Stagger Check -> Run -> Release
    """

    def __init__(self,
                 max_concurrency: int = 3,
                 startup_stagger: float = 2.0):
        """
        :param max_concurrency: 全局最大并发数
        :param startup_stagger: 启动错峰间隔 (秒)
        """
        self.max_concurrency = max_concurrency
        self.startup_stagger = startup_stagger

        # --- 状态存储 ---
        # 正在运行的任务池 {key: Entry}
        self.running_pool: Dict[str, SchedulerEntry] = {}

        # 正在排队的任务队列 (FIFO)
        self.ready_queue: Deque[SchedulerEntry] = collections.deque()

        # 正在休眠的任务集合 (仅用于监控显示) {key: Entry}
        self.sleeping_pool: Dict[str, SchedulerEntry] = {}

        # 记录 Key 的上次结束时间 {key: timestamp}
        # 如果 Key 无限多，这里需要用 LRU Cache 防止内存泄漏
        self.last_finish_times: Dict[str, float] = {}

        # 调度状态
        self.last_dispatch_time = 0.0  # 上一次允许通过的时间

        # --- 同步原语 ---
        self._lock = threading.RLock()
        # 条件变量：用于协调 队列变动、槽位释放、错峰等待
        self._condition = threading.Condition(self._lock)

    def pace(self,
             key: str,
             interval: float = 0.0,
             stop_event: Optional[threading.Event] = None) -> "FlowContext":
        """
        [工厂方法] 创建一个调度上下文。
        :param key: 任务标识符 (如 "spider/news")
        :param interval: 需要冷却/休眠的时间 (秒)
        :param stop_event: 外部中断信号，用于优雅退出休眠或排队
        """
        return FlowContext(self, key, interval, stop_event)

    # --- 内部核心逻辑 (由 Context 调用) ---

    def _request_entry(self, entry: SchedulerEntry, interval: float, stop_event: Optional[threading.Event]):
        """
        阶段 1 & 2: 执行休眠，然后进入队列排队，直到获得运行资格。
        """

        # === Phase 1: Cooldown (Smart Sleep) ===
        if interval > 0:
            last_finish = self.last_finish_times.get(entry.key, 0.0)
            now = time.time()
            time_since_last = now - last_finish

            # 核心逻辑：
            # 如果是第一次运行 (last_finish=0)，time_since_last 很大，sleep_time < 0，变为 0。
            # 如果是循环运行，time_since_last 很小，sleep_time = 剩余需要冷却的时间。
            sleep_time = interval - time_since_last

            if sleep_time > 0:
                with self._lock:
                    entry.state = TaskState.SLEEPING
                    entry.last_state_ts = time.time()
                    entry.enter_sleep_time = entry.last_state_ts
                    entry.wake_up_time = entry.enter_sleep_time + sleep_time
                    self.sleeping_pool[entry.key] = entry

                # 执行休眠
                self._interruptible_sleep(sleep_time, stop_event)

                with self._lock:
                    if entry.key in self.sleeping_pool:
                        del self.sleeping_pool[entry.key]

        # 检查是否被中断
        if stop_event and stop_event.is_set():
            raise InterruptedError(f"Task {entry.key} stopped during sleep.")

        # === Phase 2: Enqueue (Join FIFO) ===
        with self._condition:
            entry.state = TaskState.QUEUED
            entry.last_state_ts = time.time()
            entry.enter_queue_time = entry.last_state_ts
            entry.wait_reason = ""
            entry.wait_until = 0.0
            self.ready_queue.append(entry)
            # logger.debug(f"[{entry.key}] Enqueued. Position: {len(self.ready_queue)}")
            self._condition.notify_all()  # 通知可能在等待队列非空的监控线程

            # === Phase 3: Arbitration (Wait for Slot) ===
            while True:
                # 0. 中断检查
                if stop_event and stop_event.is_set():
                    # 退出前把自己从队列清理掉
                    if entry in self.ready_queue:
                        self.ready_queue.remove(entry)
                        self._condition.notify_all()  # Avoid others stuck behind head removal
                    raise InterruptedError(f"Task {entry.key} stopped during queue.")

                # 1. FIFO Check: 必须是队首
                if not self.ready_queue or self.ready_queue[0] is not entry:
                    entry.wait_reason = "FIFO"
                    entry.wait_until = 0.0
                    self._condition.wait()
                    continue

                # 2. Capacity Check: 必须有空位
                if len(self.running_pool) >= self.max_concurrency:
                    self._condition.wait()
                    entry.wait_reason = "CAPACITY"
                    entry.wait_until = 0.0
                    continue

                # 3. Stagger Check: 错峰检查
                now = time.time()
                time_since_last = now - self.last_dispatch_time
                if time_since_last < self.startup_stagger:
                    wait_time = self.startup_stagger - time_since_last
                    # logger.debug(f"[{entry.key}] Staggering for {wait_time:.2f}s...")
                    entry.wait_reason = "STAGGER"
                    entry.wait_until = now + wait_time
                    self._condition.wait(timeout=wait_time)
                    continue  # 醒来后重新检查所有条件

                # --- 获得通过资格 ---

                # 弹出队列
                self.ready_queue.popleft()

                # 加入运行池
                entry.state = TaskState.RUNNING
                entry.last_state_ts = time.time()
                entry.wait_reason = ""
                entry.wait_until = 0.0
                entry.start_run_time = entry.last_state_ts
                self.running_pool[entry.key] = entry

                # 更新全局时间
                self.last_dispatch_time = time.time()

                logger.info(f"[{entry.key}] Started running. (Pool: {len(self.running_pool)}/{self.max_concurrency})")

                # 成功启动一个任务后，唤醒其他排队者。
                # 下一个排队者醒来后，会发现 running_pool 未满，
                # 但它会因为 stagger check 而进入带有 timeout 的 wait。
                self._condition.notify_all()

                return

    def _release_entry(self, entry: SchedulerEntry):
        """
        阶段 4: 任务结束，释放资源。
        """
        with self._condition:
            if entry.key in self.running_pool:
                del self.running_pool[entry.key]
                # 录离场时间
                self.last_finish_times[entry.key] = time.time()

                duration = time.time() - entry.start_run_time
                logger.info(f"[{entry.key}] Released after {duration:.2f}s.")

                # 关键：唤醒所有在排队的线程，让它们去争抢 (但会被 FIFO 逻辑过滤)
                self._condition.notify_all()

    def _interruptible_sleep(self, duration: float, stop_event: Optional[threading.Event]):
        """辅助函数：支持 Event 中断的 Sleep"""
        if stop_event:
            stop_event.wait(duration)
        else:
            time.sleep(duration)

    # --- 监控接口 ---

    def get_status_snapshot(self, max_items_per_state: int = 50) -> Dict:
        now = time.time()
        with self._lock:
            running_list = list(self.running_pool.values())
            queued_list = list(self.ready_queue)
            sleeping_list = list(self.sleeping_pool.values())

            def limit(lst):
                if max_items_per_state and len(lst) > max_items_per_state:
                    return lst[:max_items_per_state]
                return lst

            running_out = []
            for e in limit(running_list):
                running_out.append({
                    "key": e.key,
                    "duration": round(now - e.start_run_time, 3) if e.start_run_time else 0.0,
                    "started_at": e.start_run_time,
                    "state_age": round(now - e.last_state_ts, 3) if e.last_state_ts else 0.0
                })

            queued_out = []
            # position: 1-based
            for idx, e in enumerate(limit(queued_list), start=1):
                wait_remaining = 0.0
                if e.wait_reason == "STAGGER" and e.wait_until:
                    wait_remaining = max(0.0, e.wait_until - now)

                queued_out.append({
                    "key": e.key,
                    "position": idx,
                    "wait": round(now - e.enter_queue_time, 3) if e.enter_queue_time else 0.0,
                    "wait_reason": e.wait_reason or ("HEAD_READY" if idx == 1 else "FIFO"),
                    "wait_remaining": round(wait_remaining, 3) if wait_remaining else 0.0,
                    "wait_until": e.wait_until if e.wait_until else None,
                })

            sleeping_out = []
            for e in limit(sleeping_list):
                sleeping_out.append({
                    "key": e.key,
                    "remaining": round(max(0.0, e.wake_up_time - now), 3) if e.wake_up_time else 0.0,
                    "wake_up_at": e.wake_up_time if e.wake_up_time else None,
                    "state_age": round(now - e.last_state_ts, 3) if e.last_state_ts else 0.0
                })

            return {
                "ts": now,
                "config": {
                    "max_concurrency": self.max_concurrency,
                    "startup_stagger": self.startup_stagger
                },
                "stats": {
                    "running": len(running_list),
                    "queued": len(queued_list),
                    "sleeping": len(sleeping_list)
                },
                "details": {
                    "running": running_out,
                    "queued": queued_out,
                    "sleeping": sleeping_out
                }
            }

    def update_limits(self, max_concurrency: int = None, startup_stagger: float = None):
        """动态调整参数"""
        with self._condition:
            if max_concurrency is not None:
                self.max_concurrency = max_concurrency
            if startup_stagger is not None:
                self.startup_stagger = startup_stagger
            self._condition.notify_all()  # 配置变化可能允许更多任务运行


# --- 3. 上下文管理器 (Context Manager) ---

class FlowContext(ContextDecorator):
    """
    具体的调度上下文。支持 with 和 @decorator。
    """

    def __init__(self,
                 scheduler: FlowScheduler,
                 key: str,
                 interval: float,
                 stop_event: Optional[threading.Event]):
        self.scheduler = scheduler
        self.entry = SchedulerEntry(key=key)
        self.interval = interval
        self.stop_event = stop_event

    def __enter__(self):
        # 进入时：执行 Sleep -> Queue -> Wait 逻辑
        self.scheduler._request_entry(self.entry, self.interval, self.stop_event)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出时：释放资源
        self.scheduler._release_entry(self.entry)
        # 异常向外抛出
        return False
