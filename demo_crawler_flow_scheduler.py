import time
import threading
import logging
import random
import sys
from dataclasses import dataclass
from IntelligenceCrawler.CrawlerFlowScheduler import FlowScheduler

# ================= 配置与工具 =================

# 1. 配置日志：同时显示时间、线程名和消息
#    我们将 FlowScheduler 的日志开启，这样能看到 "Started running" 的瞬间
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)

# 过滤掉 unnecessary 的日志，只保留 FlowScheduler 的
logger = logging.getLogger("FlowScheduler")
logger.setLevel(logging.INFO)


# 为了让日志更易读，我们给调度器的日志加个前缀
def setup_scheduler_logger():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s | [SCHED] %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.handlers = []  # 清除旧的
    logger.addHandler(handler)
    logger.propagate = False  # 防止双重打印


# ================= 模拟负载定义 =================

@dataclass
class SpiderProfile:
    name: str
    count: int  # 线程数
    base_interval: float  # 冷却间隔 (s)
    work_mean: float  # 工作耗时均值 (s)
    work_dev: float  # 工作耗时波动 (s)


# 更加极端的混合负载
PROFILES = [
    # [捣乱者] 极速小任务：冷却极短，如果不加控制，会瞬间淹没队列
    SpiderProfile("Stock", count=5, base_interval=0.2, work_mean=0.5, work_dev=0.1),

    # [中流砥柱] 普通任务
    SpiderProfile("News", count=10, base_interval=2.0, work_mean=6.0, work_dev=1.0),

    # [路障] 重型任务：一旦运行，长期占坑，逼迫其他人排队
    SpiderProfile("Video", count=5, base_interval=5.0, work_mean=45.0, work_dev=3.0),
]

# 调度器配置
MAX_CONCURRENCY = 5  # 只有5个浏览器窗口
STARTUP_STAGGER = 1.0  # 强制每秒只能启动1个 (防反爬关键参数)


# ================= 主程序 =================

def run_stream_simulation():
    # 假设 FlowScheduler 类已定义 (直接使用你提供的代码)
    # 此处实例化调度器
    try:
        scheduler = FlowScheduler(max_concurrency=MAX_CONCURRENCY, startup_stagger=STARTUP_STAGGER)
    except NameError:
        print("错误: 请确保 FlowScheduler 类定义在当前上下文中。")
        return

    setup_scheduler_logger()
    stop_event = threading.Event()

    print(f"\n{'=' * 80}")
    print(
        f" 启动模拟: {sum(p.count for p in PROFILES)} 线程 | 并发限制: {MAX_CONCURRENCY} | 启动间隔: {STARTUP_STAGGER}s")
    print(f" 目标: 观察 [SCHED] 日志的节奏是否均匀，以及 [STATS] 中 Queued 的积压情况")
    print(f"{'=' * 80}\n")

    # --- 工作线程逻辑 ---
    def worker_routine(profile: SpiderProfile, idx: int):
        key = f"{profile.name}-{idx}"
        while not stop_event.is_set():
            try:
                # 随机冷却
                interval = max(0.1, random.gauss(profile.base_interval, profile.base_interval * 0.2))

                with scheduler.pace(key=key, interval=interval, stop_event=stop_event):
                    # 模拟工作 (随机耗时)
                    work_time = max(0.1, random.gauss(profile.work_mean, profile.work_dev))
                    time.sleep(work_time)
            except InterruptedError:
                break
            except Exception:
                pass

    # --- 监控线程 (流式输出) ---
    def monitor_stream():
        step = 0
        while not stop_event.is_set():
            # 每 20 行打印一次表头
            if step % 20 == 0:
                print(
                    f"\n{'TIME':<8} | {'RUN/MAX':<7} | {'QUEUED':<6} | {'SLEEP':<5} | {'DETAILS (Top Long-Running Tasks)'}")
                print("-" * 80)

            snap = scheduler.get_status_snapshot()
            stats = snap['stats']
            details = snap['details']

            # 显示所有正在运行的任务（按耗时排序）
            longest_running = sorted(details['running'], key=lambda x: x['duration'], reverse=True)
            if longest_running:
                details_str = ", ".join([f"{t['key']}({t['duration']:.1f}s)" for t in longest_running])
            else:
                details_str = "-"

            timestamp = time.strftime("%H:%M:%S")

            # 使用 print 自动换行，保留历史
            # 前缀 [STATS] 用于区分日志类型
            print(
                f"{timestamp:<8} | {stats['running']}/{MAX_CONCURRENCY:<7} | {stats['queued']:<6} | {stats['sleeping']:<5} | {details_str}")

            step += 1
            time.sleep(1.0)  # 每秒采样一次

    # --- 启动所有线程 ---
    threads = []

    # 启动监控
    m = threading.Thread(target=monitor_stream, daemon=True)
    m.start()

    # 启动爬虫
    for p in PROFILES:
        for i in range(p.count):
            t = threading.Thread(target=worker_routine, args=(p, i + 1))
            t.daemon = True
            t.start()
            threads.append(t)

    # 阻塞主线程，直到用户中断
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n正在停止 (Graceful Shutdown)...")
        stop_event.set()
        time.sleep(2)  # 给一点时间让日志吐完


if __name__ == "__main__":
    run_stream_simulation()
