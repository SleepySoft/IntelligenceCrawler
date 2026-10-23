"""可复用、无副作用的抓取漏斗统计。"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from threading import Lock
from typing import Any, DefaultDict, Dict, List


class CrawlTrace:
    """记录各阶段数量、拒绝原因及有限样本，可用于 GUI 和线上治理。"""

    def __init__(self, sample_limit: int = 5):
        self.sample_limit = max(0, int(sample_limit))
        self._counters: Counter[str] = Counter()
        self._rejections: Counter[str] = Counter()
        self._samples: DefaultDict[str, List[str]] = defaultdict(list)
        self._lock = Lock()

    def add(self, metric: str, count: int = 1) -> None:
        with self._lock:
            self._counters[str(metric)] += int(count)

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._rejections.clear()
            self._samples.clear()

    def reject(self, stage: str, reason: str, value: str = "") -> None:
        key = f"{stage}.{reason}"
        with self._lock:
            self._rejections[key] += 1
            if value and len(self._samples[key]) < self.sample_limit:
                self._samples[key].append(str(value))

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(sorted(self._counters.items())),
                "rejections": dict(sorted(self._rejections.items())),
                "samples": deepcopy(dict(self._samples)),
            }

    def summary_lines(self) -> List[str]:
        snapshot = self.snapshot()
        lines = [f"{key}: {value}" for key, value in snapshot["counters"].items()]
        lines.extend(
            f"rejected.{key}: {value}"
            for key, value in snapshot["rejections"].items()
        )
        return lines
