import time
import random
import requests
from dataclasses import dataclass
from typing import Optional, Dict, Any


# ---- Status constants (must match core) ----
class Status:
    PENDING = 0
    RUNNING = 1
    SUCCESS = 2
    TEMP_FAIL = 3
    PERM_FAIL = 4
    SKIPPED = 5
    STOPPED = 6
    CACHED = 7
    IGNORED = 8


@dataclass
class RemoteSpiderSDK:
    """
    Lightweight wrapper for CGS RPC API.
    Contract aligned with backend:
      - /rpc/register_group
      - /rpc/should_crawl
      - /rpc/report_result
      - /rpc/round/lifecycle
    """
    base_url: str
    spider_name: str
    timeout: float = 5.0

    def __post_init__(self):
        self.session = requests.Session()
        # Optional: keep-alive headers
        self.session.headers.update({"Content-Type": "application/json"})

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        resp = self.session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    # ---------- RPC: register group ----------
    def register_group(self, group: str, list_url: str, name: Optional[str] = None) -> None:
        """
        group: logical group name (without spider prefix), e.g. "Remote_News"
        group_path will be normalized as "{spider}/{group}" to match core grouping.
        """
        group_path = f"{self.spider_name}/{group}".strip("/")

        payload = {
            "group_path": group_path,
            "list_url": list_url,
        }
        if name:
            payload["name"] = name

        self._post("/rpc/register_group", payload)

    # ---------- RPC: should crawl ----------
    def should_crawl(self, url: str, max_retries: int = 3) -> bool:
        payload = {"url": url, "max_retries": max_retries}
        data = self._post("/rpc/should_crawl", payload)
        return bool(data.get("should_crawl", False))

    # ---------- RPC: report result ----------
    def report_result(
        self,
        url: str,
        group: str,
        status: int,
        http_code: int = 0,
        duration: float = 0.0,
        state_msg: Optional[str] = None,
        error_msg: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> None:
        """
        status must use core Status values.
        group is logical name (without spider prefix).
        """
        group_path = f"{self.spider_name}/{group}".strip("/")

        payload = {
            "url": url,
            "group_path": group_path,
            "spider": self.spider_name,  # optional; backend can derive but keeping is ok
            "status": int(status),
            "http_code": int(http_code),
            "duration": float(duration),
            "state_msg": state_msg,
            "error_msg": error_msg,
            "file_path": file_path,
        }
        # Remove None keys to keep payload clean
        payload = {k: v for k, v in payload.items() if v is not None}

        self._post("/rpc/report_result", payload)

    # ---------- RPC: round lifecycle ----------
    def round_start(self, group: str, expected_count: int = 0) -> None:
        group_path = f"{self.spider_name}/{group}".strip("/")
        payload = {"action": "start", "group": group_path, "expected_count": int(expected_count)}
        self._post("/rpc/round/lifecycle", payload)

    def round_finish(self, group: str, next_run_delay: float = 0) -> None:
        group_path = f"{self.spider_name}/{group}".strip("/")
        payload = {"action": "finish", "group": group_path, "next_run_delay": float(next_run_delay)}
        self._post("/rpc/round/lifecycle", payload)


# ----------------------------------------------------------------------------------------------------------------------

def run_spider_process():
    API_URL = "http://localhost:8002"  # <-- match backend port
    spider_name = "rpc_worker_01"

    sdk = RemoteSpiderSDK(base_url=API_URL, spider_name=spider_name)

    group = "Remote_News"
    list_url = "http://remote-news.com/feed"
    interval = 10  # seconds, client-side schedule

    # 1) register group + anchor
    sdk.register_group(group=group, list_url=list_url, name="Remote News")

    print(f"Spider {sdk.spider_name} started. Connected to Governance via RPC.")

    while True:
        # One "round" per list fetch cycle
        sdk.round_start(group=group, expected_count=0)

        if sdk.should_crawl(list_url, max_retries=3):
            print(f"Crawling List: {list_url}")
            start = time.time()
            time.sleep(1)

            # Report list success (treat list as a task too)
            sdk.report_result(
                url=list_url,
                group=group,
                status=Status.SUCCESS,
                http_code=200,
                duration=time.time() - start,
                state_msg="List OK"
            )

            # Discover articles
            articles = [f"{list_url}/{i}" for i in range(random.randint(1, 3))]
            sdk.round_start(group=group, expected_count=len(articles))  # optional: set expected more accurately

            for art in articles:
                if not sdk.should_crawl(art, max_retries=3):
                    continue

                print(f"  > Crawling Article: {art}")
                astart = time.time()
                time.sleep(0.2)

                dice = random.random()
                if dice > 0.8:
                    # TEMP_FAIL (retryable)
                    sdk.report_result(
                        url=art,
                        group=group,
                        status=Status.TEMP_FAIL,
                        http_code=503,
                        duration=time.time() - astart,
                        error_msg="Gateway Timeout"
                    )
                elif dice > 0.1:
                    # SUCCESS
                    sdk.report_result(
                        url=art,
                        group=group,
                        status=Status.SUCCESS,
                        http_code=200,
                        duration=time.time() - astart,
                        state_msg="OK"
                    )
                else:
                    # PERM_FAIL (non-retryable)
                    sdk.report_result(
                        url=art,
                        group=group,
                        status=Status.PERM_FAIL,
                        http_code=404,
                        duration=time.time() - astart,
                        error_msg="Not Found"
                    )

        else:
            print(".", end="", flush=True)

        # End round, provide next_run_delay for UI countdown
        sdk.round_finish(group=group, next_run_delay=interval)
        time.sleep(interval)


if __name__ == "__main__":
    run_spider_process()
