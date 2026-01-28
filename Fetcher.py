#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import queue
import random
import re
import time

import requests
import threading        # Add threading for PlaywrightFetcher avoiding asyncio conflict with Newspaper3kExtractor
from typing import Dict, Optional, Callable, List
from urllib.parse import urlparse
from abc import ABC, abstractmethod

from IntelligenceCrawler.BrowserMonitor import AutoTrackedBrowser
from IntelligenceCrawler.PlaywrightActionEngine import PlaywrightActionEngine

try:
    from dateutil.parser import parse as date_parse
except ImportError:
    print("!!! IMPORT ERROR: 'python-dateutil' not found.")
    print("!!! Please install it for date filtering: pip install python-dateutil")
    date_parse = None

# --- Playwright Imports (with detailed error checking) ---
try:
    from playwright.sync_api import sync_playwright, Page, Error as PlaywrightError
    from playwright._impl._errors import TimeoutError as PlaywrightTimeoutError
except ImportError:
    print("!!! IMPORT ERROR: Could not import 'playwright.sync_api'.")
    print("!!! Please ensure playwright is installed correctly: pip install playwright")
    sync_playwright = None
    PlaywrightError = None
except Exception as e:
    print(f"!!! UNEXPECTED ERROR importing playwright: {e}")
    sync_playwright = None
    PlaywrightError = None

# --- NEW: Smart Import for playwright-stealth (v1 and v2) ---
sync_stealth = None  # For v2.x
Stealth = None  # For v1.x

try:
    # Try importing v2.x style
    from playwright_stealth import sync_stealth

    print("Imported playwright-stealth v2.x ('sync_stealth') successfully.")
except ImportError:
    print("!!! Could not import 'sync_stealth' (v2.x). Trying v1.x fallback...")
    try:
        # Try importing v1.x style
        from playwright_stealth.stealth import Stealth

        print("Imported playwright-stealth v1.x ('Stealth') successfully.")
    except ImportError:
        print("!!! IMPORT ERROR: Could not import 'playwright_stealth' v1 or v2.")
        print("!!! Please ensure it is installed: pip install playwright-stealth")
    except Exception as e:
        print(f"!!! UNEXPECTED ERROR importing playwright_stealth: {e}")
except Exception as e:
    print(f"!!! UNEXPECTED ERROR importing playwright_stealth: {e}")

# Generic check to print the user-friendly message
if not sync_playwright or (not sync_stealth and not Stealth):  # Check both
    print("\n--- Library Setup Incomplete ---")
    print("One or more required Playwright libraries failed to import.")
    print("Please check the '!!! IMPORT ERROR' messages above.")
    print("To install/reinstall, run:")
    print("  pip install playwright playwright-stealth")
    print("Then install browser binaries:")
    print("  python -m playwright install")
    print("----------------------------------\n")
    if 'sync_playwright' not in locals(): sync_playwright = None
    if 'PlaywrightError' not in locals(): PlaywrightError = None


class Fetcher(ABC):
    """
    Abstract Base Class for a content fetcher.
    Defines the interface for different fetching strategies (e.g., simple requests
    or full browser rendering) and standardizes how they are initialized and used.
    """

    @abstractmethod
    def get_content(self, url: str, **kwargs) -> Optional[bytes]:
        """
        Fetches content from a given URL.

        Args:
            url (str): The URL to fetch.
            **kwargs: Additional implementation-specific arguments.

        Returns:
            Optional[bytes]: The raw content of the response as bytes,
                             or None if fetching failed.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Cleans up any persistent resources.
        This could be a requests.Session, a Playwright browser instance,
        or any other long-lived connection.
        """
        pass


def fetcher_factory(name: str, init_params: dict) -> Fetcher:
    if name == 'RequestsFetcher':
        return RequestsFetcher(
            log_callback=init_params.get('log_callback'),
            proxy=init_params.get('proxy'),
            timeout_s=init_params.get('timeout_s', 30)
        )

    if name == 'PlaywrightFetcher':
        return PlaywrightFetcher(
            log_callback=init_params.get('log_callback'),
            proxy=init_params.get('proxy'),
            timeout_s=init_params.get('timeout_s', 30),
            stealth=init_params.get('stealth', False),
            pause_browser=init_params.get('pause_browser', False),
            render_page=init_params.get('render_page', True)
        )

    if 'Requests' in name:
        return fetcher_factory('RequestsFetcher', init_params)
    if 'Playwright' in name:
        return fetcher_factory('PlaywrightFetcher', init_params)

    raise ValueError(f"Unknown fetcher: {name}")


def also_print(log_callback):
    """A helper wrapper to ensure logs are always printed to console."""

    def wrapper(text):
        if log_callback != print:
            print(text)
        if log_callback:
            log_callback(text)

    return wrapper


class RequestsFetcher(Fetcher):
    """
    A fast, lightweight fetcher that uses the `requests` library.
    It maintains a persistent `requests.Session` for connection pooling
    and cookie handling.

    This fetcher is ideal for simple websites, APIs, XML sitemaps, and
    other resources that do not require JavaScript rendering.
    """
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
    }

    def __init__(self,
                 log_callback=print,
                 proxy: Optional[str] = None,
                 timeout_s: int = 10):
        """
        Initializes the RequestsFetcher.

        Args:
            log_callback: A callable (like print) to receive log messages.
            proxy (Optional[str]): A proxy URL string.
                Format: "protocol://user:pass@host:port"
                Examples:
                    - "http://127.0.0.1:8080"
                    - "http://user:pass@proxyserver.com:8080"
                    - "socks5://user:pass@127.0.0.1:1080"
                (For SOCKS support, `pip install "requests[socks]"` is required)
        """
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._log = also_print(log_callback)
        self.timeout = timeout_s

        # --- NEW: Proxy Configuration ---
        if proxy:
            # `requests` expects a dictionary mapping protocols to the proxy URL.
            # We use the same proxy string for both http and https traffic.
            proxies = {
                'http': proxy,
                'https': proxy
            }
            self.session.proxies.update(proxies)

            # Log the proxy server, but hide credentials for security.
            proxy_host = proxy.split('@')[-1]
            self._log(f"Using RequestsFetcher with proxy: {proxy_host}")
        else:
            self._log("Using RequestsFetcher (Fast, Simple)")

    def get_content(self, url: str, **kwargs) -> Optional[bytes]:
        """
        Fetches content from a URL using the configured requests.Session.

        Args:
            url (str): The URL to fetch.
            **kwargs: Can include 'timeout' (int) or 'headers' (dict)
                      to override session defaults for this single request.
                      Other kwargs are ignored.

        Returns:
            Optional[bytes]: The raw response content, or None on failure.
        """
        try:
            # Set a dynamic Referer header based on the target domain
            parsed_url = urlparse(url)
            referer = f"{parsed_url.scheme}://{parsed_url.netloc}/"

            # 1. 确定超时
            # (用户传入的 'timeout' 优先于 'timeout_s'，也优先于实例的 self.timeout)
            request_timeout = kwargs.get('timeout', self.timeout)

            # 2. 合并 Headers
            request_headers = self.session.headers.copy()
            request_headers['Referer'] = referer
            if 'headers' in kwargs and isinstance(kwargs['headers'], dict):
                request_headers.update(kwargs['headers'])

            response = self.session.get(
                url,
                timeout=request_timeout,
                headers=request_headers
            )
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            return response.content
        except requests.exceptions.RequestException as e:
            self._log(f"[Request Error] Failed to fetch {url}: {e}")
            return None

    def close(self):
        """Closes the persistent requests.Session."""
        self._log("Closing RequestsFetcher session.")
        self.session.close()


class PlaywrightFetcher(Fetcher):
    """
    [Refactored to run in a dedicated thread to avoid asyncio conflicts]

    A robust, slower fetcher that uses a real headless browser (Playwright)
    to render pages, execute JavaScript, and bypass anti-bot measures.

    This class launches Playwright in a separate worker thread,
    and provides a synchronous, thread-safe interface for the main thread.
    """

    def __init__(self,
                 log_callback=print,
                 proxy: Optional[str] = None,
                 timeout_s: int = 20,
                 stealth: bool = False,
                 pause_browser: bool = False,
                 render_page: bool = True,

                 default_wait_until: str = "domcontentloaded",
                 default_block_resources: Optional[dict] = None,
                 default_block_third_party: bool = False,
                 default_allowed_domains: Optional[list] = None
):
        """
        Initializes the Fetcher and starts the background Playwright worker thread.
        This method will block until the browser is successfully launched or fails.

        Args:
            log_callback (callable): Function to use for logging.
            proxy (Optional[str]): Proxy string (e.g., "http://user:pass@host:port").
            timeout_s (int): Default timeout in seconds for operations.
            stealth (bool): Whether to enable playwright-stealth.
            pause_browser (bool): If True, launches browser non-headless and
                                  calls page.pause() for debugging.
            render_page (bool): If True, gets page.content() (rendered HTML).
                                If False, gets response.body() (raw response).

            default_block_resources example:
                {
                  "image": True,
                  "media": True,
                  "font": True,
                  "stylesheet": False,
                  "script": False
                }

        """
        self._log = also_print(log_callback)
        self.timeout_ms = timeout_s * 1000  # Playwright timeout is in ms

        # --- Store config for the worker thread ---
        self.stealth_mode = stealth
        self.pause_browser = pause_browser
        self.render_page = render_page

        self.default_wait_until = default_wait_until
        self.default_block_resources = default_block_resources or {
            "image": True,
            "media": True,
            "font": True,
            "stylesheet": False,  # 样式表有时影响可见性/selector，默认不禁
            "script": False  # 禁 script 会直接破坏渲染，默认不禁
        }
        self.default_block_third_party = default_block_third_party
        self.default_allowed_domains = default_allowed_domains  # None 表示不做 allowlist

        self.proxy_config: Optional[Dict[str, str]] = None

        # --- Queues for thread communication ---
        self.job_queue: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self.startup_queue: "queue.Queue[Any]" = queue.Queue(maxsize=1)

        # --- Threading resources ---
        self.worker_thread: Optional[threading.Thread] = None

        # --- 1. Verify Library Availability ---
        if not sync_playwright:
            raise ImportError("Playwright is not installed. Please install 'playwright' and 'playwright install'.")
        if self.stealth_mode and (not sync_stealth and not Stealth):
            raise ImportError("Playwright-Stealth (v1 or v2) is not installed. Please install 'playwright-stealth'.")

        # --- 2. Parse Proxy Configuration (Robust Version) ---
        if proxy:
            try:
                if "://" not in proxy:
                    self._log(f"[Proxy Warning] No scheme found in proxy '{proxy}'. Defaulting to http://")
                    proxy = f"http://{proxy}"

                parsed_proxy = urlparse(proxy)

                # 2. 确保主机名和端口解析成功
                if not parsed_proxy.hostname or not parsed_proxy.port:
                    raise ValueError("Could not parse hostname or port from proxy string.")

                self.proxy_config = {
                    "server": f"{parsed_proxy.scheme}://{parsed_proxy.hostname}:{parsed_proxy.port}"
                }

                if parsed_proxy.username:
                    self.proxy_config["username"] = parsed_proxy.username
                if parsed_proxy.password:
                    self.proxy_config["password"] = parsed_proxy.password

                self._log(f"Playwright proxy configured: {self.proxy_config['server']}")

            except Exception as e:
                self._log(f"!!! WARNING: Invalid proxy format '{proxy}'. Ignoring proxy. Error: {e}")
                self.proxy_config = None

        # --- 3. Start Worker Thread ---
        self._log("Starting Playwright worker thread...")
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        # --- 4. Wait for Browser to Launch ---
        try:
            # Wait up to 60s for browser to start
            startup_result = self.startup_queue.get(timeout=60)
            if isinstance(startup_result, Exception):
                raise startup_result  # Re-raise the exception from the worker thread
            self._log("Playwright worker thread started successfully.")
        except queue.Empty:
            self._log("[Fatal Error] Playwright worker thread timed out on startup.")
            raise TimeoutError("Playwright worker thread failed to start in time.")

    def _create_context(self):
        """[Worker Thread] Create a long-lived context for same-site crawling."""
        if not hasattr(self, "browser") or not self.browser:
            raise RuntimeError("Browser is not initialized; cannot create context.")

        context_options = {
            "user_agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'
        }
        if self.proxy_config:
            context_options["proxy"] = self.proxy_config

        self.context = self.browser.new_context(**context_options)

        # 统一 init script 放在 context 层，保证每个 page 都继承
        # 注意：如果你使用 stealth v1/v2，它通常还需要对每个 page apply（后面 _fetch_page_content 里仍保留）
        self.context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        self._context_created_ts = time.time()
        self._context_request_count = 0
        self._consecutive_errors = 0

        self._log("[Worker] Long-lived context created.")

    def _close_context(self):
        """[Worker Thread] Close current context safely."""
        if hasattr(self, "context") and self.context:
            try:
                self.context.close()
                self._log("[Worker] Context closed.")
            except Exception as e:
                self._log(f"[Worker Warning] Error closing context: {e}")
            finally:
                self.context = None

    def _should_rotate_context(self) -> bool:
        """[Worker Thread] Decide whether to rotate context."""
        MAX_REQUESTS_PER_CONTEXT = 200  # 建议起步 200~1000
        MAX_CONTEXT_AGE_S = 30 * 60  # 建议起步 15~60分钟
        MAX_CONSECUTIVE_ERRORS = 5  # 连续错误触发轮换

        if not hasattr(self, "context") or self.context is None:
            return True

        age = time.time() - float(getattr(self, "_context_created_ts", 0.0))

        if int(getattr(self, "_context_request_count", 0)) >= MAX_REQUESTS_PER_CONTEXT:
            self._log(f"[Worker] Rotate context: request count {self._context_request_count} reached.")
            return True

        if age >= MAX_CONTEXT_AGE_S:
            self._log(f"[Worker] Rotate context: age {age:.0f}s reached.")
            return True

        if int(getattr(self, "_consecutive_errors", 0)) >= MAX_CONSECUTIVE_ERRORS:
            self._log(f"[Worker] Rotate context: consecutive errors {self._consecutive_errors} reached.")
            return True

        return False

    def _ensure_context(self):
        """[Worker Thread] Ensure context exists and is healthy; rotate if needed."""
        if self._should_rotate_context():
            self._close_context()
            self._create_context()

    def _start_playwright(self):
        """
        [Worker Thread] Initializes Playwright and launches the browser.
        Returns:
            bool: True if started successfully, False otherwise.
        """
        try:
            mode = "Stealth" if self.stealth_mode else "Standard"
            self._log(f"[Worker] Starting Playwright ({mode}, Headless: {not self.pause_browser})...")

            self.playwright = sync_playwright().start()

            headless_mode = not self.pause_browser
            # Use strict args to prevent zombie processes and memory issues
            launch_args = [
                '--disable-gpu',
                '--no-sandbox',
                '--disable-dev-shm-usage'
            ]

            real_browser = self.playwright.chromium.launch(
                headless=headless_mode,
                args=launch_args
                # 更推荐把 proxy 放这里（launch(proxy=...)），
            )

            # Use the user's custom wrapper
            self.browser = AutoTrackedBrowser(real_browser)

            # --- initialize context-related state (context created later) ---
            self.context = None
            self._context_created_ts = 0.0
            self._context_request_count = 0
            self._consecutive_errors = 0

            self._log("[Worker] Browser instance created successfully.")
            return True
        except Exception as e:
            self._log(f"[Worker Error] Failed to start Playwright: {e}")
            self._stop_playwright()  # Cleanup if partial failure
            return False

    def _stop_playwright(self):
        """[Worker Thread] Safely shuts down the Playwright browser/context and process."""
        self._log("[Worker] Stopping Playwright browser/context...")

        # --- NEW: close context first ---
        try:
            self._close_context()
        except Exception:
            pass

        # 1. Close Browser
        if hasattr(self, 'browser') and self.browser:
            try:
                self.browser.close()
                self._log("[Worker] Browser closed.")
            except Exception as e:
                self._log(f"[Worker Warning] Error closing browser: {e}")
            finally:
                self.browser = None

        # 2. Stop Playwright Driver
        if hasattr(self, 'playwright') and self.playwright:
            try:
                self.playwright.stop()
                self._log("[Worker] Playwright driver stopped.")
            except Exception as e:
                self._log(f"[Worker Warning] Error stopping playwright: {e}")
            finally:
                self.playwright = None

    def _worker_loop(self):
        """
        [Worker Thread] Main loop with Robust Lifecycle Management.
        Implements a "Restart Periodically" strategy to prevent memory leaks.
        Now also reuses a long-lived Context and rotates it periodically.
        """

        # Configuration: Restart browser after N requests to clear memory leaks
        MAX_REQUESTS_PER_BROWSER = 500

        # Signal successful thread start (initial only)
        # We try to start it once to check dependencies.
        if self._start_playwright():
            self.startup_queue.put(True)
            self._stop_playwright()  # Stop immediately, let the loop handle lifecycle
        else:
            self.startup_queue.put(RuntimeError("Worker failed to verify Playwright startup."))
            return

        shutdown_requested = False

        # --- OUTER LOOP: Manages Browser Lifecycle (Restart Logic) ---
        while not shutdown_requested:
            try:
                # 1. Start Browser Session
                if not self._start_playwright():
                    self._log("[Worker Error] Browser failed to start. Retrying in 5s...")
                    threading.Event().wait(5)
                    continue

                request_count = 0

                # Create long-lived context once per browser session
                self._create_context()

                # --- INNER LOOP: Manages Job Processing ---
                while not shutdown_requested:
                    # Check restart criteria (Leak Prevention)
                    if request_count >= MAX_REQUESTS_PER_BROWSER:
                        self._log(
                            f"[Worker] Reached limit ({MAX_REQUESTS_PER_BROWSER} jobs). Restarting browser to free memory...")
                        break  # trigger finally -> restart outer loop

                    # ensure context is healthy (rotate if needed)
                    try:
                        self._ensure_context()
                    except Exception as e:
                        self._log(f"[Worker Error] Failed to ensure context: {e}. Forcing browser restart...")
                        break

                    try:
                        job_data = self.job_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue

                    if not job_data:
                        continue

                    job_type, data, result_queue = job_data

                    # Handle Shutdown Signal
                    if job_type == 'shutdown':
                        self._log("[Worker] Shutdown signal received.")
                        result_queue.put(True)
                        shutdown_requested = True
                        break

                    # Handle Fetch Job
                    if job_type == 'get_content':
                        try:
                            content = self._fetch_page_content(data)
                            result_queue.put(content)
                        except Exception as e:
                            self._log(f"[Worker Error] Job failed: {str(e)}")
                            result_queue.put(e)
                        finally:
                            request_count += 1
                            # Optional: Force garbage collection after heavy jobs
                            # import gc; gc.collect()

            except Exception as e:
                self._log(f"[Worker Critical Error] Unhandled exception in worker loop: {e}")
                threading.Event().wait(2)

            finally:
                # ensure context closed before browser restart/exit
                try:
                    self._close_context()
                except Exception:
                    pass
                self._stop_playwright()

        self._log("[Worker] Thread exiting cleanly.")

    def get_content(self, url: str, **kwargs) -> Optional[bytes]:
        """
        Fetches content from a URL with flexible wait conditions.

        This method is synchronous and thread-safe. It sends a job to the
        background worker thread and blocks until the result is returned.

        Timeout behavior (important):
            - By default, timeout_s=None => wait indefinitely as long as the worker
              thread is still alive.
            - The method periodically wakes up every `check_interval_s` seconds to
              check if the worker thread is still alive.
            - If the worker thread dies/hangs permanently and stops responding
              (i.e., the thread is not alive), this method raises RuntimeError.
            - If timeout_s is provided (float), it behaves like a hard deadline and
              raises TimeoutError when exceeded.

        Args:
            url (str):
                The URL to fetch.

            timeout_s (Optional[float], default None):
                Overall waiting time in seconds in the main thread.
                - None: wait forever (recommended for production stability when
                  worker is reliable and Playwright has its own internal timeouts).
                - float: wait at most this many seconds for the worker result.

            check_interval_s (float, default 0.5):
                The polling interval (seconds) used to periodically check whether
                the worker thread is still alive while waiting for the job result.
                Smaller => more responsive to worker death, but slightly more CPU wakeups.

            **kwargs:
                Flexible options passed to the worker, including:

                --- Navigation / waiting strategy ---
                wait_until (str):
                    Passed to page.goto(wait_until=...). One of:
                    'load', 'domcontentloaded', 'networkidle'.
                    If not provided, uses self.default_wait_until if defined,
                    otherwise falls back to 'networkidle' (legacy behavior).

                wait_mode (str):
                    Alias of wait_until. If both provided, wait_mode takes priority.

                wait_for_selector (Optional[str]):
                    A CSS selector to wait for after navigation.
                    Best-effort: timeout will be logged but not necessarily raise
                    inside worker (depending on worker implementation).

                wait_for_timeout_s (Optional[int/float]):
                    Timeout (seconds) for the selector/function/response waits.
                    If None, defaults to self.timeout_ms.

                wait_for_function (Optional[str]):
                    A JS predicate string for page.wait_for_function().
                    Example: "() => window.__DATA__ !== undefined"
                    Best-effort by design (recommended for SPA readiness).

                wait_for_response_url (Optional[str]):
                    Wait until a network response URL contains this substring.
                    Useful when SPA loads content via XHR/fetch.

                wait_for_text (Optional[str]):
                    Wait until document.body.innerText includes this text.
                    Useful for simple readiness checks without selectors.

                extra_wait_s (int/float):
                    Extra small delay after waits (e.g., animations), in seconds.

                --- Rendering / extraction ---
                render_page (Optional[bool]):
                    Override instance-level self.render_page for this request only.
                    - True: return rendered HTML via page.content() (needs JS render)
                    - False: return raw response body via response.body() (faster)

                --- Page interactions ---
                scroll_pages (int):
                    Number of pages to scroll.
                    > 0: scroll down; < 0: scroll up; 0: no scroll.

                post_extra_action (Callable[[Page], None] | list | None):
                    Extra action after page loaded and scrolled.
                    - callable(page): do anything (click, type, evaluate...)
                    - list: if your worker supports PlaywrightActionEngine actions
                    - None: no extra action

                --- Resource blocking / performance knobs ---
                block_resources (Optional[dict]):
                    Resource type blocking map; True => abort.
                    Example:
                        {
                          "image": True,
                          "media": True,
                          "font": True,
                          "stylesheet": False,
                          "script": False
                        }
                    If None, uses self.default_block_resources if defined.

                block_third_party (bool):
                    If True, abort requests whose host is third-party relative to page URL.
                    If not provided, uses self.default_block_third_party if defined.

                allowed_domains (Optional[list[str]]):
                    Allowlist of domains. If set, abort any request not in allowlist.
                    Stronger than block_third_party. If None, no allowlist restriction.

        Returns:
            Optional[bytes]:
                The fetched page content (rendered HTML bytes or raw response bytes).

        Raises:
            RuntimeError:
                - If the worker thread is not running (dead/hung and not alive).
                - If the worker returns an Exception (re-raised here).
            TimeoutError:
                - If timeout_s is provided and exceeded.
        """
        if not self.worker_thread or not self.worker_thread.is_alive():
            raise RuntimeError(
                "Playwright worker thread is not running. Fetcher may have been closed or failed to start.")

        # Create a one-time queue to get the result back
        result_queue: "queue.Queue[Any]" = queue.Queue(maxsize=1)

        # --- wait options ---
        wait_until_val = kwargs.get('wait_until', None)
        wait_mode_val = kwargs.get('wait_mode', None)  # NEW alias
        wait_until_final = wait_mode_val or wait_until_val or self.default_wait_until

        wait_for_selector_val = kwargs.get('wait_for_selector', None)
        wait_for_timeout_s_val = kwargs.get('wait_for_timeout_s', None)
        scroll_pages_val = kwargs.get('scroll_pages', 0)
        post_extra_action = kwargs.get('post_extra_action', None)

        # --- NEW: resource blocking / domain policy ---
        block_resources = kwargs.get("block_resources", None) or self.default_block_resources
        block_third_party = kwargs.get("block_third_party", self.default_block_third_party)
        allowed_domains = kwargs.get("allowed_domains", self.default_allowed_domains)

        # --- NEW: more precise waits ---
        wait_for_function = kwargs.get("wait_for_function", None)  # JS string: "() => ..."
        wait_for_response_url = kwargs.get("wait_for_response_url", None)  # substring match
        wait_for_text = kwargs.get("wait_for_text", None)  # simple text presence
        extra_wait_s = kwargs.get("extra_wait_s", 0)  # small buffer for animation

        job_payload = {
            'url': url,
            'wait_until': wait_until_final,
            'wait_for_selector': wait_for_selector_val,
            'wait_for_timeout_ms': (wait_for_timeout_s_val * 1000) if wait_for_timeout_s_val is not None else None,
            'scroll_pages': scroll_pages_val,
            'post_extra_action': post_extra_action,

            # NEW
            'block_resources': block_resources,
            'block_third_party': block_third_party,
            'allowed_domains': allowed_domains,

            'wait_for_function': wait_for_function,
            'wait_for_response_url': wait_for_response_url,
            'wait_for_text': wait_for_text,
            'extra_wait_s': extra_wait_s,

            # allow overriding render_page per request
            'render_page': kwargs.get("render_page", None),
        }

        # 添加任何其他传入的 kwargs (未来扩展性)
        # job_payload.update(kwargs)

        # Send the job to the worker thread
        self.job_queue.put(('get_content', job_payload, result_queue))

        poll_interval_s = 10
        overall_timeout_s = 10 * 60
        deadline = time.time() + overall_timeout_s

        while True:
            # 1) 线程死了就立刻退出（避免永久等待）
            if not self.worker_thread.is_alive():
                raise RuntimeError(f"Worker thread died while waiting for result: {url}")

            # 2) 指定了 overall_timeout_s 才执行硬截止
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError(f"Playwright job for {url} timed out after {overall_timeout_s}s")

            try:
                # 3) 关键：用 poll_interval_s 做短超时轮询，让我们有机会检查 worker 状态
                result = result_queue.get(timeout=poll_interval_s)
            except queue.Empty:
                continue

            if isinstance(result, Exception):
                self._log(f"[Main Thread] Error received from worker for {url}")
                raise result

            return result

    def _is_third_party(self, target_url: str, page_url: str) -> bool:
        """Return True if target_url host differs from page_url host."""
        try:
            th = urlparse(target_url).hostname or ""
            ph = urlparse(page_url).hostname or ""
            if not th or not ph:
                return False
            return th != ph and not th.endswith("." + ph)
        except Exception:
            return False

    def _apply_page_routing(self, page, job_payload: dict):
        """
        [Worker Thread] Apply per-page routing to block resources/domains.
        Returns a function to unroute for cleanup.
        """
        block_resources = job_payload.get("block_resources") or {}
        block_third_party = bool(job_payload.get("block_third_party", False))
        allowed_domains = job_payload.get("allowed_domains", None)
        page_url = job_payload.get("url")

        def handler(route, request):
            try:
                rtype = request.resource_type  # 'document','script','image','media','font','stylesheet','xhr','fetch'...
                req_url = request.url

                # Allowlist has highest priority: if configured, block anything not in list
                if allowed_domains:
                    host = urlparse(req_url).hostname or ""
                    allowed = any(host == d or host.endswith("." + d) for d in allowed_domains)
                    if not allowed:
                        return route.abort()

                # Block third party domains if enabled
                if block_third_party and self._is_third_party(req_url, page_url):
                    return route.abort()

                # Block resource types
                if block_resources.get(rtype, False):
                    return route.abort()

                return route.continue_()
            except Exception:
                # Fail-open to avoid breaking navigation unexpectedly
                return route.continue_()

        page.route("**/*", handler)

        def cleanup():
            try:
                page.unroute("**/*", handler)
            except Exception:
                pass

        return cleanup

    def _fetch_page_content(self, job_payload: dict) -> Optional[bytes]:
        url = job_payload['url']
        wait_until = job_payload.get('wait_until', 'domcontentloaded')
        wait_for_selector = job_payload.get('wait_for_selector')
        scroll_pages = job_payload.get('scroll_pages', 0)
        post_extra_action = job_payload.get('post_extra_action', None)

        selector_timeout_ms = job_payload.get('wait_for_timeout_ms') or self.timeout_ms

        # NEW precise waits
        wait_for_function = job_payload.get("wait_for_function", None)
        wait_for_response_url = job_payload.get("wait_for_response_url", None)
        wait_for_text = job_payload.get("wait_for_text", None)
        extra_wait_s = float(job_payload.get("extra_wait_s", 0) or 0)

        # allow per-request render_page override
        render_page = job_payload.get("render_page")
        if render_page is None:
            render_page = self.render_page

        page = None
        cleanup_route = None

        try:
            self._ensure_context()
            page = self.context.new_page()

            # Apply stealth (page-level as before)
            if self.stealth_mode:
                if Stealth:
                    Stealth().apply_stealth_sync(page)
                elif sync_stealth:
                    sync_stealth(page)
                else:
                    page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            else:
                page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

            # per-page routing (block images/fonts/3rd party etc.)
            cleanup_route = self._apply_page_routing(page, job_payload)

            # --- Main navigation ---
            response = page.goto(url, timeout=self.timeout_ms, wait_until=wait_until)
            if not response or not response.ok:
                status = response.status if response else 'N/A'
                raise PlaywrightError(f"Failed to get valid response. Status: {status}")

            self._log(f"[Worker] page.goto() ok for {url} (Status: {response.status}, Wait: {wait_until})")

            # --- NEW: wait for response URL (useful for SPA data) ---
            if wait_for_response_url:
                self._log(f"[Worker] Waiting for response containing '{wait_for_response_url}'...")
                try:
                    page.wait_for_response(lambda r: wait_for_response_url in r.url, timeout=selector_timeout_ms)
                    self._log("[Worker] Target response observed.")
                except Exception as e:
                    self._log(f"[Worker Warning] wait_for_response_url timeout/error: {e}")

            # --- Existing: selector wait (best-effort) ---
            if wait_for_selector:
                self._log(f"[Worker] Waiting selector '{wait_for_selector}' (timeout {selector_timeout_ms}ms)...")
                try:
                    page.wait_for_selector(wait_for_selector, state='visible', timeout=selector_timeout_ms)
                    self._log(f"[Worker] Selector '{wait_for_selector}' ready.")
                except Exception as e:
                    self._log(f"[Worker Warning] Selector wait failed: {e} (best-effort continue)")

            # --- NEW: wait for function (best-effort) ---
            if wait_for_function:
                self._log(f"[Worker] Waiting for function: {wait_for_function} ...")
                try:
                    page.wait_for_function(wait_for_function, timeout=selector_timeout_ms)
                    self._log("[Worker] Function condition satisfied.")
                except Exception as e:
                    self._log(f"[Worker Warning] wait_for_function timeout/error: {e}")

            # --- NEW: wait for text appears (best-effort) ---
            if wait_for_text:
                self._log(f"[Worker] Waiting for text: '{wait_for_text}' ...")
                try:
                    # simplest: check body innerText contains
                    page.wait_for_function(
                        """(t) => document.body && document.body.innerText && document.body.innerText.includes(t)""",
                        arg=wait_for_text,
                        timeout=selector_timeout_ms
                    )
                    self._log("[Worker] Text condition satisfied.")
                except Exception as e:
                    self._log(f"[Worker Warning] wait_for_text timeout/error: {e}")

            # optional small buffer
            if extra_wait_s > 0:
                page.wait_for_timeout(int(extra_wait_s * 1000))

            # --- scrolling (keep your current logic; later we can optimize away from networkidle) ---
            if scroll_pages != 0:
                scroll_direction = 'down' if scroll_pages > 0 else 'up'
                self._log(f"[Worker] Scrolling {abs(scroll_pages)} pages {scroll_direction} (jitter mode)...")

                js_scroll_distance = "window.innerHeight" if scroll_pages > 0 else "-window.innerHeight"
                scroll_network_timeout = 1500  # NEW: reduce default to be faster; still best-effort

                for i in range(abs(scroll_pages)):
                    page.evaluate(f"window.scrollBy(0, {js_scroll_distance});")
                    jitter_ms = random.randint(300, 900)  # NEW: faster jitter; enough for lazyload
                    page.wait_for_timeout(jitter_ms)

                    try:
                        # best-effort, short timeout
                        page.wait_for_load_state('domcontentloaded', timeout=scroll_network_timeout)
                    except Exception:
                        pass

                try:
                    page.wait_for_timeout(300)  # small settle
                except Exception:
                    pass

            # --- post actions ---
            try:
                if post_extra_action is None:
                    pass
                elif callable(post_extra_action):
                    post_extra_action(page)
                elif isinstance(post_extra_action, list):
                    action_engine = PlaywrightActionEngine(page=page)
                    action_engine.execute(post_extra_action)
                else:
                    raise ValueError("Not support post extra action - ignore.")
            except Exception as e:
                self._log(str(e))

            # --- Extract content ---
            content_bytes: Optional[bytes] = None
            if 200 <= response.status < 300:
                if render_page:
                    self._log("[Worker] Extracting rendered page.content()...")
                    content_str = page.content()
                    content_bytes = content_str.encode("utf-8")
                else:
                    self._log("[Worker] Extracting raw response.body()...")
                    content_bytes = response.body()
            else:
                self._log("[Worker] Non-2xx response while extracting content.")

            # bookkeeping success
            self._context_request_count += 1
            self._consecutive_errors = 0
            return content_bytes

        except PlaywrightTimeoutError:
            self._log(f"[Warning] goto timeout for {url}, trying to grab content anyway.")
            if page:
                content = page.content()
            else:
                content = None

            if not content:
                self._consecutive_errors += 1
                raise ValueError("Timeout occurred AND page content was empty/invalid.")

            self._context_request_count += 1
            self._consecutive_errors = 0
            return content.encode("utf-8")

        except Exception as e:
            self._consecutive_errors += 1
            msg = str(e).lower()
            if ("target closed" in msg) or ("context closed" in msg) or ("browser" in msg and "disconnected" in msg):
                self._log("[Worker] Critical error suggests broken context; closing context to force recreation.")
                try:
                    self._close_context()
                except Exception:
                    pass

            self._log(f"[Worker Error] _fetch_page_content failed for {url}: {e}")
            raise

        finally:
            # cleanup route first
            if cleanup_route:
                try:
                    cleanup_route()
                except Exception:
                    pass
            if page:
                try:
                    page.close()
                except Exception as e:
                    self._log(f"[Worker Warning] Error closing page: {e}")

    def close(self):
        """
        [Main Thread] Shuts down the Playwright worker thread and browser.
        """
        self._log("Sending shutdown signal to worker thread...")
        if hasattr(self, 'worker_thread') and self.worker_thread and self.worker_thread.is_alive():
            try:
                # Use a queue to wait for acknowledgment
                shutdown_queue: "queue.Queue[Any]" = queue.Queue(maxsize=1)
                self.job_queue.put(('shutdown', None, shutdown_queue))
                # Wait 10s for acknowledgment
                shutdown_queue.get(timeout=10)
            except queue.Empty:
                self._log("[Warning] Worker did not acknowledge shutdown signal.")

            # Wait for thread to fully exit
            self.worker_thread.join(timeout=10)
            if self.worker_thread.is_alive():
                self._log("[Error] Worker thread failed to join.")
        self._log("PlaywrightFetcher closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()