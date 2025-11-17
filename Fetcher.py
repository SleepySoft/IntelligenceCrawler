#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import queue
import random
import re

import requests
import threading        # Add threading for PlaywrightFetcher avoiding asyncio conflict with Newspaper3kExtractor
from typing import Dict, Optional, Callable
from urllib.parse import urlparse
from abc import ABC, abstractmethod


try:
    from dateutil.parser import parse as date_parse
except ImportError:
    print("!!! IMPORT ERROR: 'python-dateutil' not found.")
    print("!!! Please install it for date filtering: pip install python-dateutil")
    date_parse = None

# --- Playwright Imports (with detailed error checking) ---
try:
    from playwright.sync_api import sync_playwright, Error as PlaywrightError
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


def also_print(log_callback):
    """A helper wrapper to ensure logs are always printed to console."""

    def wrapper(text):
        if log_callback != print:
            print(text)
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
                 render_page: bool = True):
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
        """
        self._log = also_print(log_callback)
        self.timeout_ms = timeout_s * 1000  # Playwright timeout is in ms

        # --- Store config for the worker thread ---
        self.stealth_mode = stealth
        self.pause_browser = pause_browser
        self.render_page = render_page
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

        # --- 2. Parse Proxy Configuration ---
        if proxy:
            try:
                parsed_proxy = urlparse(proxy)
                if not all([parsed_proxy.scheme, parsed_proxy.hostname, parsed_proxy.port]):
                    raise ValueError("Proxy string must include scheme, host, and port.")
                self.proxy_config = {
                    "server": f"{parsed_proxy.scheme}://{parsed_proxy.hostname}:{parsed_proxy.port}"
                }
                if parsed_proxy.username: self.proxy_config["username"] = parsed_proxy.username
                if parsed_proxy.password: self.proxy_config["password"] = parsed_proxy.password
                self._log(f"Playwright proxy configured for server: {self.proxy_config['server']}")
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

    def _start_playwright(self):
        """[Worker Thread] Initializes Playwright and launches the browser."""
        mode = "Stealth" if self.stealth_mode else "Standard"
        self._log(f"[Worker] Starting Playwright ({mode}, Headless: {not self.pause_browser})...")
        headless_mode = not self.pause_browser

        self.playwright: "Playwright" = sync_playwright().start()
        self.browser: "Browser" = self.playwright.chromium.launch(headless=headless_mode)

        log_msg = "[Worker] Headless browser started." if headless_mode \
            else "[Worker] Headful browser started (pause_browser=True)."
        self._log(log_msg)

    def _stop_playwright(self):
        """[Worker Thread] Shuts down the Playwright browser and process."""
        self._log("[Worker] Stopping Playwright browser...")
        if hasattr(self, 'browser') and self.browser:
            try:
                self.browser.close()
            except Exception as e:
                self._log(f"[Worker Warning] Error closing browser: {e}")
        if hasattr(self, 'playwright') and self.playwright:
            try:
                self.playwright.stop()
            except Exception as e:
                self._log(f"[Worker Warning] Error stopping playwright: {e}")
        self._log("[Worker] Playwright stopped.")

    def _worker_loop(self):
        """
        [Worker Thread] This is the main loop for the dedicated Playwright thread.
        It initializes Playwright and then waits for jobs.
        """
        try:
            self._start_playwright()
            self.startup_queue.put(True)  # Signal success to __init__
        except Exception as e:
            self._log(f"[Worker Error] Failed to start Playwright: {e}")
            self.startup_queue.put(e)  # Signal failure to __init__
            return  # Exit thread

        # --- Main Job Loop ---
        while True:
            try:
                # Wait for a job from the main thread
                job_data = self.job_queue.get()
                if not job_data:
                    continue

                job_type, data, result_queue = job_data

                if job_type == 'shutdown':
                    self._log("[Worker] Shutdown signal received.")
                    result_queue.put(True)  # Acknowledge shutdown
                    break  # Exit loop

                if job_type == 'get_content':
                    # 'data' is now a job_payload dictionary
                    job_payload = data
                    url = job_payload['url']
                    self._log(f"[Worker] Starting job for: {url}")
                    try:
                        # Call the *actual* fetching logic
                        content = self._fetch_page_content(job_payload)
                        result_queue.put(content)  # Send content back
                    except Exception as e:
                        self._log(f"[Worker Error] Job failed for {url}: {e}")
                        result_queue.put(e)  # Send exception back

            except Exception as e:
                self._log(f"[Worker Error] Unhandled error in worker loop: {e}")

        # --- Cleanup ---
        self._stop_playwright()
        self._log("[Worker] Thread exiting.")

    def get_content(self, url: str, **kwargs) -> Optional[bytes]:
        """
        [Main Thread] Fetches content from a URL with flexible wait conditions.

        This method is synchronous and thread-safe. It sends the request
        to the background worker thread and blocks until the result is returned.

        Args:
            url (str): The URL to fetch.
            **kwargs: Flexible options passed to the worker, including:
                wait_until (str): The 'wait_until' strategy for page.goto().
                    One of: 'load', 'domcontentloaded', 'networkidle'.
                    Defaults to 'load'.
                wait_for_selector (Optional[str]): A CSS selector to wait for
                    after the page.goto() completes. (Best-effort wait).
                wait_for_timeout_s (Optional[int]): Specific timeout in seconds
                    for the 'wait_for_selector'. If None, defaults to the
                    main 'timeout_s' defined in __init__.
                scroll_pages (int): Number of pages to scroll.
                    > 0: Scroll down (content moves up).
                    < 0: Scroll up (content moves down).
                    0: No scrolling (default).
                post_extra_action (Callable[[Page], None]): The extra action after page loaded and scrolled.

        Returns:
            Optional[bytes]: The fetched page content (HTML or raw bytes).

        Raises:
            RuntimeError: If the worker thread is not running.
            TimeoutError: If the worker thread times out responding.
            PlaywrightError: If a non-recoverable error occurs (e.g., page load failure).
        """
        if not self.worker_thread or not self.worker_thread.is_alive():
            raise RuntimeError(
                "Playwright worker thread is not running. Fetcher may have been closed or failed to start.")

        # Create a one-time queue to get the result back
        result_queue: "queue.Queue[Any]" = queue.Queue(maxsize=1)

        wait_until_val = kwargs.get('wait_until', 'networkidle')
        wait_for_selector_val = kwargs.get('wait_for_selector', None)
        wait_for_timeout_s_val = kwargs.get('wait_for_timeout_s', None)
        scroll_pages_val = kwargs.get('scroll_pages', 0)
        post_extra_action = kwargs.get('post_extra_action', None)

        # --- Create the job payload with all wait parameters ---
        job_payload = {
            'url': url,
            'wait_until': wait_until_val,
            'wait_for_selector': wait_for_selector_val,
            'wait_for_timeout_ms': (wait_for_timeout_s_val * 1000) if wait_for_timeout_s_val is not None else None,
            'scroll_pages': scroll_pages_val,
            'post_extra_action': post_extra_action
        }

        # 添加任何其他传入的 kwargs (未来扩展性)
        # job_payload.update(kwargs)

        # Send the job to the worker thread
        self.job_queue.put(('get_content', job_payload, result_queue))

        # --- 动态计算等待超时时间 ---

        # 1. 主超时时间 (默认为 20s)
        base_timeout = self.timeout_ms / 1000

        # 2. 估算滚动所需时间: 滚动次数 * (最大抖动 1s + 网络等待 3s)
        # 注意: 即使网络等待超时，Playwright 也会在 3s 后返回，所以用 3s 是安全的估算。
        scroll_time_estimate = abs(scroll_pages_val) * (1 + 3)

        # 3. 总等待超时 = 主超时 + 滚动估时 + 额外缓冲
        wait_timeout = base_timeout + scroll_time_estimate + 5  # 5s 缓冲

        # 记录日志，以便调试
        self._log(f"[Main Thread] Calculated wait_timeout: {wait_timeout:.2f}s")

        try:
            # Block and wait for the result
            result = result_queue.get(timeout=wait_timeout)

            # If the worker sent back an exception, re-raise it in the main thread
            if isinstance(result, Exception):
                self._log(f"[Main Thread] Error received from worker for {url}")
                raise result

            return result
        except queue.Empty:
            self._log(f"[Main Thread] Timeout waiting for worker response for {url}")
            raise TimeoutError(f"Playwright job for {url} timed out after {wait_timeout}s")

    def _fetch_page_content(self, job_payload: dict) -> Optional[bytes]:
        """
        [Worker Thread] The *actual* browser logic, now with flexible
        and best-effort waiting.
        """
        # --- 1. Unpack Job Payload ---
        url = job_payload['url']
        wait_until = job_payload.get('wait_until', 'load')
        wait_for_selector = job_payload.get('wait_for_selector')
        scroll_pages = job_payload.get('scroll_pages', 0)
        post_extra_action = job_payload.get('post_extra_action', None)

        # Use specific selector timeout, or fall back to the main timeout
        selector_timeout_ms = job_payload.get('wait_for_timeout_ms') or self.timeout_ms

        context = None
        page = None
        try:
            # --- 2. Create Context and Page ---
            context_options = {
                "user_agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'
            }
            if self.proxy_config:
                context_options["proxy"] = self.proxy_config

            context = self.browser.new_context(**context_options)
            page = context.new_page()

            # --- 3. Apply Stealth (if enabled) ---
            if self.stealth_mode:
                if Stealth:  # v2
                    Stealth().apply_stealth_sync(page)
                elif sync_stealth:  # v1
                    sync_stealth(page)
                else:  # Fallback
                    page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            else:
                page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

            # --- 4. Main Page Navigation (Hard Fail) ---
            response = page.goto(
                url,
                timeout=self.timeout_ms,
                wait_until=wait_until
            )

            if self.pause_browser:
                page.pause()

            # This is a hard failure. If the page didn't load, we must error.
            if not response or not response.ok:
                status = response.status if response else 'N/A'
                raise PlaywrightError(f"Failed to get valid response. Status: {status}")

            self._log(f"[Worker] page.goto() successful for {url} (Status: {response.status}, Wait: {wait_until})")

            # --- 5. Best-Effort Selector Wait (Soft Fail) ---
            if wait_for_selector:
                self._log(f"[Worker] Waiting for selector '{wait_for_selector}' (timeout: {selector_timeout_ms}ms)...")
                try:
                    page.wait_for_selector(
                        wait_for_selector,
                        state='visible',
                        timeout=selector_timeout_ms
                    )
                    self._log(f"[Worker] Found selector '{wait_for_selector}'.")
                except Exception as e:
                    # This is the "best-effort" logic. We log the warning
                    # but DO NOT raise the error.
                    self._log(f"[Worker Warning] Timeout or error waiting for selector '{wait_for_selector}': {str(e)}")
                    self._log("[Worker] Proceeding to extract content anyway (best-effort).")

            # --- 5.5. Handle Scrolling (健壮且带抖动的版本) ---
            if scroll_pages != 0:
                scroll_direction = 'down' if scroll_pages > 0 else 'up'
                self._log(
                    f"[Worker] Scrolling {abs(scroll_pages)} pages {scroll_direction} (robust + jitter mode)...")

                js_scroll_distance = "window.innerHeight" if scroll_pages > 0 else "-window.innerHeight"

                # 智能等待（networkidle）的超时时间
                scroll_network_timeout = 3000

                for i in range(abs(scroll_pages)):
                    # --- 1. 执行滚动 ---
                    page.evaluate(f"window.scrollBy(0, {js_scroll_distance});")
                    self._log(f"[Worker] Scroll {i + 1}/{abs(scroll_pages)} executed.")

                    # --- 2. 增加“人性化”时间抖动 ---
                    # 模拟人类滚动后，视线移动或反应的短暂延迟
                    # 随机在 300ms 到 1000ms (0.3s到1s) 之间暂停
                    jitter_ms = random.randint(300, 1000)
                    self._log(f"[Worker] Pausing for {jitter_ms}ms (human jitter)...")
                    page.wait_for_timeout(jitter_ms)

                    # --- 3. 智能等待（捕获异常） ---
                    # 抖动暂停后，我们再开始等待网络加载
                    # 这是你问题的核心：用 try...except 包裹
                    self._log(
                        f"[Worker] Jitter complete. Waiting for network idle (max {scroll_network_timeout}ms)...")
                    try:
                        # 尝试等待网络空闲
                        page.wait_for_load_state('networkidle', timeout=scroll_network_timeout)
                        self._log(f"[Worker] Scroll {i + 1} network is idle.")
                    except Exception as e:
                        # 【关键】如果超时（或其他错误），我们捕获它，打印日志，但不让程序崩溃
                        # 循环会继续执行下一次滚动
                        self._log(
                            f"[Worker Warning] Network not idle after scroll {i + 1} (timeout/error: {e}). Continuing loop.")

                self._log(f"[Worker] Finished all scrolling.")
                try:
                    # 等待所有懒加载的内容完成
                    page.wait_for_load_state('networkidle', timeout=5000)
                    self._log("[Worker] Network is idle after scrolling.")
                except Exception:
                    self._log(
                        "[Worker Warning] Network did not become idle after scrolling (5s timeout). Proceeding anyway.")

            if post_extra_action:
                post_extra_action(page)

            # --- 6. Extract Content ---
            # This code is now reached even if the selector times out.
            content_bytes: Optional[bytes] = None
            # Maybe there's re-direction after scrolling or post extra actions.
            if 200 <= response.status < 300:
                if self.render_page:
                    self._log("[Worker] Rendering page.content()...")
                    content_str = page.content()
                    content_bytes = content_str.encode('utf-8')
                else:
                    self._log("[Worker] Getting raw response.body()...")
                    content_bytes = response.body()
            else:
                self._log('[Worker] Detect error response when getting content.')

            # safe_filename = re.sub(r'[^\w\s-]', '', url)[:50]
            # dump_filename = f'dump_{safe_filename}.html'

            # self._log(f"[Worker DEBUG] Dumping content to {dump_filename}")
            # with open(dump_filename, 'wb') as f:
            #     f.write(content_bytes)

            context.close()
            return content_bytes

        except PlaywrightTimeoutError:
            if self._log:
                self._log(f"[Warning] Page.goto timed out for {job_payload['url']}. "
                          f"Attempting to grab content anyway.")

            # !!! 关键：超时了，但我们不在乎，我们直接尝试获取内容
            # 如果页面（如HTML）已经存在，这将成功返回
            # 确保 'page' 对象存在
            if page:
                content = page.content()
            else:
                content = None

            if not content:
                # 如果内容为空或无效，才真正抛出异常
                raise ValueError(f"Timeout occurred AND page content was empty/invalid.")

            # 如果我们拿到了内容，就假装什么都没发生，返回它
            return content.encode('utf-8')

        except Exception as e:
            # This outer catch block handles hard failures (like page.goto)
            # or failures during page.content()
            self._log(f"[Worker Error] _fetch_page_content failed for {url}: {e}")
            if context:
                context.close()
            raise e  # Re-raise to send back to main thread

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