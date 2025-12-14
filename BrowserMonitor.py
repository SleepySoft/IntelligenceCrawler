# -*- coding: utf-8 -*-
"""
Playwright Resource Monitor
===========================

This module provides a thread-safe mechanism to track Playwright Browser instances
and detect resource leaks (zombie processes) upon application exit.

It implements the Proxy Pattern to wrap the underlying Playwright Browser object,
intercepting the `.close()` method to track lifecycle events while transparently
forwarding all other method calls (like `.new_page()`, `.new_context()`) to the
real object.

Key Components:
    - BrowserTracker: A thread-safe singleton registry for active browsers.
    - AutoTrackedBrowser: A proxy wrapper that manages registration/deregistration.

Usage in main application:
    from browser_monitor import AutoTrackedBrowser

    # 1. Launch real browser
    real_browser = playwright.chromium.launch(headless=True)

    # 2. Wrap it
    browser = AutoTrackedBrowser(real_browser)

    # 3. Use 'browser' exactly as you would use 'real_browser'
    page = browser.new_page()

    # 4. Closing handles cleanup in the tracker
    browser.close()
"""

import threading
import traceback
import time
import atexit
import datetime
from typing import Dict, Any, Optional


class BrowserTracker:
    """
    A thread-safe global registry for monitoring active Playwright browser instances.

    This class maintains a record of all `AutoTrackedBrowser` instances that have
    been created but not yet closed. It registers an `atexit` handler to automatically
    report leaks and attempt cleanup when the Python interpreter shuts down.

    Attributes:
        _active_browsers (Dict[int, AutoTrackedBrowser]): A dictionary mapping object IDs 
            to browser wrapper instances.
        _lock (threading.Lock): A mutex to ensure thread-safe access to the registry.
    """
    _active_browsers: Dict[int, 'AutoTrackedBrowser'] = {}
    _lock = threading.Lock()

    @classmethod
    def register(cls, wrapper: 'AutoTrackedBrowser') -> None:
        """
        Registers a new browser wrapper instance.

        Args:
            wrapper (AutoTrackedBrowser): The browser wrapper to track.
        """
        with cls._lock:
            cls._active_browsers[id(wrapper)] = wrapper
            # Optional: Log allocation for debugging
            # print(f"[Monitor] Browser allocated. Active count: {len(cls._active_browsers)}")

    @classmethod
    def unregister(cls, wrapper: 'AutoTrackedBrowser') -> None:
        """
        Unregisters a browser wrapper instance (called when .close() is invoked).

        Args:
            wrapper (AutoTrackedBrowser): The browser wrapper to remove.
        """
        with cls._lock:
            if id(wrapper) in cls._active_browsers:
                del cls._active_browsers[id(wrapper)]
                # Optional: Log release for debugging
                # print(f"[Monitor] Browser released. Active count: {len(cls._active_browsers)}")

    @classmethod
    def report_leaks(cls) -> None:
        """
        Analyzes the registry for unclosed browsers and prints a report.

        This method is automatically called via `atexit` when the program terminates.
        If leaks are detected, it attempts to force-close the underlying real browsers
        to prevent zombie processes.

        Usage:
            # Usually called automatically, but can be invoked manually for debugging:
            BrowserTracker.report_leaks()
        """
        with cls._lock:
            count = len(cls._active_browsers)
            if count == 0:
                print("\n[Monitor] ✅ CLEAN EXIT: No browser leaks detected.")
            else:
                print(f"\n[Monitor] ⚠️ LEAK DETECTED: {count} browser(s) were NOT closed!")

                now = time.time()

                for i, (bid, wrapper) in enumerate(cls._active_browsers.items(), 1):
                    duration_sec = now - wrapper._start_ts
                    duration_str = str(datetime.timedelta(seconds=int(duration_sec)))

                    print(f"\n--- 🔴 Leaked Browser #{i} ---")
                    print(f"Start Time:  {wrapper.created_at_str}")
                    print(f"Duration:    {duration_str} (Active for this long)")
                    print(f"Location:    Created at the following code path:")
                    print("-" * 40)
                    print(wrapper.creation_stack.strip())
                    print("-" * 40)

                    try:
                        print(">> Attempting force close...")
                        wrapper.force_close_real()
                        print(">> Force close successful.")
                    except Exception as e:
                        print(f"!! Force close failed: {e}")


# Register the cleanup hook immediately upon module import
atexit.register(BrowserTracker.report_leaks)


class AutoTrackedBrowser:
    """
    A Proxy Wrapper for the Playwright Browser object.

    This class wraps a standard Playwright `Browser` instance. It behaves exactly
    like the original object (via dynamic attribute forwarding) but intercepts the
    `.close()` method to update the `BrowserTracker`.

    It captures the stack trace upon instantiation to help identify where a leaked
    browser was originally created.

    Args:
        real_browser (Browser): The actual Playwright Browser instance returned 
                                by `chromium.launch()`.

    Attributes:
        created_at (str): Timestamp of creation.
        creation_stack (str): The call stack at the moment of creation.

    Usage:
        >> # Standard Playwright setup
        >> from playwright.sync_api import sync_playwright
        >> p = sync_playwright().start()

        >> # Launch and Wrap
        >> real_browser = p.chromium.launch(headless=True)
        >> browser = AutoTrackedBrowser(real_browser)

        >> # Use exactly like a normal browser
        >> page = browser.new_page()
        >> page.goto("http://example.com")

        >> # This will close the real browser AND unregister it from the tracker
        >> browser.close()
    """

    def __init__(self, real_browser: Any):
        self._browser = real_browser

        self.created_at_str = time.strftime("%Y-%m-%d %H:%M:%S")
        self._start_ts = time.time()

        # Capture the last few frames of the stack trace to identify the caller
        self.creation_stack = "".join(traceback.format_stack(limit=10)[:-1])

        # Automatically register with the global tracker
        BrowserTracker.register(self)

    def close(self) -> None:
        """
        Closes the browser and unregisters it from the tracker.

        This overrides the underlying `close` method to ensure the tracker
        is updated before the actual resource is released.
        """
        # 1. Remove from active registry
        BrowserTracker.unregister(self)
        # 2. Perform the actual close operation
        return self._browser.close()

    def force_close_real(self) -> None:
        """
        Directly closes the underlying browser instance.

        This method is primarily used by the `BrowserTracker` during emergency
        cleanup (atexit) to ensure zombie processes are killed.
        """
        return self._browser.close()

    def __getattr__(self, name: str) -> Any:
        """
        Dynamic attribute dispatcher.

        If a method or attribute (e.g., 'new_page', 'new_context', 'version') 
        is not found on this wrapper class, Python calls this method.
        We transparently forward the request to the underlying `_browser` object.

        Args:
            name (str): The name of the attribute/method being accessed.

        Returns:
            The attribute or method from the real Playwright browser instance.
        """
        return getattr(self._browser, name)

    def __repr__(self) -> str:
        return f"<AutoTrackedBrowser wrapping {self._browser!r}>"
