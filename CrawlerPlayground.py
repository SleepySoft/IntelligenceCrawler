#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crawler Playground (v4.0)
A GUI application for discovering, fetching, and extracting web content
using various strategies and libraries.
"""
import os
import sys
import datetime
import traceback
from collections import deque
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional

# --- Core Component Imports ---

try:
    from IntelligenceCrawler.Fetcher import Fetcher, PlaywrightFetcher, RequestsFetcher
except ImportError:
    print("!!! CRITICAL: Could not import Fetcher classes.")


    # Mock classes to allow UI to load
    class Fetcher:
        pass


    class PlaywrightFetcher:
        pass


    class RequestsFetcher:
        pass

try:
    from IntelligenceCrawler.Discoverer import IDiscoverer, SitemapDiscoverer, RSSDiscoverer, ListPageDiscoverer
except ImportError:
    print("!!! CRITICAL: Could not import Discoverer classes.")


    class IDiscoverer:
        pass


    class SitemapDiscoverer:
        pass


    class RSSDiscoverer:
        pass

    class ListPageDiscoverer:
        pass

try:
    from IntelligenceCrawler.Extractor import (
        IExtractor, TrafilaturaExtractor, ReadabilityExtractor,
        Newspaper3kExtractor, GenericCSSExtractor, Crawl4AIExtractor, ExtractionResult
)

    # Store imported classes for factory
    EXTRACTOR_MAP = {
        "Trafilatura": TrafilaturaExtractor,
        "Readability": ReadabilityExtractor,
        "Newspaper3k": Newspaper3kExtractor,
        "Generic CSS": GenericCSSExtractor,
        "Crawl4AI": Crawl4AIExtractor,
    }
except ImportError:
    print("!!! CRITICAL: Could not import Extractor classes.")
    EXTRACTOR_MAP = {}


    class IExtractor:
        pass

try:
    from dateutil.parser import parse as date_parse
except ImportError:
    print("!!! IMPORT ERROR: 'python-dateutil' not found.")
    date_parse = None

# --- Playwright Imports (with detailed error checking) ---
try:
    from playwright.sync_api import sync_playwright, Error as PlaywrightError
except ImportError:
    print("!!! IMPORT ERROR: Could not import 'playwright.sync_api'.")
    sync_playwright = None
    PlaywrightError = None
except Exception as e:
    sync_playwright = None
    PlaywrightError = None

# --- Smart Import for playwright-stealth (v1 and v2) ---
sync_stealth = None  # For v2.x
Stealth = None  # For v1.x
try:
    from playwright_stealth import sync_stealth

    print("Imported playwright-stealth v2.x ('sync_stealth') successfully.")
except ImportError:
    try:
        from playwright_stealth.stealth import Stealth

        print("Imported playwright-stealth v1.x ('Stealth') successfully.")
    except ImportError:
        print("!!! IMPORT ERROR: Could not import 'playwright_stealth' v1 or v2.")
    except Exception:
        pass
except Exception:
    pass

# --- PyQt5 Imports ---
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QTreeWidget, QTreeWidgetItem, QSplitter,
    QTextEdit, QStatusBar, QTabWidget, QLabel, QFrame, QComboBox,
    QDateEdit, QCheckBox, QToolBar, QSizePolicy, QSpinBox,
    QMenu, QAction, QFileDialog, QFormLayout, QGridLayout
)
from PyQt5.QtCore import (
    Qt, QRunnable, QThreadPool, QObject, pyqtSignal, QTimer, QSettings
)
from PyQt5.QtGui import QFont, QIcon, QCursor

# --- PyQtWebEngine Imports ---
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    from PyQt5.QtCore import QUrl
except ImportError:
    print("Error: PyQtWebEngine not found. Web preview will be disabled.")
    QWebEngineView = None
    QUrl = None


SETTING_ORG = 'SleepySoft'
SETTING_APP = 'CrawlerPlayground'


# =============================================================================
#
# SECTION 1: Utility Factories
# (To create instances inside workers)
#
# =============================================================================
def create_fetcher_instance(fetcher_name: str,
                            log_callback,
                            proxy: Optional[str] = None,
                            timeout: int = 10,  # <-- NEW (in seconds)
                            **kwargs) -> Fetcher:
    """
    Factory to create a fetcher instance based on its name.
    (工厂函数：根据名称创建 fetcher 实例。)
    """
    stealth_mode = "Stealth" in fetcher_name
    pause = kwargs.get('pause_browser', False)
    render = kwargs.get('render_page', False)

    # We assume the Fetcher classes have been modified to accept 'timeout'
    # in their __init__ and apply it appropriately (e.g., to self.timeout).
    # (我们假设 Fetcher 类已被修改以在 __init__ 中接受 'timeout'。)

    if "Playwright" in fetcher_name:
        if not sync_playwright: raise ImportError("Playwright not installed.")
        if stealth_mode and (not sync_stealth and not Stealth):
            raise ImportError("Playwright-Stealth not installed.")

        return PlaywrightFetcher(
            log_callback=log_callback,
            proxy=proxy,
            timeout_s=timeout,  # <-- NEW (pass ms)
            stealth=stealth_mode,
            pause_browser=pause,
            render_page=render
        )
    else:  # "Simple (Requests)"
        return RequestsFetcher(
            log_callback=log_callback,
            proxy=proxy,
            timeout_s=timeout
        )


def create_discoverer_instance(discoverer_name: str, fetcher: Fetcher, log_callback, **kwargs) -> IDiscoverer:
    """Factory to create a discoverer instance based on its name."""
    if discoverer_name == "Sitemap":
        if 'SitemapDiscoverer' not in globals(): raise ImportError("SitemapDiscoverer not found.")
        return SitemapDiscoverer(fetcher, verbose=True)
    elif discoverer_name == "RSS":
        if 'RSSDiscoverer' not in globals(): raise ImportError("RSSDiscoverer not found.")
        return RSSDiscoverer(fetcher, verbose=True)
    elif discoverer_name == "Smart Analysis":
        if 'ListPageDiscoverer' not in globals(): raise ImportError("ListPageDiscoverer not found.")
        # 从 kwargs 获取 ai_signature
        ai_sig = kwargs.get('ai_signature', None)
        return ListPageDiscoverer(fetcher, verbose=True, ai_signature=ai_sig)
    else:
        raise ValueError(f"Unknown discoverer_name: {discoverer_name}")


def create_extractor_instance(extractor_name: str, log_callback) -> IExtractor:
    """Factory to create an extractor instance based on its name."""
    if extractor_name not in EXTRACTOR_MAP:
        raise ImportError(f"Extractor '{extractor_name}' not found or failed to import.")

    ExtractorClass = EXTRACTOR_MAP[extractor_name]
    return ExtractorClass(verbose=True)



# =============================================================================
#
# Reusable Fetcher Configuration Widget
#
# =============================================================================

class FetcherConfigWidget(QWidget):
    """
    A reusable widget encapsulating all UI controls for fetcher configuration.
    (一个可复用的窗口部件，封装了所有用于 fetcher 配置的 UI 控件。)
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # --- Create Widgets ---
        self.fetcher_combo = QComboBox()
        self.fetcher_combo.addItems([
            "Simple (Requests)",
            "Advanced (Playwright)",
            "Stealth (Playwright)"
        ])
        if not sync_playwright:
            self.fetcher_combo.model().item(1).setEnabled(False)
            self.fetcher_combo.model().item(2).setEnabled(False)
        if not sync_stealth and not Stealth:
            self.fetcher_combo.model().item(2).setEnabled(False)

        self.proxy_input = QLineEdit()
        self.proxy_input.setPlaceholderText("e.g., http://user:pass@host:port")

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(1, 300)
        self.timeout_spin.setValue(10)
        self.timeout_spin.setSuffix(" s")

        self.pause_check = QCheckBox("Pause")
        self.pause_check.setToolTip("Pauses Playwright (in headful mode) for debugging.")

        self.render_check = QCheckBox("Render")
        self.render_check.setToolTip("Fetches final rendered HTML (slower) vs. raw response (faster).")

        # --- New Playwright-specific widgets ---
        self.wait_until_label = QLabel("WaitUntil:")
        self.wait_until_combo = QComboBox()
        self.wait_until_combo.addItems(['networkidle', 'load', 'domcontentloaded', 'commit'])
        self.wait_until_combo.setToolTip("Playwright page.goto() wait_until option.")

        self.wait_selector_label = QLabel("Wait Selector:")
        self.wait_selector_input = QLineEdit()
        self.wait_selector_input.setPlaceholderText("e.g., #main-content")
        self.wait_selector_input.setToolTip("Playwright: wait for this selector to appear before returning.")

        # --- Layout ---
        # We use QGridLayout for precise alignment
        grid_layout = QGridLayout(self)
        grid_layout.setContentsMargins(0, 0, 0, 0)  # No external margins

        # Row 0
        grid_layout.addWidget(QLabel("Fetcher:"), 0, 0)
        grid_layout.addWidget(self.fetcher_combo, 0, 1)
        grid_layout.addWidget(QLabel("Timeout:"), 0, 2)
        grid_layout.addWidget(self.timeout_spin, 0, 3)
        grid_layout.addWidget(self.pause_check, 0, 4)
        grid_layout.addWidget(self.render_check, 0, 5)

        # Row 1
        grid_layout.addWidget(QLabel("Proxy:"), 1, 0)
        grid_layout.addWidget(self.proxy_input, 1, 1, 1, 5)  # Spans 5 columns

        # Row 2 (Playwright options)
        grid_layout.addWidget(self.wait_until_label, 2, 0)
        grid_layout.addWidget(self.wait_until_combo, 2, 1)
        grid_layout.addWidget(self.wait_selector_label, 2, 2)
        grid_layout.addWidget(self.wait_selector_input, 2, 3, 1, 3)  # Spans 3 columns

        # Set column stretch
        grid_layout.setColumnStretch(1, 1)
        grid_layout.setColumnStretch(3, 1)

        # --- Connect Signals ---
        self.fetcher_combo.currentTextChanged.connect(self._on_fetcher_changed)

        # --- Initial State ---
        self._on_fetcher_changed(self.fetcher_combo.currentText())

    def _on_fetcher_changed(self, text: str):
        """Show/hide Playwright options based on fetcher selection."""
        is_playwright = "Playwright" in text
        self.wait_until_label.setVisible(is_playwright)
        self.wait_until_combo.setVisible(is_playwright)
        self.wait_selector_label.setVisible(is_playwright)
        self.wait_selector_input.setVisible(is_playwright)

        # Playwright-specific checks
        self.pause_check.setEnabled(is_playwright)
        self.render_check.setEnabled(is_playwright)
        if not is_playwright:
            self.pause_check.setChecked(False)
            self.render_check.setChecked(False)  # Requests cannot render

    def set_defaults(self, fetcher_name: str, timeout: int, render: bool, proxy: str = ""):
        """Set the default values for the widget."""
        self.fetcher_combo.setCurrentText(fetcher_name)
        self.timeout_spin.setValue(timeout)
        self.render_check.setChecked(render)
        self.proxy_input.setText(proxy)

        # Ensure correct state is triggered
        self._on_fetcher_changed(fetcher_name)

    def set_render_tooltip(self, tooltip: str):
        """Allow parent to override the 'Render' checkbox tooltip."""
        self.render_check.setToolTip(tooltip)

    def get_config(self) -> Dict[str, Any]:
        """Return the current configuration as a dictionary."""
        fetcher_name = self.fetcher_combo.currentText()
        is_playwright = "Playwright" in fetcher_name

        return {
            'fetcher_name': fetcher_name,
            'proxy': self.proxy_input.text().strip() or None,
            'timeout': self.timeout_spin.value(),
            'pause': self.pause_check.isChecked() and is_playwright,
            'render': self.render_check.isChecked() and is_playwright,
            'wait_until': self.wait_until_combo.currentText() if is_playwright else 'networkidle',
            'wait_for_selector': self.wait_selector_input.text().strip() or None if is_playwright else None,
        }


# =============================================================================
#
# SECTION 2: PyQt5 Threading Workers (QRunnable)
# (Refactored to be generic)
#
# =============================================================================

class WorkerSignals(QObject):
    """Defines the signals available from a running worker thread."""
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(str)  # For sending log messages


class ChannelDiscoveryWorker(QRunnable):
    """Worker thread for Stage 1: Discovering all channels."""

    def __init__(self,
                 discoverer_name: str,
                 fetcher_name: str,
                 entry_point: Any,
                 start_date: datetime.datetime,
                 end_date: datetime.datetime,
                 proxy: Optional[str],
                 timeout: int,
                 pause_browser: bool,
                 render_page: bool,
                 ai_signature: Optional[str] = None):
        super(ChannelDiscoveryWorker, self).__init__()
        self.discoverer_name = discoverer_name
        self.fetcher_name = fetcher_name
        self.entry_point = entry_point
        self.start_date = start_date
        self.end_date = end_date
        self.proxy = proxy
        self.timeout = timeout
        self.pause_browser = pause_browser
        self.ai_signature = ai_signature
        self.render_page = render_page  # Note: This is for XML, may break parsing
        self.signals = WorkerSignals()

    def run(self):
        fetcher: Optional[Fetcher] = None
        try:
            log_callback = self.signals.progress.emit

            # 1. Create Fetcher
            # Note: Forcing render_page=False for discovery, as it's
            # almost always parsing XML/Text, not rendered HTML.
            if self.render_page:
                log_callback("[Warning] 'Render Page' is enabled for Discovery, " \
                             "this may fail XML/RSS parsing. Forcing False.")

            fetcher = create_fetcher_instance(
                self.fetcher_name,
                log_callback,
                proxy=self.proxy,
                timeout=self.timeout,
                pause_browser=self.pause_browser,
                render_page=False  # Force False for discovery
            )

            # 2. Create Discoverer
            discoverer = create_discoverer_instance(
                self.discoverer_name,
                fetcher,
                log_callback,
                ai_signature=self.ai_signature
            )

            # 3. Do the work
            channel_list = discoverer.discover_channels(
                self.entry_point,
                start_date=self.start_date,
                end_date=self.end_date
            )
            self.signals.result.emit(channel_list)

        except Exception as e:
            ex_type, ex_value, tb_str = sys.exc_info()
            self.signals.error.emit((str(ex_type), str(e), traceback.format_exc()))
        finally:
            if fetcher: fetcher.close()
            self.signals.finished.emit()


class ArticleListWorker(QRunnable):
    """Worker thread for Stage 2 (Lazy Loading): Gets articles for one channel."""

    # REFACTORED: Now accepts names
    def __init__(self,
                 discoverer_name: str,
                 fetcher_name: str,
                 channel_url: str,
                 proxy: Optional[str],
                 timeout: int,
                 pause_browser: bool,
                 render_page: bool):
        super(ArticleListWorker, self).__init__()
        self.discoverer_name = discoverer_name
        self.fetcher_name = fetcher_name
        self.channel_url = channel_url
        self.proxy = proxy
        self.timeout = timeout
        self.pause_browser = pause_browser
        self.render_page = render_page
        self.signals = WorkerSignals()

    def run(self):
        fetcher: Optional[Fetcher] = None
        try:
            log_callback = self.signals.progress.emit

            # 1. Create Fetcher
            if self.render_page:
                log_callback("[Warning] 'Render Page' is enabled for Article List, " \
                             "this may fail XML/RSS parsing. Forcing False.")

            fetcher = create_fetcher_instance(
                self.fetcher_name,
                log_callback,
                proxy=self.proxy,
                timeout=self.timeout,
                pause_browser=self.pause_browser,
                render_page=False  # Force False for discovery
            )

            # 2. Create Discoverer
            discoverer = create_discoverer_instance(self.discoverer_name, fetcher, log_callback)

            # 3. Do the work
            article_list = discoverer.get_articles_for_channel(self.channel_url)
            self.signals.result.emit({
                'channel_url': self.channel_url,
                'articles': article_list
            })
        except Exception as e:
            ex_type, ex_value, tb_str = sys.exc_info()
            self.signals.error.emit((str(ex_type), str(e), traceback.format_exc()))
        finally:
            if fetcher: fetcher.close()
            self.signals.finished.emit()


class ChannelSourceWorker(QRunnable):
    """
    Worker thread to fetch raw channel content (e.g., XML) for the text viewer.
    (REFACTORED from XmlContentWorker)
    """

    def __init__(self,
                 discoverer_name: str,  # Discoverer needed for get_content_str
                 fetcher_name: str,
                 url: str,
                 proxy: Optional[str],
                 timeout: int,
                 pause_browser: bool,
                 render_page: bool):
        super(ChannelSourceWorker, self).__init__()
        self.discoverer_name = discoverer_name
        self.fetcher_name = fetcher_name
        self.url = url
        self.proxy = proxy
        self.timeout = timeout
        self.pause_browser = pause_browser
        self.render_page = render_page
        self.signals = WorkerSignals()

    def run(self):
        fetcher: Optional[Fetcher] = None
        try:
            log_callback = self.signals.progress.emit

            # 1. Create Fetcher
            if self.render_page:
                log_callback("[Warning] 'Render Page' is enabled for Channel Source, "
                             "this may fail XML/RSS parsing. Forcing False.")

            fetcher = create_fetcher_instance(
                self.fetcher_name,
                log_callback,
                proxy=self.proxy,
                timeout=self.timeout,
                pause_browser=self.pause_browser,
                render_page=False  # Force False for discovery
            )

            # 2. Create Discoverer (only for its .get_content_str method)
            discoverer = create_discoverer_instance(self.discoverer_name, fetcher, log_callback)

            # 3. Do the work (using the generic interface method)
            content_string = discoverer.get_content_str(self.url)
            self.signals.result.emit(content_string)
        except Exception as e:
            ex_type, ex_value, tb_str = sys.exc_info()
            self.signals.error.emit((str(ex_type), str(e), traceback.format_exc()))
        finally:
            if fetcher: fetcher.close()
            self.signals.finished.emit()


# --- NEW WORKER FOR EXTRACTION (REQ 2e) ---
class ExtractionWorker(QRunnable):
    """Worker thread for Stage 3: Fetching and Extracting a single article."""

    def __init__(self,
                 fetcher_config: dict,  # <-- 接收整个配置字典
                 extractor_name: str,
                 url_to_extract: str,
                 extractor_kwargs: dict):
        super(ExtractionWorker, self).__init__()
        self.fetcher_config = fetcher_config  # 存储字典
        self.extractor_name = extractor_name
        self.url_to_extract = url_to_extract
        self.extractor_kwargs = extractor_kwargs
        self.signals = WorkerSignals()

    def run(self):
        fetcher: Optional[Fetcher] = None
        try:
            log_callback = self.signals.progress.emit

            # --- [MODIFIED] ---
            # 1. Create Fetcher
            fetcher_name = self.fetcher_config.get('fetcher_name')
            log_callback(f"Fetching {self.url_to_extract} using {fetcher_name}...")

            fetcher = create_fetcher_instance(
                fetcher_name,
                log_callback,
                proxy=self.fetcher_config.get('proxy'),
                timeout=self.fetcher_config.get('timeout'),
                pause_browser=self.fetcher_config.get('pause'),
                render_page=self.fetcher_config.get('render')
            )

            # 2. Get Content
            # 准备传递给 get_content() 的参数
            wait_until_val = self.fetcher_config.get('wait_until', 'networkidle')
            wait_for_selector_val = self.fetcher_config.get('wait_for_selector')
            # 使用主超时作为 'wait_for_timeout_s'
            wait_for_timeout_s_val = self.fetcher_config.get('timeout')

            # 假设 fetcher.get_content 签名已更新
            content_bytes = fetcher.get_content(
                self.url_to_extract,
                wait_until=wait_until_val,
                wait_for_selector=wait_for_selector_val,
                wait_for_timeout_s=wait_for_timeout_s_val
            )

            if not content_bytes:
                raise ValueError("Failed to fetch content (returned None).")
            # --- [END MODIFICATION] ---

            log_callback(f"Fetched {len(content_bytes)} bytes. Extracting using {self.extractor_name}...")

            # 3. Create Extractor
            extractor = create_extractor_instance(self.extractor_name, log_callback)

            # 4. Do the work
            markdown_result = extractor.extract(
                content_bytes,
                self.url_to_extract,
                **self.extractor_kwargs
            )
            self.signals.result.emit(markdown_result)

        except Exception as e:
            ex_type, ex_value, tb_str = sys.exc_info()
            self.signals.error.emit((str(ex_type), str(e), traceback.format_exc()))
        finally:
            if fetcher: fetcher.close()
            self.signals.finished.emit()




# =============================================================================
#
# SECTION 3: PyQt5 Main Application (GUI Refactored)
#
# =============================================================================

CODE_TEMPLATE = """# CrawlerGenerated.py - This code is generated by CrawlerPlayground

from IntelligenceCrawler.CrawlPipeline import *

def run_pipeline():
    # === 1. Initialize Components ===
    {code_d_fetcher}
    {code_e_fetcher}
    {code_discoverer}
    {code_extractor}
    
    # === 2. Define Parameters ===
    {parameters}
    {channel_filter_list}

    # === 3. Build pipeline ===
    pipeline = CrawlPipeline(
        d_fetcher=d_fetcher,
        discoverer=discoverer,
        e_fetcher=e_fetcher,
        extractor=extractor,
        log_callback=log_cb
    )

    # Step 1: Discover all channels
    pipeline.discover_channels(entry_point, start_date, end_date)

    # Step 2: Discover and fetch articles (populates pipeline.contents)
    pipeline.discover_articles(channel_filter=partial(
        common_channel_filter, channel_filter_list=channel_filter_list))

    # Step 3: Extract content and run handlers
    pipeline.extract_articles(
        article_filter=lambda url: True,
        content_handler=save_article_to_disk,
        exception_handler=lambda url, exception: None,
        **extractor_kwargs
    )


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
"""


# --- REQ 4: New Name ---
class CrawlerPlaygroundApp(QMainWindow):
    """
    Main application window for the Crawler Playground.
    Provides a UI to test Discoverer, Fetcher, and Extractor combinations.
    """

    def __init__(self):
        super().__init__()

        # --- Internal State ---
        self.discoverer_name: str = "Sitemap"

        self.discovery_fetcher_widget: Optional[FetcherConfigWidget] = None
        self.article_fetcher_widget: Optional[FetcherConfigWidget] = None

        # --- [NEW] UI attribute placeholders ---
        self.ai_signature_label: Optional[QLabel] = None
        self.ai_signature_input: Optional[QLineEdit] = None
        self.css_selector_label: Optional[QLabel] = None
        self.css_selector_input: Optional[QLineEdit] = None
        # --- [END NEW] ---

        # Cache for the *actual* entry_point (str or List[str])
        # used in the last discovery run.
        self.last_used_entry_point: Any = None

        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(QThreadPool.globalInstance().maxThreadCount() // 2 + 1)

        self.channel_item_map: Dict[str, QTreeWidgetItem] = {}
        self.log_history_view: Optional[QTextEdit] = None

        # --- NEW: Settings for URL History ---
        self.URL_HISTORY_KEY = "discovery_url_history"
        self.MAX_URL_HISTORY = 25
        # --- NEW: Settings for Proxy History (REQ 1) ---
        self.DISCOVERY_PROXY_KEY = "discovery_proxy_history"
        self.ARTICLE_PROXY_KEY = "article_proxy_history"

        # --- Initialize UI ---
        self.init_ui()
        self._load_url_history()

        settings = QSettings(SETTING_ORG, SETTING_APP)
        saved_d_proxy = settings.value(self.DISCOVERY_PROXY_KEY, "", type=str)
        saved_a_proxy = settings.value(self.ARTICLE_PROXY_KEY, "", type=str)

        if self.discovery_fetcher_widget:
            self.discovery_fetcher_widget.proxy_input.setText(saved_d_proxy)
        if self.article_fetcher_widget:
            # 在 Article 侧设置默认值和加载的代理
            self.article_fetcher_widget.set_defaults(
                fetcher_name="Stealth (Playwright)",
                timeout=20,
                render=True,
                proxy=saved_a_proxy
            )

        self.connect_signals()  # Centralize signal connections

        # --- Set initial visibility for dynamic UI ---
        self.update_generated_code()  # Show initial code
        self._update_discoverer_options_ui(self.discoverer_combo.currentText())
        self._update_extractor_options_ui(self.extractor_combo.currentText())

        self.setWindowTitle("Crawler Playground (v4.0)")
        self.setWindowIcon(QIcon.fromTheme("internet-web-browser"))

        # --- MODIFICATION (REQ 3): HiDPI-aware window sizing ---
        # Replace fixed self.setGeometry(100, 100, 1400, 900)
        try:
            # Get 80% of the *available* screen geometry (respects taskbars)
            screen_geometry = QApplication.primaryScreen().availableGeometry()
            self.setGeometry(
                screen_geometry.x() + screen_geometry.width() * 0.1,
                screen_geometry.y() + screen_geometry.height() * 0.1,
                screen_geometry.width() * 0.8,
                screen_geometry.height() * 0.8
            )
        except Exception as e:
            # Fallback for any error (e.g., no screen found)
            print(f"Could not get screen geometry, falling back to fixed size. Error: {e}")
            self.setGeometry(100, 100, 1400, 900)
        # --- END MODIFICATION ---

        self.update_generated_code()  # Show initial code

    def init_ui(self):
        """Set up the main UI layout."""

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # --- 1. Top URL Input Bar (Refactored) ---
        # --- MODIFICATION (REQ 2): Split into two rows ---

        # --- Row 1: URL, Discoverer, Date, and Action Button ---
        top_bar_row1_layout = QHBoxLayout()
        top_bar_row1_layout.setSpacing(10)

        self.url_input = QComboBox()
        self.url_input.setEditable(True)
        self.url_input.setPlaceholderText("Enter website homepage URL (e.g., https://www.example.com)")
        self.url_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.url_input.lineEdit().returnPressed.connect(self.start_channel_discovery)
        self.url_input.setContextMenuPolicy(Qt.CustomContextMenu)
        self.url_input.customContextMenuRequested.connect(self._show_url_history_context_menu)
        top_bar_row1_layout.addWidget(self.url_input, 1)  # Give it stretch factor 1

        top_bar_row1_layout.addWidget(QLabel("Discoverer:"))
        self.discoverer_combo = QComboBox()
        self.discoverer_combo.addItems(["Sitemap", "RSS", "Smart Analysis"])
        if "RSSDiscoverer" not in globals():
            self.discoverer_combo.model().item(1).setEnabled(False)
        # self.discoverer_combo.model().item(2).setEnabled(False)  # WIP
        self.discoverer_combo.setToolTip(
            "Select the discovery method:\n"
            "- Sitemap: Finds sitemap.xml from the homepage.\n"
            "- RSS: Finds <link rel='alternate'> RSS feeds from the homepage.\n\n"
            "In both cases, enter the homepage URL."
        )
        top_bar_row1_layout.addWidget(self.discoverer_combo)

        # --- [NEW] AI Signature (for Smart Analysis) ---
        self.ai_signature_label = QLabel("AI Signature:")
        self.ai_signature_input = QLineEdit()
        self.ai_signature_input.setPlaceholderText("Optional: e.g., 'a[class*=\"title\"]'")
        self.ai_signature_input.setToolTip("Manually specify the 'link fingerprint' signature.")
        self.ai_signature_input.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        top_bar_row1_layout.addWidget(self.ai_signature_label)
        top_bar_row1_layout.addWidget(self.ai_signature_input, 1)  # Give it stretch
        # --- [END NEW] ---

        self.date_filter_check = QCheckBox("Filter last:")
        self.date_filter_check.setToolTip("If checked, only discover channels/articles updated within the last X days.")
        top_bar_row1_layout.addWidget(self.date_filter_check)

        self.date_filter_days_spin = QSpinBox()
        self.date_filter_days_spin.setRange(1, 9999)
        self.date_filter_days_spin.setValue(7)
        self.date_filter_days_spin.setSuffix(" days")
        self.date_filter_days_spin.setEnabled(False)  # Disabled by default
        top_bar_row1_layout.addWidget(self.date_filter_days_spin)
        self.date_filter_check.stateChanged.connect(
            lambda state: self.date_filter_days_spin.setEnabled(state == Qt.Checked)
        )

        top_bar_row1_layout.addSpacing(15)

        # self.analyze_button = QPushButton("Discover Channels")  # Renamed
        # self.analyze_button.setStyleSheet("padding: 5px 10px;")  # Add padding
        # top_bar_row1_layout.addWidget(self.analyze_button)

        main_layout.addLayout(top_bar_row1_layout)  # Add Row 1

        # --- Row 2: Fetcher Options and Proxy ---
        top_bar_row2_layout = QHBoxLayout()
        top_bar_row2_layout.setSpacing(10)

        top_bar_row2_layout.addWidget(QLabel("Discovery Fetcher:"))

        self.discovery_fetcher_widget = FetcherConfigWidget(self)
        self.discovery_fetcher_widget.set_defaults(
            fetcher_name="Simple (Requests)",
            timeout=10,
            render=False
        )
        self.discovery_fetcher_widget.set_render_tooltip(
            "Fetches final rendered HTML (slower).\n"
            "[Discovery] Will be forced OFF to ensure XML/RSS parsing.\n"
            "[Extraction] Will be used as set."
        )
        top_bar_row2_layout.addWidget(self.discovery_fetcher_widget, 1)

        top_bar_row2_layout.addWidget(QLabel("Discovery Fetcher:"))
        self.discovery_fetcher_combo = QComboBox()
        self.discovery_fetcher_combo.addItems([
            "Simple (Requests)",
            "Advanced (Playwright)",
            "Stealth (Playwright)"
        ])
        if not sync_playwright:
            self.discovery_fetcher_combo.model().item(1).setEnabled(False)
            self.discovery_fetcher_combo.model().item(2).setEnabled(False)
        if not sync_stealth and not Stealth:
            self.discovery_fetcher_combo.model().item(2).setEnabled(False)
        top_bar_row2_layout.addWidget(self.discovery_fetcher_combo)

        self.pause_browser_check = QCheckBox("Pause Browser")
        self.pause_browser_check.setToolTip("Pauses Playwright (in headful mode) for debugging.")
        top_bar_row2_layout.addWidget(self.pause_browser_check)

        self.render_page_check = QCheckBox("Render Page")
        self.render_page_check.setToolTip(
            "Fetches final rendered HTML (slower).\n"
            "[Discovery] Will be forced OFF to ensure XML/RSS parsing.\n"
            "[Extraction] Will be used as set.")
        top_bar_row2_layout.addWidget(self.render_page_check)

        top_bar_row2_layout.addSpacing(5)
        top_bar_row2_layout.addWidget(QLabel("Timeout(s):"))
        self.discovery_timeout_spin = QSpinBox()
        self.discovery_timeout_spin.setRange(1, 300)  # 1s to 5min
        self.discovery_timeout_spin.setValue(10)  # Default 10
        self.discovery_timeout_spin.setToolTip("Fetcher timeout in seconds for discovery.")
        top_bar_row2_layout.addWidget(self.discovery_timeout_spin)

        top_bar_row2_layout.addSpacing(15)

        top_bar_row2_layout.addWidget(QLabel("Proxy:"))
        self.discovery_proxy_input = QLineEdit()
        self.discovery_proxy_input.setPlaceholderText("e.g., http://user:pass@host:port")
        self.discovery_proxy_input.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)

        # --- NEW (REQ 1): Load Discovery Proxy ---
        settings = QSettings("MyOrg", "CrawlerPlayground")
        saved_d_proxy = settings.value(self.DISCOVERY_PROXY_KEY, "", type=str)
        self.discovery_proxy_input.setText(saved_d_proxy)
        # --- END NEW ---

        top_bar_row2_layout.addWidget(self.discovery_proxy_input, 1)  # Give it stretch factor 1

        # --- [NEW] Analyze button moved to Row 2 ---
        top_bar_row2_layout.addSpacing(15)
        self.analyze_button = QPushButton("Discover Channels")
        self.analyze_button.setStyleSheet("padding: 5px 10px;")
        top_bar_row2_layout.addWidget(self.analyze_button)
        # --- [END NEW] ---

        main_layout.addLayout(top_bar_row2_layout)  # Add Row 2

        # --- END MODIFICATION (REQ 2) ---

        # --- Top-to-Bottom splitter ---
        vertical_splitter = QSplitter(Qt.Vertical)

        # --- 2. Main Content Splitter (Tree | Tabs) ---
        self.main_splitter = QSplitter(Qt.Horizontal)

        # --- 2a. Left Side: Tree Widget ---
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Discovered Channels / Articles"])
        self.main_splitter.addWidget(self.tree_widget)

        # --- 2b. Right Side: Tab Widget (Refactored) ---
        self.tab_widget = QTabWidget()

        # --- REQ 2 & 3: New Article Preview Tab ---
        self.article_preview_widget = self._create_article_preview_tab()
        if QWebEngineView:
            self.tab_widget.addTab(self.article_preview_widget, "Article Preview")
        else:
            self.tab_widget.addTab(QTextEdit("QWebEngineView not available."), "Preview (Disabled)")

        self.channel_source_viewer = QTextEdit()
        self.channel_source_viewer.setReadOnly(True)
        self.channel_source_viewer.setFont(QFont("Courier", 10))
        self.channel_source_viewer.setLineWrapMode(QTextEdit.NoWrap)
        self.tab_widget.addTab(self.channel_source_viewer, "Channel Source")  # Renamed

        self.main_splitter.addWidget(self.tab_widget)
        self.main_splitter.setSizes([400, 1000])
        vertical_splitter.addWidget(self.main_splitter)

        # --- 3. Bottom: (Code | Log) Splitter ---
        bottom_splitter = QSplitter(Qt.Horizontal)

        # --- 3a. Bottom-Left: Generated Code (REQ 5) ---
        code_box = QFrame()
        code_box.setFrameShape(QFrame.StyledPanel)
        code_layout = QVBoxLayout(code_box)
        code_label = QLabel("Generated Python Code:")  # Renamed
        code_label.setStyleSheet("font-weight: bold;")
        code_layout.addWidget(code_label)
        self.generated_code_text = QTextEdit()  # Renamed
        self.generated_code_text.setReadOnly(True)
        self.generated_code_text.setFont(QFont("Courier", 9))
        code_layout.addWidget(self.generated_code_text)
        # --- NEW: Save Code Button ---
        self.save_code_button = QPushButton(QIcon.fromTheme("document-save"), "Save Code to File...")
        self.save_code_button.setToolTip("Save the generated code above to a Python file (e.g., CrawlerGenerated.py)")
        code_layout.addWidget(self.save_code_button)
        bottom_splitter.addWidget(code_box)

        # --- 3b. Bottom-Right: Log History ---
        log_box = QFrame()
        log_box.setFrameShape(QFrame.StyledPanel)
        log_layout = QVBoxLayout(log_box)
        log_label = QLabel("Log History:")
        log_label.setStyleSheet("font-weight: bold;")
        log_layout.addWidget(log_label)
        self.log_history_view = QTextEdit()
        self.log_history_view.setReadOnly(True)
        self.log_history_view.setFont(QFont("Courier", 9))
        self.log_history_view.setLineWrapMode(QTextEdit.NoWrap)
        log_layout.addWidget(self.log_history_view)
        bottom_splitter.addWidget(log_box)

        bottom_splitter.setSizes([600, 600])
        vertical_splitter.addWidget(bottom_splitter)
        vertical_splitter.setSizes([700, 200])
        main_layout.addWidget(vertical_splitter, 1)

        # --- 4. Status Bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Enter a URL and select a discoverer.")

        self.setCentralWidget(main_widget)

    def _create_article_preview_tab(self) -> QWidget:
        """Helper function to build the complex Article Preview tab."""
        # This 'main_widget' is what the tab.addTab() receives.
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(5)
        layout.setContentsMargins(0, 5, 0, 0)  # Keep top margin

        # --- Create the main horizontal splitter ---
        self.article_splitter = QSplitter(Qt.Horizontal)
        self.article_splitter.setOpaqueResize(False)  # FIX for webview flicker

        # --- Build the Left Pane (URL Bar + Web View) ---
        left_pane_widget = QWidget()
        left_layout = QVBoxLayout(left_pane_widget)
        left_layout.setSpacing(5)
        left_layout.setContentsMargins(0, 0, 5, 0)  # Right margin

        left_toolbar = QToolBar("Article URL")
        left_toolbar.addWidget(QLabel("URL:"))
        self.article_url_input = QLineEdit()
        self.article_url_input.setPlaceholderText("Select an article from the tree...")
        left_toolbar.addWidget(self.article_url_input)
        self.article_go_button = QPushButton("Go")
        left_toolbar.addWidget(self.article_go_button)

        left_layout.addWidget(left_toolbar)  # Add toolbar to left pane

        if QWebEngineView:
            self.web_view = QWebEngineView()
        else:
            self.web_view = QTextEdit("QWebEngineView not available. Install PyQtWebEngine.")
            self.web_view.setReadOnly(True)

        left_layout.addWidget(self.web_view, 1)  # Add webview (stretches)

        # --- Build the Right Pane (Tools + Markdown View) ---
        right_pane_widget = QWidget()
        right_layout = QVBoxLayout(right_pane_widget)
        right_layout.setSpacing(5)
        right_layout.setContentsMargins(5, 0, 0, 0)  # Left margin

        # --- Toolbar 1: Fetcher Settings ---
        fetcher_toolbar = QToolBar("Fetcher Tools")

        self.article_fetcher_widget = FetcherConfigWidget(self)
        fetcher_toolbar.addWidget(self.article_fetcher_widget)

        # --- Toolbar 2: Extractor Settings ---
        extractor_toolbar = QToolBar("Extractor Tools")
        extractor_toolbar.layout().setSpacing(5)
        extractor_toolbar.addWidget(QLabel("Extractor:"))
        self.extractor_combo = QComboBox()
        available_extractors = sorted(EXTRACTOR_MAP.keys())
        if available_extractors:
            self.extractor_combo.addItems(available_extractors)
            if "Trafilatura" in available_extractors:
                self.extractor_combo.setCurrentText("Trafilatura")
        else:
            self.extractor_combo.addItem("No Extractors Found")
            self.extractor_combo.setEnabled(False)
        extractor_toolbar.addWidget(self.extractor_combo)

        self.css_selector_label = QLabel("Selectors:")
        self.css_selector_input = QLineEdit()
        self.css_selector_input.setPlaceholderText("e.g., article.content, .post-body")
        self.css_selector_input.setToolTip("CSS selectors (comma-separated) for Generic CSS Extractor")
        self.css_selector_input.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        extractor_toolbar.addWidget(self.css_selector_label)
        extractor_toolbar.addWidget(self.css_selector_input)

        self.extractor_analyze_button = QPushButton("Analyze")
        extractor_toolbar.addWidget(self.extractor_analyze_button)

        # --- Add both toolbars to the right layout ---
        right_layout.addWidget(fetcher_toolbar)
        right_layout.addWidget(extractor_toolbar)

        # --- NEW: Vertical Splitter for Markdown and Metadata ---
        self.output_splitter = QSplitter(Qt.Vertical)

        # --- Markdown view (Top) ---
        self.markdown_output_view = QTextEdit()
        self.markdown_output_view.setReadOnly(True)
        self.markdown_output_view.setFont(QFont("Courier", 10))
        self.markdown_output_view.setLineWrapMode(QTextEdit.NoWrap)
        self.markdown_output_view.setPlaceholderText("Extracted Markdown content will appear here...")
        self.output_splitter.addWidget(self.markdown_output_view)

        # --- Metadata view (Bottom) ---
        self.metadata_output_view = QTextEdit()  # <-- NEW WIDGET
        self.metadata_output_view.setReadOnly(True)
        self.metadata_output_view.setFont(QFont("Courier", 10))
        self.metadata_output_view.setLineWrapMode(QTextEdit.NoWrap)
        self.metadata_output_view.setPlaceholderText("Extracted metadata (JSON) will appear here...")
        self.output_splitter.addWidget(self.metadata_output_view)

        # Set initial sizes for the new splitter
        self.output_splitter.setSizes([700, 300])  # 70% Markdown, 30% Meta

        right_layout.addWidget(self.output_splitter, 1)  # Add splitter (stretches)

        # --- Add panes to splitter ---
        self.article_splitter.addWidget(left_pane_widget)
        self.article_splitter.addWidget(right_pane_widget)
        self.article_splitter.setSizes([800, 500])  # Adjust initial sizes

        layout.addWidget(self.article_splitter, 1)  # Add splitter to main layout
        return main_widget

    def connect_signals(self):
        """Centralize all signal/slot connections."""
        # Top Bar
        self.url_input.lineEdit().returnPressed.connect(self.start_channel_discovery)
        self.url_input.lineEdit().textChanged.connect(self.on_url_input_changed)
        self.analyze_button.clicked.connect(self.start_channel_discovery)
        self.discoverer_combo.currentTextChanged.connect(self._update_discoverer_options_ui)

        # Tree
        self.tree_widget.itemClicked.connect(self.on_tree_item_clicked)
        self.tree_widget.itemExpanded.connect(self.on_tree_item_expanded)

        # Article Preview Tab
        self.article_go_button.clicked.connect(self.on_article_go_clicked)
        self.article_url_input.returnPressed.connect(self.on_article_go_clicked)
        self.extractor_analyze_button.clicked.connect(self.start_extraction_analysis)
        self.extractor_combo.currentTextChanged.connect(self._update_extractor_options_ui)

        self.discoverer_combo.currentTextChanged.connect(self.update_generated_code)

        # 连接新 widget 内部的 ComboBox
        if self.discovery_fetcher_widget:
            self.discovery_fetcher_widget.fetcher_combo.currentTextChanged.connect(self.update_generated_code)
        if self.article_fetcher_widget:
            self.article_fetcher_widget.fetcher_combo.currentTextChanged.connect(self.update_generated_code)

        self.extractor_combo.currentTextChanged.connect(self.update_generated_code)
        self.tree_widget.itemChanged.connect(self.update_generated_code_from_tree)

        self.save_code_button.clicked.connect(self._save_generated_code)

    def set_loading_state(self, is_loading: bool, message: str = ""):
        """Enable/Disable UI controls during threaded operations."""
        # Top bar
        self.url_input.setEnabled(not is_loading)
        self.analyze_button.setEnabled(not is_loading)
        self.discoverer_combo.setEnabled(not is_loading)

        if self.discovery_fetcher_widget:
            self.discovery_fetcher_widget.setEnabled(not is_loading)

        # Tree
        self.tree_widget.setEnabled(not is_loading)

        # Article Tab (partially)
        self.extractor_analyze_button.setEnabled(not is_loading)

        if self.article_fetcher_widget:
            self.article_fetcher_widget.setEnabled(not is_loading)

        if is_loading:
            self.status_bar.showMessage(message)
            # Find the button that was pressed
            if "Discover" in message:
                self.analyze_button.setText("Discovering...")
            elif "Extracting" in message:
                self.extractor_analyze_button.setText("Analyzing...")

            if self.log_history_view:
                self.log_history_view.append(f"--- {message} ---")
        else:
            self.status_bar.showMessage(message or "Ready.")
            self.analyze_button.setText("Discover Channels")
            self.extractor_analyze_button.setText("Analyze")
            if self.log_history_view and message:
                self.log_history_view.append(f"--- {message} ---")

    def clear_all_controls(self):
        """Reset the UI to its initial state."""
        self.tree_widget.clear()
        self.channel_item_map.clear()
        self.channel_source_viewer.clear()
        self.generated_code_text.clear()

        # --- MODIFICATION: Clear only text, not history list ---
        # Do NOT clear the user's input. They may want to run it again
        # or see what generated the current results.
        # self.url_input.setCurrentIndex(-1)
        # self.url_input.clearEditText()

        if self.log_history_view:
            self.log_history_view.clear()
        if self.web_view and QUrl:
            self.web_view.setUrl(QUrl("about:blank"))
        self.article_url_input.clear()
        self.markdown_output_view.clear()
        self.metadata_output_view.clear()
        self.update_generated_code()

    def append_log_history(self, message: str):
        """Appends a message to the log history text area."""
        if self.log_history_view:
            self.log_history_view.append(message)

    # --- Threaded Action Starters ---
    def start_channel_discovery(self):
        """Slot for 'Discover Channels' button. (Refactored for Unified Flow)"""

        # 1. 获取 UI 值
        url_input_text = self.url_input.currentText().strip()
        self.discoverer_name = self.discoverer_combo.currentText()  # 存储选择

        if not url_input_text:
            self.status_bar.showMessage("Error: Please enter a URL or list of URLs.")
            return

        # 2. 解析输入文本
        potential_urls = url_input_text.split()
        http_urls = [u for u in potential_urls if u.startswith("http")]

        # 3. [核心] 决定要传递给 Worker 的 entry_point 是什么类型
        entry_point_for_worker: Any = None

        # 你的 "on_url_input_changed" 辅助函数（如果已添加）
        # 会确保多行粘贴在此时已经是空格分隔的字符串

        if self.discoverer_name == "RSS":
            # RSS 模式：可以处理 List[str] 或 str
            if len(http_urls) > 1:
                # 情况 A: RSS + 多个 URL -> 传递 List[str]
                entry_point_for_worker = http_urls
                self.append_log_history(f"[Info] RSS mode: Passing {len(http_urls)} URLs list to worker.")

            elif len(http_urls) == 1:
                # 情况 B: RSS + 单个 URL -> 传递 str
                entry_point_for_worker = http_urls[0]
                if http_urls[0] != url_input_text:
                    self.url_input.setText(http_urls[0])  # 清理 UI

            else:
                # 情况 C: RSS + 非 http 字符串 (如 example.com 或单个 feed) -> 传递 str
                # 我们假设 RSS 的 _handle_single_url 可以处理 'example.com' (如果不能，就添加 https://)
                # 为了安全起见，我们添加 https://
                entry_point_for_worker = "https://" + url_input_text
                self.url_input.setText(entry_point_for_worker)
                self.append_log_history("[Info] Non-http string detected, adding 'https://'.")

        elif self.discoverer_name == "Sitemap":
            # Sitemap 模式：必须是单个 str
            if len(http_urls) > 1:
                # 情况 D: Sitemap + 多个 URL -> 错误
                self.status_bar.showMessage("Error: Sitemap mode only supports a *single* homepage URL.")
                self.append_log_history("[Error] SitemapDiscoverer requires a single string URL. Input has multiple.")
                return  # 停止执行

            elif len(http_urls) == 1:
                # 情况 E: Sitemap + 单个 URL -> 传递 str
                entry_point_for_worker = http_urls[0]
                if http_urls[0] != url_input_text:
                    self.url_input.setText(http_urls[0])  # 清理 UI

            else:
                # 情况 F: Sitemap + 非 http 字符串 (example.com) -> 传递 str
                entry_point_for_worker = "https://" + url_input_text
                self.url_input.setText(entry_point_for_worker)
                self.append_log_history("[Info] Non-http string detected, adding 'https://'.")

        else:
            # 其他 (未来的) Discoverer，我们假定它们使用单个 str
            self.append_log_history(
                f"[Warning] Unknown discoverer '{self.discoverer_name}'. Defaulting to single URL logic.")
            if len(http_urls) > 1:
                self.append_log_history(f"  > Using *first* URL only: {http_urls[0]}")

            entry_point_for_worker = http_urls[0] if http_urls else "https://" + url_input_text
            if entry_point_for_worker != self.url_input.currentText():
                self.url_input.setText(entry_point_for_worker)

        # 4. 清理并准备 Worker
        self.clear_all_controls()
        self._save_url_history(self.url_input.currentText())

        # --- (获取日期过滤器设置 - 这部分逻辑不变) ---
        start_date: Optional[datetime.datetime] = None
        end_date: Optional[datetime.datetime] = None

        if self.date_filter_check.isChecked():
            days_ago = self.date_filter_days_spin.value()
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days_ago)
            self.append_log_history(f"Applying date filter: Last {days_ago} days "
                                    f"(since {start_date.strftime('%Y-%m-%d')})")
        # --- (日期设置结束) ---

        # 存储策略
        fetcher_config = self.discovery_fetcher_widget.get_config()

        self.set_loading_state(True, f"Discovering {self.discoverer_name} channels...")
        self.update_generated_code()  # 更新代码片段

        ai_signature_str = self.ai_signature_input.text().strip() or None
        if self.discoverer_name != "Smart Analysis":
            ai_signature_str = None  # 确保只在 Smart Analysis 时传递

        self.last_used_entry_point = entry_point_for_worker

        worker = ChannelDiscoveryWorker(
            discoverer_name=self.discoverer_name,
            fetcher_name=fetcher_config['fetcher_name'],
            entry_point=entry_point_for_worker,
            start_date=start_date,
            end_date=end_date,
            proxy=fetcher_config['proxy'],
            timeout=fetcher_config['timeout'],
            pause_browser=fetcher_config['pause'],
            render_page=fetcher_config['render'],
            ai_signature=ai_signature_str
        )

        worker.signals.result.connect(self.on_channel_discovery_result)
        worker.signals.finished.connect(self.on_channel_discovery_finished)
        worker.signals.error.connect(self.on_worker_error)
        worker.signals.progress.connect(self.status_bar.showMessage)
        worker.signals.progress.connect(self.append_log_history)

        self.thread_pool.start(worker)

    def start_article_loading(self, channel_item: QTreeWidgetItem, channel_url: str):
        """Starts the Stage 2 (Lazy Loading) worker for a specific channel."""
        channel_item.takeChildren()  # Remove dummy
        loading_item = QTreeWidgetItem(["Loading articles..."])
        channel_item.addChild(loading_item)
        channel_item.setExpanded(True)
        self.status_bar.showMessage(f"Loading articles for {channel_url}...")

        fetcher_config = self.discovery_fetcher_widget.get_config()

        worker = ArticleListWorker(
            discoverer_name=self.discoverer_name,
            fetcher_name=fetcher_config['fetcher_name'],
            channel_url=channel_url,
            proxy=fetcher_config['proxy'],
            timeout=fetcher_config['timeout'],
            pause_browser=fetcher_config['pause'],
            render_page=fetcher_config['render']
        )

        worker.signals.result.connect(self.on_article_list_result)
        worker.signals.finished.connect(self.on_worker_finished)
        worker.signals.error.connect(self.on_worker_error)
        worker.signals.progress.connect(self.status_bar.showMessage)
        worker.signals.progress.connect(self.append_log_history)

        self.thread_pool.start(worker)

    def start_channel_source_loading(self, url: str):
        """Starts worker to fetch raw channel source (e.g., XML) for the viewer."""
        self.channel_source_viewer.setPlainText(f"Loading source from {url}...")
        self.tab_widget.setCurrentWidget(self.channel_source_viewer)

        fetcher_config = self.discovery_fetcher_widget.get_config()

        worker = ChannelSourceWorker(
            discoverer_name=self.discoverer_name,
            fetcher_name=fetcher_config['fetcher_name'],
            url=url,
            proxy=fetcher_config['proxy'],
            timeout=fetcher_config['timeout'],
            pause_browser=fetcher_config['pause'],
            render_page=fetcher_config['render']
        )

        worker.signals.result.connect(self.on_channel_source_result)
        worker.signals.finished.connect(self.on_worker_finished)
        worker.signals.error.connect(self.on_worker_error)
        worker.signals.progress.connect(self.status_bar.showMessage)
        worker.signals.progress.connect(self.append_log_history)

        self.thread_pool.start(worker)

    def start_extraction_analysis(self):
        """Slot for the 'Analyze' button in the Article Preview tab."""
        url = self.article_url_input.text().strip()
        if not url:
            self.status_bar.showMessage("Error: No article URL to analyze.")
            return

        fetcher_config = self.article_fetcher_widget.get_config()
        extractor_name = self.extractor_combo.currentText()

        # Get kwargs from our helper function
        extractor_kwargs = self._get_current_extractor_args(extractor_name)

        if extractor_name == "Generic CSS" and not extractor_kwargs.get('selectors'):
            self.append_log_history("[Warning] Generic CSS Extractor running with no selectors provided.")
        # --- [END MODIFICATION] ---

        self.markdown_output_view.setPlainText(f"Starting analysis on {url}...")
        self.metadata_output_view.setPlainText("Waiting for analysis to complete...")
        self.set_loading_state(True, f"Extracting {url} with {extractor_name}...")
        self.update_generated_code()  # Update code snippet

        worker = ExtractionWorker(
            fetcher_config=fetcher_config,
            extractor_name=extractor_name,
            url_to_extract=url,
            extractor_kwargs=extractor_kwargs
        )

        worker.signals.result.connect(self.on_extraction_result)
        worker.signals.finished.connect(self.on_extraction_finished)
        worker.signals.error.connect(self.on_worker_error)
        worker.signals.progress.connect(self.status_bar.showMessage)
        worker.signals.progress.connect(self.append_log_history)

        self.thread_pool.start(worker)

    # --- Thread Result Slots ---

    def on_channel_discovery_result(self, channel_list: List[str]):
        """Slot for ChannelDiscoveryWorker 'result' signal."""
        if not channel_list:
            self.status_bar.showMessage("No channels found.")
            return

        self.tree_widget.setDisabled(True)
        self.channel_queue = deque(channel_list)
        QTimer.singleShot(0, self.add_channels_to_tree)

    def add_channels_to_tree(self):
        """Process a chunk of channels to add to the tree."""
        count = 0
        while self.channel_queue and count < 100:
            channel_url = self.channel_queue.popleft()
            item = QTreeWidgetItem([channel_url])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Unchecked)
            item.setData(0, Qt.UserRole, {
                'type': 'channel', 'url': channel_url, 'loaded': False
            })
            item.addChild(QTreeWidgetItem())  # Dummy child for lazy loading
            self.tree_widget.addTopLevelItem(item)
            self.channel_item_map[channel_url] = item
            count += 1

        if self.channel_queue:
            QTimer.singleShot(0, self.add_channels_to_tree)
        else:
            self.tree_widget.setDisabled(False)
            self.status_bar.showMessage(f"Found {len(self.channel_item_map)} channels. Click to load articles.")
            self.update_generated_code()  # Update code now that tree is populated

    def on_channel_discovery_finished(self):
        """Slot for *ChannelDiscoveryWorker* 'finished' signal."""
        self.set_loading_state(False, "Discovery complete.")

    def on_article_list_result(self, result: Dict[str, Any]):
        """Slot for ArticleListWorker 'result' signal."""
        channel_url = result['channel_url']
        article_list = result['articles']
        parent_item = self.channel_item_map.get(channel_url)
        if not parent_item: return
        data = parent_item.data(0, Qt.UserRole)
        data['loaded'] = True
        parent_item.setData(0, Qt.UserRole, data)
        parent_item.takeChildren()
        if not article_list:
            parent_item.addChild(QTreeWidgetItem(["No articles found in this channel."]))
        else:
            for article_url in article_list:
                child_item = QTreeWidgetItem([article_url])
                child_item.setData(0, Qt.UserRole, {'type': 'article', 'url': article_url})
                parent_item.addChild(child_item)
        parent_item.setExpanded(True)
        self.status_bar.showMessage(f"Loaded {len(article_list)} articles for {channel_url}", 5000)

    def on_channel_source_result(self, content_string: str):
        """Slot for ChannelSourceWorker 'result' signal."""
        self.channel_source_viewer.setPlainText(content_string)

    def on_extraction_result(self, result: ExtractionResult):
        """Slot for ExtractionWorker 'result' signal."""
        import json

        if result.error:
            error_msg = f"--- EXTRACTION FAILED ---\n\n{result.error}"
            self.markdown_output_view.setPlainText(error_msg)
            self.metadata_output_view.setPlainText(error_msg)
            self.append_log_history(f"[Error] Extraction failed: {result.error}")
        else:
            # Set Markdown content
            self.markdown_output_view.setPlainText(result.markdown_content or "[No Markdown Content Extracted]")

            # Set Metadata content (as pretty-printed JSON)
            try:
                metadata_str = json.dumps(
                    result.metadata,
                    indent=2,
                    ensure_ascii=False,
                    default=str  # Handle non-serializable types like datetime
                )
                self.metadata_output_view.setPlainText(metadata_str)
            except Exception as e:
                self.metadata_output_view.setPlainText(f"Could not serialize metadata: {e}\n\n{result.metadata}")

    def on_extraction_finished(self):
        """Slot for *ExtractionWorker* 'finished' signal."""
        self.set_loading_state(False, "Extraction complete.")

    def on_worker_finished(self):
        """Generic 'finished' slot for sub-tasks."""
        if not self.analyze_button.isEnabled():
            if self.thread_pool.activeThreadCount() == 0:
                self.status_bar.showMessage("Task complete. Ready.", 3000)

    def on_worker_error(self, error: tuple):
        """Slot for any worker's 'error' signal."""
        ex_type, message, tb = error
        error_msg = f"Error: {ex_type}: {message}"
        self.status_bar.showMessage(error_msg)

        if self.log_history_view:
            self.log_history_view.append(f"--- Worker Error ---")
            self.log_history_view.append(error_msg)
            self.log_history_view.append(tb)
            self.log_history_view.append(f"--------------------")

        print(f"--- Worker Error ---")
        print(tb)
        print(f"--------------------")

        # Re-enable UI if a main task fails
        self.set_loading_state(False, f"Error occurred. {message}")

    # --- UI Event Handlers ---

    def on_url_input_changed(self, text: str):
        """
        Slot to normalize multi-line pastes in the URL bar *only* for RSS mode.
        (槽函数：仅在 RSS 模式下规范化 URL 栏中的多行粘贴。)
        """

        # 仅当 "RSS" 被选中时才启用此功能
        if self.discoverer_combo.currentText() != "RSS":
            return

        # 检查是否存在换行符，这通常意味着多行粘贴
        if '\n' in text or '\r' in text:
            self.append_log_history("[Info] Multi-line paste detected. Normalizing to space-separated list.")

            # 规范化：按任何空白（包括换行）拆分，然后用单个空格连接
            normalized_text = " ".join(text.split())

            # 阻止信号以防止无限递归
            self.url_input.lineEdit().blockSignals(True)
            self.url_input.lineEdit().setText(normalized_text)
            self.url_input.lineEdit().blockSignals(False)

            # 将光标移到末尾
            self.url_input.lineEdit().end(False)

    def on_tree_item_expanded(self, item: QTreeWidgetItem):
        """
        Handles the 'itemExpanded' signal.
        This is now the *only* trigger for lazy-loading articles.
        (处理 'itemExpanded' 信号。)
        (这是现在懒加载文章的 *唯一* 触发器。)
        """
        if not self.tree_widget.isEnabled(): return
        data = item.data(0, Qt.UserRole)
        if not data: return

        item_type = data.get('type')
        url = data.get('url')

        # We only care about expanding "channel" items
        # (我们只关心 "channel" 项的展开)
        if item_type == 'channel':
            # Check if it has the dummy child or is already loading
            # (检查它是否有虚拟子项或已在加载)
            if item.childCount() == 1 and "Loading" in item.child(0).text(0):
                return  # Already loading (已在加载)

            # Check the 'loaded' flag we set
            # (检查我们设置的 'loaded' 标志)
            if data.get('loaded') == False:
                # This is the first time it's being expanded, load data
                # (这是它第一次被展开，加载数据)
                self.start_article_loading(item, channel_url=url)

    def on_tree_item_clicked(self, item: QTreeWidgetItem, column: int):
        """
        Handles clicks on any tree item (channel or article).

        [MODIFIED] Now performs a "hit test". It ignores clicks
        on the checkbox or expand-arrow, only responding to
        clicks on the main item text.
        (处理对任何树项目（频道或文章）的点击。)
        ([已修改] 现在执行“点击测试”。它忽略对复选框或)
        (展开箭头的点击，只响应对主项目文本的点击。)
        """

        # --- [NEW] Hit Test Logic ---
        # (新增 点击测试逻辑)

        # Get the click position relative to the tree widget's viewport
        # (获取相对于树控件视口的点击位置)
        pos = self.tree_widget.viewport().mapFromGlobal(QCursor.pos())

        # Get the item's full visual rectangle
        # (获取项目的完整可视化矩形)
        visual_rect = self.tree_widget.visualItemRect(item)

        # This is the X-coordinate where the "main" part (text/label)
        # of the item begins.
        # It accounts for the expand-arrow's indentation.
        # (这是项目“主要”部分（文本/标签）开始的 X 坐标。)
        # (它考虑了展开箭头的缩进。)
        text_start_x = visual_rect.x() + self.tree_widget.indentation()

        # [HEURISTIC] Add a buffer for the checkbox itself (approx 20px)
        # (启发式) 为复选框本身添加一个缓冲区（约 20px）
        # This is not perfect, but robust enough.
        # (这不完美，但足够稳健。)
        if item.flags() & Qt.ItemIsUserCheckable:
            text_start_x += 20

        if pos.x() < text_start_x:
            # Click was on the checkbox or expander
            # (点击发生在复选框或展开器上)
            # We *only* want the checkbox to trigger itemChanged
            # (我们 *只* 希望复选框触发 itemChanged)
            # and the expander to trigger itemExpanded.
            # (而展开器触发 itemExpanded。)
            # So, we do *nothing* in itemClicked.
            # (因此，我们在 itemClicked 中 *不执行任何操作*。)
            return

            # --- [END NEW] ---

        # If we are here, the click was on the *text* part
        # (如果我们在这里，说明点击的是 *文本* 部分)
        if not self.tree_widget.isEnabled(): return
        data = item.data(0, Qt.UserRole)
        if not data: return

        item_type = data.get('type')
        url = data.get('url')

        if item_type == 'channel':
            # --- [MODIFIED] ---
            # The article loading logic has been MOVED
            # to on_tree_item_expanded.
            # (文章加载逻辑已移至 on_tree_item_expanded。)
            # --- [END MODIFIED] ---

            # We still want to load the source XML on a text click
            # (我们仍然希望在文本点击时加载源码 XML)
            self.start_channel_source_loading(url=url)

        elif item_type == 'article':
            # --- (This logic is unchanged and correct) ---
            # (此逻辑未更改且正确)
            self.article_url_input.setText(url)
            self.markdown_output_view.clear()
            self.metadata_output_view.clear()
            self.update_generated_code()

            if self.web_view and QUrl:
                self._apply_proxy_to_webview()
                self.web_view.setUrl(QUrl(url))
                self.web_view.setFocus()
                self.tab_widget.setCurrentWidget(self.article_preview_widget)
                self.status_bar.showMessage(f"Loading page: {url}", 3000)

    def on_article_go_clicked(self):
        """Handles clicks on the 'Go' button in the article tab."""
        if self.web_view and QUrl:
            self._apply_proxy_to_webview()
            url = self.article_url_input.text()
            self.web_view.setUrl(QUrl(url))
            self.web_view.setFocus()

    def _apply_proxy_to_webview(self):
        """
        Reads the proxy from the 'article_proxy_input' and applies it
        to the QWebEngineView instance.
        (读取 'article_proxy_input' 中的代理并将其应用于 QWebEngineView 实例。)
        """
        if not self.web_view or not QUrl:
            # 如果 webview 不可用，则不执行任何操作
            return

        proxy_str = self.article_proxy_input.text().strip()

        # 获取 web_view 关联的 profile
        # (我们使用 page().profile() 而不是 defaultProfile() 以确保获取正确的实例)
        profile = self.web_view.page().profile()

        if not proxy_str:
            # 字符串为空，清除代理设置
            profile.setHttpProxy(QUrl())
            self.append_log_history("[Webview] 代理已清除。使用系统设置。")
            return

        # 检查用户是否提供了 scheme (如 http://, socks5://)
        # 如果没有，默认为 http://
        if "://" not in proxy_str:
            proxy_str = "http://" + proxy_str
            self.append_log_history(f"[Webview] 未检测到代理协议，默认为 http://")

        proxy_url = QUrl(proxy_str)

        if not proxy_url.isValid() or not proxy_url.host():
            # 处理无效URL (例如 "http://")
            profile.setHttpProxy(QUrl())  # 为安全起见清除它
            self.append_log_history(f"[Webview] 代理 URL 无效: {proxy_str}。代理已清除。")
            return

        # 应用代理
        # 注意：尽管方法名叫 'setHttpProxy'，
        # 但它会根据 QUrl 的 scheme (http, https, socks5) 正确处理代理。
        profile.setHttpProxy(proxy_url)

        self.append_log_history(f"[Webview] 代理已设置为: {proxy_url.toString()}")

    def update_generated_code_from_tree(self, item: QTreeWidgetItem, column: int):
        """Wrapper to call code gen when tree checkstate changes."""
        data = item.data(0, Qt.UserRole)
        if data and data.get('type') == 'channel':
            self.update_generated_code()

    # --- REQ 5: Code Generation ---
    def update_generated_code(self):
        """
        Orchestrator for code generation.
        Gathers all UI settings into a config dict, then generates
        the corresponding Python code script.
        (代码生成的协调器。
         将所有UI设置收集到一个配置字典中，然后生成相应的Python代码脚本。)
        """
        try:
            # Step 1: Read all UI controls into a structured dictionary
            # (第 1 步：将所有 UI 控件读入结构化字典)
            config_dict = self._build_config_dict()

            # Step 2: Pass the dictionary to the code generator
            # (第 2 步：将字典传递给代码生成器)
            code_script = self.generate_code_from_config(config_dict)

            # Step 3: Display the generated code
            # (第 3 步：显示生成的代码)
            self.generated_code_text.setPlainText(code_script)

        except Exception as e:
            # Show any error during generation in the code block itself
            # (在代码块本身中显示生成期间的任何错误)
            error_msg = f"# Failed to generate code:\n# {type(e).__name__}: {e}\n\n"
            error_msg += traceback.format_exc()
            self.generated_code_text.setPlainText(error_msg)

    def _get_current_extractor_args(self, extractor_name: str) -> dict:
        """
        Retrieves extractor-specific arguments from the UI.
        (从UI检索特定于提取器的参数。)
        """
        # --- [MODIFIED] ---
        if extractor_name == "Generic CSS":
            if hasattr(self, 'css_selector_input') and self.css_selector_input:
                selector_str = self.css_selector_input.text().strip()
                if selector_str:
                    # 按逗号分割，并去除每个选择器的空白
                    selectors_list = [s.strip() for s in selector_str.split(',') if s.strip()]
                    return {
                        'selectors': selectors_list
                    }
            # Fallback if UI not ready or input is empty
            return {'selectors': ['body']}  # Default fallback
        return {}  # Default for other extractors

    def _generate_channel_filter_list_code(self) -> str:
        """
        Generates the Python code string for 'channel_filter_list'.

        This list is based on the user-checked channels in the tree,
        but contains "filter keys" (e.g., 'it/news_sitemap.xml')
        instead of full URLs.
        """
        def get_filter_key(url: str) -> str:
            """
            Helper function: Creates a simple, more unique filter key from a URL.

            Examples:
            - '.../sitemap/it/news_sitemap.xml' -> 'it/news_sitemap.xml'
            - '.../feeds/main.rss'               -> 'feeds/main.rss'
            - '.../feeds/'                       -> 'feeds'
            - '.../sitemap.xml'                  -> 'sitemap.xml'
            - 'https://example.com/' (as feed)    -> 'example.com'
            """
            try:
                parsed_url = urlparse(url)
                path = parsed_url.path

                # [FIXED] If path is just '/' or empty, this is a root URL.
                # Use the netloc (domain) as the key.
                if not path or path == '/':
                    # e.g., 'www.example.com'
                    return parsed_url.netloc or url  # Fallback to full url if netloc is also empty

                # Strip a trailing slash if it exists (e.g., /feeds/ -> /feeds)
                if path.endswith('/'):
                    path = path[:-1]

                # Get the filename (e.g., 'news_sitemap.xml' or 'feeds')
                filename = os.path.basename(path)

                # Get the parent directory (e.g., '/sitemap/it' or '/')
                parent_dir_path = os.path.dirname(path)

                # If the parent is not the root, get its name (e.g., 'it')
                if parent_dir_path and parent_dir_path != '/':
                    parent_folder = os.path.basename(parent_dir_path)
                    # Combine them: 'it/news_sitemap.xml'
                    return f"{parent_folder}/{filename}"
                else:
                    # Parent is root, just use the filename (e.g., 'sitemap.xml')
                    return filename

            except Exception:
                # Fallback on any error
                return url

        # 1. Iterate the tree, get "keys" for all checked channels
        checked_keys = []
        for i in range(self.tree_widget.topLevelItemCount()):
            item = self.tree_widget.topLevelItem(i)
            if not item:
                continue

            data = item.data(0, Qt.UserRole)

            # Ensure it is a "channel" and it is checked
            if (data and
                    data.get('type') == 'channel' and
                    item.checkState(0) == Qt.Checked):

                url = data.get('url')
                if url:
                    checked_keys.append(get_filter_key(url))

        # 2. Generate the code string
        if not checked_keys:
            # Generate an empty list if no channels were checked
            return "channel_filter_list = []"

        # Format into a clean Python list string
        list_str = "channel_filter_list = [\n"
        # Sort and de-duplicate the keys
        for key in sorted(list(set(checked_keys))):
            # repr() automatically handles quotes (e.g., 'it/news_sitemap.xml')
            list_str += f"        {repr(key)},\n"
        list_str += "    ]"
        return list_str

    def _build_config_dict(self) -> dict:
        """
        Reads all UI controls and builds the standardized config dictionary.
        (读取所有UI控件并构建标准化的配置字典。)
        """
        # --- 1. Discoverer Configuration ---
        d_fetcher_config_dict = self.discovery_fetcher_widget.get_config()
        discovery_fetcher_name = d_fetcher_config_dict['fetcher_name']

        discoverer_fetcher_params = {
            "class": discovery_fetcher_name,
            "parameters": {
                "proxy": d_fetcher_config_dict['proxy'],
                "timeout": d_fetcher_config_dict['timeout'],
                "stealth": "Stealth" in discovery_fetcher_name,
                "pause_browser": d_fetcher_config_dict['pause'],
                "render_page": False  # Hardcoded False for discovery
            }
        }

        discoverer_name = self.discoverer_combo.currentText()
        discoverer_args = {
            # Read from our cache, not the live (and potentially empty) UI control.
            # This 'entry_point' key now matches the 'Any' type (str or List[str]).
            "entry_point": self.last_used_entry_point,
            # --- MODIFICATION: Store new date filter state ---
            "date_filter_enabled": self.date_filter_check.isChecked(),
            "date_filter_days": self.date_filter_days_spin.value(),
            "ai_signature": self.ai_signature_input.text().strip() or None
        }

        # --- 2. Extractor Configuration ---
        e_fetcher_config_dict = self.article_fetcher_widget.get_config()
        article_fetcher_name = e_fetcher_config_dict['fetcher_name']

        extractor_fetcher_params = {
            "class": article_fetcher_name,
            "parameters": {
                "proxy": e_fetcher_config_dict['proxy'],
                "timeout": e_fetcher_config_dict['timeout'],
                "stealth": "Stealth" in article_fetcher_name,
                "pause_browser": e_fetcher_config_dict['pause'],
                "render_page": e_fetcher_config_dict['render']
            }
        }

        extractor_name = self.extractor_combo.currentText()

        # 合并 CSS args 和 Playwright wait args
        extractor_args = self._get_current_extractor_args(extractor_name)

        # 将 Playwright-wait 参数添加到 extractor_kwargs 中
        # 生成的脚本中的 CrawlPipeline 将需要解析这些
        extractor_args['wait_until'] = e_fetcher_config_dict['wait_until']
        extractor_args['wait_for_selector'] = e_fetcher_config_dict['wait_for_selector']
        # 传递超时，以便 get_content 可以使用它
        extractor_args['wait_for_timeout_s'] = e_fetcher_config_dict['timeout']

        # --- 3. Assemble Final Config ---
        config = {
            "discoverer": {
                "class": discoverer_name,
                "args": discoverer_args,
                "fetcher": discoverer_fetcher_params
            },
            "extractor": {
                "class": extractor_name,
                "args": extractor_args,  # 现在包含 wait-args
                "fetcher": extractor_fetcher_params
            }
        }
        return config

    def generate_code_from_config(self, config: dict) -> str:
        """
        Takes a configuration dict and generates a simple, runnable
        Python script based on the new 'CrawlProcess' architecture.
        (根据新的 'CrawlProcess' 架构，获取配置字典并生成一个简单的、
         可运行的Python脚本。)
        """
        # --- 1. Class Name Mappings (The "Table" Lookup) ---
        DISCOVERER_CLASS_MAP = {
            "Sitemap": "SitemapDiscoverer",
            "RSS": "RSSDiscoverer",
            "Smart Analysis": "ListPageDiscoverer"
        }

        FETCHER_CLASS_MAP = {
            "Simple (Requests)": "RequestsFetcher",
            "Advanced (Playwright)": "PlaywrightFetcher",
            "Stealth (Playwright)": "PlaywrightFetcher"
        }

        # 1. 生成渠道过滤器列表代码
        channel_list_code = self._generate_channel_filter_list_code()

        # --- 2. Get Discovery Config ---
        d_config = config['discoverer']
        d_fetcher_config = d_config['fetcher']
        d_class_name = DISCOVERER_CLASS_MAP.get(d_config['class'], "UnknownDiscoverer")
        d_fetcher_class_name = FETCHER_CLASS_MAP.get(d_fetcher_config['class'], "UnknownFetcher")

        # --- Build Discovery Fetcher Args ---
        d_fetcher_params = d_fetcher_config['parameters']
        d_arg_list = ['log_callback=log_cb']

        if 'Playwright' in d_fetcher_class_name:
            # Match PlaywrightFetcher.__init__
            d_arg_list.append(f"proxy={repr(d_fetcher_params.get('proxy'))}")
            d_arg_list.append(f"timeout_s={repr(d_fetcher_params.get('timeout'))}")  # Use 'timeout_s'
            d_arg_list.append(f"stealth={repr(d_fetcher_params.get('stealth'))}")
            d_arg_list.append(f"pause_browser={repr(d_fetcher_params.get('pause_browser'))}")
            d_arg_list.append(f"render_page={repr(d_fetcher_params.get('render_page'))}")
        elif 'Requests' in d_fetcher_class_name:
            # Match RequestsFetcher.__init__
            d_arg_list.append(f"proxy={repr(d_fetcher_params.get('proxy'))}")
            d_arg_list.append(f"timeout_s={repr(d_fetcher_params.get('timeout'))}")  # Use 'timeout_s'

        d_fetcher_args_str = ", ".join(d_arg_list)

        # --- 3. Get Extraction Config ---
        e_config = config['extractor']
        e_fetcher_config = e_config['fetcher']
        e_class_name = "UnknownExtractor"
        if e_config['class'] in EXTRACTOR_MAP:
            e_class_name = EXTRACTOR_MAP[e_config['class']].__name__
        e_fetcher_class_name = FETCHER_CLASS_MAP.get(e_fetcher_config['class'], "UnknownFetcher")

        # --- [FIXED] Build Extraction Fetcher Args ---
        e_fetcher_params = e_fetcher_config['parameters']
        e_arg_list = ['log_callback=log_cb']

        if 'Playwright' in e_fetcher_class_name:
            # Match PlaywrightFetcher.__init__
            e_arg_list.append(f"proxy={repr(e_fetcher_params.get('proxy'))}")
            e_arg_list.append(f"timeout_s={repr(e_fetcher_params.get('timeout'))}")  # Use 'timeout_s'
            e_arg_list.append(f"stealth={repr(e_fetcher_params.get('stealth'))}")
            e_arg_list.append(f"pause_browser={repr(e_fetcher_params.get('pause_browser'))}")
            e_arg_list.append(f"render_page={repr(e_fetcher_params.get('render_page'))}")
        elif 'Requests' in e_fetcher_class_name:
            # Match RequestsFetcher.__init__
            e_arg_list.append(f"proxy={repr(e_fetcher_params.get('proxy'))}")
            e_arg_list.append(f"timeout_s={repr(e_fetcher_params.get('timeout'))}")  # Use 'timeout_s'

        e_fetcher_args_str = ", ".join(e_arg_list)
        e_kwargs_str = repr(e_config['args'])  # Extractor-specific args

        # --- 4. Build {parameters} String ---
        code_d_fetcher = f"d_fetcher = {d_fetcher_class_name}({d_fetcher_args_str})"
        code_e_fetcher = f"e_fetcher = {e_fetcher_class_name}({e_fetcher_args_str})"

        # Note: Renamed variables to avoid conflict with 'pipeline.' call
        code_discoverer = ""
        if d_class_name == "ListPageDiscoverer":
            ai_sig_val = d_config['args'].get('ai_signature')
            code_discoverer = f"discoverer = {d_class_name}(fetcher=d_fetcher, verbose=True, ai_signature={repr(ai_sig_val)})"
        else:
            # Default for Sitemap, RSS, etc.
            code_discoverer = f"discoverer = {d_class_name}(fetcher=d_fetcher, verbose=True)"

        code_extractor = f"extractor = {e_class_name}(verbose=True)"

        entry_point_value = d_config['args'].get('entry_point')
        parameters = f"entry_point = {repr(entry_point_value)}\n"
        if d_config['args']['date_filter_enabled']:
            parameters += f"    days_ago = {d_config['args']['date_filter_days']}\n"
            parameters += "    end_date = datetime.datetime.now()\n"
            parameters += "    start_date = end_date - datetime.timedelta(days=days_ago)\n"
        else:
            parameters += "    start_date = None\n"
            parameters += "    end_date = None\n"
        parameters += f"    extractor_kwargs = {e_kwargs_str}"

        # --- 5. Format the final code ---
        code = CODE_TEMPLATE.format(
            code_d_fetcher=code_d_fetcher,
            code_e_fetcher=code_e_fetcher,
            code_discoverer=code_discoverer,
            code_extractor=code_extractor,
            parameters=parameters,
            channel_filter_list=channel_list_code
        )

        return code

    def closeEvent(self, event):
        """Ensure threads are cleaned up on exit."""
        self.status_bar.showMessage("Shutting down... waiting for tasks...")

        settings = QSettings(SETTING_ORG, SETTING_APP)
        if self.discovery_fetcher_widget:
            settings.setValue(self.DISCOVERY_PROXY_KEY, self.discovery_fetcher_widget.proxy_input.text())
        if self.article_fetcher_widget:
            settings.setValue(self.ARTICLE_PROXY_KEY, self.article_fetcher_widget.proxy_input.text())

        self.thread_pool.waitForDone(3000)
        self.thread_pool.clear()
        event.accept()

    # --- NEW: Slot for Save Code Button ---
    def _save_generated_code(self):
        """Saves the content of the generated code text box to a file."""
        code_content = self.generated_code_text.toPlainText()
        if not code_content:
            self.status_bar.showMessage("Nothing to save.", 3000)
            return

        # Open "Save As" dialog
        default_filename = "CrawlerGenerated.py"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Generated Code",
            default_filename,
            "Python Files (*.py);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(code_content)
                self.status_bar.showMessage(f"Code saved to {file_path}", 5000)
            except Exception as e:
                self.status_bar.showMessage(f"Error saving file: {e}", 5000)
                self.append_log_history(f"[Error] Failed to save code: {e}")

    # --- [NEW] UI Helper Functions (for dynamic widgets) ---

    def _update_discoverer_options_ui(self, discoverer_name: str):
        """Shows/hides discoverer-specific options based on selection."""
        is_smart = (discoverer_name == "Smart Analysis")
        if self.ai_signature_label:
            self.ai_signature_label.setVisible(is_smart)
        if self.ai_signature_input:
            self.ai_signature_input.setVisible(is_smart)

    def _update_extractor_options_ui(self, extractor_name: str):
        """Shows/hides extractor-specific options based on selection."""
        is_generic_css = (extractor_name == "Generic CSS")
        if self.css_selector_label:
            self.css_selector_label.setVisible(is_generic_css)
        if self.css_selector_input:
            self.css_selector_input.setVisible(is_generic_css)

    # --- NEW: URL History Management Methods ---

    def _load_url_history(self):
        """Loads URL history from QSettings into the ComboBox."""
        settings = QSettings(SETTING_ORG, SETTING_APP)
        history = settings.value(self.URL_HISTORY_KEY, [], type=list)
        if history:
            self.url_input.addItems(history)
            self.url_input.setCurrentIndex(-1)  # Show placeholder

    def _save_url_history(self, url: str):
        """Saves a new URL to the top of the history and QSettings."""
        if not url:
            return

        # 1. Find if item already exists
        found_index = self.url_input.findText(url, Qt.MatchFixedString)

        # 2. Remove if exists
        if found_index >= 0:
            self.url_input.removeItem(found_index)

        # 3. Add to top
        self.url_input.insertItem(0, url)
        self.url_input.setCurrentText(url)  # Ensure it's the selected item

        # 4. Trim history if over limit
        while self.url_input.count() > self.MAX_URL_HISTORY:
            self.url_input.removeItem(self.MAX_URL_HISTORY)

        # 5. Persist to QSettings
        new_history = [self.url_input.itemText(i) for i in range(self.url_input.count())]
        settings = QSettings(SETTING_ORG, SETTING_APP)
        settings.setValue(self.URL_HISTORY_KEY, new_history)

    def _show_url_history_context_menu(self, pos):
        """Shows a right-click context menu for the URL ComboBox."""
        menu = QMenu(self)
        clear_action = menu.addAction("Clear History")

        action = menu.exec_(self.url_input.mapToGlobal(pos))

        if action == clear_action:
            self._clear_url_history()

    def _clear_url_history(self):
        """Clears the ComboBox and the QSettings history."""
        self.url_input.clear()  # Clears the list
        self.url_input.clearEditText()  # Clears the typed text

        settings = QSettings(SETTING_ORG, SETTING_APP)
        settings.setValue(self.URL_HISTORY_KEY, [])
        self.status_bar.showMessage("URL history cleared.")


# =============================================================================
#
# SECTION 4: Main Execution
#
# =============================================================================

if __name__ == "__main__":
    if not QWebEngineView:
        print("\n--- WARNING ---")
        print("PyQtWebEngine not found. The Article web preview will be disabled.")
        print("Please install it for full functionality: pip install PyQtWebEngine")

    if not sync_playwright:
        print("\n--- WARNING ---")
        print("Playwright not found. 'Advanced' and 'Stealth' fetchers will be disabled.")
        print("Please install it: pip install playwright && python -m playwright install")

    app = QApplication(sys.argv)

    app.setOrganizationName(SETTING_ORG)
    app.setApplicationName(SETTING_APP)

    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    main_window = CrawlerPlaygroundApp()
    main_window.show()

    sys.exit(app.exec_())
