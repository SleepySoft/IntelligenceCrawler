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

try:
    from CrawlerCodeGenerator import CrawlerCodeGenerator
except Exception as e:
    print(str(e))
    from .CrawlerCodeGenerator import CrawlerCodeGenerator


# --- Core Component Imports ---

try:
    from IntelligenceCrawler.Fetcher import Fetcher, PlaywrightFetcher, RequestsFetcher, fetcher_factory
except ImportError:
    print("!!! CRITICAL: Could not import Fetcher classes.")

    # Mock classes to allow UI to load
    class Fetcher: pass
    class PlaywrightFetcher: pass
    class RequestsFetcher: pass
try:
    from IntelligenceCrawler.Discoverer import IDiscoverer, SitemapDiscoverer, RSSDiscoverer, ListPageDiscoverer, \
    discoverer_factory
except ImportError:
    print("!!! CRITICAL: Could not import Discoverer classes.")
    class IDiscoverer: pass
    class SitemapDiscoverer: pass
    class RSSDiscoverer: pass
    class ListPageDiscoverer: pass
try:
    from IntelligenceCrawler.Extractor import (
        IExtractor, PassThroughExtractor, TrafilaturaExtractor, ReadabilityExtractor,
        Newspaper3kExtractor, GenericCSSExtractor, Crawl4AIExtractor, ExtractionResult, extractor_factory
    )
    # Store imported classes for factory
    EXTRACTOR_MAP = {
        "PassThrough": PassThroughExtractor,
        "Trafilatura": TrafilaturaExtractor,
        "Readability": ReadabilityExtractor,
        "Newspaper3k": Newspaper3kExtractor,
        "Generic CSS": GenericCSSExtractor,
        "Crawl4AI": Crawl4AIExtractor,
    }
except ImportError:
    print("!!! CRITICAL: Could not import Extractor classes.")
    class IExtractor: pass
    EXTRACTOR_MAP = {}

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
    QMenu, QAction, QFileDialog, QFormLayout, QGridLayout, QDialogButtonBox, QDialog, QListView, QAbstractScrollArea,
    QMessageBox
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
# Reusable Fetcher Configuration Widget
#
# =============================================================================

class FetcherConfigWidget(QWidget):
    """
    A reusable widget encapsulating all UI controls for fetcher configuration.
    (一个可复用的窗口部件，封装了所有用于 fetcher 配置的 UI 控件。)

    [MODIFIED] Now supports 'one_row' or 'two_row' (multi-row grid) layout.
    ([已修改] 现在支持 'one_row' 或 'two_row' (多行网格) 布局。)
    """

    def __init__(self, layout_style: str = 'two_row', parent: Optional[QWidget] = None):
        """
        Initialize the widget.
        (初始化窗口部件。)

        Args:
            layout_style (str): 'one_row' or 'two_row'.
                                'one_row' uses a QHBoxLayout.
                                'two_row' (default) uses a QGridLayout.
            parent (Optional[QWidget]): Parent widget.
        """
        super().__init__(parent)
        self.layout_style = layout_style

        # --- Create Widgets (This part is unchanged) ---
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

        self.wait_until_label = QLabel("WaitUntil:")
        self.wait_until_combo = QComboBox()
        self.wait_until_combo.addItems(['networkidle', 'load', 'domcontentloaded', 'commit'])
        self.wait_until_combo.setToolTip("Playwright page.goto() wait_until option.")

        self.wait_selector_label = QLabel("Wait Selector:")
        self.wait_selector_input = QLineEdit()
        self.wait_selector_input.setPlaceholderText("e.g., #main-content")
        self.wait_selector_input.setToolTip("Playwright: wait for this selector to appear before returning.")

        # --- [NEW] Scroll Pages Widget ---
        self.scroll_pages_label = QLabel("Scroll:")
        self.scroll_pages_spin = QSpinBox()
        self.scroll_pages_spin.setRange(-100, 100)
        self.scroll_pages_spin.setValue(0)
        self.scroll_pages_spin.setToolTip("How many 'pages' to scroll to load lazy content.\n"
                                          "> 0: Scroll Down (load more)\n"
                                          "< 0: Scroll Up\n"
                                          "0: Disabled")

        # --- Conditional Layout ---
        if self.layout_style == 'one_row':
            # --- 'one_row' LAYOUT (using QHBoxLayout) ---
            # (用于 Discovery bar)
            layout = QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(5)  # 控件间的紧凑间距

            layout.addWidget(QLabel("Fetcher:"))
            layout.addWidget(self.fetcher_combo)
            layout.addWidget(QLabel("Timeout:"))
            layout.addWidget(self.timeout_spin)
            layout.addWidget(self.pause_check)
            layout.addWidget(self.render_check)

            layout.addWidget(QLabel("Proxy:"))
            layout.addWidget(self.proxy_input, 1)  # 代理输入框占满剩余空间

            # 添加 Playwright 控件 (它们默认隐藏)
            layout.addWidget(self.wait_until_label)
            layout.addWidget(self.wait_until_combo)
            layout.addWidget(self.wait_selector_label)
            layout.addWidget(self.wait_selector_input, 1)  # 选择器输入框也占满空间

            # --- Add scroll widgets to one_row layout ---
            layout.addWidget(self.scroll_pages_label)
            layout.addWidget(self.scroll_pages_spin)

        else:
            # --- 'two_row' LAYOUT (using QGridLayout) ---
            grid_layout = QGridLayout(self)
            grid_layout.setContentsMargins(0, 0, 0, 0)
            grid_layout.setSpacing(5)  # 增加控件间距

            # --- Row 0 ---
            grid_layout.addWidget(QLabel("Fetcher:"), 0, 0)
            grid_layout.addWidget(self.fetcher_combo, 0, 1)  # Col 1
            grid_layout.addWidget(QLabel("Timeout:"), 0, 2)
            grid_layout.addWidget(self.timeout_spin, 0, 3)  # Col 3
            grid_layout.addWidget(self.pause_check, 0, 4)  # Col 4
            grid_layout.addWidget(self.render_check, 0, 5)  # Col 5

            # --- Add Scroll to Row 0 (to keep it 2 rows) ---
            grid_layout.addWidget(self.scroll_pages_label, 0, 6)  # Col 6
            grid_layout.addWidget(self.scroll_pages_spin, 0, 7)   # Col 7

            # --- Row 1 ---
            grid_layout.addWidget(QLabel("Proxy:"), 1, 0)
            grid_layout.addWidget(self.proxy_input, 1, 1)  # Col 1

            # [FIX] Playwright 控件现在位于第 1 行 (Row 1)
            grid_layout.addWidget(self.wait_until_label, 1, 2)
            grid_layout.addWidget(self.wait_until_combo, 1, 3)  # Col 3
            grid_layout.addWidget(self.wait_selector_label, 1, 4)
            grid_layout.addWidget(self.wait_selector_input, 1, 5)  # Col 5

            # --- Set Column Stretches ---
            # (设置列的拉伸，使输入框和下拉框可以扩展)
            grid_layout.setColumnStretch(1, 2)  # (Fetcher Combo / Proxy Input)
            grid_layout.setColumnStretch(3, 1)  # (Timeout Spin / WaitUntil Combo)
            grid_layout.setColumnStretch(5, 2)  # (Render Check / Wait Selector Input)
            grid_layout.setColumnStretch(7, 1)  # [NEW] (Scroll Spin)

        # --- Connect Signals (Unchanged) ---
        self.fetcher_combo.currentTextChanged.connect(self._on_fetcher_changed)

        # --- Initial State (Unchanged) ---
        self._on_fetcher_changed(self.fetcher_combo.currentText())

    def _on_fetcher_changed(self, text: str):
        """Show/hide Playwright options based on fetcher selection."""
        # (This method works for both layouts without modification)
        is_playwright = "Playwright" in text
        self.wait_until_label.setVisible(is_playwright)
        self.wait_until_combo.setVisible(is_playwright)
        self.wait_selector_label.setVisible(is_playwright)
        self.wait_selector_input.setVisible(is_playwright)

        # --- [NEW] Add scroll widgets to show/hide logic ---
        self.scroll_pages_label.setVisible(is_playwright)
        self.scroll_pages_spin.setVisible(is_playwright)
        # --- [END NEW] ---

        self.pause_check.setEnabled(is_playwright)
        self.render_check.setEnabled(is_playwright)
        if not is_playwright:
            self.pause_check.setChecked(False)
            self.render_check.setChecked(False)
            self.scroll_pages_spin.setValue(0)  # [NEW] Reset to 0 if not PW

    def set_defaults(self, fetcher_name: str, timeout: int, render: bool, proxy: str = ""):
        """Set the default values for the widget."""
        self.fetcher_combo.setCurrentText(fetcher_name)
        self.timeout_spin.setValue(timeout)
        self.render_check.setChecked(render)
        self.proxy_input.setText(proxy)
        self.scroll_pages_spin.setValue(0) # 确保滚动默认为0
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
            'scroll_pages': self.scroll_pages_spin.value() if is_playwright else 0
        }

    def load_from_config(self, fetcher_name: str, init_params: dict, runtime_kwargs: dict):
        """
        Smartly loads configuration back into the widget controls.
        """
        # 1. 设置 Fetcher 类型 (这将触发 _on_fetcher_changed 信号，更新 UI 可见性)
        # 我们需要把类名 (RequestsFetcher) 映射回 UI 名称 (Simple (Requests))
        # 这需要反向映射，或者我们在 UI ComboBox 里存 UserData

        # 简单的反向查找逻辑：
        target_ui_name = None
        for i in range(self.fetcher_combo.count()):
            ui_text = self.fetcher_combo.itemText(i)
            # 这里的判断逻辑需要和你的生成逻辑对应
            if "Requests" in fetcher_name and "Requests" in ui_text:
                target_ui_name = ui_text
                break
            if "Playwright" in fetcher_name:
                if init_params.get("stealth") and "Stealth" in ui_text:
                    target_ui_name = ui_text
                    break
                elif not init_params.get("stealth") and "Advanced" in ui_text:
                    target_ui_name = ui_text
                    break

        if target_ui_name:
            self.fetcher_combo.setCurrentText(target_ui_name)

        # 2. 填充初始化参数 (Init Params)
        self.proxy_input.setText(init_params.get('proxy') or "")
        self.timeout_spin.setValue(int(init_params.get('timeout_s', 30)))

        # Playwright 特有
        if "Playwright" in fetcher_name:
            self.pause_check.setChecked(init_params.get('pause_browser', False))
            self.render_check.setChecked(init_params.get('render_page', False))

        # 3. 填充运行时参数 (Runtime Kwargs)
        if "Playwright" in fetcher_name:
            self.wait_until_combo.setCurrentText(runtime_kwargs.get('wait_until', 'networkidle'))
            self.wait_selector_input.setText(runtime_kwargs.get('wait_for_selector') or "")
            self.scroll_pages_spin.setValue(runtime_kwargs.get('scroll_pages', 0))


class AdjustableWidthComboBox(QComboBox):
    """
    修正版：确保下拉列表宽度不小于 QComboBox 自身宽度，并限制最大宽度。
    """

    def __init__(self, parent=None, max_dropdown_width=800):
        super().__init__(parent)
        self.max_dropdown_width = max_dropdown_width

        list_view = QListView()
        self.setView(list_view)

        # 启用水平滚动条
        list_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # 关键设置：限制下拉列表的最大宽度
        # 这样即使内容超长，列表也不会溢出屏幕
        list_view.setMaximumWidth(self.max_dropdown_width)

    def showPopup(self):
        """
        覆盖 showPopup 方法，在下拉列表弹出前动态设置其最小宽度。
        """
        # 1. 获取 QComboBox 自身的当前宽度
        combobox_width = self.width()

        # 2. 获取下拉列表的视图
        list_view = self.view()

        # 3. 【关键修正】设置最小宽度：
        #    确保下拉列表至少和 QComboBox 自身一样宽。
        #    如果 QComboBox 很长，列表就会跟着长。
        list_view.setMinimumWidth(combobox_width)

        # 4. 调用基类的 showPopup 方法
        super().showPopup()


class SignatureInspectorDialog(QDialog):
    """
    A dialog to display signature groups and select the best one.
    (一个用于显示签名组并选择最佳签名的对话框。)
    """

    def __init__(self, groups_data: List[Dict[str, Any]], page_url: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.groups_data = groups_data
        self.selected_signature: Optional[str] = None

        self.setWindowTitle(f"Signature Inspector - {page_url}")
        self.setMinimumSize(800, 600)  # 设置一个合理的最小尺寸
        self.setLayout(QVBoxLayout())

        # 1. 帮助文本
        self.layout().addWidget(QLabel(
            "Double-click or select a signature and press OK. "
            "Clicking a sample link also selects its parent signature."
        ))

        # 2. 树形控件
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Signature / Sample Text", "Count / Sample URL"])
        self.tree.setColumnWidth(0, 500)  # 签名列更宽
        self.layout().addWidget(self.tree, 1)

        self._populate_tree()

        # 3. 按钮
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        # OK 按钮默认禁用，直到用户选择
        self.ok_button = self.button_box.button(QDialogButtonBox.Ok)
        self.ok_button.setEnabled(False)
        self.layout().addWidget(self.button_box)

        # 4. 连接信号
        self.tree.itemClicked.connect(self.on_item_clicked)
        self.tree.itemDoubleClicked.connect(self.on_item_double_clicked)

    def _populate_tree(self):
        """Fills the tree with signature groups and their sample links."""
        self.tree.clear()
        for group in self.groups_data:
            # 创建父项 (签名)
            parent_item = QTreeWidgetItem([
                group.get('signature', 'N/A'),
                str(group.get('count', 0))
            ])
            # [关键] 将签名字符串存储在 UserRole 中，以便轻松检索
            parent_item.setData(0, Qt.UserRole, group.get('signature'))
            parent_item.setToolTip(0, f"Signature:\n{group.get('signature')}")
            parent_item.setToolTip(1, f"Count: {group.get('count', 0)}")
            self.tree.addTopLevelItem(parent_item)

            # 创建子项 (样本链接)
            for link in group.get('sample_links', []):
                child_item = QTreeWidgetItem([
                    link.get('text', '[no text]'),
                    link.get('href', 'N/A')
                ])
                child_item.setToolTip(0, f"Sample Text:\n{link.get('text')}")
                child_item.setToolTip(1, f"Sample URL:\n{link.get('href')}")
                parent_item.addChild(child_item)

        self.tree.expandAll()
        self.tree.resizeColumnToContents(0)
        self.tree.resizeColumnToContents(1)

    def on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Selects the parent signature when any item is clicked."""
        parent = item
        # 循环查找顶层父项
        while parent.parent():
            parent = parent.parent()

        # 检索存储的签名
        signature = parent.data(0, Qt.UserRole)
        if signature:
            self.selected_signature = signature
            self.ok_button.setEnabled(True)
            # (可选) 设置窗口状态栏文本，如果它有的话
            # self.statusBar().showMessage(f"Selected: {signature}")

    def on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Double-clicking accepts the selection."""
        self.on_item_clicked(item, column)  # 确保选中
        if self.selected_signature:
            self.accept()  # 接受对话框

    def get_selected_signature(self) -> Optional[str]:
        """Public method to retrieve the result."""
        return self.selected_signature


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
    """
    Worker thread for Stage 1: Discovering all channels.
    Refactored to use config dict and factories.
    """

    def __init__(self,
                 discoverer_config: Dict[str, Any],  # 接收整个 discoverer 配置块
                 entry_point: Any,  # URL 列表
                 start_date: Optional[datetime.datetime],
                 end_date: Optional[datetime.datetime]):
        super(ChannelDiscoveryWorker, self).__init__()
        self.config = discoverer_config
        self.entry_point = entry_point
        self.start_date = start_date
        self.end_date = end_date
        self.signals = WorkerSignals()

    def run(self):
        fetcher: Optional[Fetcher] = None
        try:
            log_callback = self.signals.progress.emit

            # --- 1. Setup Fetcher using Factory ---
            fetcher_cfg = self.config.get('fetcher', {})
            fetcher_params = fetcher_cfg.get('parameters', {}).copy()

            # 注入运行时回调
            fetcher_params['log_callback'] = log_callback

            fetcher_name = fetcher_cfg.get('class')
            fetcher = fetcher_factory(fetcher_name, fetcher_params)

            # --- 2. Setup Discoverer using Factory ---
            discoverer_name = self.config.get('class')
            discoverer_args = self.config.get('args', {}).copy()

            # 注入运行时依赖
            discoverer_args['fetcher'] = fetcher
            discoverer_args['verbose'] = True

            discoverer = discoverer_factory(discoverer_name, discoverer_args)

            # --- 3. Execution ---
            # 运行时参数 (如 wait_until)
            runtime_kwargs = self.config.get('fetcher_kwargs', {})

            channel_list = discoverer.discover_channels(
                self.entry_point,
                start_date=self.start_date,
                end_date=self.end_date,
                fetcher_kwargs=runtime_kwargs
            )
            self.signals.result.emit(channel_list)

        except Exception as e:
            ex_type, ex_value, tb_str = sys.exc_info()
            self.signals.error.emit((str(ex_type), str(e), traceback.format_exc()))
        finally:
            if fetcher: fetcher.close()
            self.signals.finished.emit()


class ArticleListWorker(QRunnable):
    """
    Worker thread for Stage 2: Gets articles for one channel.
    """

    def __init__(self,
                 discoverer_config: Dict[str, Any],
                 channel_url: str):
        super(ArticleListWorker, self).__init__()
        self.config = discoverer_config
        self.channel_url = channel_url
        self.signals = WorkerSignals()

    def run(self):
        fetcher: Optional[Fetcher] = None
        try:
            log_callback = self.signals.progress.emit

            # --- 1. Setup Fetcher ---
            fetcher_cfg = self.config.get('fetcher', {})
            # 注意：列表抓取通常也需要渲染，所以沿用 discovery 的配置
            # 或者，如果逻辑上列表抓取需要强制渲染，可以在这里修改 fetcher_params
            fetcher_params = fetcher_cfg.get('parameters', {}).copy()
            fetcher_params['log_callback'] = log_callback

            fetcher = fetcher_factory(fetcher_cfg.get('class'), fetcher_params)

            # --- 2. Setup Discoverer ---
            discoverer_name = self.config.get('class')
            discoverer_args = self.config.get('args', {}).copy()
            discoverer_args['fetcher'] = fetcher

            discoverer = discoverer_factory(discoverer_name, discoverer_args)

            # --- 3. Execution ---
            runtime_kwargs = self.config.get('fetcher_kwargs', {})

            article_list = discoverer.get_articles_for_channel(
                self.channel_url,
                fetcher_kwargs=runtime_kwargs
            )
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


class ExtractionWorker(QRunnable):
    """
    Worker thread for Stage 3: Fetching and Extracting.
    """

    def __init__(self,
                 extractor_config: Dict[str, Any],  # 接收整个 extractor 配置块
                 url_to_extract: str):
        super(ExtractionWorker, self).__init__()
        self.config = extractor_config
        self.url_to_extract = url_to_extract
        self.signals = WorkerSignals()

    def run(self):
        fetcher: Optional[Fetcher] = None
        try:
            log_callback = self.signals.progress.emit

            # --- 1. Setup Fetcher ---
            fetcher_cfg = self.config.get('fetcher', {})
            fetcher_params = fetcher_cfg.get('parameters', {}).copy()
            fetcher_params['log_callback'] = log_callback

            fetcher = fetcher_factory(fetcher_cfg.get('class'), fetcher_params)

            # --- 2. Get Content ---
            runtime_kwargs = self.config.get('fetcher_kwargs', {})

            # 从 kwargs 提取 fetcher.get_content 需要的参数
            # 注意：timeout_s 在 params 里有，但 get_content 有时也需要 wait_for_timeout_s
            # 我们可以直接把整个 runtime_kwargs 传进去，只要 Fetcher 支持 **kwargs
            content_bytes = fetcher.get_content(
                self.url_to_extract,
                **runtime_kwargs
            )

            if not content_bytes:
                raise ValueError("Failed to fetch content (returned None).")

            # --- 3. Setup Extractor ---
            extractor_name = self.config.get('class')
            extractor_args = self.config.get('args', {}).copy()
            extractor_args['verbose'] = True

            extractor = extractor_factory(extractor_name, extractor_args)

            # --- 4. Execution ---
            # Extractor.extract 通常接收 bytes, url, 以及额外的提取参数 (如 selectors)
            # 这些参数应该已经在 extractor_args 里了，或者需要单独拆分
            # 在 _build_config_dict 中，selectors 放在 args 里

            # 注意：某些 Extractor 的 extract 方法参数不同。
            # GenericCSSExtractor 需要 selectors 列表。
            # 这里我们将 args 作为 kwargs 传给 extract 方法
            extract_runtime_args = extractor_args.copy()
            # 移除 verbose，因为它是 __init__ 参数
            if 'verbose' in extract_runtime_args: del extract_runtime_args['verbose']

            markdown_result = extractor.extract(
                content_bytes,
                self.url_to_extract,
                **extract_runtime_args
            )
            self.signals.result.emit(markdown_result)

        except Exception as e:
            ex_type, ex_value, tb_str = sys.exc_info()
            self.signals.error.emit((str(ex_type), str(e), traceback.format_exc()))
        finally:
            if fetcher: fetcher.close()
            self.signals.finished.emit()


class ChannelSourceWorker(QRunnable):
    """
    Worker thread to fetch raw channel content (e.g., XML) for the text viewer.
    Refactored to use config dict and factories.
    """

    def __init__(self,
                 discoverer_config: Dict[str, Any],  # 接收标准配置
                 url: str):
        super(ChannelSourceWorker, self).__init__()
        self.config = discoverer_config
        self.url = url
        self.signals = WorkerSignals()

    def run(self):
        fetcher: Optional[Fetcher] = None
        try:
            log_callback = self.signals.progress.emit

            # 1. Setup Fetcher via Factory
            fetcher_cfg = self.config.get('fetcher', {})
            # 源码查看通常不需要渲染，为了速度和 XML 解析稳定性，强制 render=False
            # 但如果你希望跟 UI 设置保持一致，就直接用 copy。这里我们做一个特殊处理：
            fetcher_params = fetcher_cfg.get('parameters', {}).copy()
            fetcher_params['log_callback'] = log_callback
            fetcher_params['render_page'] = False  # Force False for source viewing

            fetcher = fetcher_factory(fetcher_cfg.get('class'), fetcher_params)

            # 2. Setup Discoverer via Factory
            discoverer_name = self.config.get('class')
            discoverer_args = self.config.get('args', {}).copy()
            discoverer_args['fetcher'] = fetcher

            discoverer = discoverer_factory(discoverer_name, discoverer_args)

            # 3. Execution
            # 获取运行时的 kwargs (如 headers, cookies 等，虽然目前主要是 wait_until)
            runtime_kwargs = self.config.get('fetcher_kwargs', {})

            # 使用 Discoverer 的通用接口 get_content_str
            content_string = discoverer.get_content_str(
                self.url,
                fetcher_kwargs=runtime_kwargs
            )
            self.signals.result.emit(content_string)

        except Exception as e:
            ex_type, ex_value, tb_str = sys.exc_info()
            self.signals.error.emit((str(ex_type), str(e), traceback.format_exc()))
        finally:
            if fetcher: fetcher.close()
            self.signals.finished.emit()


class SignatureAnalysisWorker(QRunnable):
    """
    Worker thread to analyze a list page and get all signature groups.
    Refactored to use config dict and factories.
    """

    def __init__(self,
                 discoverer_config: Dict[str, Any],  # 接收标准配置
                 url_to_analyze: str):
        super(SignatureAnalysisWorker, self).__init__()
        self.config = discoverer_config
        self.url_to_analyze = url_to_analyze
        self.signals = WorkerSignals()

    def run(self):
        fetcher: Optional[Fetcher] = None
        try:
            log_callback = self.signals.progress.emit
            log_callback(f"Starting signature analysis on {self.url_to_analyze}...")

            # 1. Setup Fetcher via Factory
            fetcher_cfg = self.config.get('fetcher', {})
            fetcher_params = fetcher_cfg.get('parameters', {}).copy()
            fetcher_params['log_callback'] = log_callback

            fetcher = fetcher_factory(fetcher_cfg.get('class'), fetcher_params)

            # 2. Setup Discoverer (Must be ListPageDiscoverer)
            # 即使 config 里写的是其他名字，这里逻辑上也必须是 ListPage。
            # 但既然 UI 做了限制，我们可以信任 config['class'] 就是 ListPageDiscoverer
            discoverer_name = self.config.get('class')
            discoverer_args = self.config.get('args', {}).copy()
            discoverer_args['fetcher'] = fetcher

            # 强制创建一个 Discoverer 实例
            discoverer = discoverer_factory(discoverer_name, discoverer_args)

            # 双重检查类型，防止运行时错误
            if not hasattr(discoverer, 'get_signature_groups'):
                raise ValueError(f"Discoverer '{discoverer_name}' does not support signature analysis.")

            # 3. Execution
            runtime_kwargs = self.config.get('fetcher_kwargs', {})

            groups_data = discoverer.get_signature_groups(
                self.url_to_analyze,
                fetcher_kwargs=runtime_kwargs
            )

            self.signals.result.emit(groups_data)

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

        self.manual_specified_signature_label: Optional[QLabel] = None
        self.manual_specified_signature_input: Optional[QLineEdit] = None
        self.scope_selector_label: Optional[QLabel] = None
        self.scope_selector_input: Optional[QLineEdit] = None

        self.css_selector_label: Optional[QLabel] = None
        self.css_selector_input: Optional[QLineEdit] = None

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
        self.create_menu()
        self._load_url_history()

        settings = QSettings(SETTING_ORG, SETTING_APP)
        settings.remove("mainWindowGeometry")
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
            print(f"Warning: Could not get screen geometry, falling back to fixed size. Error: {e}")
            self.setGeometry(100, 100, 1400, 900)

        default_width = 1200
        default_height = 800
        self.resize(default_width, default_height)

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

        self.url_input = AdjustableWidthComboBox(max_dropdown_width=1200)
        self.url_input.setEditable(True)
        self.url_input.setMaximumWidth(1200)
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
        self.manual_specified_signature_label = QLabel("AI Signature:")
        self.manual_specified_signature_input = QLineEdit()
        self.manual_specified_signature_input.setPlaceholderText("Optional: e.g., 'a[class*=\"title\"]'")
        self.manual_specified_signature_input.setToolTip("Manually specify the 'link fingerprint' signature.")
        self.manual_specified_signature_input.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        top_bar_row1_layout.addWidget(self.manual_specified_signature_label)
        top_bar_row1_layout.addWidget(self.manual_specified_signature_input, 1)  # Give it stretch

        self.scope_selector_label = QLabel("Scope:")
        self.scope_selector_input = QLineEdit()
        self.scope_selector_input.setPlaceholderText("e.g., '#main-content'")
        self.scope_selector_input.setToolTip("Limit discovery to this CSS selector (e.g., div.news-list).")
        # 让它稍微短一点，stretch factor 设为 0 或者 1，视情况而定
        self.scope_selector_input.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)

        top_bar_row1_layout.addWidget(self.scope_selector_label)
        top_bar_row1_layout.addWidget(self.scope_selector_input, 1)

        self.inspect_signature_button = QPushButton("Inspect...")
        self.inspect_signature_button.setToolTip(
            "Analyze the entry URL to find all possible link signatures.\n"
            "(Requires 'Smart Analysis' mode)"
        )
        self.inspect_signature_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        top_bar_row1_layout.addWidget(self.inspect_signature_button)
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

        self.discovery_fetcher_widget = FetcherConfigWidget(layout_style='one_row', parent=self)
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

        code_layout_button_line = QHBoxLayout()

        self.refresh_code_button = QPushButton("Refresh")
        self.refresh_code_button.setToolTip("Re-generated code based on current configuration.")
        code_layout_button_line.addWidget(self.refresh_code_button, 1)

        self.load_code_button = QPushButton(QIcon.fromTheme("document-open"), "Load Config...")
        self.load_code_button.setToolTip("Load a saved CrawlerConfig.py and restore UI settings.")
        self.load_code_button.clicked.connect(self.load_config_from_file)
        code_layout_button_line.addWidget(self.load_code_button, 1)  # Add with stretch 1

        self.save_code_button = QPushButton(QIcon.fromTheme("document-save"), "Save Code to File...")
        self.save_code_button.setToolTip("Save the generated code above to a Python file (e.g., CrawlerConfig.py)")
        code_layout_button_line.addWidget(self.save_code_button, 99)

        code_layout.addLayout(code_layout_button_line)
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

    def create_menu(self):
        # 1. 获取菜单栏
        menubar = self.menuBar()

        # 2. 创建 'File' 菜单
        file_menu = menubar.addMenu('&File')

        # 3. 添加 Load Action
        load_action = QAction(QIcon.fromTheme("document-open"), '&Load Configuration...', self)
        load_action.setShortcut('Ctrl+O')
        load_action.setStatusTip('Load a saved configuration file')
        load_action.triggered.connect(self.load_config_from_file)
        file_menu.addAction(load_action)

        # 4. 添加 Save Action
        save_action = QAction(QIcon.fromTheme("document-save"), '&Save Configuration...', self)
        save_action.setShortcut('Ctrl+S')
        save_action.setStatusTip('Save current configuration to file')
        save_action.triggered.connect(self._save_generated_code)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        # 5. 添加 Exit Action
        exit_action = QAction(QIcon.fromTheme("application-exit"), '&Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

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

        self.article_fetcher_widget = FetcherConfigWidget(layout_style='two_row', parent=self)
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
        self.inspect_signature_button.clicked.connect(self.start_signature_inspection)

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
        self.refresh_code_button.clicked.connect(self._re_generated_code)

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

    def start_channel_discovery(self):
        """Slot for 'Discover Channels' button. (Refactored: Uniform List Input)"""

        # 1. 获取 UI 输入
        raw_text = self.url_input.currentText().strip()
        self.discoverer_name = self.discoverer_combo.currentText()

        if not raw_text:
            self.status_bar.showMessage("Error: Please enter a URL or list of URLs.")
            return

        # 2. 解析输入
        entry_point_urls = raw_text.split()

        self.append_log_history(f"[Info] Dispatching {len(entry_point_urls)} URL(s) to {self.discoverer_name}...")

        # 3. 清理 UI 和保存历史
        self.clear_all_controls()
        self._save_url_history(raw_text)

        # =========================================================
        # 在生成配置之前，必须先更新“最后使用的入口点”缓存
        # 这样 _build_config_dict() 才能读到最新的值，
        # 从而保证 Worker 和 代码生成器 都能拿到这次提交的 URL。
        # =========================================================
        self.last_used_entry_point = entry_point_urls

        # 4. 现在可以安全地生成配置了
        full_config = self._build_config_dict()
        discoverer_config = full_config['discoverer']

        # 5. 准备日期过滤器
        start_date: Optional[datetime.datetime] = None
        end_date: Optional[datetime.datetime] = None

        if discoverer_config.get('date_filter', {}).get('enabled'):
            days = discoverer_config['date_filter']['days']
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days)

        # 6. 启动 Worker
        # 注意：这里可以直接传 self.last_used_entry_point (它现在是新的了)
        # 也可以传 entry_point_urls，两者现在相等。
        worker = ChannelDiscoveryWorker(
            discoverer_config=discoverer_config,
            entry_point=self.last_used_entry_point,
            start_date=start_date,
            end_date=end_date
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

        full_config = self._build_config_dict()
        discoverer_config = full_config['discoverer']

        worker = ArticleListWorker(
            discoverer_config=discoverer_config,
            channel_url=channel_url
        )

        worker.signals.result.connect(self.on_article_list_result)
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

        full_config = self._build_config_dict()
        extractor_config = full_config['extractor']

        url = self.article_url_input.text().strip()

        worker = ExtractionWorker(
            extractor_config=extractor_config,
            url_to_extract=url
        )

        worker.signals.result.connect(self.on_extraction_result)
        worker.signals.finished.connect(self.on_subtask_finished)
        worker.signals.error.connect(self.on_worker_error)
        worker.signals.progress.connect(self.status_bar.showMessage)
        worker.signals.progress.connect(self.append_log_history)

        self.thread_pool.start(worker)

    def start_channel_source_loading(self, url: str):
        """Starts worker to fetch raw channel source (e.g., XML) for the viewer."""
        self.channel_source_viewer.setPlainText(f"Loading source from {url}...")
        self.tab_widget.setCurrentWidget(self.channel_source_viewer)

        # [REFACTORED] Use unified config
        full_config = self._build_config_dict()
        discoverer_config = full_config['discoverer']

        worker = ChannelSourceWorker(
            discoverer_config=discoverer_config,
            url=url
        )

        worker.signals.result.connect(self.on_channel_source_result)
        worker.signals.finished.connect(self.on_worker_finished)
        worker.signals.error.connect(self.on_worker_error)
        worker.signals.progress.connect(self.status_bar.showMessage)
        worker.signals.progress.connect(self.append_log_history)

        self.thread_pool.start(worker)

    def start_signature_inspection(self):
        """
        Slot for the 'Inspect...' button.
        Starts the SignatureAnalysisWorker.
        """
        if self.discoverer_combo.currentText() != "Smart Analysis":
            self.status_bar.showMessage("Error: Signature inspection only works with 'Smart Analysis' discoverer.")
            return

        url = self.url_input.currentText().strip()
        if not url:
            self.status_bar.showMessage("Error: Please enter a URL to inspect.")
            return

        self.set_loading_state(True, f"Inspecting signatures for {url}...")

        # [REFACTORED] Use unified config
        full_config = self._build_config_dict()
        discoverer_config = full_config['discoverer']

        # 注意：_build_config_dict 已经根据 UI 状态获取了 scope_selector 等参数
        # 并放入了 discoverer_config['args'] 中，所以这里不需要再手动获取 scope 并传入 Worker

        worker = SignatureAnalysisWorker(
            discoverer_config=discoverer_config,
            url_to_analyze=url
        )

        worker.signals.result.connect(self.on_signature_inspection_result)
        worker.signals.finished.connect(self.on_subtask_finished)
        worker.signals.error.connect(self.on_worker_error)
        worker.signals.progress.connect(self.status_bar.showMessage)
        worker.signals.progress.connect(self.append_log_history)

        self.thread_pool.start(worker)

    def on_signature_inspection_result(self, groups_data: List[Dict[str, Any]]):
        """
        Slot for SignatureAnalysisWorker 'result' signal.
        Shows the SignatureInspectorDialog.
        """
        self.set_loading_state(False, "Signature inspection complete.")

        if not groups_data:
            self.status_bar.showMessage("Inspection found 0 signature groups.")
            self.append_log_history("[Inspect] No signature groups found.")
            return

        self.append_log_history(f"[Inspect] Found {len(groups_data)} signature groups. Opening dialog...")

        url = self.url_input.currentText().strip()  # 获取 URL 用于对话框标题
        dialog = SignatureInspectorDialog(groups_data, url, self)

        # 以模态方式执行对话框
        if dialog.exec_() == QDialog.Accepted:
            selected_sig = dialog.get_selected_signature()
            if selected_sig:
                self.manual_specified_signature_input.setText(selected_sig)
                self.status_bar.showMessage(f"AI Signature set from inspector.")
                self.append_log_history(f"[Inspect] User selected signature: {selected_sig}")
            else:
                self.append_log_history("[Inspect] Dialog accepted but no signature selected.")
        else:
            self.append_log_history("[Inspect] User cancelled signature selection.")

    # --- Thread Result Slots ---

    def on_channel_discovery_result(self, channel_list: List[str]):
        """Slot for ChannelDiscoveryWorker 'result' signal."""
        self.last_used_entry_point = channel_list
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

    def on_subtask_finished(self):
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
                self.web_view.setUrl(QUrl(url))
                self.web_view.setFocus()
                self.tab_widget.setCurrentWidget(self.article_preview_widget)
                self.status_bar.showMessage(f"Loading page: {url}", 3000)

    def on_article_go_clicked(self):
        """Handles clicks on the 'Go' button in the article tab."""
        if self.web_view and QUrl:
            url = self.article_url_input.text()
            self.web_view.setUrl(QUrl(url))
            self.web_view.setFocus()

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
            code_script = CrawlerCodeGenerator().generate_code_from_config(config_dict)

            # Step 3: Display the generated code
            # (第 3 步：显示生成的代码)
            self.generated_code_text.setPlainText(code_script)

        except Exception as e:
            # Show any error during generation in the code block itself
            # (在代码块本身中显示生成期间的任何错误)
            error_msg = f"# Failed to generate code:\n# {type(e).__name__}: {e}\n\n"
            error_msg += traceback.format_exc()
            self.generated_code_text.setPlainText(error_msg)
            print(traceback.format_exc())
            print(error_msg)

    def _build_channel_filter_config(self) -> dict:
        """
        Generates the configuration dictionary for 'channel_filter_list'.
        This dictionary contains the "filter keys" (e.g., 'it/news_sitemap.xml')
        based on the user-checked channels in the tree.

        (生成 'channel_filter_list' 的配置字典。
         该字典包含基于用户在树中勾选的渠道的“过滤键”。)
        """

        def get_filter_key(url: str) -> str:
            """
            Helper function: Creates a simple, more unique filter key from a URL.
            (辅助函数：从 URL 创建一个简单、更独特的过滤键。)
            """
            try:
                parsed_url = urlparse(url)
                path = parsed_url.path

                if not path or path == '/':
                    return parsed_url.netloc or url

                if path.endswith('/'):
                    path = path[:-1]

                filename = os.path.basename(path)
                parent_dir_path = os.path.dirname(path)

                if parent_dir_path and parent_dir_path != '/':
                    parent_folder = os.path.basename(parent_dir_path)
                    return f"{parent_folder}/{filename}"
                else:
                    return filename

            except Exception:
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

        # Sort and de-duplicate the keys
        final_keys = sorted(list(set(checked_keys)))

        return {
            "channel_filter_keys": final_keys
        }

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

    def _build_config_dict(self) -> dict:
        """
        Reads all UI controls and builds the standardized config dictionary.
        Now uses standardized class names for Factories.
        """
        # --- 1. Discoverer Configuration ---
        d_fetcher_config_dict = self.discovery_fetcher_widget.get_config()

        # [Mapping] UI Name -> Class Name
        discovery_fetcher_class = self._map_ui_fetcher_to_class(d_fetcher_config_dict['fetcher_name'])

        # Fetcher Init Parameters (for Factory)
        discoverer_fetcher_params = {
            "proxy": d_fetcher_config_dict['proxy'],
            "timeout_s": d_fetcher_config_dict['timeout'],  # Factory expects timeout_s
            "stealth": "Stealth" in d_fetcher_config_dict['fetcher_name'],
            "pause_browser": d_fetcher_config_dict['pause'],
            "render_page": d_fetcher_config_dict['render']
        }

        # Fetcher Runtime Parameters (for .get_content / .discover)
        discoverer_fetcher_kwargs = {
            'wait_until': d_fetcher_config_dict.get('wait_until'),
            'wait_for_selector': d_fetcher_config_dict.get('wait_for_selector'),
            'wait_for_timeout_s': d_fetcher_config_dict.get('timeout'),
            'scroll_pages': d_fetcher_config_dict.get('scroll_pages', 0)
        }

        discoverer_ui_name = self.discoverer_combo.currentText()
        discoverer_class = self._map_ui_discoverer_to_class(discoverer_ui_name)

        discoverer_args = {
            "entry_point": self.last_used_entry_point,
            # Specific args for ListPageDiscoverer
            "scope_selector": self.scope_selector_input.text().strip() or None,
            "manual_specified_signature": self.manual_specified_signature_input.text().strip() or None,
            # Common args
            "verbose": True
        }

        # --- 2. Extractor Configuration ---
        e_fetcher_config_dict = self.article_fetcher_widget.get_config()
        article_fetcher_class = self._map_ui_fetcher_to_class(e_fetcher_config_dict['fetcher_name'])

        extractor_fetcher_params = {
            "proxy": e_fetcher_config_dict['proxy'],
            "timeout_s": e_fetcher_config_dict['timeout'],
            "stealth": "Stealth" in e_fetcher_config_dict['fetcher_name'],
            "pause_browser": e_fetcher_config_dict['pause'],
            "render_page": e_fetcher_config_dict['render']
        }

        extractor_fetcher_kwargs = {
            'wait_until': e_fetcher_config_dict['wait_until'],
            'wait_for_selector': e_fetcher_config_dict['wait_for_selector'],
            'wait_for_timeout_s': e_fetcher_config_dict['timeout'],
            'scroll_pages': e_fetcher_config_dict['scroll_pages']
        }

        extractor_name = self.extractor_combo.currentText()
        # Extractor args (init args + extract args mixed, separated by Worker logic if needed)
        extractor_args = self._get_current_extractor_args(extractor_name)

        # --- 3. Assemble Final Config ---
        config = {
            "discoverer": {
                "class": discoverer_class,
                "args": discoverer_args,
                "fetcher": {
                    "class": discovery_fetcher_class,
                    "parameters": discoverer_fetcher_params
                },
                "fetcher_kwargs": discoverer_fetcher_kwargs,
                # Extra meta info
                "date_filter": {
                    "enabled": self.date_filter_check.isChecked(),
                    "days": self.date_filter_days_spin.value()
                }
            },
            "extractor": {
                "class": extractor_name,  # Usually matches class name directly
                "args": extractor_args,
                "fetcher": {
                    "class": article_fetcher_class,
                    "parameters": extractor_fetcher_params
                },
                "fetcher_kwargs": extractor_fetcher_kwargs,
            },
            "channel_filter": self._build_channel_filter_config()
        }
        return config


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
        default_filename = "CrawlerConfig.py"
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

    def _re_generated_code(self):
        self.update_generated_code()

    # --- [NEW] UI Helper Functions (for dynamic widgets) ---

    def _update_discoverer_options_ui(self, discoverer_name: str):
        """Shows/hides discoverer-specific options based on selection."""
        is_smart = (discoverer_name == "Smart Analysis")
        if self.manual_specified_signature_label:
            self.manual_specified_signature_label.setVisible(is_smart)
        if self.manual_specified_signature_input:
            self.manual_specified_signature_input.setVisible(is_smart)
        if self.scope_selector_label:
            self.scope_selector_label.setVisible(is_smart)
        if self.scope_selector_input:
            self.scope_selector_input.setVisible(is_smart)
        if hasattr(self, 'inspect_signature_button'):
            self.inspect_signature_button.setVisible(is_smart)

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

    def _map_ui_fetcher_to_class(self, ui_name: str) -> str:
        if "Playwright" in ui_name:
            return "PlaywrightFetcher"
        return "RequestsFetcher"

    def _map_ui_discoverer_to_class(self, ui_name: str) -> str:
        mapping = {
            "Sitemap": "SitemapDiscoverer",
            "RSS": "RSSDiscoverer",
            "Smart Analysis": "ListPageDiscoverer"
        }
        return mapping.get(ui_name, ui_name)

        # 在 CrawlerPlaygroundApp 类中添加

    def load_config_from_file(self):
        """
        Loads a python config file and updates the UI.
        """
        # 1. 打开文件选择器
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "Python Files (*.py)"
        )
        if not file_path:
            return

        try:
            # 2. 动态加载 Python 文件 (即执行它以获取字典)
            import importlib.util
            spec = importlib.util.spec_from_file_location("loaded_config", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if not hasattr(module, 'CRAWLER_CONFIG'):
                raise ValueError("The file does not contain 'CRAWLER_CONFIG'.")

            config = module.CRAWLER_CONFIG

            # ==========================================
            # 3. 开始 UI 同步 (顺序非常重要)
            # ==========================================

            # --- A. 恢复 Entry Point (URL) ---
            entry_points = config.get('entry_points', [])
            if entry_points:
                # 假设我们只取第一个，或者把列表拼接成字符串
                url_text = " ".join(entry_points) if isinstance(entry_points, list) else str(entry_points)

                # 更新输入框
                self.url_input.setCurrentText(url_text)
                # 关键：同步更新缓存，否则下次生成会出错
                self.last_used_entry_point = entry_points

                # --- B. 恢复 Discoverer ---
            # B1. 先设置类型 (触发 UI 变化)
            disc_class = config.get('discoverer_name', 'SitemapDiscoverer')

            # 简单的名称映射回 UI
            ui_disc_map = {
                'SitemapDiscoverer': 'Sitemap',
                'RSSDiscoverer': 'RSS',
                'ListPageDiscoverer': 'Smart Analysis'
            }
            ui_disc_name = ui_disc_map.get(disc_class)
            if ui_disc_name:
                self.discoverer_combo.setCurrentText(ui_disc_name)

            # B2. 设置 Discoverer 参数 (Smart Analysis 特有)
            disc_init = config.get('discoverer_init_param', {})
            if disc_class == 'ListPageDiscoverer':
                if self.manual_specified_signature_input:
                    self.manual_specified_signature_input.setText(disc_init.get('manual_specified_signature') or "")
                if self.scope_selector_input:
                    self.scope_selector_input.setText(disc_init.get('scope_selector') or "")

            # --- C. 恢复 Discovery Fetcher ---
            self.discovery_fetcher_widget.load_from_config(
                fetcher_name=config.get('d_fetcher_name', ''),
                init_params=config.get('d_fetcher_init_param', {}),
                runtime_kwargs=config.get('d_fetcher_kwargs', {})
            )

            # --- D. 恢复 Extractor ---
            # D1. 设置类型
            ext_class = config.get('extractor_name', 'TrafilaturaExtractor')
            # 假设 UI 中的名字基本和类名对应 (去除 'Extractor' 后缀或查表)
            # 这里做一个简单处理：
            target_ext_ui = None
            for i in range(self.extractor_combo.count()):
                ui_text = self.extractor_combo.itemText(i)
                # 比如 TrafilaturaExtractor -> Trafilatura
                if ui_text in ext_class:
                    target_ext_ui = ui_text
                    break

            if target_ext_ui:
                self.extractor_combo.setCurrentText(target_ext_ui)

            # D2. 设置 Extractor 参数 (如 Generic CSS 的 selector)
            # 注意：Extractor 的参数在生成代码时可能分散在 init_param 和 kwargs 里
            # 根据你的生成逻辑，Generic CSS 的 selectors 应该在 kwargs 里
            ext_kwargs = config.get('extractor_kwargs', {})
            if "Generic" in ext_class:
                selectors = ext_kwargs.get('selectors', [])
                if selectors:
                    self.css_selector_input.setText(", ".join(selectors))

            # --- E. 恢复 Article Fetcher ---
            self.article_fetcher_widget.load_from_config(
                fetcher_name=config.get('e_fetcher_name', ''),
                init_params=config.get('e_fetcher_init_param', {}),
                runtime_kwargs=config.get('e_fetcher_kwargs', {})
            )

            # --- F. 恢复 Date Filter ---
            period = config.get('period_filter', (None, None))
            # period 可能是 (datetime, datetime) 或者 (None, None)
            if period and period[0]:
                self.date_filter_check.setChecked(True)
                # 计算天数差
                delta = period[1] - period[0]
                self.date_filter_days_spin.setValue(max(1, delta.days))
            else:
                self.date_filter_check.setChecked(False)

            self.status_bar.showMessage(f"Configuration loaded from {file_path}", 5000)

            # 最后：强制刷新一下代码预览，确保“加载”后的状态和“生成”的代码一致
            self.update_generated_code()

        except Exception as e:
            self.status_bar.showMessage(f"Failed to load config: {e}")
            print(traceback.format_exc())
            QMessageBox.warning(self, "Load Error", f"Could not load configuration:\n{str(e)}")


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
