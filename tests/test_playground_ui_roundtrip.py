import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTreeWidgetItem

from IntelligenceCrawler.CrawlerPlayground import CrawlerPlaygroundApp


def test_real_playground_load_build_and_generate_round_trip(tmp_path):
    config_path = tmp_path / "crawler_config_ui_test.py"
    config_path.write_text(
        """
def keep_article(url: str) -> bool:
    return '/news/' in url

CRAWLER_CONFIG = {
    'entry_points': {'news': 'https://example.test/feed'},
    'discoverer_name': 'RSSDiscoverer',
    'discoverer_init_param': {'verbose': False, 'hidden': 'preserve'},
    'd_fetcher_name': 'RequestsFetcher',
    'd_fetcher_init_param': {'log_callback': print, 'proxy': None, 'timeout_s': 10},
    'd_fetcher_kwargs': {},
    'e_fetcher_name': 'RequestsFetcher',
    'e_fetcher_init_param': {'log_callback': print, 'proxy': None, 'timeout_s': 10},
    'e_fetcher_kwargs': {},
    'extractor_name': 'Trafilatura',
    'extractor_init_param': {'verbose': False},
    'extractor_kwargs': {},
    'url_filter': {'allowed_hosts': ['example.test'], 'path_includes': ['/news/']},
    'url_filter_samples': [
        {'url': 'https://example.test/news/a', 'allowed': True},
        {'url': 'https://outside.test/news/a', 'allowed': False},
    ],
    'period_filter': (None, None),
    'channel_filter': None,
    'article_filter': keep_article,
    'content_handler': None,
    'exception_handler': None,
}
""",
        encoding="utf-8",
    )
    app = QApplication.instance() or QApplication([])
    window = CrawlerPlaygroundApp()
    try:
        window.load_config_path(config_path)
        rebuilt = window._build_flat_config_dict()
        assert rebuilt["discoverer_init_param"]["hidden"] == "preserve"
        assert callable(rebuilt["article_filter"])
        assert rebuilt["url_filter"]["allowed_hosts"] == ["example.test"]
        assert "passed: 2" in window.diagnostics_view.toPlainText()

        generated = window.generated_code_text.toPlainText()
        namespace = {}
        exec(compile(generated, "<playground-generated>", "exec"), namespace)
        generated_config = namespace["CRAWLER_CONFIG"]
        assert generated_config["article_filter"]("https://example.test/news/a")
        assert generated_config["url_filter"] == rebuilt["url_filter"]
        assert window._config_state.changes(
            window._collect_ui_config_projection()
        ) == []

        channel = QTreeWidgetItem(["https://example.test/feed"])
        channel.setData(0, Qt.UserRole, {
            "type": "channel", "url": "https://example.test/feed",
        })
        channel.setFlags(channel.flags() | Qt.ItemIsUserCheckable)
        channel.setCheckState(0, Qt.Checked)
        window.tree_widget.addTopLevelItem(channel)
        selected = window._build_flat_config_dict()["channel_filter"]
        window._pending_channel_filter_keys = selected["channel_list_filter"]
        window.tree_widget.clear()
        assert window._build_flat_config_dict()["channel_filter"] == selected
    finally:
        window.close()
