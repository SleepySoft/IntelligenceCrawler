from pathlib import Path

from IntelligenceCrawler.ProjectCatalog import scan_crawler_catalog


def test_catalog_scans_configs_without_executing_them(tmp_path):
    task_root = tmp_path / "CrawlTasks"
    news = task_root / "news"
    news.mkdir(parents=True)
    (news / "crawler_config_demo.py").write_text(
        """
raise RuntimeError('catalog must not execute config modules')
CRAWLER_CONFIG = {
    'discoverer_name': 'RSSDiscoverer',
    'extractor_name': 'Trafilatura',
    'entry_points': {'news': 'https://example.test/feed'},
    'url_filter': {'allowed_hosts': ['example.test']},
    'url_filter_samples': [{'url': 'https://example.test/news/a', 'allowed': True}],
    'article_filter': _custom_filter,
}
""",
        encoding="utf-8",
    )
    (news / "task_crawl_special.py").write_text(
        "\ufefffrom crawler_config_demo import CRAWLER_CONFIG\n",
        encoding="utf-8",
    )
    collector = tmp_path / "collector.json"
    collector.write_text(
        '{"collector":{"disabled_tasks":["task_crawl_special"]}}',
        encoding="utf-8",
    )

    entries = scan_crawler_catalog(task_root, collector)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.name == "demo"
    assert entry.group == "news"
    assert entry.discoverer == "RSSDiscoverer"
    assert entry.extractor == "Trafilatura"
    assert entry.entry_count == 1
    assert entry.has_declarative_filter
    assert entry.has_custom_filter
    assert entry.regression_sample_count == 1
    assert entry.task_path.name == "task_crawl_special.py"
    assert entry.disabled


def test_catalog_skips_invalid_python_and_marks_config_only(tmp_path):
    task_root = tmp_path / "tasks"
    task_root.mkdir()
    (task_root / "crawler_config_bad.py").write_text("not valid [", encoding="utf-8")
    good = task_root / "crawler_config_good.py"
    good.write_text("CRAWLER_CONFIG = {'entry_points': []}\n", encoding="utf-8")

    entries = scan_crawler_catalog(task_root)

    assert [entry.config_path for entry in entries] == [good.resolve()]
    assert entries[0].task_path is None
