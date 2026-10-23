import datetime

from IntelligenceCrawler.CrawlerCodeGenerator import CrawlerCodeGenerator
from IntelligenceCrawler.PlaygroundConfigState import PlaygroundConfigState


def _allow_news(url: str) -> bool:
    return "/news/" in url


def _baseline():
    return {
        "discoverer_name": "RSSDiscoverer",
        "discoverer_init_param": {"verbose": False, "hidden": "keep"},
        "d_fetcher_name": "RequestsFetcher",
        "d_fetcher_init_param": {"timeout_s": 10, "hidden": "keep"},
        "d_fetcher_kwargs": {},
        "article_filter": _allow_news,
        "content_handler": None,
        "period_filter": (
            datetime.datetime(2026, 10, 1),
            datetime.datetime(2026, 10, 2),
        ),
    }


def _projection():
    return {
        "discoverer_name": "RSSDiscoverer",
        "discoverer_init_param": {"verbose": True},
        "d_fetcher_name": "RequestsFetcher",
        "d_fetcher_init_param": {"timeout_s": 10, "proxy": None},
        "d_fetcher_kwargs": {},
        "article_filter": None,
        "content_handler": None,
        "date_filter_enabled": False,
        "date_filter_days": 7,
    }


def test_loaded_config_is_unchanged_without_ui_edits():
    state = PlaygroundConfigState()
    state.capture(_baseline(), _projection())

    assert state.merge(_projection()) == _baseline()


def test_single_ui_edit_preserves_hidden_and_runtime_fields():
    state = PlaygroundConfigState()
    state.capture(_baseline(), _projection())
    edited = _projection()
    edited["d_fetcher_init_param"]["timeout_s"] = 25

    rebuilt = state.merge(edited)

    assert rebuilt["d_fetcher_init_param"] == {"timeout_s": 25, "hidden": "keep"}
    assert rebuilt["discoverer_init_param"]["hidden"] == "keep"
    assert rebuilt["article_filter"] is _allow_news
    assert rebuilt["period_filter"] == _baseline()["period_filter"]


def test_generated_module_level_filter_remains_executable():
    generated = CrawlerCodeGenerator().generate_code_from_config(_baseline())
    namespace = {}

    exec(compile(generated, "<generated-config>", "exec"), namespace)

    article_filter = namespace["CRAWLER_CONFIG"]["article_filter"]
    assert article_filter("https://example.test/news/a")
    assert not article_filter("https://example.test/sports/a")
