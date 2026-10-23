from IntelligenceCrawler.CrawlPipeline import CrawlPipeline
from IntelligenceCrawler.CrawlTrace import CrawlTrace
from IntelligenceCrawler.URLFilter import (
    evaluate_article_url,
    evaluate_url_filter,
    filter_article_urls,
    normalize_url_filter,
    validate_url_filter_samples,
)


RULES = {
    "allowed_hosts": ["news.example.com"],
    "path_includes": ["/world/", "/politics/"],
    "path_excludes": ["/opinion/"],
    "query_policy": "reject",
}


def test_declarative_filter_reports_stable_rejection_reasons():
    assert evaluate_url_filter("https://news.example.com/world/a", RULES).allowed
    assert evaluate_url_filter(
        "https://other.example.com/world/a", RULES
    ).reason == "host_not_allowed"
    assert evaluate_url_filter(
        "https://news.example.com/sports/a", RULES
    ).reason == "path_not_included"
    assert evaluate_url_filter(
        "https://news.example.com/world/opinion/a", RULES
    ).reason == "path_excluded"
    assert evaluate_url_filter(
        "https://news.example.com/world/a?ref=home", RULES
    ).reason == "query_rejected"


def test_host_boundary_is_exact_unless_subdomains_are_enabled():
    rules = {"allowed_hosts": ["example.com"]}
    assert evaluate_url_filter("https://example.com/news/a", rules).allowed
    assert not evaluate_url_filter("https://news.example.com/news/a", rules).allowed

    rules["include_subdomains"] = True
    assert evaluate_url_filter("https://news.example.com/news/a", rules).allowed
    assert not evaluate_url_filter("https://notexample.com/news/a", rules).allowed


def test_empty_filter_normalizes_to_none_and_invalid_policy_fails():
    assert normalize_url_filter({}) is None
    assert normalize_url_filter({"allowed_hosts": []}) is None

    try:
        normalize_url_filter({"query_policy": "strip"})
    except ValueError as exc:
        assert "query_policy" in str(exc)
    else:
        raise AssertionError("invalid query policy must fail")


def test_declarative_and_custom_filters_are_combined():
    custom = lambda url: not url.endswith("/blocked")
    accepted, rejected = filter_article_urls([
        "https://news.example.com/world/ok",
        "https://news.example.com/world/blocked",
        "https://other.example.com/world/ok",
    ], RULES, custom)

    assert accepted == ["https://news.example.com/world/ok"]
    assert rejected == [
        ("https://news.example.com/world/blocked", "custom_filter_rejected"),
        ("https://other.example.com/world/ok", "host_not_allowed"),
    ]


class _Fetcher:
    def begin_pipeline(self):
        pass

    def end_pipeline(self):
        pass

    def close(self):
        pass

    def get_content(self, url, **_kwargs):
        return url.encode()


class _Discoverer:
    def get_articles_for_channel(self, channel_url, fetcher_kwargs=None):
        return [
            "https://news.example.com/world/ok",
            "https://news.example.com/world/opinion/a",
            "https://other.example.com/world/a",
        ]


class _Extractor:
    def extract(self, content, url, **_kwargs):
        return {"url": url, "size": len(content)}


def test_pipeline_uses_same_filter_and_exposes_funnel_diagnostics():
    trace = CrawlTrace(sample_limit=2)
    pipeline = CrawlPipeline(
        name="test",
        d_fetcher=_Fetcher(),
        discoverer=_Discoverer(),
        e_fetcher=_Fetcher(),
        extractor=_Extractor(),
        log_callback=lambda *_args: None,
        trace=trace,
    )
    pipeline.channels = ["https://news.example.com/feed"]
    pipeline.discover_articles(fetcher_kwargs={})
    pipeline.extract_articles(url_filter=RULES)

    snapshot = trace.snapshot()
    assert [url for url, _group, _result in pipeline.contents] == [
        "https://news.example.com/world/ok"
    ]
    assert snapshot["counters"]["articles_discovered"] == 3
    assert snapshot["counters"]["articles_accepted"] == 1
    assert snapshot["counters"]["extraction_succeeded"] == 1
    assert snapshot["rejections"] == {
        "article_filter.host_not_allowed": 1,
        "article_filter.path_excluded": 1,
    }
    assert len(snapshot["samples"]["article_filter.path_excluded"]) == 1


def test_single_url_evaluation_matches_batch_filter():
    url = "https://news.example.com/world/ok"
    decision = evaluate_article_url(url, RULES)
    accepted, rejected = filter_article_urls([url], RULES)
    assert decision.allowed
    assert accepted == [url]
    assert rejected == []


def test_embedded_regression_samples_report_only_failures():
    config = {
        "url_filter": RULES,
        "url_filter_samples": [
            {"url": "https://news.example.com/world/a", "allowed": True},
            {"url": "https://other.example.com/world/a", "allowed": False},
            {"url": "https://news.example.com/sports/a", "allowed": True},
        ],
    }

    failures = validate_url_filter_samples(config)

    assert len(failures) == 1
    assert failures[0]["index"] == 2
    assert failures[0]["reason"] == "path_not_included"
