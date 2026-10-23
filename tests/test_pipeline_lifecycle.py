import types
import threading

import pytest

from IntelligenceCrawler.CrawlPipeline import CrawlPipeline, drive_pipeline_batch
from IntelligenceCrawler.Fetcher import Fetcher, PlaywrightFetcher


class RecordingFetcher(Fetcher):
    def __init__(self, events):
        self.events = events

    def begin_pipeline(self):
        self.events.append("begin")

    def end_pipeline(self):
        self.events.append("end")

    def get_content(self, url, **kwargs):
        return b""

    def close(self):
        self.events.append("close")


def test_drive_pipeline_batch_deduplicates_fetcher_and_ends_on_error():
    events = []
    fetcher = RecordingFetcher(events)
    pipeline = CrawlPipeline(
        name="test",
        d_fetcher=fetcher,
        discoverer=None,
        e_fetcher=fetcher,
        extractor=None,
        log_callback=lambda *_: None,
    )

    def fail_discovery(**_kwargs):
        events.append("work")
        raise RuntimeError("boom")

    pipeline.discover_channels = fail_discovery

    with pytest.raises(RuntimeError, match="boom"):
        drive_pipeline_batch(pipeline, {"entry_points": []})

    assert events == ["begin", "work", "end"]
    assert pipeline._pipeline_active is False


def test_drive_pipeline_batch_stops_between_stages_and_releases_fetcher():
    events = []
    stop_event = threading.Event()
    fetcher = RecordingFetcher(events)
    pipeline = CrawlPipeline(
        name="test",
        d_fetcher=fetcher,
        discoverer=None,
        e_fetcher=fetcher,
        extractor=None,
        log_callback=lambda *_: None,
    )

    def stop_after_discovery(**_kwargs):
        events.append("discover")
        stop_event.set()

    pipeline.discover_channels = stop_after_discovery
    pipeline.discover_articles = lambda **_kwargs: events.append("articles")

    with pytest.raises(InterruptedError, match="was stopped"):
        drive_pipeline_batch(pipeline, {"entry_points": []}, stop_event=stop_event)

    assert events == ["begin", "discover", "end"]
    assert pipeline._pipeline_active is False


def test_playwright_runtime_is_lazy_and_restarts_for_next_pipeline():
    fetcher = PlaywrightFetcher(log_callback=lambda *_: None)
    starts = []
    stops = []

    def fake_start(self):
        starts.append("start")
        self.playwright = object()
        self.browser = object()
        self.context = None
        return True

    def fake_create_context(self):
        self.context = object()
        self._context_created_ts = 0.0
        self._context_request_count = 0
        self._consecutive_errors = 0

    def fake_fetch(self, payload):
        assert self.playwright is not None
        assert self.browser is not None
        assert self.context is not None
        return payload["url"].encode("utf-8")

    def fake_stop(self):
        if any(getattr(self, name, None) is not None
               for name in ("playwright", "browser", "context")):
            stops.append("stop")
        self.context = None
        self.browser = None
        self.playwright = None

    fetcher._start_playwright = types.MethodType(fake_start, fetcher)
    fetcher._create_context = types.MethodType(fake_create_context, fetcher)
    fetcher._fetch_page_content = types.MethodType(fake_fetch, fetcher)
    fetcher._stop_playwright = types.MethodType(fake_stop, fetcher)

    try:
        fetcher.begin_pipeline()
        assert starts == []
        assert fetcher.get_content("first") == b"first"
        assert starts == ["start"]
        fetcher.end_pipeline()
        assert stops == ["stop"]

        fetcher.begin_pipeline()
        assert fetcher.get_content("second") == b"second"
        assert starts == ["start", "start"]
        fetcher.end_pipeline()
        assert stops == ["stop", "stop"]
    finally:
        fetcher.close()

    assert not fetcher.worker_thread.is_alive()


def test_trial_article_limit_is_applied_inside_pipeline_driver():
    events = []
    fetcher = RecordingFetcher(events)
    pipeline = CrawlPipeline(
        name="trial",
        d_fetcher=fetcher,
        discoverer=None,
        e_fetcher=fetcher,
        extractor=None,
        log_callback=lambda *_: None,
    )

    def discover_channels(**_kwargs):
        pipeline.channels = ["channel"]

    def discover_articles(**_kwargs):
        pipeline.articles = [(f"article-{index}", "group") for index in range(4)]

    def extract_articles(**_kwargs):
        events.append([url for url, _group in pipeline.articles])

    pipeline.discover_channels = discover_channels
    pipeline.discover_articles = discover_articles
    pipeline.extract_articles = extract_articles

    drive_pipeline_batch(pipeline, {"entry_points": []}, max_articles=2)

    assert events == ["begin", ["article-0", "article-1"], "end"]
    assert pipeline.trace.snapshot()["counters"]["articles_limited_out"] == 2
