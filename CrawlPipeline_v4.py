from __future__ import annotations

import os
import datetime
import traceback
import tldextract
from functools import partial
from dataclasses import dataclass
from contextlib import nullcontext
from urllib.parse import urlparse
from collections import defaultdict
from typing import List, Optional, Callable, Any, Tuple, Dict, Iterator
from typing import Callable, Dict, Iterator, List, Optional, Tuple, DefaultDict

from IntelligenceCrawler.Persistence import save_extraction_result_as_md
from IntelligenceCrawler.CrawlerGovernanceCore import GovernanceManager, CrawlSession
from IntelligenceCrawler.Discoverer import IDiscoverer, discoverer_factory
from IntelligenceCrawler.Extractor import IExtractor, ExtractionResult, extractor_factory
from IntelligenceCrawler.Fetcher import Fetcher, fetcher_factory


def format_exception_with_traceback(exception: Exception) -> str:
    """从异常对象生成格式化的 traceback"""
    if exception.__traceback__ is None:
        return f"{type(exception).__name__}: {exception}\n(No traceback)"

    tb_lines = traceback.format_exception(
        type(exception),
        exception,
        exception.__traceback__,
        limit=None
    )
    return ''.join(tb_lines)


def format_exception_compact(exception: Exception, max_frames: int = 5) -> str:
    if exception.__traceback__ is None:
        return f"{type(exception).__name__}: {exception}"

    tb_lines = traceback.format_exception(
        type(exception),
        exception,
        exception.__traceback__,
        limit=max_frames
    )
    return ''.join(tb_lines)


@dataclass(frozen=True)
class ChannelJob:
    """A deferred unit of work keyed by channel_url."""
    channel_url: str
    run: Callable[[], Tuple[List[str], Optional[Exception]]]
    # run() returns: (items, exception)


@dataclass(frozen=True)
class ArticleJob:
    """A deferred unit of work keyed by article_url (with its channel_group)."""
    article_url: str
    channel_group: str
    run: Callable[[], Tuple[Optional["ExtractionResult"], Optional[Exception]]]
    # run() returns: (result_or_none, exception)


class CrawlPipeline:
    """
    A stateful pipeline that encapsulates the 3-stage process of
    Discovering channels, Fetching articles, and Extracting content.
    """

    def __init__(self,
                 name: str,
                 d_fetcher: Fetcher,
                 discoverer: IDiscoverer,
                 e_fetcher: Fetcher,
                 extractor: IExtractor,
                 log_callback: Callable[..., None] = print,
                 crawler_governor: Optional[GovernanceManager] = None):
        """
        Initializes the pipeline with all required components.

        Args:
            d_fetcher: Fetcher instance for the Discoverer.
            discoverer: IDiscoverer instance.
            e_fetcher: Fetcher instance for the Extractor.
            extractor: IExtractor instance.
            log_callback: A function (like print or a GUI logger) to send logs to.
        """
        self.name = name
        self.d_fetcher = d_fetcher
        self.discoverer = discoverer
        self.e_fetcher = e_fetcher
        self.extractor = extractor
        self.log = log_callback
        self.crawler_governor = crawler_governor or GovernanceManager()

            # --- State Properties ---
        self.channels: List[str] = []
        self.articles: List[str] = []
        self.contents: List[Tuple[str, ExtractionResult]] = []

    def shutdown(self):
        """Gracefully closes both fetcher instances."""
        self.log("--- 5. Shutting down fetchers ---")
        try:
            if self.d_fetcher: self.d_fetcher.close()
        except Exception as e:
            self.log(f"[Error] Failed to close discovery fetcher: {e}")

        try:
            # Avoid closing the same fetcher twice if they are the same instance
            if self.e_fetcher and self.e_fetcher is not self.d_fetcher:
                self.e_fetcher.close()
        except Exception as e:
            self.log(f"[Error] Failed to close extraction fetcher: {e}")


    def discover_channel_jobs(
        self,
        entry_point: str | List[str],
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        fetcher_kwargs: Optional[dict] = None
    ) -> Iterator[ChannelJob]:
        """Yield deferred jobs that discover channels for each entry point URL."""
        if isinstance(entry_point, str):
            entry_point = [entry_point]

        fetcher_kwargs = fetcher_kwargs or {}

        for channel_url in entry_point:
            # Create a runner bound to the current loop variables (avoid late binding).
            def _runner(url=channel_url, sd=start_date, ed=end_date, kwargs=fetcher_kwargs):
                try:
                    channels_found = self.discoverer.discover_channels(
                        entry_point=url,
                        start_date=sd,
                        end_date=ed,
                        fetcher_kwargs=kwargs
                    )
                    return channels_found, None
                except Exception as e:
                    return [], e

            yield ChannelJob(channel_url=channel_url, run=_runner)

    def discover_channels(
        self,
        entry_point: str | List[str],
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        fetcher_kwargs: Optional[dict] = None
    ) -> List[str]:
        """
        Step 1: Discovers all channels from a list of entry point URLs.
        Clears all internal state.
        """
        if isinstance(entry_point, str):
            entry_point = [entry_point]

        self.log(f"--- 1. Discovering Channels from {len(entry_point)} entry point(s) ---")

        channels: List[str] = []

        for job in self.discover_channel_jobs(entry_point, start_date, end_date, fetcher_kwargs):
            channel_url = job.channel_url
            channels_found, exception = job.run()

            if exception is None:
                channels.extend(channels_found)
                self.log(f"Found {len(channels_found)} channels from {channel_url}")
            else:
                full_traceback = format_exception_with_traceback(exception)
                self.log(f"[Error] Failed to discover from {channel_url}: \n{full_traceback}")

        # De-duplicate while preserving order
        self.channels = list(dict.fromkeys(channels))
        self.log(f"Found {len(self.channels)} unique channels in total.")
        return self.channels

    def discover_articles_jobs(
        self,
        channel_urls: List[str],
        fetcher_kwargs: Optional[dict] = None
    ) -> Iterator[ChannelJob]:
        """Yield deferred jobs that discover article URLs for each channel URL."""
        fetcher_kwargs = fetcher_kwargs or {}

        for channel_url in channel_urls:
            def _runner(url=channel_url, kwargs=fetcher_kwargs):
                try:
                    self.log(f"Processing Channel: {url}")
                    articles_in_channel = self.discoverer.get_articles_for_channel(url, kwargs)
                    # De-duplicate within the channel
                    articles_in_channel = list(set(articles_in_channel))
                    self.log(f"Found {len(articles_in_channel)} articles in channel.")
                    return articles_in_channel, None
                except Exception as e:
                    return [], e

            yield ChannelJob(channel_url=channel_url, run=_runner)

    def discover_articles(
        self,
        channel_tables: Optional[Dict] = None,
        channel_filter: Optional[Callable[[str], bool]] = None,
        fetcher_kwargs: Optional[dict] = None
    ) -> List[Tuple[str, str]]:
        """
        Step 2: Discovers article URLs from channels.
        Populates self.articles as [(article_url, channel_group), ...].
        """
        self.log(f"--- 2. Discovering Articles from {len(self.channels)} Channels ---")

        seen_articles: set[str] = set()
        discovered_results: List[Tuple[str, str]] = []
        channel_tables = channel_tables or {}

        discover_channels: List[str] = []
        for channel_url in self.channels:
            if channel_filter and not channel_filter(channel_url):
                self.log(f"Skipping channel (filtered): {channel_url}")
                continue
            discover_channels.append(channel_url)

        for job in self.discover_articles_jobs(discover_channels, fetcher_kwargs):
            channel_url = job.channel_url
            channel_group = channel_tables.get(channel_url, "default")

            context = nullcontext()
            if self.crawler_governor:
                self.crawler_governor.register_group_metadata(channel_group, channel_url)
                context = self.crawler_governor.transaction(channel_url, channel_group)

            with context as task:
                articles_in_channel, exception = job.run()
                if exception is None:
                    count_new = 0
                    for article_url in articles_in_channel:
                        if article_url not in seen_articles:
                            seen_articles.add(article_url)
                            discovered_results.append((article_url, channel_group))
                            count_new += 1
                    if task: task.success()
                    self.log(f"Found {count_new} new articles in channel {channel_url}.")
                else:
                    if task: task.fail_temp(state_msg=f"Fail by exception: {str(exception)}")
                    full_traceback = format_exception_with_traceback(exception)
                    self.log(f"[Error] Failed to discover articles from {channel_url}: \n{full_traceback}")

        self.articles = discovered_results
        self.log(f"Discovered {len(self.articles)} unique articles.")
        return self.articles

    def extract_articles_jobs(
        self,
        article_urls: List[str],
        channel_group: str,
        fetcher_kwargs: Optional[dict] = None,
        extractor_kwargs: Optional[dict] = None
    ) -> Iterator[ArticleJob]:
        """Yield deferred jobs that fetch and extract content for each article URL."""
        fetcher_kwargs = fetcher_kwargs or {}
        extractor_kwargs = extractor_kwargs or {}

        for article_url in article_urls:
            def _runner(url=article_url, fk=fetcher_kwargs, ek=extractor_kwargs):
                try:
                    self.log(f"Processing: {url}")
                    content = self.e_fetcher.get_content(url, **fk)
                    if not content:
                        self.log(f"Skipped (no content): {url}")
                        return None, None

                    self.log(f"Fetched {len(content)} bytes. Extracting...")
                    result = self.extractor.extract(content, url, **ek)
                    return result, None

                except Exception as e:
                    return None, e

            yield ArticleJob(article_url=article_url, channel_group=channel_group, run=_runner)


    def extract_articles(
        self,
        article_filter: Optional[Callable[[str, str], bool]] = None,
        content_handler: Optional[Callable[[str, "ExtractionResult"], None]] = None,
        exception_handler: Optional[Callable[[str, Exception], None]] = None,
        fetcher_kwargs: Optional[dict] = None,
        extractor_kwargs: Optional[dict] = None
    ) -> List[Tuple[str, "ExtractionResult"]]:
        """
        Step 3: Fetches and extracts content from all discovered articles.
        Populates self.contents and calls optional handlers.
        """
        fetcher_kwargs = fetcher_kwargs or {}
        extractor_kwargs = extractor_kwargs or {}

        self.log(f"--- 3. Fetching & Extracting {len(self.articles)} Articles ---")

        # Group article URLs by channel_group
        grouped: DefaultDict[str, List[str]] = defaultdict(list)
        for article_url, channel_group in self.articles:
            grouped[channel_group].append(article_url)

        contents: List[Tuple[str, "ExtractionResult"]] = []

        for channel_group, article_urls in grouped.items():
            # Apply per-group filtering
            extract_article_urls: List[str] = []
            for article_url in article_urls:
                if article_filter and not article_filter(article_url, channel_group):
                    self.log(f"Skipping article (filtered): {article_url}")
                    continue
                extract_article_urls.append(article_url)

            for job in self.extract_articles_jobs(
                extract_article_urls,
                channel_group,
                fetcher_kwargs=fetcher_kwargs,
                extractor_kwargs=extractor_kwargs
            ):
                article_url = job.article_url

                ctx = nullcontext()
                if self.crawler_governor:
                    # Here transaction is keyed by article_url, grouped by channel_group.
                    self.crawler_governor.register_group_metadata(channel_group, article_url)
                    ctx = self.crawler_governor.transaction(article_url, channel_group)

                with ctx:
                    result, exception = job.run()

                if exception is None:
                    if result is None:
                        # No content is not an error; it is a skip.
                        continue
                    contents.append((article_url, result))
                    if content_handler:
                        content_handler(article_url, result)
                else:
                    if exception_handler:
                        exception_handler(article_url, exception)
                    full_traceback = format_exception_with_traceback(exception)
                    self.log(f"[Error] Failed to extract {article_url}: \n{full_traceback}")

        self.contents = contents
        self.log(f"Extracted {len(self.contents)} articles successfully.")
        return self.contents

