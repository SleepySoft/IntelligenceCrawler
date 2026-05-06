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
from typing import Callable, Dict, Iterator, List, Optional, Tuple, DefaultDict

from IntelligenceCrawler.Persistence import save_extraction_result_as_md
from IntelligenceCrawler.CrawlerGovernanceCore import GovernanceManager, CrawlSession
from IntelligenceCrawler.Discoverer import IDiscoverer, discoverer_factory
from IntelligenceCrawler.Extractor import IExtractor, ExtractionResult, extractor_factory
from IntelligenceCrawler.Fetcher import Fetcher, fetcher_factory


# --- Configuration ---
# Define the root directory where all articles will be saved
BASE_OUTPUT_DIR = "CRAWLER_OUTPUT"


def auto_group_name_by_url(url_list):
    # 1. 临时分组容器
    groups = {}

    for url in url_list:
        extracted = tldextract.extract(url)
        domain = extracted.domain
        path = urlparse(url).path

        if domain not in groups:
            groups[domain] = []

        # 存入原始信息
        groups[domain].append({
            'url': url,
            'path': path
        })

    # 2. 结果字典
    result_dict = {}

    for domain, items in groups.items():
        paths = [x['path'] for x in items]

        # 计算公共前缀（文件夹层级）
        if len(paths) > 1:
            common = os.path.commonprefix(paths)
            if '/' in common:
                common = common[:common.rfind('/') + 1]
        else:
            # 单个链接时，以其父目录为公共前缀
            common = os.path.dirname(paths[0]) + '/'

        for item in items:
            # 1. 切掉公共路径
            short_path = item['path'].replace(common, "", 1).lstrip('/')

            # 2. 【关键】去掉文件后缀 (.xml, .html 等)
            name_no_ext = os.path.splitext(short_path)[0]

            # 3. 组装 Value: 域名/纯文件名
            group_name = f"{domain}/{name_no_ext}"

            # 4. 写入字典: Key=URL, Value=短名
            result_dict[item['url']] = group_name

    return result_dict


def format_exception_with_traceback(exception: Exception) -> str:
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
        self.crawler_governor = crawler_governor

            # --- State Properties ---
        self.channels: List[str] = []
        self.articles: List[str] = []
        self.contents: List[Tuple[str, ExtractionResult]] = []

    def reset(self):
        self.channels = []
        self.articles = []
        self.contents = []

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

            _url = channel_url
            _sd, _ed = start_date, end_date
            _kwargs_snapshot = dict(fetcher_kwargs)

            def _runner(url=_url, sd=_sd, ed=_ed, kwargs=_kwargs_snapshot):
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
        self.reset()

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
            _url = channel_url
            _kwargs_snapshot = dict(fetcher_kwargs)
            def _runner(url=_url, kwargs=_kwargs_snapshot):
                try:
                    self.log(f"Processing Channel: {url}")
                    articles_in_channel = self.discoverer.get_articles_for_channel(
                        channel_url=url, fetcher_kwargs=kwargs)
                    # De-duplicate within the channel
                    articles_in_channel = list(dict.fromkeys(articles_in_channel))
                    return articles_in_channel, None
                except Exception as e:
                    self.log(f"Discover article job exception.")
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

        filtered_channels: List[str] = []
        for channel_url in self.channels:
            if channel_filter and not channel_filter(channel_url):
                self.log(f"Skipping channel (filtered): {channel_url}")
                continue
            filtered_channels.append(channel_url)

        for job in self.discover_articles_jobs(filtered_channels, fetcher_kwargs):
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
            _url = article_url
            _f_kwargs_snapshot = dict(fetcher_kwargs)
            _e_kwargs_snapshot = dict(extractor_kwargs)
            def _runner(url=article_url, fk=_f_kwargs_snapshot, ek=_e_kwargs_snapshot):
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
        article_filter: Optional[Callable[[str], bool]] = None,
        content_handler: Optional[Callable[[str, str, "ExtractionResult"], bool]] = None,
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

        contents: List[Tuple[str, str, "ExtractionResult"]] = []

        for channel_group, article_urls in grouped.items():
            # Apply per-group filtering
            extract_article_urls: List[str] = []
            for article_url in article_urls:
                if article_filter and not article_filter(article_url):
                    self.log(f"Skipping article (filtered): {article_url}")
                    continue
                extract_article_urls.append(article_url)

            if self.crawler_governor:
                self.crawler_governor.start_round(channel_group, len(extract_article_urls))

            for job in self.extract_articles_jobs(
                extract_article_urls,
                channel_group,
                fetcher_kwargs=fetcher_kwargs,
                extractor_kwargs=extractor_kwargs
            ):
                article_url = job.article_url

                ctx = nullcontext()
                if self.crawler_governor:
                    ctx = self.crawler_governor.transaction(article_url, channel_group)

                with ctx as task:
                    if self.crawler_governor and not self.crawler_governor.should_crawl(article_url):
                        task.ignore(state_msg='Already fetched - Ignore.')
                        continue

                    result, exception = job.run()

                    if exception is None:
                        if result:
                            contents.append((article_url, channel_group, result))
                            if content_handler:
                                content_handler(article_url, channel_group, result)
                            if task: task.success()
                        else:
                            # No content is not an error; it is a skip.
                            if task: task.skip()
                    else:
                        if isinstance(exception, CrawlSession.Flow): raise exception        # Handled by context

                        full_traceback = format_exception_with_traceback(exception)
                        self.log(f"[Error] Failed to extract {article_url}: \n{full_traceback}")

                        # External exception handler does not exist or returns False causes permanent fail.
                        if not exception_handler or not exception_handler(article_url, exception):
                            if task: task.fail_perm(state_msg=f"Fail by exception: {str(exception)}")

            if self.crawler_governor:
                self.crawler_governor.finish_round(channel_group)

        self.contents = contents
        self.log(f"Extracted {len(self.contents)} articles successfully.")
        return self.contents


# ----------------------------------------------------------------------------------------------------------------------

def common_channel_filter(channel_url: str, channel_filter_list: List[str]) -> bool:
    """
    Checks if a given channel URL matches the filter list based on its "key".
    (根据“key”检查给定的频道 URL 是否与过滤器列表匹配。)

    This logic MUST mirror the 'get_filter_key' logic from the GUI's
    _generate_channel_filter_list_code method.
    (此逻辑必须与 GUI 的 _generate_channel_filter_list_code
     方法中的 'get_filter_key' 逻辑相匹配。)
    """

    # If the list is empty, the filter is disabled (pass all)
    # (如果列表为空，则禁用过滤器（全部通过）)
    if not channel_filter_list:
        return True

    # --- New Key Generation Logic ---
    key_to_check = ""
    try:
        parsed_url = urlparse(channel_url)
        path = parsed_url.path

        # If path is just '/' or empty, this is a root URL.
        # Use the netloc (domain) as the key.
        if not path or path == '/':
            key_to_check = parsed_url.netloc or channel_url  # Fallback
        else:
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
                key_to_check = f"{parent_folder}/{filename}"
            else:
                # Parent is root, just use the filename (e.g., 'sitemap.xml')
                key_to_check = filename

    except Exception:
        key_to_check = channel_url  # Fallback
    # --- End of New Logic ---

    # Return True if the extracted key is in the allowed list
    # (如果提取的名称在允许列表中，则返回 True)
    is_allowed = key_to_check in channel_filter_list
    # log_cb(f"Checking filter for {channel_url}... Key: {key_to_check}... Allowed: {is_allowed}")
    return is_allowed


def save_article_to_disk(
        url: str,
        channel_group: str,
        result: ExtractionResult,
        in_markdown: bool = True,
        in_pdf: bool = True
):
    if in_markdown:
        save_extraction_result_as_md(url, result, save_image=True, root_dir=BASE_OUTPUT_DIR)
    # PDF export has issue. Do not use.
    # if in_pdf:
    #     save_extraction_result_as_pdf(result, root_dir=BASE_OUTPUT_DIR)


# ----------------------------------------------------------------------------------------------------------------------

def build_pipeline(
        name: str,
        config: dict,
        log_callback: Callable[..., None],
        crawler_governor: GovernanceManager
):
    d_fetcher_name = config.get('d_fetcher_name', 'N/A')
    d_fetcher_init_param = config.get('d_fetcher_init_param', {})
    d_fetcher = fetcher_factory(d_fetcher_name, d_fetcher_init_param)

    e_fetcher_name = config.get('e_fetcher_name', 'N/A')
    e_fetcher_init_param = config.get('e_fetcher_init_param', {})
    e_fetcher = fetcher_factory(e_fetcher_name, e_fetcher_init_param)

    discoverer_name = config.get('discoverer_name', 'N/A')
    discoverer_init_param = config.get('discoverer_init_param', {})
    discoverer = discoverer_factory(discoverer_name, { 'fetcher': d_fetcher, **discoverer_init_param} )

    extractor_name = config.get('extractor_name', 'N/A')
    extractor_init_param = config.get('extractor_init_param', {})
    extractor = extractor_factory(extractor_name, extractor_init_param)

    pipeline = CrawlPipeline(
        name = name,
        d_fetcher=d_fetcher,
        discoverer=discoverer,
        e_fetcher=e_fetcher,
        extractor=extractor,
        log_callback=log_callback,
        crawler_governor=crawler_governor
    )
    return pipeline


def drive_pipeline_batch(pipeline: CrawlPipeline, config: dict):
    entry_points = config.get('entry_points', {})
    start_date, end_date = config.get('period_filter', (None, None))
    d_fetcher_kwargs = config.get('d_fetcher_kwargs', {})

    if isinstance(entry_points, dict):
        channel_tables = {value: f"{pipeline.name}/{key}" for key, value in entry_points.items()}
        entry_points_list = list(entry_points.values())
    else:
        channel_tables = auto_group_name_by_url(entry_points)
        entry_points_list = entry_points

    # ============== 1. Discover Channels ==============

    pipeline.discover_channels(
        entry_point=entry_points_list,
        start_date=start_date,
        end_date=end_date,
        fetcher_kwargs=d_fetcher_kwargs)

    # ============== 2. Discover Articles ==============

    # Only support channel_list_filter
    channel_filter = config.get('channel_filter', {})
    if channel_filter and 'channel_list_filter' in channel_filter:
        channel_list_filter_params = channel_filter['channel_list_filter']
        channel_filter = partial(common_channel_filter, channel_filter_list=channel_list_filter_params)
    else:
        channel_filter = None

    pipeline.discover_articles(
        channel_tables=channel_tables,
        channel_filter=channel_filter,
        fetcher_kwargs=d_fetcher_kwargs)

    # =============== 3. Extract Articles ===============

    article_filter = config.get('article_filter', None)
    content_handler = config.get('content_handler', None)
    exception_handler = config.get('exception_handler', None)

    e_fetcher_kwargs = config.get('e_fetcher_kwargs', { })
    extractor_kwargs = config.get('extractor_kwargs', { })

    pipeline.extract_articles(
        article_filter=article_filter,
        content_handler=content_handler,
        exception_handler=exception_handler,
        fetcher_kwargs=e_fetcher_kwargs,
        extractor_kwargs=extractor_kwargs
    )


def run_pipeline(
        name: str = 'default',
        config: dict = None,
        log_callback: Callable[..., None] = print,
        crawler_governor: Optional[GovernanceManager] = None):
    if not config:
        raise ValueError("Config is required for pipeline.")
    pipeline = build_pipeline(name, config, log_callback, crawler_governor)
    drive_pipeline_batch(pipeline, config)


# ----------------------------------------------------------------------------------------------------------------------
