# CrawlPipeline.py

import os
import datetime
import traceback
from urllib.parse import urlparse
from typing import List, Optional, Callable, Any, Tuple

# Import for generated code
from functools import partial
from IntelligenceCrawler.Fetcher import Fetcher, RequestsFetcher, PlaywrightFetcher
from IntelligenceCrawler.Extractor import (
    ExtractionResult, IExtractor, TrafilaturaExtractor, ReadabilityExtractor,
    Newspaper3kExtractor, GenericCSSExtractor, Crawl4AIExtractor)
from IntelligenceCrawler.Discoverer import IDiscoverer, SitemapDiscoverer, RSSDiscoverer, ListPageDiscoverer
from IntelligenceCrawler.Persistence import save_extraction_result_as_md, save_extraction_result_as_pdf


log_cb = print


# --- Configuration ---
# Define the root directory where all articles will be saved
BASE_OUTPUT_DIR = "CRAWLER_OUTPUT"


class CrawlPipeline:
    """
    A stateful pipeline that encapsulates the 3-stage process of
    Discovering channels, Fetching articles, and Extracting content.
    """

    def __init__(self,
                 d_fetcher: Fetcher,
                 discoverer: IDiscoverer,
                 e_fetcher: Fetcher,
                 extractor: IExtractor,
                 log_callback: Callable[..., None] = print):
        """
        Initializes the pipeline with all required components.

        Args:
            d_fetcher: Fetcher instance for the Discoverer.
            discoverer: IDiscoverer instance.
            e_fetcher: Fetcher instance for the Extractor.
            extractor: IExtractor instance.
            log_callback: A function (like print or a GUI logger) to send logs to.
        """
        self.d_fetcher = d_fetcher
        self.discoverer = discoverer
        self.e_fetcher = e_fetcher
        self.extractor = extractor
        self.log = log_callback

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

    def discover_channels(self,
                          entry_point: str | List[str],
                          start_date: Optional[datetime.datetime] = None,
                          end_date: Optional[datetime.datetime] = None,
                          fetcher_kwargs: Optional[dict] = None) -> List[str]:
        """
        Step 1: Discovers all channels from a list of entry point URLs.
        Clears all internal state.
        """
        if isinstance(entry_point, str):
            entry_point = [entry_point]

        self.log(f"--- 1. Discovering Channels from {len(entry_point)} entry point(s) ---")

        channels = []
        for url in entry_point:
            self.log(f"Scanning entry point: {url}")
            try:
                channels_found = self.discoverer.discover_channels(
                    entry_point=url,
                    start_date=start_date,
                    end_date=end_date,
                    fetcher_kwargs=fetcher_kwargs
                )
                channels.extend(channels_found)
                self.log(f"Found {len(channels_found)} channels from this entry point.")
            except Exception as e:
                self.log(f"[Error] Failed to discover from {url}: {e}\n{traceback.format_exc()}")

        # De-duplicate the list while preserving order
        self.channels = list(dict.fromkeys(channels))
        self.log(f"Found {len(self.channels)} unique channels in total.")
        return self.channels

    def discover_articles(self,
                          channel_filter: Optional[Callable[[str], bool]] = None,
                          fetcher_kwargs: Optional[dict] = None) -> List[str]:
        """
        Step 2: Discovers article URLs from channels and fetches their content.
        Populates self.contents.
        """
        self.log(f"--- 2. Discovering & Articles from {len(self.channels)} Channels ---")

        articles = []
        for channel_url in self.channels:
            if channel_filter and not channel_filter(channel_url):
                self.log(f"Skipping channel (filtered): {channel_url}")
                continue

            self.log(f"Processing Channel: {channel_url}")
            try:
                articles_in_channel = self.discoverer.get_articles_for_channel(channel_url, fetcher_kwargs)
                self.log(f"Found {len(articles_in_channel)} articles in channel.")
                articles.extend(articles_in_channel)
            except Exception as e:
                self.log(f"[Error] Failed to process channel {channel_url}: {e}\n{traceback.format_exc()}")

        self.articles = list(dict.fromkeys(articles))
        # [FIX] This log message was incorrect, moved log to extract_articles
        # self.log(f"Fetched {len(self.articles)} article contents.")
        self.log(f"Discovered {len(self.articles)} unique articles.")
        return self.articles

    def extract_articles(self,
                         article_filter: Optional[Callable[[str], bool]] = None,
                         content_handler: Optional[Callable[[str, ExtractionResult], None]] = None,
                         exception_handler: Optional[Callable[[str, Exception], None]] = None,
                         **extractor_kwargs: Any) -> List[Tuple[str, ExtractionResult]]:
        """
        Step 3: Extracts content from all fetched articles.
        Populates self.articles and calls optional handlers.
        """
        self.log(f"--- 3. Fetching & Extracting {len(self.articles)} Articles ---")

        contents = []
        for article_url in self.articles:
            if article_filter and not article_filter(article_url):
                self.log(f"Skipping article (filtered): {article_url}")
                continue

            self.log(f"Processing: {article_url}")

            try:
                content = self.e_fetcher.get_content(article_url)
                if not content:
                    self.log(f"Skipped (no content): {article_url}")
                    continue

                self.log(f"  -> Fetched {len(content)} bytes. Extracting...")
                result = self.extractor.extract(content, article_url, **extractor_kwargs)
                contents.append((article_url, result))  # Store the final result

                if content_handler:
                    content_handler(article_url, result)  # Pass full result to handler
            except Exception as e:
                self.log(f"[Error] Failed to extract {article_url}: {e}")
                if exception_handler:
                    exception_handler(article_url, e)  # Pass URL and exception

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

        # [FIX] If path is just '/' or empty, this is a root URL.
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
        result: ExtractionResult,
        in_markdown: bool = True,
        in_pdf: bool = True
):
    if in_markdown:
        save_extraction_result_as_md(url, result, save_image=True, root_dir=BASE_OUTPUT_DIR)
    if in_pdf:
        save_extraction_result_as_pdf(result, root_dir=BASE_OUTPUT_DIR)
