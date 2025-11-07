# CrawlPipeline.py

import re
import os
import json
import hashlib
import datetime
import traceback
from urllib.parse import urlparse
from typing import List, Optional, Callable, Any, Tuple, Dict

# Import for generated code
from functools import partial
from IntelligenceCrawler.Fetcher import Fetcher, RequestsFetcher, PlaywrightFetcher
from IntelligenceCrawler.Extractor import (
    ExtractionResult, IExtractor, TrafilaturaExtractor, ReadabilityExtractor,
    Newspaper3kExtractor, GenericCSSExtractor, Crawl4AIExtractor)
from IntelligenceCrawler.Discoverer import IDiscoverer, SitemapDiscoverer, RSSDiscoverer


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
                          end_date: Optional[datetime.datetime] = None) -> List[str]:
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
                    end_date=end_date
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
                          channel_filter: Optional[Callable[[str], bool]] = None) -> List[str]:
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
                articles_in_channel = self.discoverer.get_articles_for_channel(channel_url)
                self.log(f"Found {len(articles_in_channel)} articles in channel.")
                articles.extend(articles_in_channel)
            except Exception as e:
                self.log(f"[Error] Failed to process channel {channel_url}: {e}\n{traceback.format_exc()}")

        self.articles = list(dict.fromkeys(articles))
        self.log(f"Fetched {len(self.articles)} article contents.")
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
        self.log(f"--- 3. Extracting {len(self.contents)} Articles ---")

        contents = []
        for article_url in self.articles:
            if article_filter and not article_filter(article_url):
                self.log(f"Skipping article (filtered): {article_url}")
                continue

            self.log(f"Fetching: {article_url}")

            try:
                content = self.e_fetcher.get_content(article_url)
                if not content:
                    self.log(f"Skipped (no content): {article_url}")
                    continue

                result = self.extractor.extract(content, article_url, **extractor_kwargs)
                contents.append((article_url, result))      # Store the final result

                if content_handler:
                    content_handler(article_url, result)    # Pass full result to handler
            except Exception as e:
                self.log(f"[Error] Failed to extract {article_url}: {e}")
                if exception_handler:
                    exception_handler(article_url, e)       # Pass URL and exception

        self.contents = contents
        self.log(f"Extracted {len(self.contents)} articles successfully.")
        return self.contents


# ----------------------------------------------------------------------------------------------------------------------

def common_channel_filter(channel_url: str, channel_filter_list: List[str]) -> bool:
    """
    Checks if a given channel URL matches the filter list based on its "name".
    (根据“名称”检查给定的频道 URL 是否与过滤器列表匹配。)

    :param channel_url: The channel URL to check.
    :param channel_filter_list: The list of allowed "names" (e.g., "feed.xml").
    :return: True if the filter passes, False otherwise.
    """

    # If the list is empty, the filter is disabled (pass all)
    # (如果列表为空，则禁用过滤器（全部通过）)
    if not channel_filter_list:
        return True

    try:
        # --- This logic MUST mirror the logic in _get_checked_channel_filter_list ---
        # (此逻辑必须与 _get_checked_channel_filter_list 中的逻辑相匹配)

        parsed_url = urlparse(channel_url)
        path = parsed_url.path
        hostname = parsed_url.netloc

        # Get the last part of the path, e.g., "feed.xml" or "blog"
        # (获取路径的最后一部分，例如 "feed.xml" 或 "blog")
        cleaned_path = path.rstrip('/')
        name = cleaned_path.split('/')[-1]

        # If path was just "/" or empty, use the hostname as the name
        # (如果路径只是 "/" 或为空，则使用主机名作为名称)
        filter_name = name or hostname

        # Return True if the extracted name is in the allowed list
        # (如果提取的名称在允许列表中，则返回 True)
        return filter_name in channel_filter_list

    except Exception:
        # On error, safely filter out (return False)
        # (出错时，安全过滤掉（返回 False）)
        return False


# ----------------------------------------------------------------------------------------------------------------------

def _slugify_filename(text: str) -> str:
    """
    Sanitize a string to be used as a valid filename.
    (e.g., "My Post / 1?" -> "my-post-1")

    :param text: The string to sanitize.
    :return: A filesystem-safe string.
    """
    if not text:
        return "untitled"

    # 1. Convert to lowercase
    text = str(text).lower()

    # 2. Remove special characters (keep alphanumeric, spaces, hyphens)
    text = re.sub(r'[^\w\s-]', '', text, flags=re.ASCII)

    # 3. Replace whitespace and hyphens with a single hyphen
    text = re.sub(r'[\s_-]+', '-', text).strip('-')

    # 4. Truncate to a reasonable length (e.g., 100 chars)
    return text[:100]


def _get_safe_basename(url: str, metadata: Dict[str, Any]) -> str:
    """
    Create a safe base filename (no extension) for the article.
    It tries to find a human-readable name and falls back to a hash.

    :param url: The source URL of the article.
    :param metadata: The extracted metadata dictionary.
    :return: A filesystem-safe base filename.
    """

    # Strategy 1: Use the title if it exists
    title = metadata.get('title')
    if title:
        slug = _slugify_filename(title)
        # Avoid empty slugs if the title was just symbols
        if slug:
            return slug

    # Strategy 2: Use the last part of the URL path
    try:
        path_filename = os.path.basename(urlparse(url).path)
        if path_filename:
            # Remove extension (e.g., .html, .aspx)
            base = os.path.splitext(path_filename)[0]
            slug = _slugify_filename(base)
            if slug:
                return slug
    except Exception:
        pass  # Fallback to hash

    # Strategy 3: (Fallback) Use a hash of the URL for a guaranteed unique name
    return hashlib.md5(url.encode('utf-8')).hexdigest()


def save_article_to_disk(url: str, result: ExtractionResult):
    """
    This is the content_handler implementation for the CrawlPipeline.

    It saves the extracted article markdown and metadata to a
    categorized folder structure:

    CRAWLER_OUTPUT/
        └── <domain.com>/
            ├── <article-slug>.md
            └── <article-slug>.meta.json

    :param url: The source URL of the article.
    :param result: The ExtractionResult object from the extractor.
    """

    # 1. Do not save if extraction failed or content is empty
    if not result.success:
        log_cb(f"[Handler] SKIPPING (Failure): {url}. Reason: {result.error}")
        return

    if not result.markdown_content or not result.markdown_content.strip():
        log_cb(f"[Handler] SKIPPING (No Content): {url}. Markdown is empty.")
        return

    try:
        # 2. Determine Category (Folder) from domain
        parsed_url = urlparse(url)
        domain = parsed_url.netloc or "unknown_domain"
        # Sanitize domain to remove ports (e.g., "localhost:8000")
        domain = domain.split(':', 1)[0]

        # 3. Determine safe filename
        base_filename = _get_safe_basename(url, result.metadata)

        # 4. Define full file paths
        # Structure: CRAWLER_OUTPUT / <domain> / <base_filename>
        article_dir = os.path.join(BASE_OUTPUT_DIR, domain)

        md_filepath = os.path.join(article_dir, f"{base_filename}.md")
        meta_filepath = os.path.join(article_dir, f"{base_filename}.meta.json")

        # 5. Create the directory if it doesn't exist
        os.makedirs(article_dir, exist_ok=True)

        # 6. Save Markdown content
        # We'll add a simple header to the Markdown file
        with open(md_filepath, 'w', encoding='utf-8') as f:
            f.write(f"# {result.metadata.get('title', base_filename)}\n\n")
            f.write(f"**Source:** <{url}>\n")

            pub_date = result.metadata.get('date')
            if pub_date:
                f.write(f"**Published:** {pub_date}\n")

            f.write("\n---\n\n")
            f.write(result.markdown_content)

        # 7. Save Metadata
        with open(meta_filepath, 'w', encoding='utf-8') as f:
            # Add the source URL to the metadata for reference
            result.metadata['__source_url'] = url
            # Add the save path for reference
            result.metadata['__saved_md_path'] = md_filepath

            json.dump(result.metadata, f, indent=2, ensure_ascii=False, default=str)

        log_cb(f"[Handler] SAVED: {url}\n    -> {md_filepath}")

    except Exception as e:
        log_cb(f"[Handler] CRITICAL ERROR saving {url}: {e}")
        traceback.print_exc()
