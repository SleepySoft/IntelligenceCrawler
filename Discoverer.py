import re
import json
import logging
import datetime
import traceback
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup, Tag
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from usp.tree import sitemap_from_str
from collections import deque, defaultdict
from urllib.parse import urlparse, urljoin
from typing import Set, List, Dict, Any, Optional, Deque, Tuple

# --- RSS/HTML Parsing Imports ---
import feedparser

# --- Date Imports (for interface compatibility) ---
try:
    from dateutil.parser import parse as date_parse
except ImportError:
    print("!!! IMPORT ERROR: 'python-dateutil' not found.")
    print("!!! Please install it for date filtering: pip install python-dateutil")
    date_parse = None

logger = logging.getLogger(__name__)


class IDiscoverer(ABC):
    """
    Abstract base class for a discovery component.
    (发现组件的抽象基类)

    Its role is to find "channels" (like leaf sitemaps or RSS feeds)
    and then extract individual article URLs from those channels.
    It relies on an injected Fetcher for all network operations.
    (它的职责是找到“频道”(如叶子sitemap或RSS feed)，
     然后从这些频道中提取单独的文章URL。
     它依赖注入的 Fetcher 来执行所有网络操作。)
    """

    def __init__(self, fetcher: "Fetcher", verbose: bool = True):
        """
        Initializes the discoverer.
        (初始化发现器)

        :param fetcher: An instance of a Fetcher implementation.
        :param verbose: Toggles detailed logging.
        """
        self.fetcher = fetcher
        self.verbose = verbose
        self.log_messages: List[str] = []

    @abstractmethod
    def _log(self, message: str, indent: int = 0):
        """
        Provides a unified logging mechanism.
        (提供统一的日志记录机制)

        Note: We make this abstract so concrete classes *must*
        implement it, ensuring logging is available.
        (我们将其设为抽象，因此具体类*必须*实现它，以确保日志记录可用。)
        """
        pass

    @abstractmethod
    def discover_channels(self,
                          entry_point: Any,
                          start_date: Optional[datetime.datetime] = None,
                          end_date: Optional[datetime.datetime] = None,
                          fetcher_kwargs: Optional[Dict[str, Any]] = None
                          ) -> List[str]:
        """
        Stage 1: Discovers all "channels" (e.g., leaf sitemaps, RSS feeds)
        from a main entry point.
        (阶段1：从主入口点发现所有“频道”（例如叶子sitemap、RSS feed）)

        :param entry_point: The entry point for discovery. The type depends
                            on the concrete implementation (e.g., a URL string
                            for Sitemap, a URL string OR a List[str] for RSS).
        :param start_date: (Optional) Filter to include only channels
                           relevant *after* this date.
        :param end_date: (Optional) Filter to include only channels
                         relevant *before* this date.
        :param fetcher_kwargs: (Optional) A dictionary of keyword arguments
                               to pass to the fetcher's get_content() method.
        :return: A list of string URLs, each representing a "channel"
                 that contains article links.
        """
        pass

    @abstractmethod
    def get_articles_for_channel(self,
                                 channel_url: str,
                                 fetcher_kwargs: Optional[Dict[str, Any]] = None
                                 ) -> List[str]:
        """
        Stage 2: Fetches and parses a single "channel" URL (found in Stage 1)
        to extract all individual article URLs it contains.
        (阶段2：获取并解析在阶段1中找到的单个“频道”URL，
         以提取其包含的所有单独的文章URL。)

        :param channel_url: The URL of a single channel
                            (e.g., a leaf sitemap or an RSS feed URL).
        :param fetcher_kwargs: (Optional) A dictionary of keyword arguments
                               to pass to the fetcher's get_content() method.
        :return: A list of string URLs for individual articles.
        """
        pass

    def get_content_str(self,
                        url: str,
                        fetcher_kwargs: Optional[Dict[str, Any]] = None
                        ) -> str:
        """
        Helper to get raw content as a string for display or debugging.
        (辅助函数：获取原始内容的字符串用于显示或调试。)

        This can be a concrete method in the base class as it only
        depends on the fetcher.
        (这可以是基类中的一个具体方法，因为它只依赖于 fetcher。)
        """
        self.log_messages.clear()
        self._log(f"Fetching raw content for: {url}")
        content = self.fetcher.get_content(url, **(fetcher_kwargs or {}))
        if content:
            try:
                return content.decode('utf-8', errors='ignore')
            except Exception as e:
                self._log(f"Error decoding content: {e}")
                return f"Error decoding content: {e}"
        return f"Failed to fetch content from {url}"


def discoverer_factory(name: str, init_params: dict):
    if name == 'RSSDiscoverer':
        return RSSDiscoverer(
            fetcher=init_params.get('fetcher'),
            verbose=init_params.get('verbose', False)
        )

    if name == 'SitemapDiscoverer':
        return SitemapDiscoverer(
            fetcher=init_params.get('fetcher'),
            verbose=init_params.get('verbose', False)
        )

    if name == 'ListPageDiscoverer':
        return ListPageDiscoverer(
            fetcher=init_params.get('fetcher'),
            verbose=init_params.get('verbose', False),
            min_group_count=init_params.get('min_group_count', 5),
            scope_selector=init_params.get('scope_selector'),
            manual_specified_signature=init_params.get('manual_specified_signature'),
            extraction_mode=init_params.get('extraction_mode', 'auto'),
            smart_scope_mode=init_params.get('smart_scope_mode', True),
        )

    if name == 'Sitemap': return discoverer_factory('SitemapDiscoverer', init_params)
    if name == 'RSS': return discoverer_factory('RSSDiscoverer', init_params)
    if name == 'Smart Analysis': return discoverer_factory('ListPageDiscoverer', init_params)

    raise ValueError(f"Unknown discoverer: {name}")


class SitemapDiscoverer(IDiscoverer):
    """
    Discovers articles by parsing sitemap.xml files.
    (通过解析 sitemap.xml 文件发现文章。)

    Requires a 'Fetcher' instance to be injected upon initialization.
    All network I/O is delegated to self.fetcher.

    v3.7 (Refactor): Now includes date filtering to avoid processing
    stale sitemap indexes.
    """
    NAMESPACES = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

    def __init__(self, fetcher: "Fetcher", verbose: bool = True):
        """
        Initializes the sitemap discoverer.
        (初始化 sitemap 发现器。)
        :param fetcher: An instance of a class that implements the Fetcher ABC.
        :param verbose: Whether to print detailed log messages.
        """
        super().__init__(fetcher, verbose)

        # --- State Properties ---
        self.all_article_urls: Set[str] = set()
        self.leaf_sitemaps: Set[str] = set()
        self.to_process_queue: Deque[str] = deque()
        self.processed_sitemaps: Set[str] = set()

        # --- NEW: Check for dateutil library ---
        if not date_parse:
            self._log("[Warning] 'python-dateutil' not found. Date filtering will be disabled.")

    def _log(self, message: str, indent: int = 0):
        """Unified logging function."""
        log_msg = f"{' ' * (indent * 4)}{message}"
        self.log_messages.append(log_msg)
        if self.verbose:
            print(log_msg)

    def _discover_sitemap_entry_points(self, homepage_url: str) -> List[str]:
        """Step 1 (Internal): Automatically discover sitemap entry points."""
        self._log(f"Auto-discovering sitemap entry points for {homepage_url}...")
        try:
            parsed_home = urlparse(homepage_url)
            base_url = f"{parsed_home.scheme}://{parsed_home.netloc}"
        except Exception as e:
            self._log(f"[Error] Could not parse homepage URL: {e}")
            return []

        # Path 1: Check robots.txt (Preferred)
        robots_url = urljoin(base_url, '/robots.txt')
        self._log(f"Checking robots.txt: {robots_url}", 1)
        robots_content_bytes = self.fetcher.get_content(robots_url)

        sitemap_urls = []
        if robots_content_bytes:
            try:
                sitemap_urls = re.findall(
                    r"^Sitemap:\s*(.+)$",
                    robots_content_bytes.decode('utf-8', errors='ignore'),
                    re.IGNORECASE | re.MULTILINE
                )
                sitemap_urls = [url.strip() for url in sitemap_urls]
                if sitemap_urls:
                    self._log(f"Found {len(sitemap_urls)} sitemap(s) in robots.txt: {sitemap_urls}", 1)
                    return sitemap_urls
            except Exception as e:
                self._log(f"Error parsing robots.txt: {e}", 1)

        # Path 2: Guess default paths (Fallback)
        self._log("No sitemaps found in robots.txt. Guessing default paths...", 1)
        return [
            urljoin(base_url, '/sitemap_index.xml'),
            urljoin(base_url, '/sitemap.xml')
        ]

    # --- Date parsing and checking helper ---
    def _parse_and_check_date(self,
                              lastmod_str: Optional[str],
                              start_date: Optional[datetime.datetime],
                              end_date: Optional[datetime.datetime]) -> bool:
        """
        Checks if a sitemap's lastmod date is within the desired range.
        Returns True if it should be processed, False if it should be skipped.
        """

        # Rule 1: If no date library, we can't filter. Process everything.
        if not date_parse:
            return True

            # Rule 2: If no date limits are set by the user, always process.
        if not start_date and not end_date:
            return True

        # Rule 3: If the sitemap has no <lastmod>, process it (our fallback).
        if not lastmod_str:
            self._log("      > No <lastmod> date found. Including by default.", 3)
            return True

        try:
            # Attempt to parse the date string (e.g., "2025-11-01T18:23:17+00:00")
            sitemap_date = date_parse(lastmod_str)

            # --- Timezone Handling (CRITICAL for correct comparison) ---
            # Make sure sitemap_date is timezone-aware (assume UTC if naive)
            if sitemap_date.tzinfo is None:
                sitemap_date = sitemap_date.replace(tzinfo=datetime.timezone.utc)

            # Make sure start_date is timezone-aware (assume UTC if naive)
            start_date_aware = start_date
            if start_date and start_date.tzinfo is None:
                start_date_aware = start_date.replace(tzinfo=datetime.timezone.utc)

            # Make sure end_date is timezone-aware (assume UTC if naive)
            end_date_aware = end_date
            if end_date and end_date.tzinfo is None:
                end_date_aware = end_date.replace(tzinfo=datetime.timezone.utc)
            # --- End Timezone Handling ---

            # Rule 4: Check against start_date
            if start_date_aware and sitemap_date < start_date_aware:
                self._log(
                    f"      > SKIPPING: Date {sitemap_date.date()} is older than start date {start_date_aware.date()}",
                    3)
                return False

            # Rule 5: Check against end_date
            if end_date_aware and sitemap_date > end_date_aware:
                self._log(
                    f"      > SKIPPING: Date {sitemap_date.date()} is newer than end date {end_date_aware.date()}", 3)
                return False

            # Rule 6: It's within range
            self._log(f"      > Date {sitemap_date.date()} is within range. Including.", 3)
            return True

        except Exception as e:
            # If parsing fails (e.g., "invalid date format"), process it just to be safe.
            self._log(f"      > Warning: Could not parse date '{lastmod_str}'. Error: {e}. Including by default.", 3)
            return True

    # --- _parse_sitemap_xml now returns richer data ---
    def _parse_sitemap_xml(self, xml_content: bytes, sitemap_url: str) -> Dict[str, List[Any]]:
        """
        Parses Sitemap XML content with a fallback mechanism.

        Returns a dict:
        {
            'pages': List[str],  // List of page URLs
            'sub_sitemaps': List[Dict[str, Optional[str]]] // List of {'loc': url, 'lastmod': date_str}
        }
        """
        pages: List[str] = []
        # --- UPDATED: sub_sitemaps is now a list of dicts ---
        sub_sitemaps: List[Dict[str, Optional[str]]] = []

        try:
            self._log("    Trying to parse with [ultimate-sitemap-parser]...", 1)
            parsed_sitemap = sitemap_from_str(xml_content.decode('utf-8', errors='ignore'))

            for page in parsed_sitemap.all_pages():
                pages.append(page.url)

            # --- UPDATED: Extract lastmod along with loc ---
            for sub_sitemap in parsed_sitemap.all_sub_sitemaps():
                lastmod_str = sub_sitemap.lastmod.isoformat() if sub_sitemap.lastmod else None
                sub_sitemaps.append({
                    'loc': sub_sitemap.url,
                    'lastmod': lastmod_str
                })
            self._log(f"    [USP Success] Found {len(pages)} pages and {len(sub_sitemaps)} sub-sitemaps.", 1)

        except Exception as e:
            self._log(f"    [USP Failed] Library parsing error: {e}", 1)
            self._log("    --> Initiating [Manual ElementTree] fallback...", 1)
            try:
                root = ET.fromstring(xml_content)
                index_nodes = root.findall('ns:sitemap', self.NAMESPACES)

                if index_nodes:
                    # --- UPDATED: Extract lastmod along with loc ---
                    for node in index_nodes:
                        loc_node = node.find('ns:loc', self.NAMESPACES)
                        lastmod_node = node.find('ns:lastmod', self.NAMESPACES)

                        loc_text = loc_node.text if loc_node is not None and loc_node.text else None
                        lastmod_text = lastmod_node.text if lastmod_node is not None and lastmod_node.text else None

                        if loc_text:
                            sub_sitemaps.append({
                                'loc': loc_text,
                                'lastmod': lastmod_text
                            })
                    self._log(f"    [Manual Fallback] Found {len(sub_sitemaps)} sub-sitemaps.", 1)

                url_nodes = root.findall('ns:url', self.NAMESPACES)
                if url_nodes:
                    for node in url_nodes:
                        loc = node.find('ns:loc', self.NAMESPACES)
                        if loc is not None and loc.text:
                            pages.append(loc.text)
                    self._log(f"    [Manual Fallback] Found {len(pages)} pages.", 1)

                if not index_nodes and not url_nodes:
                    self._log("    [Manual Fallback] Failed: No <sitemap> or <url> tags found.", 1)
            except ET.ParseError as xml_e:
                self._log(f"    [Manual Fallback] Failed: Could not parse XML. Error: {xml_e}", 1)

        return {'pages': pages, 'sub_sitemaps': sub_sitemaps}

    def discover_channels(self,
                          entry_point: Any,
                          start_date: Optional[datetime.datetime] = datetime.datetime.now() - datetime.timedelta(days=7),
                          end_date: Optional[datetime.datetime] = datetime.datetime.now(),
                          fetcher_kwargs: Optional[Dict[str, Any]] = None
                          ) -> List[str]:
        """
        STAGE 1: Discover all "channels" (leaf sitemaps containing articles).

        :param entry_point: The root URL of the website (MUST be a string).
        :param start_date: (Optional) The earliest date to include sitemaps from.
        :param end_date: (Optional) The latest date to include sitemaps from.
        :param fetcher_kwargs: (Optional) A dictionary of keyword arguments
                               to pass to the fetcher's get_content() method.
        """
        if entry_point and isinstance(entry_point, (list, tuple, set)):
            entry_point = entry_point[0]
        if not isinstance(entry_point, str):
            self._log(f"[Error] SitemapDiscoverer requires a string URL as an entry_point. Got {type(entry_point)}.")
            return []

        homepage_url = entry_point

        self._log(f"--- STAGE 1: Discovering Channels for {homepage_url} ---")
        if start_date or end_date:
            self._log(
                f"Filtering sitemaps between: {start_date.date() if start_date else 'Beginning'} and {end_date.date() if end_date else 'Today'}")

        self.log_messages.clear()
        self.leaf_sitemaps.clear()
        self.to_process_queue.clear()
        self.processed_sitemaps.clear()

        kwargs_to_pass = fetcher_kwargs or {}

        initial_sitemaps = self._discover_sitemap_entry_points(homepage_url)
        if not initial_sitemaps:
            self._log("Could not find any sitemap entry points.")
            return []

        self.to_process_queue.extend(initial_sitemaps)

        while self.to_process_queue:
            # --- UPDATED: Limit queue size to prevent infinite loops on bad sites ---
            if len(self.to_process_queue) > 5000:
                self._log("[Error] Queue size exceeds 5000. Aborting to prevent infinite loop.")
                break

            sitemap_url = self.to_process_queue.popleft()
            if sitemap_url in self.processed_sitemaps:
                continue
            self.processed_sitemaps.add(sitemap_url)

            # 在抓取(fetch)之前，先检查 URL 字符串本身
            if not self._check_url_against_date_range(sitemap_url, start_date, end_date):
                continue

            self._log(f"\n--- Analyzing index: {sitemap_url} ---")
            xml_content = self.fetcher.get_content(sitemap_url, **kwargs_to_pass)

            if not xml_content:
                self._log("  Failed to fetch, skipping.", 1)
                continue

            parse_result = self._parse_sitemap_xml(xml_content, sitemap_url)

            if parse_result['sub_sitemaps']:
                self._log(f"  > Found {len(parse_result['sub_sitemaps'])} sub-indexes. Filtering by date...", 2)

                valid_sitemaps_to_queue = []
                for sitemap_info in parse_result['sub_sitemaps']:
                    loc = sitemap_info['loc']
                    lastmod = sitemap_info['lastmod']

                    self._log(f"    - Checking: {loc}", 3)

                    # Use the new helper function to decide
                    if self._parse_and_check_date(lastmod, start_date, end_date):
                        valid_sitemaps_to_queue.append(loc)

                self._log(
                    f"  > Queuing {len(valid_sitemaps_to_queue)} out of {len(parse_result['sub_sitemaps'])} sub-indexes.",
                    2)
                self.to_process_queue.extend(valid_sitemaps_to_queue)

            if parse_result['pages']:
                self._log(f"  > Found {len(parse_result['pages'])} pages. Marking as 'Channel'.", 2)
                # This is a leaf node, so we just add it.
                # The *date* of the sitemap file itself doesn't matter here,
                # only that it contains article URLs.
                self.leaf_sitemaps.add(sitemap_url)

        self._log(f"\nStage 1 Complete: Discovered {len(self.leaf_sitemaps)} total channels.")
        return list(self.leaf_sitemaps)

    # --- (get_articles_for_channel & get_xml_content_str are unchanged) ---
    def get_articles_for_channel(self,
                                 channel_url: str,
                                 fetcher_kwargs: Optional[Dict[str, Any]] = None
                                 ) -> List[str]:
        """
        Helper for Stage 2 (Lazy Loading): Gets pages for ONE specific channel.
        """
        self.log_messages.clear()
        self._log(f"--- STAGE 2: Fetching articles for {channel_url} ---")
        xml_content = self.fetcher.get_content(channel_url, **(fetcher_kwargs or {}))
        if not xml_content:
            return []

        # Note: This *could* also be modified to filter articles by date
        # but for now it just returns all articles from the channel.
        parse_result = self._parse_sitemap_xml(xml_content, channel_url)
        self._log(f"  > Found {len(parse_result['pages'])} articles.")
        return parse_result['pages']

    def get_xml_content_str(self,
                            url: str,
                            fetcher_kwargs: Optional[Dict[str, Any]] = None
                            ) -> str:
        """Helper to get raw XML as a string for display."""
        self.log_messages.clear()
        self._log(f"Fetching XML content for: {url}")
        content = self.fetcher.get_content(url, **(fetcher_kwargs or {}))
        if content:
            try:
                return content.decode('utf-8', errors='ignore')
            except Exception as e:
                self._log(f"Error decoding XML: {e}")
                return f"Error decoding XML: {e}"
        return f"Failed to fetch content from {url}"

    def _check_url_against_date_range(self,
                                      sitemap_url: str,
                                      start_date: Optional[datetime.datetime],
                                      end_date: Optional[datetime.datetime]) -> bool:
        """
        [新功能] 检查 sitemap URL 字符串本身是否包含日期信息，并判断是否在范围内。
        返回 True (应该处理) 或 False (应该跳过)。
        """
        # 规则 1: 如果没有日期库或日期范围，无法过滤，必须处理。
        if not date_parse or (not start_date and not end_date):
            return True

        # 规则 2: 尝试从 URL 中匹配日期
        # 匹配: 2024-01-04 | 2025-November-1 | 2025 (必须紧跟 .xml)
        pattern = r"(\d{4}-\d{2}-\d{2})|(\d{4}-[A-Za-z]+-\d{1,2})|(\d{4})(?=\.xml)"
        match = re.search(pattern, sitemap_url)

        # 规则 3: URL 中没有可识别的日期，必须处理 (依赖后续的 lastmod)
        if not match:
            return True

        date_str = match.group(0)

        try:
            # --- 统一处理时区 (从 _parse_and_check_date 复制) ---
            start_date_aware = start_date
            if start_date and start_date.tzinfo is None:
                start_date_aware = start_date.replace(tzinfo=datetime.timezone.utc)

            end_date_aware = end_date
            if end_date and end_date.tzinfo is None:
                end_date_aware = end_date.replace(tzinfo=datetime.timezone.utc)
            # --- 时区处理结束 ---

            # 规则 4: 特殊处理纯年份 (例如 "2025")
            if len(date_str) == 4 and date_str.isdigit():
                year = int(date_str)
                # 该 URL 代表的开始时间 (e.g., 2025-01-01 00:00:00)
                sitemap_year_start = datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc)
                # 该 URL 代表的结束时间 (e.g., 2025-12-31 23:59:59)
                sitemap_year_end = datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc) - datetime.timedelta(
                    seconds=1)

                # 4a: 如果用户的开始日期在这一年的结束之后 (e.g., 2026-01-01)，跳过
                if start_date_aware and start_date_aware > sitemap_year_end:
                    self._log(
                        f"  > SKIPPING (URL): Year {date_str} is older than start date {start_date_aware.date()}", 1)
                    return False

                # 4b: 如果用户的结束日期在这一年的开始之前 (e.g., 2024-12-31)，跳过
                if end_date_aware and end_date_aware < sitemap_year_start:
                    self._log(
                        f"  > SKIPPING (URL): Year {date_str} is newer than end date {end_date_aware.date()}", 1)
                    return False

                # 4c: 年份有重叠，处理
                self._log(f"  > (URL) Year {date_str} overlaps with date range. Processing.", 1)
                return True

            # 规则 5: 处理标准日期 (YYYY-MM-DD 或 YYYY-Month-D)
            sitemap_date = date_parse(date_str)
            if sitemap_date.tzinfo is None:
                sitemap_date = sitemap_date.replace(tzinfo=datetime.timezone.utc)

            # 5a: 检查开始日期
            if start_date_aware and sitemap_date < start_date_aware:
                self._log(
                    f"  > SKIPPING (URL): Date {sitemap_date.date()} is older than start date {start_date_aware.date()}",
                    1)
                return False

            # 5b: 检查结束日期
            if end_date_aware and sitemap_date > end_date_aware:
                self._log(
                    f"  > SKIPPING (URL): Date {sitemap_date.date()} is newer than end date {end_date_aware.date()}", 1)
                return False

            # 5c: 在范围内
            self._log(f"  > (URL) Date {sitemap_date.date()} is within range. Processing.", 1)
            return True

        except Exception as e:
            # 解析失败，宁可抓错也别放过
            self._log(f"  > Warning: Could not parse date '{date_str}' from URL. Error: {e}. Processing anyway.", 1)
            return True


# =======================================================================
# == 2. RSS PARSING UTILITIES (From your RSS Fetcher file)
# =======================================================================

class RssMeta(BaseModel):
    """Pydantic model for RSS feed metadata."""
    title: str = ''  # The title of channel (maybe)
    link: str = ''  # Not the feed link. I have no idea.
    description: str = ''  # Description of this feed
    language: str = ''  # Like: zh-cn
    updated: object | None = None  # ?


class RssItem(BaseModel):
    """Pydantic model for a single RSS item/entry."""
    title: str  # The title of article
    link: str  # The link of article
    published: object | None  # Published time
    authors: list  # Authors but in most case it's empty
    description: str  # Description of this article
    guid: str  # In most case it's empty
    categories: list  # In most case it's empty
    media: object | None  # ......


class FeedData(BaseModel):
    """Pydantic model for the complete parsed feed data."""
    meta: RssMeta
    entries: List[RssItem]
    errors: List[str]
    fatal: bool


def sanitize_html(raw: str) -> str:
    """Strips HTML tags and returns clean text."""
    return BeautifulSoup(raw, "html.parser").get_text(separator=" ", strip=True)


def extract_media(entry) -> list:
    """Extracts media enclosures and media:content from a feed entry."""
    media = []
    # Process enclosure tags
    for enc in entry.get("enclosures", []):
        if enc.get("type", "").startswith(("image/", "video/", "audio/")):
            media.append({
                "url": enc["href"],
                "type": enc["type"],
                "length": enc.get("length", 0)
            })
    # Process media_content extensions
    for mc in entry.get("media_content", []):
        media.append({
            "url": mc["url"],
            "type": mc.get("type", "unknown"),
            "width": mc.get("width", 0),
            "height": mc.get("height", 0)
        })
    return media


def parse_feed(content: str) -> FeedData:
    """
    Parses RSS/Atom content string using feedparser and returns a FeedData object.

    :param content: The original RSS/Atom XML content string.
    :return: A FeedData object containing metadata, entries, and any errors.
    """
    errors = []
    try:
        parsed = feedparser.parse(content)

        if parsed.get("bozo", 0) == 1:
            exception = parsed.get("bozo_exception", Exception("Unknown parsing error"))
            errors.append(str(exception))
            logger.error(f'Feed XML parse fail: {str(exception)}')

        meta = RssMeta(
            title=parsed.feed.get("title", ""),
            link=parsed.feed.get("link", ""),
            description=parsed.feed.get("description", ""),
            language=parsed.feed.get("language", "zh-cn"),
            updated=parsed.feed.get("updated_parsed", None)
        )

        # Process article items
        entries = []
        for entry in parsed.entries:
            authors = []
            for author_data in entry.get("authors", []):
                if author := author_data.get("name", '').strip():
                    authors.append(author)
            item = RssItem(
                title=entry.get("title", "Untitled"),
                link=entry.get("link", ""),
                published=entry.get("published_parsed", entry.get("updated_parsed", None)),
                authors=authors,
                description=sanitize_html(entry.get("description", "")),
                guid=entry.get("id", ""),
                categories=entry.get("tags", []),
                media=extract_media(entry)
            )
            entries.append(item)

        feed_data = FeedData(
            meta=meta,
            entries=entries,
            errors=errors,
            fatal=False
        )
        return feed_data

    except Exception as e:
        error_text = f"Exception: {str(e)}"
        errors.append(error_text)
        logger.error(error_text, exc_info=True)

        return FeedData(
            meta=RssMeta(),
            entries=[],
            errors=errors,
            fatal=True
        )


# =======================================================================
# == 3. CONCRETE IMPLEMENTATION: RSSDiscoverer
# =======================================================================

class RSSDiscoverer(IDiscoverer):
    """
    Implements the IDiscoverer interface for finding articles via
    RSS and Atom feeds.

    Stage 1 (discover_channels): Acts as a dispatcher based on the
    type of 'entry_point'.
      - If entry_point is List[str]: Uses them as a direct list of feeds.
      - If entry_point is str: Scrapes a homepage to find <link> tags
        OR treats the string as a direct feed URL.

    Stage 2 (get_articles_for_channel): Fetches a single RSS/Atom feed URL
    and parses it to extract all article links.
    """

    def __init__(self, fetcher: "Fetcher", verbose: bool = True):
        """
        Initializes the RSS discoverer.

        :param fetcher: An instance of a Fetcher implementation.
        :param verbose: Toggles detailed logging.
        """
        super().__init__(fetcher, verbose)
        self._log("Initialized RSSDiscoverer.")

        # Standard RSS/Atom MIME types to look for
        self.FEED_MIME_TYPES = {
            'application/rss+xml',
            'application/atom+xml',
            'application/xml',
            'text/xml',
            'application/feed+json',  # Also include JSON feeds
        }

    def _log(self, message: str, indent: int = 0):
        """
        Unified logging function.

        :param message: The log message.
        :param indent: The indentation level (multiplied by 4 spaces).
        """
        log_msg = f"{' ' * (indent * 4)}{message}"
        self.log_messages.append(log_msg)
        if self.verbose:
            print(log_msg)

    def _handle_direct_list(self, feed_urls: List[str]) -> List[str]:
        """
        Handles the "direct use" case where a list of feeds is provided.
        (处理“直接使用”模式，即提供了一个 feed 列表)
        """
        self._log(f"--- STAGE 1: Using direct list of {len(feed_urls)} RSS feeds ---")
        self.log_messages.clear()
        found_feeds_set: Set[str] = set()

        for url in feed_urls:
            if isinstance(url, str) and url.strip():
                abs_url = url.strip()
                found_feeds_set.add(abs_url)
                self._log(f"  > Added direct feed: {abs_url}", 1)
            else:
                self._log(f"  > [Warning] Skipping non-string item in feed list: {url}", 1)

        self._log(f"\nStage 1 Complete: Confirmed {len(found_feeds_set)} direct RSS channels.")
        return list(found_feeds_set)

    def _handle_single_url(self,
                           entry_point_url: str,
                           start_date: Optional[datetime.datetime] = None,
                           end_date: Optional[datetime.datetime] = None,
                           fetcher_kwargs: Optional[Dict[str, Any]] = None
                           ) -> List[str]:
        """
        Handles the "discovery" case from a single URL (homepage or feed).
        (处理“发现”模式，即从单个URL（主页或feed）开始)
        """
        self._log(f"--- STAGE 1: Discovering RSS Channels from {entry_point_url} ---")
        if start_date or end_date:
            self._log("[Info] 'start_date' and 'end_date' are ignored by RSSDiscoverer.", 1)

        found_feeds_set: Set[str] = set()

        # 1. Fetch the content
        content_bytes = self.fetcher.get_content(entry_point_url, **(fetcher_kwargs or {}))
        if not content_bytes:
            self._log(f"[Error] Failed to fetch content: {entry_point_url}", 1)
            return []

        try:
            content_str = content_bytes.decode('utf-8', errors='ignore')
        except Exception as e:
            self._log(f"[Error] Failed to decode content: {e}", 1)
            return []

        # 2. Check if the content is XML *before* trying to parse as HTML
        content_start_check = content_str.lstrip()
        if content_start_check.startswith(('<?xml', '<rss', '<feed')):
            self._log(f"Input URL appears to be an XML feed directly.", 1)
            self._log("Skipping HTML <link> tag discovery.", 2)
            found_feeds_set.add(entry_point_url)
            self._log(f"\nStage 1 Complete: Discovered 1 (self) RSS channel.")
            return list(found_feeds_set)

        self._log(f"Content does not look like XML. Proceeding with HTML parsing...", 1)

        # 3. Parse the HTML with BeautifulSoup (only if it wasn't XML)
        try:
            self._log(f"Parsing HTML from {entry_point_url}...", 1)
            soup = BeautifulSoup(content_str, 'html.parser')  # Use content_str

            # 4. Find all <link rel="alternate"> tags
            link_tags = soup.find_all(
                'link',
                rel='alternate',
                type=lambda t: t in self.FEED_MIME_TYPES
            )

            if not link_tags:
                self._log("No <link rel='alternate'> tags found.", 1)

            # 5. Extract and resolve URLs
            for tag in link_tags:
                href = tag.get('href')
                if not href:
                    continue

                # Resolve relative URLs (e.g., "/feed.xml") to absolute URLs
                absolute_url = urljoin(entry_point_url, href)

                if absolute_url not in found_feeds_set:
                    self._log(f"Found feed URL: {absolute_url}", 2)
                    found_feeds_set.add(absolute_url)

        except Exception as e:
            self._log(f"[Error] Failed during HTML parsing: {e}", 1)
            traceback.print_exc()

        self._log(f"\nStage 1 Complete: Discovered {len(found_feeds_set)} total RSS channels.")
        return list(found_feeds_set)

    # --- (This is now a dispatcher) ---
    def discover_channels(self,
                          entry_point: Any,  # <-- 适配新接口
                          start_date: Optional[datetime.datetime] = None,
                          end_date: Optional[datetime.datetime] = None,
                          fetcher_kwargs: Optional[Dict[str, Any]] = None
                          ) -> List[str]:
        """
        STAGE 1: Discovers all RSS/Atom feed URLs ("channels").
        This method is a dispatcher.

        - If 'entry_point' is a list:
          It treats it as a direct list of feed URLs.
          (Mode: "Directly use existing RSS Feeds")

        - If 'entry_point' is a string:
          It scrapes the URL as a homepage or treats it as a single feed.
          (Mode: "Discover RSS")

        :param entry_point: A homepage URL (str) OR a direct list of
                            feed URLs (List[str]).
        :param start_date: (Optional) Ignored by this discoverer.
        :param end_date: (Optional) Ignored by this discoverer.
        :param fetcher_kwargs: (Optional) A dictionary of keyword arguments
                               to pass to the fetcher's get_content() method.
        :return: A list of discovered RSS/Atom feed URLs.
        """
        self.log_messages.clear()

        if isinstance(entry_point, list):
            # --- "直接使用现成的RSS Feeds" 模式 ---
            return self._handle_direct_list(entry_point)

        elif isinstance(entry_point, str):
            # --- "发现RSS" 模式 ---
            return self._handle_single_url(entry_point, start_date, end_date, fetcher_kwargs)

        else:
            self._log(f"[Error] Invalid entry_point type: {type(entry_point)}. Must be str or List[str].")
            return []

    def get_articles_for_channel(self,
                                 channel_url: str,
                                 fetcher_kwargs: Optional[Dict[str, Any]] = None
                                 ) -> List[str]:
        """
        STAGE 2: Fetches and parses a single RSS/Atom feed ("channel")
        to extract all individual article URLs it contains.

        (This method was correct and did not need modification)

        :param channel_url: The URL of a single RSS/Atom feed.
        :param fetcher_kwargs: (Optional) A dictionary of keyword arguments
                               to pass to the fetcher's get_content() method.
        :return: A list of string URLs for individual articles.
        """
        self.log_messages.clear()
        self._log(f"--- STAGE 2: Fetching articles for RSS channel {channel_url} ---")

        article_urls: List[str] = []

        # 1. Fetch the raw XML content
        xml_content_bytes = self.fetcher.get_content(channel_url, **(fetcher_kwargs or {}))
        if not xml_content_bytes:
            self._log("[Error] Failed to fetch feed content.", 1)
            return []

        try:
            # feedparser prefers a string
            xml_content_str = xml_content_bytes.decode('utf-8', errors='ignore')
        except Exception as e:
            self._log(f"[Error] Failed to decode XML content: {e}", 1)
            return []

        # 2. Parse the feed using the utility function
        self._log("Parsing feed content with feedparser...", 1)
        feed_data = parse_feed(xml_content_str)

        if not feed_data:
            self._log(f"[Error] Failed to parse feed, 'feed_data' is None.", 1)
            return []

        # 检查 feedparser 的标准错误标志
        if hasattr(feed_data, 'bozo') and feed_data.bozo:
            self._log(f"[Warning] Feed is ill-formed (bozo=1). Errors: {feed_data.bozo_exception}", 1)
            # 即使格式不佳，也经常可以继续

        if not hasattr(feed_data, 'entries'):
            self._log(f"[Error] Parsed feed data has no 'entries' attribute.", 1)
            return []

        # 3. Extract the links from each entry
        for entry in feed_data.entries:
            if hasattr(entry, 'link') and entry.link and isinstance(entry.link, str):
                article_urls.append(entry.link)
            else:
                entry_title = entry.title if hasattr(entry, 'title') else 'N/A'
                self._log(f"[Warning] Found entry without a valid link: '{entry_title}'", 2)

        self._log(f"  > Found {len(article_urls)} articles in this channel.")
        return article_urls


# --- Begin: 改进后的 Link Fingerprint Data Models ---

class LinkFingerprint(BaseModel):
    """
    代表一个链接及其结构上下文。
    """
    href: str = Field(description="完整的、绝对的 URL")
    text: str = Field(description="链接的可见文本")
    signature: str = Field(description="此链接的结构化路径签名")


class ClusterFeatures(BaseModel):
    """
    用于存储一个集群的统计特征，避免重复计算。
    """
    avg_text_len: float = 0.0
    avg_href_depth: float = 0.0
    text_uniqueness: float = 0.0  # (0.0 到 1.0) 1.0 = 完全独特


class LinkGroup(BaseModel):
    """
    代表共享相同签名的一个链接集群。
    """
    signature: str = Field(description="共享的结构化路径签名")
    count: int = Field(description="此签名出现的次数")
    all_links: List[LinkFingerprint] = Field(description="此集群中的所有链接")

    @property
    def sample_links(self) -> List[LinkFingerprint]:
        """返回用于AI提示或调试的样本链接。"""
        return self.all_links[:5]


# --- End: Data Models ---


# --- Core Logic: Heuristic Dictionaries & Regex ---

# Filter 1: Junk Content (立即排除)
FILTER_JUNK: Set[str] = {
    'nav', 'menu', 'footer', 'copyright', 'legal', 'privacy', 'social', 'breadcrumb',
    'login', 'register', 'search', 'skip'
}

# Filter 2: Secondary Content (立即排除)
FILTER_SECONDARY: Set[str] = {
    'widget', 'sidebar', 'aside', 'ad', 'banner', 'comment', 'meta',
    'theme', 'themes', 'topic', 'topics', 'tag', 'tags', 'category', 'categories',
    'related', 'recommend', 'pagination'
}

# Final Scoring: Weighted Keywords (权重调整)
FINAL_SCORING_MAP: Dict[str, int] = {
    # 强正向
    'article': 30, 'post': 30, 'entry': 25, 'story': 25, 'news': 25,
    'feed': 20, 'main': 15, 'list': 10, 'item': 10,
    # 结构
    'h2': 10, 'h3': 8, 'title': 15, 'headline': 15,
    # 负向
    'top': -20, 'popular': -15, 'li': -5, 'nav': -50  # li 权重调低, nav 强负向
}

# Fingerprint: "Noisy" classes to ignore (来自你的定义)
NOISY_CLASSES: Set[str] = {
    'ng-scope', 'ng-binding', 'ng-isolate-scope', 'react-target',
    'odd', 'even', 'active', 'selected', 'hidden', 'visible', 'first', 'last'
}

# 编译后的正则表达式，用于签名泛化
RE_NUMERIC_CLASS = re.compile(r'\d+')
RE_UTILITY_CLASS = re.compile(
    r'^(p|m)(t|b|l|r|x|y)?-\S+|^[wh]-\S+|^bg-\S+|^text-\S+|^font-\S+|^border-\S+|^rounded-\S+'
    r'|^flex|^grid|^block|^inline|^hidden'
)


# --- End: Dictionaries & Regex ---


class ListPageDiscoverer(IDiscoverer):

    def __init__(self,
                 fetcher: "Fetcher",
                 verbose: bool = True,
                 min_group_count: int = 5,
                 scope_selector: Optional[str] = None,
                 manual_specified_signature: Optional[str] = None,
                 extraction_mode: str = "auto",
                 smart_scope_mode: bool = True):
        super().__init__(fetcher, verbose)

        self.log_messages: List[str] = []
        self.min_group_count = min_group_count

        # scope_selector 的语义由 extraction_mode 决定：
        #   plain_list = 列表容器
        #   card       = 文章卡片
        #   signature  = 扫描范围
        #   auto       = 自动尝试
        self.scope_selector = scope_selector

        # 兼容旧配置：若未显式指定 extraction_mode，但 smart_scope_mode=False，
        # 则映射为 signature 模式，保持旧行为。
        if extraction_mode == "auto" and not smart_scope_mode:
            self.extraction_mode = "signature"
        else:
            self.extraction_mode = extraction_mode or "auto"

        # 如果用户或 AI 已经明确指定 signature，则它的优先级最高。
        # 这种情况下不会优先走 SmartScope，以避免覆盖明确配置。
        self.manual_specified_signature = manual_specified_signature

        self.analysis_cache: Dict[str, Tuple[BeautifulSoup, List[LinkGroup]]] = {}

    def _log(self, message: str, indent: int = 0):
        log_msg = f"{' ' * (indent * 4)}{message}"
        self.log_messages.append(log_msg)
        if self.verbose:
            print(log_msg)

    def _analyze_page(self,
                      url: str,
                      fetcher_kwargs: Optional[Dict[str, Any]] = None
                      ) -> Tuple[Optional[BeautifulSoup], List[LinkGroup]]:
        if not fetcher_kwargs and url in self.analysis_cache:
            self._log(f"  [Cache] 从缓存加载: {url}", indent=1)
            return self.analysis_cache[url]

        if fetcher_kwargs:
            self._log(f"  [Info] 提供了 Fetcher kwargs，绕过缓存。", indent=1)

        self._log(f"  [Fetch] 开始抓取: {url}", indent=1)
        content = self.fetcher.get_content(url, **(fetcher_kwargs or {}))
        if not content:
            return None, []
        self._log(f"  [Parse] 正在解析 HTML (lxml)...", indent=1)
        soup = BeautifulSoup(content, 'lxml')
        self._log(f"  [Analyze] 步骤 1: 生成结构化路径签名...", indent=1)
        fingerprints = self._generate_fingerprints(soup, url)
        self._log(f"  [Analyze] 步骤 2: 聚类链接...", indent=1)
        groups = self._cluster_fingerprints(fingerprints)
        self._log(f"  [Analyze] 已完成. 发现 {len(fingerprints)} 个链接, 聚类为 {len(groups)} 组.", indent=1)

        # 仅在没有 fetcher_kwargs 时才写入缓存。
        # 原因：
        #   fetcher_kwargs 可能代表特殊请求参数、cookie、headers、代理、渲染策略等。
        #   如果把这些结果写入普通 url 缓存，后续不带 kwargs 的分析可能拿到错误结果。
        if not fetcher_kwargs:
            self.analysis_cache[url] = (soup, groups)

        return soup, groups

    def _normalize_classes(self, classes: List[str]) -> List[str]:
        """
        [新] 签名泛化助手：
        1. 移除噪音/状态类 (odd, even, active...)
        2. 移除工具类 (mt-4, p-2, flex...)
        3. 泛化数字 (item-123 -> item-N)
        """
        normalized = set()
        for c in classes:
            c_lower = c.lower()
            if c_lower in NOISY_CLASSES:
                continue
            if RE_UTILITY_CLASS.match(c_lower):
                continue

            # [修复] 移除纯数字或以数字开头的 class (例如 '987789', '2xl:text')
            # 这些类既会导致 CSS 选择器非法，也会导致聚类碎片化
            if c_lower[0].isdigit():
                continue

            # 泛化包含数字的类 (item-123 -> item-N)
            # 假设 RE_NUMERIC_CLASS = re.compile(r'\d+')
            c_normalized = RE_NUMERIC_CLASS.sub('N', c_lower)
            normalized.add(c_normalized)

        return sorted(list(normalized))

    def _get_structural_signature(self, tag: Tag, max_depth: int = 5) -> str:
        """
        [重写] 核心改进：生成一个从 <a> 标签向上的结构化路径。
        忽略无意义的 <div>/<span> 包装器。
        """
        path = []
        current = tag
        for _ in range(max_depth):
            if not current or current.name == 'body':
                break

            name = current.name

            # 仅处理有意义的 class
            classes = self._normalize_classes(current.get('class', []))

            # 忽略无 class/id 的纯包装元素
            if name in {'div', 'span'} and not classes and not current.get('id'):
                current = current.parent
                continue

            # 使用 id (如果存在) 作为强特征, 但通常不用于列表
            # tag_id = current.get('id')
            # if tag_id:
            #     path.append(f"{name}#{tag_id}") # ID 过于具体，暂不使用

            # [关键修复]
            if classes:
                # 在将类名用于 CSS 选择器之前，必须转义特殊字符。
                # 尤其是 Tailwind CSS 等框架使用的 ':' 和 '/'。
                escaped_classes = [
                    c.replace(':', r'\:').replace('/', r'\/')
                    for c in classes
                ]
                class_str = f".{'.'.join(escaped_classes)}"
            else:
                class_str = ""

            path.append(f"{name}{class_str}")

            current = current.parent
            if not current:
                break

        # 反转路径，使其成为 "parent > child" 的 CSS 选择器格式
        return " > ".join(reversed(path))

    def _generate_fingerprints(self, soup: BeautifulSoup, base_url: str) -> List[LinkFingerprint]:
        fingerprints = []
        seen_hrefs = set()

        LINK_ATTRS = ['href', 'data-link', 'data-url', 'data-href', 'ng-href', 'data-ng-href']

        # 1. 确定扫描范围 roots
        if self.scope_selector:
            self._log(f"  [Scope] 用户指定了搜索范围: '{self.scope_selector}'", indent=1)
            try:
                scope_roots = soup.select(self.scope_selector)
            except Exception as e:
                self._log(f"  [Scope] Error: 选择器语法错误: {e}", indent=2)
                return []

            if not scope_roots:
                self._log(f"  [Scope] Warning: 在页面中未找到符合 '{self.scope_selector}' 的元素。停止扫描。", indent=2)
                return []

            self._log(f"  [Scope] 命中 {len(scope_roots)} 个范围节点。", indent=1)

        else:
            default_root = soup.find('main') or soup.find('article') or soup.body
            if default_root is None:
                return []
            scope_roots = [default_root]

        potential_link_tags: Set[Tag] = set()

        def add_if_potential_link(tag: Tag):
            if not isinstance(tag, Tag):
                return

            if tag.name == 'a':
                potential_link_tags.add(tag)
                return

            for attr in LINK_ATTRS:
                if attr == 'href':
                    continue
                if tag.has_attr(attr):
                    potential_link_tags.add(tag)
                    return

        # 2. 对每个 scope root 扫描
        for root in scope_roots:
            # 关键：先检查 root 自身
            add_if_potential_link(root)

            # 再检查 descendants
            for a_tag in root.find_all('a'):
                potential_link_tags.add(a_tag)

            for attr in LINK_ATTRS:
                if attr == 'href':
                    continue
                for tag in root.find_all(attrs={attr: True}):
                    potential_link_tags.add(tag)

        # 3. 遍历候选链接节点
        for item_tag in potential_link_tags:
            href = ""
            text = ""

            for attr in LINK_ATTRS:
                if item_tag.has_attr(attr):
                    if item_tag.name == 'a' and attr != 'href':
                        continue

                    potential_href = item_tag[attr].strip()
                    if potential_href and ('http' in potential_href or potential_href.startswith('/')):
                        href = potential_href
                        break

            if not href:
                continue

            text = item_tag.get_text(strip=True)

            if not text:
                text = item_tag.get('data-title', '').strip()

            if not text:
                continue

            if href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue

            try:
                full_url = urljoin(base_url, href)
            except Exception:
                continue

            if full_url in seen_hrefs:
                continue

            if urlparse(full_url).path == urlparse(base_url).path:
                continue

            seen_hrefs.add(full_url)

            signature = self._get_structural_signature(item_tag)

            fingerprints.append(
                LinkFingerprint(
                    href=full_url,
                    text=text,
                    signature=signature
                )
            )

        self._log(f"  [Analyze] 步骤 1: 已完成. 发现 {len(fingerprints)} 个潜在链接.", indent=1)
        return fingerprints

    def _cluster_fingerprints(self, fingerprints: List[LinkFingerprint]) -> List[LinkGroup]:
        groups_map = defaultdict(list)
        for fp in fingerprints:
            groups_map[fp.signature].append(fp)

        link_groups = [
            LinkGroup(signature=sig, count=len(fps), all_links=fps)  # [修改] 使用 all_links
            for sig, fps in groups_map.items()
        ]
        link_groups.sort(key=lambda g: g.count, reverse=True)
        return link_groups

    # --- Core Logic: Decision (重构) ---

    def _get_signature_tokens(self, signature: str) -> Set[str]:
        # 'div.content > h3.title' -> {'div', 'content', 'h3', 'title'}
        # '.' '>' '#' ' ' 都是分隔符
        return set(re.split(r'[._ >#-]', signature.lower()))

    def _get_cluster_features(self, group: LinkGroup) -> ClusterFeatures:
        """
        [新] 计算一个集群的所有统计特征。
        """
        if not group.all_links:
            return ClusterFeatures()

        features = ClusterFeatures()
        all_texts = [fp.text for fp in group.all_links if fp.text]

        # 1. 文本长度 和 唯一性
        if all_texts:
            try:
                features.avg_text_len = sum(len(t) for t in all_texts) / len(all_texts)
                # [关键特征]
                features.text_uniqueness = len(set(all_texts)) / len(all_texts)
            except ZeroDivisionError:
                pass  # 保持为 0

        # 2. Href 深度
        try:
            total_depth = 0
            for fp in group.all_links:
                path = urlparse(fp.href).path.strip('/')
                if path:
                    total_depth += path.count('/') + 1
            features.avg_href_depth = total_depth / len(group.all_links)
        except ZeroDivisionError:
            pass  # 保持为 0

        return features

    def _guess_by_heuristics(self, groups: List[LinkGroup]) -> Optional[LinkGroup]:
        """
        [重构] 启发式逻辑：
        1. "淘汰赛" (使用更强的过滤器)
        2. "决赛评分" (使用 "文本唯一性" 作为核心权重)
        """
        self._log("    [Decision] 开始 '淘汰赛' 启发式...", indent=1)

        candidates = list(groups)

        # --- 预选: 存储特征以备后用 ---
        self._log(f"    [Round 0] 预计算 {len(candidates)} 个集群的特征...", indent=1)
        candidates_with_features: List[Tuple[LinkGroup, ClusterFeatures]] = []
        for g in candidates:
            if g.count < self.min_group_count:
                continue
            features = self._get_cluster_features(g)
            candidates_with_features.append((g, features))

        self._log(f"    [Round 0] 初始候选: {len(candidates_with_features)} (淘汰了 count < {self.min_group_count} 的)",
                  indent=1)
        if not candidates_with_features:
            return None

        # --- Round 1: Junk Filter ---
        self._log(f"    [Round 1] '垃圾内容过滤器' (候选: {len(candidates_with_features)})...", indent=1)
        candidates_round_1 = []
        for group, features in candidates_with_features:
            tokens = self._get_signature_tokens(group.signature)
            if tokens.intersection(FILTER_JUNK):
                reason = tokens.intersection(FILTER_JUNK).pop()
                self._log(f"      - [淘汰] {group.signature} (原因: 含 '{reason}')", indent=2)
            else:
                candidates_round_1.append((group, features))
        self._log(f"    [Round 1] 幸存: {len(candidates_round_1)}", indent=1)
        if not candidates_round_1:
            return None

        # --- Round 2: Secondary Content Filter ---
        self._log(f"    [Round 2] '次要内容过滤器' (候选: {len(candidates_round_1)})...", indent=1)
        candidates_round_2 = []
        for group, features in candidates_round_1:
            tokens = self._get_signature_tokens(group.signature)
            if tokens.intersection(FILTER_SECONDARY):
                reason = tokens.intersection(FILTER_SECONDARY).pop()
                self._log(f"      - [淘汰] {group.signature} (原因: 含 '{reason}')", indent=2)
            else:
                candidates_round_2.append((group, features))
        self._log(f"    [Round 2] 幸存: {len(candidates_round_2)}", indent=1)
        if not candidates_round_2:
            return None

        # --- Round 3: Content Sanity Filter (强化) ---
        self._log(f"    [Round 3] '内容合理性过滤器' (候选: {len(candidates_round_2)})...", indent=1)
        candidates_round_3 = []
        for group, features in candidates_round_2:
            if features.avg_text_len < 10:  # 标题平均长度至少 10
                self._log(f"      - [淘汰] {group.signature} (原因: 平均文本长度太短 {features.avg_text_len:.1f})",
                          indent=2)
            elif features.text_uniqueness < 0.3:  # [关键] 文本唯一性太低 (例如 "阅读更多")
                self._log(f"      - [淘汰] {group.signature} (原因: 文本唯一性太低 {features.text_uniqueness:.2f})",
                          indent=2)
            else:
                candidates_round_3.append((group, features))
        self._log(f"    [Round 3] 幸存: {len(candidates_round_3)}", indent=1)
        if not candidates_round_3:
            return None

        # --- Finals: Final Scoring (重构权重) ---
        self._log(f"    [Finals] '决赛评分' (候选: {len(candidates_round_3)})...", indent=1)
        best_group = None
        best_score = -999

        for group, features in candidates_round_3:
            score = 0
            log_details = []

            # 1. 文本唯一性 (最高权重)
            uniqueness_score = 0
            if features.text_uniqueness > 0.9:
                uniqueness_score = 100  # 强正向信号（几乎都是标题）
            elif features.text_uniqueness < 0.5:
                uniqueness_score = -200  # 强负向（很多重复）
            score += uniqueness_score
            log_details.append(f"唯一性得分 {uniqueness_score}")

            # 2. 签名关键词得分 (高权重)
            tokens = self._get_signature_tokens(group.signature)
            signature_score = 0
            for token in tokens:
                signature_score += FINAL_SCORING_MAP.get(token, 0)
            score += signature_score
            log_details.append(f"签名得分 {signature_score}")

            # 3. 数量得分 (中等权重)
            count_score = group.count * 2  # 权重降低
            score += count_score
            log_details.append(f"数量得分 {count_score:.0f}")

            # 4. 文本长度得分 (中等权重)
            text_len_score = min(features.avg_text_len * 0.5, 20.0)  # 权重调整
            score += text_len_score
            log_details.append(f"文本长度得分 {text_len_score:.1f}")

            # 5. Href 深度得分 (低权重)
            depth_score = features.avg_href_depth * 3  # 权重降低
            score += depth_score
            log_details.append(f"深度得分 {depth_score:.1f}")

            self._log(f"      - [评分] {group.signature} | 总分: {score:.1f}", indent=2)
            self._log(f"        (详情: {', '.join(log_details)})", indent=2)

            if score > best_score:
                best_score = score
                best_group = group

        if not best_group:
            self._log("    [Finals] 决赛评分未能选出获胜者。", indent=1)
            return None

        self._log(f"    [Winner] {best_group.signature} (总分: {best_score:.1f})", indent=1)
        return best_group

    def _find_group_by_signature(self, groups: List[LinkGroup], signature: str) -> Optional[LinkGroup]:
        for group in groups:
            if group.signature == signature:
                return group
        return None

    def _extract_links_by_signature(self, soup: BeautifulSoup, signature: str, base_url: str) -> List[str]:
        self._log(f"    [Extract] 正在使用 CSS 选择器提取链接: '{signature}'...", indent=1)

        LINK_ATTRS = ['href', 'data-link', 'data-url', 'data-href', 'ng-href', 'data-ng-href']

        # 1. 确定提取范围
        if self.scope_selector:
            try:
                roots = soup.select(self.scope_selector)
            except Exception as e:
                self._log(f"    [Extract] Scope selector 语法错误: {e}", indent=1)
                return []

            if not roots:
                self._log(f"    [Extract] 未找到 scope_selector 对应节点。", indent=1)
                return []
        else:
            roots = [soup]

        final_links = []
        seen_hrefs = set()

        # 2. 只在 scope 内 select
        for root in roots:
            try:
                link_tags = root.select(signature)
            except Exception as e:
                self._log(f"    [Extract] CSS selector 语法错误: {e}", indent=1)
                return []

            for tag in link_tags:
                if not isinstance(tag, Tag):
                    continue

                href = ""

                for attr in LINK_ATTRS:
                    if tag.has_attr(attr):
                        potential_href = tag[attr].strip()
                        if potential_href and not potential_href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                            href = potential_href
                            break

                if not href:
                    continue

                try:
                    full_url = urljoin(base_url, href)
                except Exception:
                    continue

                if full_url not in seen_hrefs:
                    final_links.append(full_url)
                    seen_hrefs.add(full_url)

        self._log(f"    [Extract] 成功提取 {len(final_links)} 个链接。", indent=1)
        return final_links

    def discover_channels(self,
                          entry_point: Any,
                          start_date: Optional[datetime.datetime] = None,
                          end_date: Optional[datetime.datetime] = None,
                          fetcher_kwargs: Optional[Dict[str, Any]] = None
                          ) -> List[str]:
        # List Page Discoverer does not discover channels.
        return list(entry_point) if isinstance(entry_point, (list, tuple, set)) else [str(entry_point)]

    def _extract_by_signature_heuristic(self,
                                        soup: BeautifulSoup,
                                        groups: List[LinkGroup],
                                        base_url: str) -> List[str]:
        """
        使用旧 signature 聚类 + heuristic 推测主文章列表。
        """
        if not groups:
            self._log("  分析失败或未找到链接组。", indent=1)
            return []

        self._log(
            "  [Decision] 使用旧 heuristic 从链接结构中推测主文章列表。",
            indent=1
        )

        winning_group = self._guess_by_heuristics(groups)

        if not winning_group:
            self._log("  [Decision] 无法确定获胜组。", indent=1)
            return []

        self._log(
            f"  [Extract] 获胜签名: {winning_group.signature} "
            f"(数量: {winning_group.count})",
            indent=1
        )

        final_links = self._extract_links_by_signature(
            soup,
            winning_group.signature,
            base_url
        )

        if not final_links and winning_group.all_links:
            self._log(
                "    [Extract] CSS 选择器提取失败，回退到使用已存储的链接。",
                indent=1
            )
            final_links = [fp.href for fp in winning_group.all_links]

        return final_links

    def get_articles_for_channel(self,
                                 channel_url: str,
                                 fetcher_kwargs: Optional[Dict[str, Any]] = None
                                 ) -> List[str]:
        self.log_messages.clear()
        self._log(f"开始从频道 (列表页) 提取文章: {channel_url}")

        soup, groups = self._analyze_page(channel_url, fetcher_kwargs=fetcher_kwargs)
        if not soup:
            self._log("  分析失败。", indent=1)
            return []

        # ------------------------------------------------------------
        # 1. 手动 signature 优先级最高
        # ------------------------------------------------------------
        if self.manual_specified_signature:
            self._log(
                f"  [Decision] 使用手动指定 signature: '{self.manual_specified_signature}'",
                indent=1
            )

            if not groups:
                self._log("  [Decision] 未找到任何签名组，无法使用手动 signature。", indent=1)
                return []

            winning_group = self._find_group_by_signature(
                groups,
                self.manual_specified_signature
            )

            if not winning_group:
                self._log("  [Decision] 手动 signature 未匹配到任何 group。", indent=1)
                return []

            final_links = self._extract_links_by_signature(
                soup,
                winning_group.signature,
                channel_url
            )

            if not final_links and winning_group.all_links:
                self._log(
                    "    [Extract] CSS 选择器提取失败，回退到 group 内已存储链接。",
                    indent=1
                )
                final_links = [fp.href for fp in winning_group.all_links]

            return final_links

        # ------------------------------------------------------------
        # 2. 根据 extraction_mode 分流
        # ------------------------------------------------------------
        mode = self.extraction_mode

        if mode == "plain_list":
            self._log("  [Decision] 使用 Plain List 模式提取。", indent=1)
            return self._extract_articles_from_plain_lists(soup, channel_url)

        if mode == "card":
            self._log("  [Decision] 使用 Card 模式提取。", indent=1)
            return self._extract_articles_from_scoped_cards(soup, channel_url)

        if mode == "signature":
            self._log("  [Decision] 使用 Signature 模式提取。", indent=1)
            return self._extract_by_signature_heuristic(soup, groups, channel_url)

        if mode == "auto":
            if self.scope_selector:
                self._log("  [Decision] Auto 模式: 先尝试 Plain List。", indent=1)
                links = self._extract_articles_from_plain_lists(soup, channel_url)
                if links:
                    return links

                self._log("  [Decision] Auto 模式: Plain List 失败，尝试 Card。", indent=1)
                links = self._extract_articles_from_scoped_cards(soup, channel_url)
                if links:
                    return links

            self._log("  [Decision] Auto 模式: 回退到 Signature Heuristic。", indent=1)
            return self._extract_by_signature_heuristic(soup, groups, channel_url)

        self._log(f"  [Error] 未知的 extraction_mode: {mode}", indent=1)
        return []

    def _extract_articles_from_plain_lists(self,
                                           soup: BeautifulSoup,
                                           base_url: str) -> List[str]:
        """
        Plain List 模式。
        从传统 ul/ol > li > a 新闻列表中提取文章链接。
        """
        if not self.scope_selector:
            roots = [soup]
        else:
            try:
                roots = soup.select(self.scope_selector)
            except Exception as e:
                self._log(f"  [PlainList] scope_selector 语法错误: {e}", indent=1)
                return []

            if not roots:
                self._log(f"  [PlainList] 未找到 scope_selector: {self.scope_selector}", indent=1)
                return []

        lists: List[Tag] = []
        for root in roots:
            if isinstance(root, Tag):
                if root.name in ('ul', 'ol'):
                    lists.append(root)
                else:
                    # 优先查找直接子级，减少深层误匹配
                    direct = [c for c in root.find_all(['ul', 'ol'], recursive=False) if isinstance(c, Tag)]
                    if direct:
                        lists.extend(direct)
                    else:
                        lists.extend([c for c in root.find_all(['ul', 'ol']) if isinstance(c, Tag)])

        final_links: List[str] = []
        seen_urls: Set[str] = set()

        skipped_image = 0
        skipped_no_link = 0
        skipped_low_score = 0
        skipped_duplicate = 0

        for ul in lists:
            for li in ul.find_all('li', recursive=False):
                if not isinstance(li, Tag):
                    continue

                # 跳过明显是图片/缩略图/广告区的 li
                if self._is_image_only_li(li):
                    skipped_image += 1
                    continue

                candidates: List[Tag] = []
                if li.name == 'a':
                    candidates.append(li)
                candidates.extend(li.find_all('a'))

                if not candidates:
                    skipped_no_link += 1
                    continue

                # 对 li 内所有 a 打分，选最佳
                scored: List[Tuple[int, Tag, str, str]] = []
                for a_tag in candidates:
                    href = a_tag.get("href", "").strip()
                    text = self._clean_link_text(a_tag)
                    score = self._score_link_in_card(a_tag, li, base_url)
                    scored.append((score, a_tag, href, text))

                scored.sort(key=lambda x: x[0], reverse=True)
                best_score, best_tag, best_href, best_text = scored[0]

                if best_score < 0:
                    skipped_low_score += 1
                    continue

                try:
                    full_url = urljoin(base_url, best_href)
                except Exception:
                    continue

                if full_url in seen_urls:
                    skipped_duplicate += 1
                    continue

                seen_urls.add(full_url)
                final_links.append(full_url)

        self._log(
            f"  [PlainList] 最终提取 {len(final_links)} 个链接。"
            f"lists={len(lists)}, image_li={skipped_image}, no_link={skipped_no_link}, "
            f"low_score={skipped_low_score}, duplicate={skipped_duplicate}",
            indent=1
        )
        return final_links

    def _is_image_only_li(self, li: Tag) -> bool:
        """
        判断一个 li 是否主要是图片/缩略图/广告区，应被跳过。
        """
        if not isinstance(li, Tag):
            return False

        li_classes = " ".join(li.get("class", [])).lower()
        li_id = (li.get("id") or "").lower()

        image_keywords = [
            'image', 'img', 'photo', 'thumbnail', 'thumb', 'pic', 'figure',
            '广告', 'ad-', '-ad', 'banner', 'sprite',
        ]

        for kw in image_keywords:
            if kw in li_classes or kw in li_id:
                return True

        # 如果 li 内没有有意义的文本，且只有图片/图标，也视为 image-only
        text_content = li.get_text(strip=True)
        has_meaningful_text = len(text_content) >= 3 and not text_content.startswith('http')

        if li.find('img') and not has_meaningful_text:
            return True

        return False

    def get_signature_groups(self,
                             page_url: str,
                             fetcher_kwargs: Optional[Dict[str, Any]] = None
                             ) -> List[Dict[str, Any]]:
        """
        为 UI 提供分析结果。

        运行 (或从缓存获取) 页面分析，并返回一个简化的、
        适合 UI 显示的 "签名组" 列表。

        Args:
            page_url (str): 要分析的列表页 URL。
            fetcher_kwargs (Optional[Dict...]): [新增] 传递给 fetcher 的参数

        Returns:
            List[Dict[str, Any]]:
            一个字典列表，每个字典代表一个签名组，包含:
            - "signature" (str): 签名
            - "count" (int): 链接数量
            - "sample_links" (List[Dict]): 示例链接 (href, text)
        """
        self.log_messages.clear()
        self._log(f"开始为 UI 分析签名组: {page_url}")

        try:
            # 1. 运行核心分析 (这将使用缓存，如果存在)
            soup, groups = self._analyze_page(page_url, fetcher_kwargs=fetcher_kwargs)

            if not groups:
                self._log(f"  分析未找到任何链接组。", indent=1)
                return []

            # 2. 将 Pydantic 模型转换为简单的字典列表
            #    我们使用 `sample_links` 属性，而不是 `all_links`，以保持数据量可控
            results_for_ui = []
            for group in groups:
                # 将 LinkFingerprint (Pydantic) 转换为 dict
                sample_links_as_dicts = [
                    {"href": fp.href, "text": fp.text}
                    for fp in group.sample_links  # 使用 @property
                ]

                group_data = {
                    "signature": group.signature,
                    "count": group.count,
                    "sample_links": sample_links_as_dicts
                }
                results_for_ui.append(group_data)

            self._log(f"  分析完成. 返回 {len(results_for_ui)} 个签名组。", indent=1)

            # 结果已按数量降序排列 (来自 _cluster_fingerprints)
            return results_for_ui

        except Exception as e:
            # 确保导入 traceback
            import traceback
            self._log(f"  [Error] get_signature_groups 失败: {str(e)}")
            self._log(traceback.format_exc())
            return []

    def _class_text(self, tag: Tag) -> str:
        """
        返回标签 class 的小写文本形式，方便做规则判断。
        """
        if not isinstance(tag, Tag):
            return ""
        return " ".join(tag.get("class", [])).lower()

    def _ancestor_class_text(self, tag: Tag, max_depth: int = 6) -> str:
        """
        收集当前标签及其若干层父节点的 class 文本。
        用于判断链接是否位于 headline/title/image/label 等区域。
        """
        parts = []
        current = tag

        for _ in range(max_depth):
            if not current or not isinstance(current, Tag):
                break

            parts.append(self._class_text(current))
            current = current.parent

        return " ".join(parts).lower()

    def _clean_link_text(self, tag: Tag) -> str:
        """
        清洗链接文本。
        """
        if not isinstance(tag, Tag):
            return ""

        text = tag.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _infer_cards_from_scope_root(self, root: Tag) -> List[Tag]:
        """
        当 scope_selector 只命中一个大容器时，尝试在容器内部自动推断文章卡片。

        例如用户传了：
            .ArticleList

        但真正的卡片是：
            li.ArticleHeadlineListWrap

        这个函数会尝试在大容器里找重复的 li/article/item/card 节点。
        """
        if not isinstance(root, Tag):
            return []

        candidate_selectors = [
            "li",
            "article",
            "[class*='ListWrap']",
            "[class*='listwrap']",
            "[class*='ListItem']",
            "[class*='listitem']",
            "[class*='Item']",
            "[class*='item']",
            "[class*='Card']",
            "[class*='card']",
        ]

        best_cards: List[Tag] = []

        for selector in candidate_selectors:
            try:
                cards = root.select(selector)
            except Exception:
                continue

            # 只保留里面有链接的节点
            cards = [
                c for c in cards
                if isinstance(c, Tag) and (c.name == "a" or c.find("a"))
            ]

            if len(cards) > len(best_cards):
                best_cards = cards

        return best_cards

    def _score_link_in_card(self, a_tag: Tag, card: Tag, base_url: str) -> int:
        """
        对一张文章卡片内部的某个 <a> 链接打分。
        分数越高，越可能是这张卡片的主链接。

        重要语义：
        - SmartScope 的目标不是只提取 /articles/*.html。
        - SmartScope 的目标是：scope_selector 命中的每张卡片，尽量抽取一个“主链接”。
        - 因此 /topics/、/special/ 这类列表卡片链接不应被强行过滤。
        """
        if not isinstance(a_tag, Tag):
            return -10_000

        href = a_tag.get("href", "").strip()
        if not href:
            return -10_000

        href_lower = href.lower()
        if href_lower.startswith(("#", "javascript:", "mailto:", "tel:")):
            return -10_000

        try:
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
        except Exception:
            return -10_000

        score = 0

        path = parsed.path.lower()
        text = self._clean_link_text(a_tag)
        text_len = len(text)

        self_class = self._class_text(a_tag)
        ancestor_class = self._ancestor_class_text(a_tag)

        # ------------------------------------------------------------
        # 1. URL 形态评分：文章 URL 加分，但不排斥 topic/special
        # ------------------------------------------------------------

        if "/articles/" in path:
            score += 120

        if path.endswith(".html"):
            score += 40

        if re.search(r"/articles/[A-Za-z0-9]+\.html", parsed.path):
            score += 80

        # special 页面也可能是页面主卡片
        if "/special/" in path:
            score += 50

        # topics 是主题页，不一定是普通文章，但如果页面上就是一个卡片，也应允许提取
        if "/topics/" in path:
            score += 10

        # 明显列表页稍微降权，但不要一票否决
        if "/list.html" in path:
            score -= 40

        if "/rensai/" in path:
            score -= 30

        # ------------------------------------------------------------
        # 2. 结构位置评分
        # ------------------------------------------------------------

        # 标题区域是最强信号
        if "headline" in ancestor_class:
            score += 160

        if "title" in ancestor_class:
            score += 100

        # 朝日的覆盖链接，也可能是主链接
        if "articleheadlinelink" in self_class:
            score += 60

        # 图片链接通常不是最佳语义链接，但如果没有标题链接，也可以作为 fallback
        if a_tag.find("img"):
            score -= 50

        if "image" in ancestor_class:
            score -= 40

        # 标签、分类、作者、meta 等上下文降权
        bad_context_keywords = [
            "rensailabel",
            "label",
            "tag",
            "category",
            "breadcrumb",
            "meta",
            "author",
            "series",
            "related",
            "recommend",
        ]

        for kw in bad_context_keywords:
            if kw in self_class or kw in ancestor_class:
                score -= 80

        # 注意：不要把 topic 作为强负向。
        # 因为有些页面里 topic card 本身就是列表项。

        # ------------------------------------------------------------
        # 3. 文本质量评分
        # ------------------------------------------------------------

        if text_len >= 12:
            score += 70
        elif text_len >= 6:
            score += 35
        elif text_len > 0:
            score += 10
        else:
            score -= 80

        low_value_texts = {
            "写真・図版",
            "画像",
            "photo",
            "image",
            "read more",
            "more",
        }

        if text.lower() in low_value_texts:
            score -= 100

        # ------------------------------------------------------------
        # 4. 标题标签内的链接强加分
        # ------------------------------------------------------------

        parent = a_tag.parent
        if isinstance(parent, Tag) and parent.name in {"h1", "h2", "h3", "h4"}:
            score += 140

        # ------------------------------------------------------------
        # 5. 外链轻微降权
        # ------------------------------------------------------------

        base_host = urlparse(base_url).netloc
        link_host = parsed.netloc

        if base_host and link_host and base_host != link_host:
            score -= 20

        return score

    def _extract_articles_from_scoped_cards(self,
                                            soup: BeautifulSoup,
                                            base_url: str) -> List[str]:
        """
        SmartScope 模式。

        当用户指定 scope_selector 且 smart_scope_mode=True 时启用。

        语义说明：
        - scope_selector 命中多个节点：
            每个节点视为一张候选卡片，系统在每张卡片内部选择一个主链接。
        - scope_selector 命中一个大容器：
            尝试在容器内部自动推断多张卡片。
        - scope_selector 直接命中 <a>：
            每个 <a> 自身就是候选主链接。

        注意：
        - SmartScope 不等于“只抓 /articles/*.html”。
        - SmartScope 更接近“抓列表页中每张卡片的主链接”。
        - 所以 /topics/、/special/ 等只要是卡片主链接，也会被保留。
        - 如果想严格只保留普通文章 URL，应在更上层增加 URL filter，而不是在这里硬编码。
        """
        if not self.scope_selector:
            return []

        try:
            cards = soup.select(self.scope_selector)
        except Exception as e:
            self._log(f"  [SmartScope] scope_selector 语法错误: {e}", indent=1)
            return []

        if not cards:
            self._log(f"  [SmartScope] 未找到 scope_selector: {self.scope_selector}", indent=1)
            return []

        cards = [c for c in cards if isinstance(c, Tag)]

        self._log(
            f"  [SmartScope] scope_selector 命中 {len(cards)} 个节点。",
            indent=1
        )

        # 如果只命中一个大容器，尝试自动推断内部 card。
        if len(cards) == 1:
            inferred_cards = self._infer_cards_from_scope_root(cards[0])

            if len(inferred_cards) >= self.min_group_count:
                self._log(
                    f"  [SmartScope] scope 似乎是列表容器，自动推断出 {len(inferred_cards)} 张候选卡片。",
                    indent=1
                )
                cards = inferred_cards

        final_links: List[str] = []
        seen_urls: Set[str] = set()

        skipped_no_link = 0
        skipped_low_score = 0
        skipped_duplicate = 0

        # 这个阈值不要太高。
        # 现在 SmartScope 的目标是“每张卡片尽量抽主链接”，不是只抽传统文章。
        min_accept_score = 20

        for idx, card in enumerate(cards):
            if not isinstance(card, Tag):
                continue

            candidates: List[Tag] = []

            # 如果 card 本身就是 <a>
            if card.name == "a":
                candidates.append(card)

            # card 内部所有 <a>
            candidates.extend(card.find_all("a"))

            if not candidates:
                skipped_no_link += 1
                self._log(
                    f"    [SmartScope] card #{idx} 没有 <a>，跳过。",
                    indent=2
                )
                continue

            scored: List[Tuple[int, Tag, str, str]] = []

            for a_tag in candidates:
                href = a_tag.get("href", "").strip()
                text = self._clean_link_text(a_tag)
                score = self._score_link_in_card(a_tag, card, base_url)
                scored.append((score, a_tag, href, text))

            scored.sort(key=lambda x: x[0], reverse=True)

            best_score, best_tag, best_href, best_text = scored[0]

            if best_score < min_accept_score:
                skipped_low_score += 1

                # 打印 top3，方便看为什么没选。
                top3 = []
                for s, _, h, t in scored[:3]:
                    top3.append(f"score={s}, text='{t[:30]}', href='{h[:80]}'")

                self._log(
                    f"    [SmartScope] card #{idx} 最高分过低，跳过。"
                    f"最高分={best_score}, top={top3}",
                    indent=2
                )
                continue

            try:
                full_url = urljoin(base_url, best_href)
            except Exception:
                continue

            if full_url in seen_urls:
                skipped_duplicate += 1
                self._log(
                    f"    [SmartScope] card #{idx} 与前面重复，跳过: {full_url}",
                    indent=2
                )
                continue

            seen_urls.add(full_url)
            final_links.append(full_url)

            self._log(
                f"    [SmartScope] card #{idx} 选择主链接: "
                f"score={best_score}, text='{best_text[:50]}', url={full_url}",
                indent=2
            )

        self._log(
            f"  [SmartScope] 最终提取 {len(final_links)} 个链接。"
            f"cards={len(cards)}, no_link={skipped_no_link}, "
            f"low_score={skipped_low_score}, duplicate={skipped_duplicate}",
            indent=1
        )

        return final_links

    # --- AI Helper Method (已更新以使用 'sample_links' 属性) ---
    def generate_ai_discovery_prompt(self, entry_point_url: str) -> Optional[str]:
        self.log_messages.clear()
        self._log(f"正在为以下地址生成 AI 发现提示: {entry_point_url}")
        soup, groups = self._analyze_page(entry_point_url)
        if not soup:
            return None
        if not groups:
            return None
        page_title = soup.title.string.strip() if soup.title else ""
        prompt = self._prepare_ai_prompt(groups, page_title, entry_point_url)
        self._log(f"  成功生成 AI 提示。", indent=1)
        return prompt

    def _prepare_ai_prompt(self, groups: List[LinkGroup], page_title: str, page_url: str) -> str:
        # [修复] 手动构建字典以正确包含 'sample_links' 属性
        groups_data = []
        for g in groups:
            groups_data.append({
                "signature": g.signature,
                "count": g.count,
                # 'sample_links' 是一个属性，我们在这里调用它
                "sample_links": [fp.model_dump() for fp in g.sample_links]
            })

        payload = {"page_url": page_url, "page_title": page_title, "link_groups": groups_data}
        json_payload = json.dumps(payload, indent=2, ensure_ascii=False)
        system_prompt = """You are a professional web structure analysis engine. Your task is to analyze a JSON input...
(AI prompt details omitted for brevity)...
**Task:**
Analyze the following JSON data and **return only** the `signature` string of the group you believe is the **main article list**. If none is found, return `null`.
"""
        return f"{system_prompt}\n\n**Input Data:**\n```json\n{json_payload}\n```"
