#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extractor Module:
Defines the IExtractor interface and provides multiple implementations
for extracting main content from HTML and converting it to Markdown.
"""
import re
import copy
import json
import html2text
import traceback
import lxml.etree
import unicodedata
from urllib.parse import urljoin
from bs4 import BeautifulSoup, Tag
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict, Any, Optional, Literal

# --- Library Import Checks ---
# These imports are optional. Implementations will check if they
# were successful before attempting to run.

try:
    from readability import Document

    print("Success: Imported 'readability-lxml'. ReadabilityExtractor is available.")
except ImportError:
    Document = None
    print("!!! FAILED to import 'readability-lxml'. ReadabilityExtractor will NOT be available.")
    print("!!! Please install it: pip install readability-lxml")

try:
    import trafilatura

    print("Success: Imported 'trafilatura'. TrafilaturaExtractor is available.")
except ImportError:
    trafilatura = None
    print("!!! FAILED to import 'trafilatura'. TrafilaturaExtractor will NOT be available.")
    print("!!! Please install it: pip install trafilatura")

try:
    from newspaper import Article

    print("Success: Imported 'newspaper'. Newspaper3kExtractor is available.")
except ImportError:
    Article = None
    print("!!! FAILED to import 'newspaper'. Newspaper3kExtractor will NOT be available.")
    print("!!! Please install it: pip install newspaper3k")

try:
    from crawl4ai.crawler import Crawler
    from crawl4ai.extraction import SmartExtractor

    print("Success: Imported 'crawl4ai'. Crawl4AIExtractor is available.")
except ImportError:
    Crawler = None
    SmartExtractor = None
    print("!!! FAILED to import 'crawl4ai'. Crawl4AIExtractor will NOT be available.")
    print("!!! Please install it: pip install crawl4ai")

# --- Type Alias for Unicode Sanitizer ---
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias
# --- NEW: Pydantic model for a standardized extraction result ---
try:
    from pydantic import BaseModel, Field, computed_field

    print("Success: Imported 'pydantic'. IExtractor will use ExtractionResult.")
except ImportError:
    print("!!! FAILED to import 'pydantic'. ExtractionResult will be a dict.")
    print("!!! Please install it: pip install pydantic")


    # Define fallback classes if pydantic isn't available
    # This allows the app to still run, albeit without type validation
    class BaseModel:
        def json(self, **kwargs):
            import json
            return json.dumps(self.__dict__, **kwargs)


    def Field(default=None, **kwargs):
        return default


# ----------------------------------------------------------------------------------------------------------------------

class ExtractionResult(BaseModel):
    """
    Standardized return object for all IExtractor implementations.
    (所有 IExtractor 实现的标准返回对象。)
    """
    markdown_content: str = Field(
        default="",
        description="The main content in Markdown format.",
        repr=False  # 1. 不在 __repr__ 中显示完整内容，避免刷屏
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted metadata (e.g., title, author, date)."
    )
    error: Optional[str] = Field(
        default=None,
        description="An error message if extraction failed."
    )

    @computed_field(repr=True)
    @property
    def content_preview(self) -> str:
        """A truncated preview of the content for repr."""
        if not self.markdown_content:
            return "[No Content]"

        cleaned_content = self.markdown_content.replace('\n', ' ')
        if len(cleaned_content) > 100:
            return cleaned_content[:100] + "..."
        return cleaned_content

    @property
    def success(self) -> bool:
        """Returns True if the extraction was successful (no error)."""
        return self.error is None

    def __str__(self):
        """
        Provides a comprehensive, human-readable summary.
        (提供一个全面且易读的摘要。)
        """

        # 1. 失败情况 (不变)
        if not self.success:
            return f"[Extraction FAILED]\n└── Error: {self.error}"

        # --- 2. 成功的情况 (根据您的反馈调整) ---
        output = ["[Extraction SUCCESS]"]

        # 明确显示标题 (无论有无)
        title = self.metadata.get('title')
        if title:
            output.append(f"├── Title: {title}")
        else:
            output.append(f"├── Title: [No Title Found]")

        # 明确显示内容预览 (无论有无)
        if self.markdown_content:
            # 清理换行符并截断
            preview_str = self.markdown_content.replace('\n', ' ').strip()
            if len(preview_str) > 70:
                preview_str = preview_str[:70] + "..."
            elif not preview_str:
                preview_str = "[Content is whitespace]"
            output.append(f"├── Content: \"{preview_str}\"")
        else:
            output.append(f"├── Content: [No Content]")

        # 3. 附加元数据信息 (作为最后一项)
        if self.metadata:
            try:
                meta_json = json.dumps(self.metadata, indent=2, ensure_ascii=False, default=str)
                meta_lines = [f"│   {line}" for line in meta_json.splitlines()]
                output.append(f"└── Metadata:\n" + "\n".join(meta_lines))
            except Exception as e:
                output.append(f"└── Metadata: [Error serializing: {e}]")
        else:
            output.append("└── Metadata: [None]")

        return "\n".join(output)

NormalizationForm: TypeAlias = Literal["NFC", "NFD", "NFKC", "NFKD"]


# =======================================================================
# == 1. UNICODE SANITIZER UTILITY (Your Code)
# =======================================================================

def sanitize_unicode_string(
        text: str,
        max_length: int = 10240,
        normalize_form: NormalizationForm = 'NFKC',
        allow_emoji: bool = False
) -> Optional[str]:
    """
    Sanitizes and cleans input string by removing Unicode variation selectors,
    combining characters, and other potentially dangerous Unicode features.
    (This function is from your provided code, with minor type hint fixes.)

    Args:
        text: Input string to be sanitized
        max_length: Maximum allowed input length (defense against bomb attacks)
        normalize_form: Unicode normalization form (NFKC recommended).
        allow_emoji: Whether to preserve emoji characters

    Returns:
        Sanitized string or None if input exceeds max_length
    """
    if not text:
        return ""

    # Defense against character bomb attacks
    if len(text) > max_length:
        text = text[:max_length]

    # Unicode normalization
    try:
        normalized = unicodedata.normalize(normalize_form, text)
    except ValueError as e:
        raise ValueError(f"Invalid normalization form: {normalize_form}") from e

    # Regex pattern for comprehensive filtering
    variation_selector_ranges = (
        r'\u180B-\u180D'  # Mongolian variation selectors
        r'\uFE00-\uFE0F'  # Unicode variation selectors
        r'[\uDB40-\uDBFF][\uDC00-\uDFFF]'  # Surrogate pairs handling
    )

    emoji_block = (r'\U0001F000-\U0001FAFF'  # Basic block (Note: \U for 32-bit)
                   r'\u231A-\u231B'  # Watch symbols
                   r'\u23E9-\u23FF'  # Control symbols
                   ) if not allow_emoji else ''

    danger_pattern = re.compile(
        r'['
        r'\u0000-\u001F\u007F-\u009F' +  # Control characters
        r'\u0300-\u036F' +  # Combining diacritics
        r'\u200B-\u200D\u202A-\u202E' +  # Zero-width/control characters
        r'\uFFF0-\uFFFF'  # Special purpose characters
        + emoji_block +
        variation_selector_ranges +
        r']',
        flags=re.UNICODE
    )

    sanitized = danger_pattern.sub('', normalized)
    sanitized = re.sub(r'[\uFE00-\uFE0F]', '', sanitized)  # Final check
    return sanitized.strip()


# =======================================================================
# == 2. ABSTRACT BASE CLASS (Interface)
# =======================================================================

class IExtractor(ABC):
    """
    Abstract base class for a content extractor.
    (内容提取器的抽象基类)

    The role of an extractor is to take raw HTML content and a URL,
    extract the main article content, and return it as a
    clean Markdown string.
    (提取器的职责是接收原始HTML内容和URL，
     提取主要文章内容，并将其作为干净的Markdown字符串返回。)
    """

    def __init__(self, verbose: bool = True):
        """
        Initializes the extractor.
        (初始化提取器。)

        :param verbose: Toggles detailed logging.
        """
        self.verbose = verbose
        self.log_messages: List[str] = []

    def _log(self, message: str, indent: int = 0):
        """
        Provides a unified logging mechanism.
        (提供统一的日志记录机制。)
        """
        log_msg = f"{' ' * (indent * 4)}{message}"
        self.log_messages.append(log_msg)
        if self.verbose:
            print(log_msg)

    @abstractmethod
    def extract(self, content: bytes, url: str, **kwargs) -> ExtractionResult:
        """
        Extracts the main content and metadata from raw HTML bytes.
        (从原始HTML字节中提取主要内容和元数据。)

        :param content: The raw HTML content as bytes.
                        (作为字节的原始HTML内容。)
        :param url: The original URL (for context and resolving relative links).
                    (原始URL（用于上下文和解析相对链接）。)
        :param kwargs: Implementation-specific options (e.g., CSS selectors).
                       (特定于实现的选项（例如CSS选择器）。)
        :return: An ExtractionResult object containing markdown, metadata, and errors.
                 (一个包含Markdown、元数据和错误的 ExtractionResult 对象。)
        """
        pass


# =======================================================================
# == 3. IMPLEMENTATIONS
# =======================================================================

class TrafilaturaExtractor(IExtractor):
    """
    Extractor implementation using the 'trafilatura' library.
    (使用 'trafilatura' 库的提取器实现。)

    This is a highly effective algorithmic extractor that natively
    supports Markdown output.
    (这是一个高效的算法提取器，原生支持Markdown输出。)
    """

    def extract(self, content: bytes, url: str, **kwargs) -> ExtractionResult:
        """
        Extracts content and metadata *separately* using trafilatura 2.0.0.
        Calls extract() twice to ensure clean separation.

        :param content: Raw HTML bytes.
        :param url: The original URL.
        :param kwargs: Passed to `trafilatura.extract()`.
        :return: ExtractionResult
        """
        if include_images := kwargs.pop('include_images', False):
            self._log(f"Extracting with TrafilaturaExtractor (Markdown + Images) from {url}")
        else:
            self._log(f"Extracting with TrafilaturaExtractor from {url}")

        if not trafilatura:
            error_str = "[Error] Trafilatura library not found."
            self._log(error_str)
            return ExtractionResult(error=error_str)

        # Remove fetcher parameters
        kwargs.pop('fetcher_kwargs', None)
        kwargs.pop('extractor_kwargs', None)

        try:
            # --- 1. 第一次调用：只获取干净的 Markdown 内容 ---
            kwargs_content = kwargs.copy()
            kwargs_content.pop('output_format', None)
            kwargs_content.pop('with_metadata', None)  # 确保不请求元数据

            markdown = trafilatura.extract(
                content,
                url=url,
                output_format='markdown',
                include_links=True,
                include_images=include_images,
                **kwargs_content
            )

            # --- 2. 第二次调用：只获取元数据 (通过 JSON) ---
            kwargs_meta = kwargs.copy()
            # 清除所有内容相关的参数
            kwargs_meta.pop('output_format', None)
            kwargs_meta.pop('include_links', None)
            kwargs_meta.pop('include_tables', None)

            json_string = trafilatura.extract(
                content,
                url=url,
                output_format='json',
                with_metadata=True,
                **kwargs_meta
            )

            # --- 3. 解析 JSON 字符串以提取元数据 ---
            metadata = {}
            if json_string:
                try:
                    data_dict = json.loads(json_string)
                    # JSON 输出会把元数据和 'text' 放在同一个字典里
                    # 我们只提取元数据，忽略 'text'
                    metadata = {
                        'title': data_dict.get('title'),
                        'author': data_dict.get('author'),
                        'date': data_dict.get('date'),
                        'sitename': data_dict.get('sitename'),
                        'tags': data_dict.get('tags'),
                        'fingerprint': data_dict.get('fingerprint'),
                        'id': data_dict.get('id'),
                        'license': data_dict.get('license'),
                        'description': data_dict.get('description'),
                        'image': data_dict.get('image'),
                        'url': data_dict.get('url'),
                        'hostname': data_dict.get('hostname'),
                    }
                    # 移除值为 None 的键，保持 metadata 字典干净
                    metadata = {k: v for k, v in metadata.items() if v is not None}

                except json.JSONDecodeError as json_err:
                    self._log(f"[Warning] Trafilatura JSON output was invalid: {json_err}")
                except Exception as e:
                    self._log(f"[Warning] Failed to parse metadata JSON: {e}")

            if markdown is None and not metadata:
                self._log("[Info] Trafilatura returned None for both content and metadata.")
                return ExtractionResult(error="Trafilatura failed to extract content.")

            return ExtractionResult(
                markdown_content=sanitize_unicode_string(markdown or ""),
                metadata=metadata  # 这是干净的 metadata 字典
            )

        except Exception as e:
            error_str = f"Trafilatura failed: {e}"
            self._log(f"[Error] {error_str}")
            self._log(traceback.format_exc())
            return ExtractionResult(error=error_str)


class ReadabilityExtractor(IExtractor):
    """
    Extractor implementation using 'readability-lxml'.
    (使用 'readability-lxml' 的提取器实现。)

    This is a refactored version of your `clean_html_content` function.
    It uses the Readability algorithm to find the main content HTML,
    then converts that HTML to Markdown.
    (这是您 `clean_html_content` 函数的重构版本。
     它使用Readability算法找到主要内容HTML，然后将其转换为Markdown。)
    """

    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.converter = html2text.HTML2Text()
        self.converter.ignore_links = False
        self.converter.ignore_images = False  # Markdown should include images
        self.converter.body_width = 0  # Don't wrap lines

    def extract(self, content: bytes, url: str, **kwargs) -> ExtractionResult:
        """
        Extracts content using readability-lxml.

        :param content: Raw HTML bytes.
        :param url: The original URL (ignored, but part of interface).
        :param kwargs: Ignored by this implementation.
        :return: Markdown string.
        """
        self._log(f"Extracting with ReadabilityExtractor from {url}")
        if not Document:
            error_str = "[Error] readability-lxml library not found."
            self._log(error_str)
            return ExtractionResult(error=error_str)

        try:
            html_str = content.decode('utf-8', errors='ignore')
            doc = Document(html_str)
            main_content_html = doc.summary()

            # Try to get the title from the Document object
            metadata = {'title': doc.title()}

            markdown = self.converter.handle(main_content_html)
            return ExtractionResult(
                markdown_content=sanitize_unicode_string(markdown),
                metadata=metadata
            )
        except Exception as e:
            error_str = f"Readability failed: {e}"
            self._log(f"[Error] {error_str}")
            self._log(traceback.format_exc())
            return ExtractionResult(error=error_str)


class Newspaper3kExtractor(IExtractor):
    """
    Extractor implementation using the 'newspaper3k' library.
    (使用 'newspaper3k' 库的提取器实现。)

    Newspaper3k only extracts plain text. This implementation
    works around this by:
    1. Letting newspaper3k parse the HTML.
    2. Grabbing the 'top_node' (the lxml element it found).
    3. Converting that element back to HTML.
    4. Converting the resulting HTML to Markdown.
    (Newspaper3k 只能提取纯文本。此实现通过以下方式解决：
     1. 让 newspaper3k 解析HTML。
     2. 获取 'top_node' (它找到的 lxml 元素)。
     3. 将该元素转换回HTML。
     4. 将生成的HTML转换为Markdown。)
    """

    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.converter = html2text.HTML2Text()
        self.converter.ignore_links = False
        self.converter.ignore_images = False
        self.converter.body_width = 0

    def extract(self, content: bytes, url: str, **kwargs) -> ExtractionResult:
        """
        Extracts content using newspaper3k.

        :param content: Raw HTML bytes.
        :param url: The original URL (required by newspaper).
        :param kwargs: Ignored by this implementation.
        :return: Markdown string.
        """
        self._log(f"Extracting with Newspaper3kExtractor from {url}")
        if not Article:
            error_str = "[Error] newspaper3k library not found."
            self._log(error_str)
            return ExtractionResult(error=error_str)

        try:
            html_str = content.decode('utf-8', errors='ignore')
            article = Article(url)
            article.set_html(html_str)
            article.parse()

            if article.top_node is None:
                self._log("[Info] Newspaper3k could not find a top_node.")
                return ""  # Failed to find content

            # Convert the main lxml node back to HTML
            if article.top_node is None:
                self._log("[Info] Newspaper3k could not find a top_node.")
                return ExtractionResult(error="Newspaper3k failed to find content.")

                # Convert the main lxml node back to HTML
            main_content_html = lxml.etree.tostring(article.top_node, encoding='unicode')
            markdown = self.converter.handle(main_content_html)

            # --- MODIFICATION: Extract rich metadata ---
            metadata = {
                'title': article.title,
                'authors': article.authors,
                'publish_date': article.publish_date,
                'top_image': article.top_image,
                'movies': article.movies,
                'keywords': article.keywords,
                'summary': article.summary,
            }

            return ExtractionResult(
                markdown_content=sanitize_unicode_string(markdown),
                metadata=metadata
            )
        except Exception as e:
            error_str = f"Newspaper3k failed: {e}"
            self._log(f"[Error] {error_str}")
            self._log(traceback.format_exc())
            return ExtractionResult(error=error_str)


class GenericCSSExtractor(IExtractor):
    """
    Extractor implementation based on user-provided CSS selectors.
    (基于用户提供的CSS选择器的提取器实现。)

    This is a refactored version of your `html_content_converter` function.
    It requires 'selectors' to be passed in the `kwargs`.
    (这是您 `html_content_converter` 函数的重构版本。
     它要求在 `kwargs` 中传入 'selectors'。)
    """

    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.converter = html2text.HTML2Text()
        self.converter.ignore_links = False
        self.converter.ignore_images = False
        self.converter.body_width = 0

    def extract(self, content: bytes, url: str, **kwargs) -> ExtractionResult:
        """
        Extracts content using specific CSS selectors.

        :param content: Raw HTML bytes.
        :param url: The original URL (ignored).
        :param kwargs: Must contain:
                       - 'selectors' (str or List[str]): CSS selector(s) for target content.
                       - 'exclude_selectors' (Optional[List[str]]): CSS selector(s) to remove.
        :return: Markdown string.
        """
        self._log(f"Extracting with GenericCSSExtractor from {url}")

        # --- Get required arguments from kwargs ---
        selectors = kwargs.get('selectors')
        if isinstance(selectors, str):
            selectors = [selectors]
        elif not selectors:
            error_str = "GenericCSSExtractor requires 'selectors' argument in kwargs."
            self._log(f"[Error] {error_str}")
            return ExtractionResult(error=error_str)

        exclude_selectors = kwargs.get('exclude_selectors', [])
        if isinstance(exclude_selectors, str):
            exclude_selectors = [exclude_selectors]

        # --- This is your logic from html_content_converter ---
        try:
            html_str = content.decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html_str, 'html.parser')

            extracted_elements = []
            for selector in selectors:
                elements = soup.select(selector)
                for element in elements:
                    element_copy = copy.copy(element)  # Work on a copy

                    # Remove excluded elements
                    for ex_selector in exclude_selectors:
                        for unwanted in element_copy.select(ex_selector):
                            unwanted.decompose()

                    extracted_elements.append(element_copy)

            if not extracted_elements:
                self._log("[Info] No elements found for the given selectors.")
                return ""

            # Convert all found elements to Markdown and join them
            markdown_parts = [
                self.converter.handle(str(el)).strip()
                for el in extracted_elements
            ]
            full_markdown = '\n\n'.join(markdown_parts)

            return ExtractionResult(
                markdown_content=sanitize_unicode_string(full_markdown),
                metadata={'source': 'Generic CSS Selector'}
            )

        except Exception as e:
            error_str = f"GenericCSSExtractor failed: {e}"
            self._log(f"[Error] {error_str}")
            return ExtractionResult(error=error_str)


class Crawl4AIExtractor(IExtractor):
    """
    Extractor implementation using the 'crawl4ai' library.
    (使用 'crawl4ai' 库的提取器实现。)

    WARNING: This extractor IGNORES the pre-fetched `content`
    because crawl4ai must run its own browser instance to
    analyze the page for AI extraction. It will RE-FETCH the `url`.
    (警告：此提取器会忽略预先获取的 `content`，
     因为 crawl4ai 必须运行自己的浏览器实例来分析页面以进行AI提取。
     它将重新抓取 `url`。)
    """

    def __init__(self, model_name: str = 'gpt-3.5-turbo', verbose: bool = True):
        """
        :param model_name: The AI model to use (e.g., 'gpt-3.5-turbo', 'gpt-4o').
        """
        super().__init__(verbose)
        self.model_name = model_name

    def extract(self, content: bytes, url: str, **kwargs) -> ExtractionResult:
        """
        Extracts content using crawl4ai's SmartExtractor.

        :param content: IGNORED.
        :param url: The URL to crawl and extract from.
        :param kwargs: Passed to `SmartExtractor`.
                       Example: `extraction_prompt="Extract only user comments"`
        :return: Markdown string.
        """
        self._log(f"Extracting with Crawl4AIExtractor from {url}")
        if not Crawler or not SmartExtractor:
            error_str = "crawl4ai library not found."
            self._log(f"[Error] {error_str}")
            return ExtractionResult(error=error_str)

        self._log("[Warning] Crawl4AIExtractor ignores pre-fetched content and is re-fetching the URL.")

        try:
            extractor = SmartExtractor(
                model=self.model_name,
                **kwargs
            )

            crawler = Crawler(extractor=extractor)
            result = crawler.run(url)

            if result and (result.markdown or result.structured_data):
                # crawl4ai returns markdown AND structured_data (which is our metadata)
                metadata = result.structured_data or {}
                metadata['source'] = 'Crawl4AI'

                return ExtractionResult(
                    markdown_content=sanitize_unicode_string(result.markdown),
                    metadata=metadata
                )

            self._log("[Info] Crawl4AI ran but returned no markdown or data.")
            return ExtractionResult(error="Crawl4AI returned no content.")
        except Exception as e:
            error_str = f"Crawl4AIExtractor failed: {e}"
            self._log(f"[Error] {error_str}")
            self._log(traceback.format_exc())
            return ExtractionResult(error=error_str)


# --- Begin: 链接指纹和聚类的数据模型 ---

class LinkFingerprint(BaseModel):
    """
    Represents a single link and its structural context.
    (代表单个链接及其结构上下文。)
    """
    href: str = Field(description="完整的、绝对路径的URL")
    text: str = Field(description="链接的可见文本")
    signature: str = Field(description="该链接的结构指纹 (例如 'h2.title.post-title')")


class LinkGroup(BaseModel):
    """
    Represents a cluster of links sharing the same signature.
    (代表共享相同指纹的链接聚类。)
    """
    signature: str = Field(description="共享的结构指纹")
    count: int = Field(description="该指纹出现的次数")
    sample_links: List[LinkFingerprint] = Field(description="该组的链接示例 (最多5个)")


# --- End: 数据模型 ---


class ArticleListExtractor(IExtractor):
    """
    Extractor implementation for discovering and extracting article lists from a webpage.
    (用于发现和提取网页文章列表的提取器实现。)

    Implements the "link fingerprint" strategy.
    (实现了“链接指纹”策略。)
    """

    def __init__(self, verbose: bool = True, min_group_count: int = 3):
        """
        初始化列表提取器。
        :param min_group_count: 启发式猜测时，一个组至少需要多少个链接才被考虑。
        """
        super().__init__(verbose)
        self.min_group_count = min_group_count

    # --- 步骤 1: 生成链接指纹 ---

    def _get_structural_signature(self, tag: Tag) -> str:
        """
        Calculates the "structural signature" for an <a> tag based on its parent.
        (根据 <a> 标签的父节点计算其“结构指纹”。)

        The signature is `tag.class1.class2` of the immediate parent.
        (指纹是其直接父节点的 `标签名.class1.class2`。)
        """
        parent = tag.parent
        if not parent or parent.name == 'body':
            return 'body'

        name = parent.name
        classes = sorted(parent.get('class', []))

        if classes:
            return f"{name}.{'.'.join(classes)}"
        return name

    def _generate_fingerprints(self, soup: BeautifulSoup, base_url: str) -> List[LinkFingerprint]:
        """
        Step 1: Analyzes the DOM and generates a LinkFingerprint for every valid <a> tag.
        (步骤 1: 分析DOM，并为每个有效的 <a> 标签生成一个 LinkFingerprint。)
        """
        fingerprints = []
        seen_hrefs = set()

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()

            # 过滤无效链接
            if not href or href.startswith('#') or href.startswith('javascript:'):
                continue

            # 解析为绝对URL
            try:
                full_url = urljoin(base_url, href)
            except Exception:
                continue  # 忽略格式错误的URL

            # 过滤重复链接
            if full_url in seen_hrefs:
                continue
            seen_hrefs.add(full_url)

            text = a_tag.get_text(strip=True)
            signature = self._get_structural_signature(a_tag)

            fingerprints.append(LinkFingerprint(
                href=full_url,
                text=text,
                signature=signature
            ))

        return fingerprints

    # --- 步骤 2: 链接指纹聚类 ---

    def _cluster_fingerprints(self, fingerprints: List[LinkFingerprint]) -> List[LinkGroup]:
        """
        Step 2: Groups fingerprints by their signature.
        (步骤 2: 按指纹对链接进行分组。)
        """
        groups_map = defaultdict(list)
        for fp in fingerprints:
            groups_map[fp.signature].append(fp)

        link_groups = []
        for signature, fps_list in groups_map.items():
            link_groups.append(LinkGroup(
                signature=signature,
                count=len(fps_list),
                sample_links=fps_list[:5]  # 仅存储前5个作为示例
            ))

        # 按数量降序排序，最重要的组排在最前面
        link_groups.sort(key=lambda g: g.count, reverse=True)
        return link_groups

    # --- 步骤 3: 启发式猜测 ---

    def _guess_by_heuristics(self, groups: List[LinkGroup]) -> Optional[LinkGroup]:
        """
        Step 3: Guesses the main article list using a scoring-based heuristic.
        (步骤 3: 使用基于评分的启发式规则猜测主要文章列表。)

        This is the non-AI logic.
        (这是非AI逻辑。)
        """
        self._log("  [Heuristics] 启动启发式评分...", indent=1)
        best_group = None
        best_score = -99

        # 定义正面和负面信号词
        POSITIVE_SIG_KEYWORDS = ['article', 'post', 'item', 'entry', 'headline', 'title', 'feed', 'story']
        POSITIVE_TAG_KEYWORDS = ['h2', 'h3']
        NEGATIVE_SIG_KEYWORDS = [
            'nav', 'menu', 'header', 'head', 'foot', 'copyright', 'legal', 'privacy',
            'sidebar', 'aside', 'widget', 'ad', 'banner', 'comment', 'meta', 'tag', 'category'
        ]
        NEGATIVE_TEXT_KEYWORDS = ['关于我们', '联系我们', '首页', '隐私政策', 'home', 'about', 'contact', 'privacy']

        for group in groups:
            score = 0
            sig_lower = group.signature.lower()

            # 规则 1: 数量必须达标
            if group.count < self.min_group_count:
                self._log(f"    - [{sig_lower}]: 数量太少 ({group.count}), 跳过。", indent=1)
                continue

            score += group.count  # 数量越多，分数越高

            # 规则 2: 签名关键词
            if any(kw in sig_lower for kw in POSITIVE_SIG_KEYWORDS):
                score += 30
            if any(kw in sig_lower for kw in POSITIVE_TAG_KEYWORDS):
                score += 15
            if any(kw in sig_lower for kw in NEGATIVE_SIG_KEYWORDS):
                score -= 50

            # 规则 3: 样本链接文本
            if group.sample_links:
                avg_text_len = sum(len(fp.text) for fp in group.sample_links) / len(group.sample_links)

                # 标题通常不会太短
                if avg_text_len > 10:
                    score += 15
                if avg_text_len < 5:  # 可能是 "阅读更多" 或 "..."
                    score -= 10

                # 检查导航/页脚的常见文本
                sample_texts_lower = [fp.text.lower() for fp in group.sample_links]
                if any(kw in t for t in sample_texts_lower for kw in NEGATIVE_TEXT_KEYWORDS):
                    score -= 30

            self._log(f"    - [{sig_lower}]: 最终得分 {score}", indent=1)

            if score > best_score:
                best_score = score
                best_group = group

        if best_score <= 0:
            self._log("  [Heuristics] 启发式猜测失败：没有组的分数 > 0。", indent=1)
            return None

        self._log(f"  [Heuristics] 获胜者: {best_group.signature} (得分: {best_score})", indent=1)
        return best_group

    # --- 步骤 4: AI Prompt 准备 ---

    def _prepare_ai_prompt(self, groups: List[LinkGroup], page_title: str, page_url: str) -> str:
        """
        Step 4: Prepares the JSON payload and the system prompt for the AI.
        (步骤 4: 为AI准备JSON负载和系统提示。)
        """

        # 将Pydantic模型转换为字典
        groups_data = [g.model_dump() for g in groups]

        payload = {
            "page_url": page_url,
            "page_title": page_title,
            "link_groups": groups_data
        }

        json_payload = json.dumps(payload, indent=2, ensure_ascii=False)

        system_prompt = """你是一个专业的网页结构分析引擎。你的任务是分析一个JSON输入，该JSON代表了网页上所有链接的分组情况。你需要找出哪一个分组是该页面的**主要文章列表**。

**决策标准:**
1.  **排除导航和页脚：** 签名（signature）中包含`nav`, `menu`, `footer`, `copyright`的，或者链接文本（text）为“首页”、“关于我们”、“隐私政策”的，**不是**主列表。
2.  **排除侧边栏和部件：** 签名（signature）中包含`sidebar`, `widget`, `aside`, `ad`的，或者链接文本为“热门文章”、“标签云”的，**不是**主列表。
3.  **识别文章特征：**
    * `count`（数量）通常较高（例如 > 5）。
    * 链接文本（text）看起来像**文章标题**（例如：“xxx的评测”、“xxx宣布了新功能”）。
    * `href`（链接）看起来像**文章的永久链接**（例如：`/post/slug-name`或`/article/12345.html`），而不是分类链接（`/category/tech`）。
4.  **识别签名：** 主列表的签名通常是`article`, `post`, `item`, `entry`, `feed`或`h2`, `h3`等。

**任务:**
分析以下JSON数据，并**仅返回**你认为是**主要文章列表**的那个分组的`signature`字符串。如果找不到，请返回`null`。
"""

        full_prompt = f"{system_prompt}\n\n**输入数据:**\n```json\n{json_payload}\n```"
        return full_prompt

    # --- 辅助方法 ---

    def _find_group_by_signature(self, groups: List[LinkGroup], signature: str) -> Optional[LinkGroup]:
        """按签名查找已聚类的组。"""
        for group in groups:
            if group.signature == signature:
                return group
        return None

    def _extract_links_by_signature(self, soup: BeautifulSoup, signature: str, base_url: str) -> List[str]:
        """
        Using the winning signature, extract all corresponding links.
        (使用获胜的签名，提取所有对应的链接。)
        """
        self._log(f"  正在使用CSS选择器 '{signature}' 提取...", indent=2)
        parent_elements = soup.select(signature)

        final_links = []
        seen_hrefs = set()

        for parent in parent_elements:
            # 在父节点内查找第一个有效链接
            a_tag = parent.find('a', href=True)

            if a_tag:
                href = a_tag['href'].strip()
                if not href or href.startswith('#') or href.startswith('javascript:'):
                    continue

                try:
                    full_url = urljoin(base_url, href)
                    if full_url not in seen_hrefs:
                        final_links.append(full_url)
                        seen_hrefs.add(full_url)
                except Exception:
                    continue  # 忽略格式错误的URL

        return final_links

    # --- 主提取方法 ---

    def extract(self, content: bytes, url: str, **kwargs) -> ExtractionResult:
        """
        Orchestrates the entire list extraction process.
        (协调整个列表提取过程。)

        :param content: The raw HTML content as bytes. (原始HTML字节)
        :param url: The original URL. (原始URL)
        :param kwargs:
            - use_ai (bool): If True, triggers AI mode. (如果为True，触发AI模式)
            - ai_signature (str): The response from the AI (the winning signature). (AI的响应，即获胜的签名)
        :return: An ExtractionResult.
        """
        self.log_messages = []  # 重置日志
        self._log(f"开始列表提取: {url}")

        use_ai = kwargs.get('use_ai', False)
        ai_signature = kwargs.get('ai_signature', None)

        try:
            self._log("正在解析HTML (使用 lxml)...")
            soup = BeautifulSoup(content, 'lxml')
            page_title = soup.title.string.strip() if soup.title else ""

            # 步骤 1: 生成指纹
            self._log("步骤 1: 正在生成链接指纹...")
            fingerprints = self._generate_fingerprints(soup, url)
            if not fingerprints:
                return ExtractionResult(error="页面上未找到任何有效链接")
            self._log(f"  找到 {len(fingerprints)} 个有效链接。", indent=1)

            # 步骤 2: 聚类
            self._log("步骤 2: 正在聚类指纹...")
            groups = self._cluster_fingerprints(fingerprints)
            if not groups:
                return ExtractionResult(error="无法对链接进行聚类")
            self._log(f"  聚类为 {len(groups)} 个唯一的签名组。", indent=1)

            # --- 决策阶段 ---
            winning_group: Optional[LinkGroup] = None

            if use_ai:
                self._log("步骤 3: [AI 模式] 启动...")
                if ai_signature:
                    # AI模式 - 步骤 2: 接收到AI的签名
                    self._log(f"  接收到AI决策: '{ai_signature}'", indent=1)
                    winning_group = self._find_group_by_signature(groups, ai_signature)
                    if not winning_group:
                        return ExtractionResult(error=f"AI返回的签名 '{ai_signature}' 在聚类组中未找到。")
                else:
                    # AI模式 - 步骤 1: 生成Prompt
                    self._log("  未提供AI签名。正在生成Prompt...", indent=1)
                    prompt = self._prepare_ai_prompt(groups, page_title, url)
                    self._log("  已生成Prompt。请使用 metadata.ai_prompt 调用您的AI服务。", indent=1)
                    # 将groups也返回，以便AI调用失败时回退
                    groups_data = [g.model_dump() for g in groups]
                    return ExtractionResult(
                        markdown_content="# AI Prompt 已生成\n\n请查看 `metadata.ai_prompt` 字段，并使用AI服务获取 `signature`。",
                        metadata={
                            "title": "AI Prompt 请求",
                            "ai_prompt_required": True,
                            "ai_prompt": prompt,
                            "link_groups": groups_data
                        }
                    )
            else:
                # 启发式模式
                self._log("步骤 3: [启发式模式] 正在猜测主列表...")
                winning_group = self._guess_by_heuristics(groups)
                if not winning_group:
                    debug_info = "\n".join([f"  - {g.signature} (Count: {g.count})" for g in groups[:10]])
                    return ExtractionResult(error=f"启发式规则未能确定主列表。检测到的顶级组:\n{debug_info}")

            # --- 提取阶段 ---
            if not winning_group:
                # 这是一个理论上不应该发生的路径，但作为保险
                return ExtractionResult(error="未能确定获胜的链接组。")

            self._log(f"步骤 4: 获胜签名 '{winning_group.signature}' (数量: {winning_group.count})")
            self._log("步骤 5: 正在提取最终链接...")
            final_links = self._extract_links_by_signature(soup, winning_group.signature, url)

            if not final_links:
                return ExtractionResult(error=f"获胜签名 '{winning_group.signature}' 未能提取到任何链接。")

            self._log(f"  成功提取 {len(final_links)} 个链接。")

            # 构建 Markdown 输出
            md_content = f"# 提取的文章列表\n\n源: {url}\n签名: `{winning_group.signature}`\n\n"
            for link in final_links:
                # 尝试从指纹中找到原始文本，如果找不到就用URL作为文本
                link_text = next((fp.text for fp in winning_group.sample_links if fp.href == link), None)
                if not link_text or len(link_text) < 5:  # 如果文本太短或没有
                    md_content += f"- {link}\n"
                else:
                    md_content += f"- [{link_text}]({link})\n"

            metadata = {
                "title": f"文章列表: {page_title if page_title else url}",
                "source_url": url,
                "winning_signature": winning_group.signature,
                "extracted_links_count": len(final_links),
                "extracted_links": final_links,
                "all_groups": [g.model_dump() for g in groups]  # 包含所有组的调试信息
            }

            return ExtractionResult(markdown_content=md_content, metadata=metadata)

        except Exception as e:
            self._log(f"提取过程中发生严重错误: {e}")
            import traceback
            self._log(traceback.format_exc())
            return ExtractionResult(error=f"提取失败: {e}")
